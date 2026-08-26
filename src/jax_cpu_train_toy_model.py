"""Multi-Node JAX CPU Toy Model Training Script.

Demonstrates true multi-node distributed data-parallel (DDP) training on GKE CPUs:
1. Distributed cluster coordination via jax.distributed.initialize
2. Multi-device SPMD Mesh across nodes and virtual CPU devices
3. Replicated model parameters and sharded data batches
4. Cross-node gradient all-reduce and optimizer step (Flax + Optax)
5. End-to-end loss convergence verification
"""

import os
import sys
import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax.linen as nn
import optax


# ------------------------------------------------------------------------------
# 1. Model Definition (Lightweight 3-Layer MLP Classifier)
# ------------------------------------------------------------------------------
class ToyMLPClassifier(nn.Module):
    hidden_dim: int = 64
    num_classes: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, name="dense1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name="dense2")(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes, name="dense_out")(x)
        return x


# ------------------------------------------------------------------------------
# 2. Synthetic Distributed Dataset Generator
# ------------------------------------------------------------------------------
def generate_synthetic_shard(rank: int, num_samples: int = 1024, num_features: int = 16, num_classes: int = 4):
    """Generates a separable synthetic dataset shard for the local node."""
    np.random.seed(1337 + rank * 100)
    
    # Create distinct class centers
    centers = np.array([
        [2.0 if (i % num_classes == c) else -2.0 for i in range(num_features)]
        for c in range(num_classes)
    ], dtype=np.float32)

    labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int32)
    features = centers[labels] + np.random.normal(0.0, 0.5, size=(num_samples, num_features)).astype(np.float32)

    return features, labels


# ------------------------------------------------------------------------------
# 3. Main Training Execution
# ------------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("🚀 Initializing Multi-Node JAX CPU Toy Model Training...")
    print("=" * 75)

    coordinator_address = os.getenv("COORDINATOR_ADDRESS")
    num_processes = int(os.getenv("NUM_PROCESSES", "1"))
    process_id = int(os.getenv("PROCESS_ID", "0"))

    coordinator_bind_address = None
    if coordinator_address and ":" in coordinator_address:
        port = coordinator_address.split(":")[-1]
        if process_id == 0:
            coordinator_bind_address = f"0.0.0.0:{port}"

    if coordinator_address:
        print(f"Connecting to coordinator at {coordinator_address} (Rank {process_id}/{num_processes}, bind: {coordinator_bind_address})")
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            coordinator_bind_address=coordinator_bind_address,
            num_processes=num_processes,
            process_id=process_id
        )
    else:
        print("COORDINATOR_ADDRESS not set, attempting automatic JAX distributed initialization...")
        jax.distributed.initialize()

    rank = jax.process_index()
    total_ranks = jax.process_count()
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    total_devices = len(global_devices)

    print(f"\n[Rank {rank}/{total_ranks}] JAX Distributed Initialized Successfully!")
    print(f"[Rank {rank}] Local Virtual Devices ({len(local_devices)}): {local_devices}")
    print(f"[Rank {rank}] Total Global Devices in Cluster ({total_devices}): {global_devices}\n")

    # Hyperparameters
    num_features = 16
    num_classes = 4
    num_epochs = 10
    local_samples = 1024
    local_batch_size = 128
    global_batch_size = local_batch_size * total_ranks
    learning_rate = 0.02

    if rank == 0:
        print("-" * 75)
        print(f"📦 Training Configuration:")
        print(f"   - Global Cluster Nodes:     {total_ranks}")
        print(f"   - Total Virtual Devices:    {total_devices}")
        print(f"   - Local Batch per Host:     {local_batch_size}")
        print(f"   - Global Batch Size:        {global_batch_size}")
        print(f"   - Number of Epochs:         {num_epochs}")
        print(f"   - Learning Rate:            {learning_rate}")
        print("-" * 75)

    # 1. Setup SPMD Device Mesh
    mesh = Mesh(np.array(global_devices), axis_names=('data',))
    replicated_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P('data'))

    # 2. Initialize Model & Optimizer
    model = ToyMLPClassifier(hidden_dim=64, num_classes=num_classes)
    init_rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, num_features), dtype=jnp.float32)

    # Initialize weights identically on all hosts
    params = model.init(init_rng, dummy_input)['params']
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    # Replicate parameters across all devices
    with mesh:
        params = jax.device_put(params, replicated_sharding)
        opt_state = jax.device_put(opt_state, replicated_sharding)

    # 3. Generate Local Data Shards
    local_x, local_y = generate_synthetic_shard(rank, num_samples=local_samples, num_features=num_features, num_classes=num_classes)
    num_batches = local_samples // local_batch_size

    # 4. Define Sharded Loss & Train Step
    def loss_fn(p, batch_features, batch_targets):
        logits = model.apply({'params': p}, batch_features)
        one_hot = jax.nn.one_hot(batch_targets, num_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == batch_targets)
        return loss, acc

    @functools.partial(jax.jit, in_shardings=(replicated_sharding, replicated_sharding, data_sharding, data_sharding),
                                out_shardings=(replicated_sharding, replicated_sharding, replicated_sharding, replicated_sharding))
    def train_step(p, opt_s, batch_features, batch_targets):
        (loss_val, acc_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch_features, batch_targets)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss_val, acc_val

    # 5. Training Loop
    if rank == 0:
        print("\n🏁 Starting Distributed Multi-Node Training Loop...\n")

    epoch_losses = []
    epoch_accs = []

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        step_losses = []
        step_accs = []

        # Shuffle local shard each epoch
        perm = np.random.permutation(local_samples)
        shuffled_x = local_x[perm]
        shuffled_y = local_y[perm]

        for step in range(num_batches):
            b_start = step * local_batch_size
            b_end = b_start + local_batch_size

            local_bx = shuffled_x[b_start:b_end]
            local_by = shuffled_y[b_start:b_end]

            # Shard the host-local arrays into global SPMD arrays across all nodes
            global_bx = multihost_utils.host_local_array_to_global_array(local_bx, mesh, P('data', None))
            global_by = multihost_utils.host_local_array_to_global_array(local_by, mesh, P('data'))

            with mesh:
                params, opt_state, loss_val, acc_val = train_step(params, opt_state, global_bx, global_by)
                loss_val.block_until_ready()

            step_losses.append(float(loss_val))
            step_accs.append(float(acc_val))

        epoch_duration = time.time() - epoch_start_time
        avg_loss = float(np.mean(step_losses))
        avg_acc = float(np.mean(step_accs)) * 100.0
        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)

        if rank == 0:
            print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:5.1f}% | Time: {epoch_duration*1000:6.1f}ms")

    # 6. Convergence & Multi-Node Proof Verification
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    final_acc = epoch_accs[-1]
    loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100.0

    if rank == 0:
        print("\n" + "=" * 75)
        print("📊 Training Results Summary:")
        print(f"   - Initial Loss (Epoch 1):  {initial_loss:.4f}")
        print(f"   - Final Loss (Epoch {num_epochs}):   {final_loss:.4f}")
        print(f"   - Loss Reduction:          {loss_reduction_pct:.1f}%")
        print(f"   - Final Cluster Accuracy:  {final_acc:.1f}%")
        print("=" * 75)

        if final_loss < 0.20 and loss_reduction_pct > 70.0:
            print("✅ MATHEMATICAL CONVERGENCE VERIFIED!")
            print("   Distributed all-reduce backpropagation successfully trained the model across all nodes.")
        else:
            print("❌ CONVERGENCE FAILED: Loss did not decrease as expected.")

    # Global barrier sync before exiting
    multihost_utils.sync_global_devices("toy_model_training_complete")
    print(f"[Rank {rank}] Multi-Node JAX CPU Model Training Finished Successfully!\n")


if __name__ == "__main__":
    main()

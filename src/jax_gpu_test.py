import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

def main():
    print("=" * 70)
    print("Initializing Multi-Node JAX GPU Cluster...")
    print("=" * 70)

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

    print(f"\n[Rank {rank}/{total_ranks}] JAX Distributed Initialized Successfully!")
    print(f"[Rank {rank}] Local GPU Devices ({len(local_devices)}): {local_devices}")
    print(f"[Rank {rank}] Total Global Devices in Cluster ({len(global_devices)}): {global_devices}\n")

    # Step 1: Mathematical Proof via psum
    expected_sum = (total_ranks * (total_ranks + 1)) / 2.0
    rank_value = float(rank + 1)

    print(f"[Rank {rank}] Input Rank Value: {rank_value} | Expected Cluster Sum: {expected_sum}")

    devices_array = np.array(jax.devices())
    num_nodes = total_ranks
    devices_per_node = len(local_devices)

    device_mesh = devices_array.reshape((num_nodes, devices_per_node))
    mesh = Mesh(device_mesh, axis_names=('data', 'model'))

    local_input = np.full((1, devices_per_node), rank_value, dtype=np.float32)
    sharded_x = multihost_utils.host_local_array_to_global_array(
        local_input,
        mesh,
        P('data', 'model')
    )

    @jax.jit
    def compute_allreduce(x):
        return jnp.sum(x, axis=0)

    print(f"[Rank {rank}] Executing JAX GPU SPMD gradient all-reduce over NCCL...")
    with mesh:
        synced_sum = compute_allreduce(sharded_x)
        synced_sum.block_until_ready()

    actual_sum = float(synced_sum.addressable_data(0)[0])
    print(f"[Rank {rank}] Synchronized all-reduce Output: {actual_sum} (Expected: {expected_sum})")

    if abs(actual_sum - expected_sum) < 1e-3:
        print(f"[Rank {rank}] ✅ MATHEMATICAL VERIFICATION PASSED! GPU NCCL All-Reduce verified.")
    else:
        print(f"[Rank {rank}] ❌ VERIFICATION FAILED! Expected {expected_sum}, got {actual_sum}")

    # Barrier sync
    multihost_utils.sync_global_devices("gpu_training_complete")
    print(f"\n[Rank {rank}] Multi-Node JAX GPU Test Completed Successfully!\n")

if __name__ == "__main__":
    main()

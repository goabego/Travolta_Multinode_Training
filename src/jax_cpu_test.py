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
    print("Initializing Multi-Node JAX CPU Cluster...")
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
    print(f"[Rank {rank}] Local Virtual CPU Devices ({len(local_devices)}): {local_devices}")
    print(f"[Rank {rank}] Total Global Devices in Cluster ({len(global_devices)}): {global_devices}\n")

    # Step 1: Mathematical Proof via pmean / psum
    # Rank i passes (i + 1). Total expected sum across N ranks = N * (N + 1) / 2
    expected_sum = (total_ranks * (total_ranks + 1)) / 2.0
    rank_value = float(rank + 1)

    print(f"[Rank {rank}] Input Rank Value: {rank_value} | Expected Cluster Sum: {expected_sum}")

    # Set up Mesh across all processes and local devices
    devices_array = np.array(jax.devices())
    num_nodes = total_ranks
    devices_per_node = len(local_devices)

    device_mesh = devices_array.reshape((num_nodes, devices_per_node))
    mesh = Mesh(device_mesh, axis_names=('data', 'model'))

    # Construct host-local array of shape (1, devices_per_node)
    local_input = np.full((1, devices_per_node), rank_value, dtype=np.float32)
    sharded_x = multihost_utils.host_local_array_to_global_array(
        local_input,
        mesh,
        P('data', 'model')
    )

    @jax.jit
    def compute_allreduce(x):
        # Synchronize and sum values along axis 0 ('data' axis across all nodes)
        return jnp.sum(x, axis=0)

    print(f"[Rank {rank}] Executing JAX SPMD cross-node all-reduce sum...")
    with mesh:
        synced_sum = compute_allreduce(sharded_x)
        synced_sum.block_until_ready()

    actual_sum = float(synced_sum.addressable_data(0)[0])
    print(f"[Rank {rank}] Synchronized all-reduce Output: {actual_sum} (Expected: {expected_sum})")

    if abs(actual_sum - expected_sum) < 1e-3:
        print(f"[Rank {rank}] ✅ MATHEMATICAL VERIFICATION PASSED! Multi-node JAX coordination verified.")
    else:
        print(f"[Rank {rank}] ❌ VERIFICATION FAILED! Expected {expected_sum}, got {actual_sum}")

    # Barrier sync
    multihost_utils.sync_global_devices("cpu_training_complete")
    print(f"\n[Rank {rank}] Multi-Node JAX CPU Test Completed Successfully!\n")

if __name__ == "__main__":
    main()

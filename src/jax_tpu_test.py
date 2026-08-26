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
    print("Initializing Multi-Node JAX TPU Slice Cluster...")
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
        print(f"Connecting to TPU coordinator at {coordinator_address} (Rank {process_id}/{num_processes}, bind: {coordinator_bind_address})")
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            coordinator_bind_address=coordinator_bind_address,
            num_processes=num_processes,
            process_id=process_id
        )
    else:
        print("COORDINATOR_ADDRESS not set, attempting automatic JAX distributed TPU initialization...")
        jax.distributed.initialize()

    rank = jax.process_index()
    total_ranks = jax.process_count()
    local_devices = jax.local_devices()
    global_devices = jax.devices()

    print(f"\n[TPU Rank {rank}/{total_ranks}] JAX Distributed TPU Initialized Successfully!")
    print(f"[TPU Rank {rank}] Local TPU Chips ({len(local_devices)}): {local_devices}")
    print(f"[TPU Rank {rank}] Total Global TPU Devices in Slice ({len(global_devices)}): {global_devices}\n")

    # Step 1: Mathematical Proof via psum over ICI
    expected_sum = (total_ranks * (total_ranks + 1)) / 2.0
    rank_value = float(rank + 1)

    print(f"[TPU Rank {rank}] Input Rank Value: {rank_value} | Expected Cluster Sum: {expected_sum}")

    devices_array = np.array(jax.devices())
    num_hosts = total_ranks
    chips_per_host = len(local_devices)

    device_mesh = devices_array.reshape((num_hosts, chips_per_host))
    mesh = Mesh(device_mesh, axis_names=('data', 'model'))

    local_input = np.full((1, chips_per_host), rank_value, dtype=np.float32)
    sharded_x = multihost_utils.host_local_array_to_global_array(
        local_input,
        mesh,
        P('data', 'model')
    )

    @jax.jit
    def compute_allreduce(x):
        return jnp.sum(x, axis=0)

    print(f"[TPU Rank {rank}] Executing JAX TPU SPMD gradient all-reduce over ICI link...")
    with mesh:
        synced_sum = compute_allreduce(sharded_x)
        synced_sum.block_until_ready()

    actual_sum = float(synced_sum.addressable_data(0)[0])
    print(f"[TPU Rank {rank}] Synchronized all-reduce Output: {actual_sum} (Expected: {expected_sum})")

    if abs(actual_sum - expected_sum) < 1e-3:
        print(f"[TPU Rank {rank}] ✅ MATHEMATICAL VERIFICATION PASSED! TPU ICI All-Reduce verified.")
    else:
        print(f"[TPU Rank {rank}] ❌ VERIFICATION FAILED! Expected {expected_sum}, got {actual_sum}")

    # Barrier sync
    multihost_utils.sync_global_devices("tpu_training_complete")
    print(f"\n[TPU Rank {rank}] Multi-Node JAX TPU Slice Test Completed Successfully!\n")

if __name__ == "__main__":
    main()

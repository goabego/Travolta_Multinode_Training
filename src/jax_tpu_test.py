import os
import sys
import time
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

def main():
    print("=" * 70)
    print("Initializing Multi-Node JAX TPU Slice Cluster...")
    print("=" * 70)

    coordinator_address = os.getenv("COORDINATOR_ADDRESS")
    num_processes = int(os.getenv("NUM_PROCESSES", "1"))
    process_id = int(os.getenv("PROCESS_ID", "0"))

    if coordinator_address:
        print(f"Connecting to TPU coordinator at {coordinator_address} (Rank {process_id}/{num_processes})")
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
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
    rank_value = jnp.array([float(rank + 1)], dtype=jnp.float32)

    print(f"[TPU Rank {rank}] Input Rank Value: {rank_value[0]} | Expected Cluster Sum: {expected_sum}")

    devices_array = jax.devices()
    num_hosts = total_ranks
    chips_per_host = len(local_devices)

    device_mesh = devices_array.reshape((num_hosts, chips_per_host))
    mesh = Mesh(device_mesh, axis_names=('data', 'model'))

    @jax.jit
    def compute_allreduce(x):
        return jax.lax.psum(x, axis_name='data')

    print(f"[TPU Rank {rank}] Executing JAX TPU lax.psum gradient all-reduce over ICI link...")
    with mesh:
        sharded_x = jax.device_put(rank_value, NamedSharding(mesh, P('data')))
        synced_sum = compute_allreduce(sharded_x)
        synced_sum.block_until_ready()

    actual_sum = float(synced_sum[0])
    print(f"[TPU Rank {rank}] Synchronized psum Output: {actual_sum} (Expected: {expected_sum})")

    if abs(actual_sum - expected_sum) < 1e-3:
        print(f"[TPU Rank {rank}] ✅ MATHEMATICAL VERIFICATION PASSED! TPU ICI All-Reduce verified.")
    else:
        print(f"[TPU Rank {rank}] ❌ VERIFICATION FAILED! Expected {expected_sum}, got {actual_sum}")

    # Barrier sync
    multihost_utils.sync_global_devices("tpu_training_complete")
    print(f"\n[TPU Rank {rank}] Multi-Node JAX TPU Slice Test Completed Successfully!\n")

if __name__ == "__main__":
    main()

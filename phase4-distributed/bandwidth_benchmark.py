# bandwidth_benchmark.py
# Exercise 4.3 — Measure real NVLink AllReduce bandwidth
#
# What this measures:
#   - Achieved AllReduce bandwidth at different tensor sizes (1MB → 2GB)
#   - Compared against A100 SXM NVLink spec: 600 GB/s bidirectional
#
# Bus bandwidth formula (industry standard, matches NCCL benchmark):
#   bandwidth = 2 × (world_size-1)/world_size × tensor_size_bytes / time
#
# The 2 × (N-1)/N factor accounts for both phases of ring AllReduce:
#   - Each element crosses the ring once in Reduce-Scatter
#   - Each element crosses the ring once in AllGather
#   At N=4: factor = 2 × 3/4 = 1.5

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# A100 SXM NVLink bidirectional spec — what we're comparing against
NVLINK_PEAK_GBS = 600.0


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def benchmark_allreduce(rank, world_size, size_bytes, n_iters=50):
    """
    Measure AllReduce bandwidth for a tensor of size_bytes.

    Args:
        size_bytes: tensor size in bytes
        n_iters: number of timed iterations (more = more stable measurement)

    Returns:
        (mean_bandwidth_gbs, mean_time_ms) — only meaningful on rank 0
    """

    # Allocate tensor — float32 = 4 bytes per element
    n_elements = size_bytes // 4
    tensor = torch.randn(n_elements, device=rank)

    # Warmup: run a few iterations before timing
    # GPU has startup overhead on first kernel launch — warmup amortizes this
    # Without warmup, first iteration skews the average significantly
    WARMUP = 10
    for _ in range(WARMUP):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()  # wait for all warmup kernels to complete

    # Timed iterations
    # cuda.Event is more accurate than time.perf_counter for GPU timing
    # perf_counter measures wall clock; cuda.Event measures GPU execution time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    # Barrier: ensure all ranks start timing simultaneously
    dist.barrier()

    start_event.record()
    for _ in range(n_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_event.record()

    # synchronize: block CPU until GPU finishes all operations
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds
    elapsed_ms = start_event.elapsed_time(end_event) / n_iters

    # Bus bandwidth formula
    # 2 × (N-1)/N accounts for both Reduce-Scatter and AllGather phases
    # At N=4: 2 × 0.75 = 1.5 — each byte effectively travels 1.5x across the ring
    bus_bw_gbs = (2 * (world_size - 1) / world_size * size_bytes / 1e9) / (elapsed_ms / 1e3)

    return bus_bw_gbs, elapsed_ms


def run(rank, world_size):
    setup(rank, world_size)

    # Test across 5 orders of magnitude of tensor size
    # 1MB → 2GB — covers latency-dominated to bandwidth-saturated regimes
    test_sizes = {
        "1 MB":    1   * 1024**2,
        "16 MB":   16  * 1024**2,
        "64 MB":   64  * 1024**2,
        "256 MB":  256 * 1024**2,
        "512 MB":  512 * 1024**2,
        "1 GB":    1   * 1024**3,
        "2 GB":    2   * 1024**3,
    }

    if rank == 0:
        print(f"\n{'Size':>10} | {'Time (ms)':>10} | {'BW (GB/s)':>10} | {'% of Peak':>10}")
        print("-" * 50)

    for label, size_bytes in test_sizes.items():
        bw, elapsed_ms = benchmark_allreduce(rank, world_size, size_bytes)

        # Only rank 0 prints — all ranks measure the same bandwidth
        # since AllReduce is a collective (all ranks participate equally)
        if rank == 0:
            pct_peak = (bw / NVLINK_PEAK_GBS) * 100
            print(f"{label:>10} | {elapsed_ms:>10.3f} | {bw:>10.1f} | {pct_peak:>9.1f}%")

    if rank == 0:
        print(f"\nReference: A100 SXM NVLink peak = {NVLINK_PEAK_GBS} GB/s")

    cleanup()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

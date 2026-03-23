import torch
import time

def benchmark(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters

print("=" * 60)
print("RTX 4090 theoretical peaks:")
print("  FP32 compute:   ~82.6 TFLOPS")
print("  Memory BW:      ~1008 GB/s")
print("  Ridge point:    ~82 FLOP/byte")
print("=" * 60)

# MEMORY-BOUND: Element-wise multiply
size = 1 << 25  # 32M elements
a = torch.randn(size, device='cuda')
b = torch.randn(size, device='cuda')

elapsed = benchmark(lambda: a * b)

bytes_transferred = 3 * size * 4
flops = size
achieved_bw_gb = bytes_transferred / elapsed / 1e9
arithmetic_intensity = flops / bytes_transferred

print(f"\n[MEMORY-BOUND] Element-wise multiply, {size/1e6:.0f}M elements")
print(f"  Elapsed:              {elapsed*1000:.3f} ms")
print(f"  Achieved bandwidth:   {achieved_bw_gb:.1f} GB/s  (peak: ~1008 GB/s)")
print(f"  % of peak BW:         {achieved_bw_gb/1008*100:.1f}%")
print(f"  Arithmetic intensity: {arithmetic_intensity:.3f} FLOP/byte")
print(f"  -> This is memory-bound (intensity << ridge point of ~82)")

# COMPUTE-BOUND: Matrix multiply
n = 8192
A = torch.randn(n, n, device='cuda', dtype=torch.float16)
B = torch.randn(n, n, device='cuda', dtype=torch.float16)

elapsed = benchmark(lambda: torch.matmul(A, B))

flops_matmul = 2 * n**3
bytes_matmul = 3 * n**2 * 2
arithmetic_intensity_mm = flops_matmul / bytes_matmul
achieved_tflops = flops_matmul / elapsed / 1e12
achieved_pct = achieved_tflops / 330 * 100

print(f"\n[COMPUTE-BOUND] FP16 matmul, N={n}")
print(f"  Elapsed:              {elapsed*1000:.3f} ms")
print(f"  Achieved TFLOPS:      {achieved_tflops:.1f}  (FP16 tensor core peak: ~330)")
print(f"  % of peak compute:    {achieved_pct:.1f}%")
print(f"  Arithmetic intensity: {arithmetic_intensity_mm:.0f} FLOP/byte")
print(f"  -> This is compute-bound (intensity >> ridge point of ~82)")

# TRANSITION: where does matmul flip?
print(f"\n[TRANSITION ANALYSIS] Arithmetic intensity vs. matrix size (FP16):")
print(f"  {'N':>6}  {'Intensity (FLOP/byte)':>22}  {'Regime':>15}")
for n_test in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    flops_t = 2 * n_test**3
    bytes_t = 3 * n_test**2 * 2
    intensity = float(flops_t) / float(bytes_t)
    regime = "COMPUTE-BOUND" if intensity > 82 else "memory-bound"
    marker = " <- flip point" if 60 < intensity < 120 else ""
    print(f"  {n_test:>6}  {intensity:>22.1f}  {regime:>15}{marker}")

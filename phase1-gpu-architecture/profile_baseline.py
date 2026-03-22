import torch
import time

# Allocate on GPU
a = torch.randn(4096, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')

# Warm up — first kernel launch has JIT overhead, don't time it
_ = torch.matmul(a, b)
torch.cuda.synchronize()

# Time it
start = time.perf_counter()
for _ in range(100):
    c = torch.matmul(a, b)
torch.cuda.synchronize()  # Wait for GPU to finish before stopping the clock
elapsed = (time.perf_counter() - start) / 100

# Calculate TFLOPS
# Matrix multiply A(m,k) @ B(k,n) = 2*m*n*k FLOPs
flops = 2 * 4096**3
tflops = flops / elapsed / 1e12
print(f"Matrix size:  4096 x 4096")
print(f"Elapsed:      {elapsed*1000:.2f} ms per matmul")
print(f"Throughput:   {tflops:.2f} TFLOPS")
print(f"\nReference peaks:")
print(f"  RTX 3090 FP32: ~35.6 TFLOPS")
print(f"  RTX 4090 FP32: ~82.6 TFLOPS")
print(f"  RTX 3090 TF32: ~142 TFLOPS (with tensor cores)")
print(f"\nYou're hitting: {tflops:.1f} TFLOPS")

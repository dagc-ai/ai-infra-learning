import torch
import triton
import triton.language as tl
import time

# ── Triton kernel ──────────────────────────────────────────────────────────────
# Unlike CUDA where one thread = one element,
# one Triton "program" = one BLOCK of elements.
# The compiler handles the thread-level mapping internally.

@triton.jit
def add_kernel(
    x_ptr,                     # pointer to first input vector
    y_ptr,                     # pointer to second input vector
    output_ptr,                # pointer to output vector
    n_elements,                # total number of elements
    BLOCK_SIZE: tl.constexpr,  # elements per program (compile-time constant)
):
    # Which block am I? (analogous to blockIdx.x in CUDA)
    pid = tl.program_id(axis=0)

    # Compute the range of indices this block is responsible for
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Bounds check: last block may extend past end of array
    mask = offsets < n_elements

    # Load from HBM — masked load returns 0 for out-of-bounds
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    # Store result back to HBM
    tl.store(output_ptr + offsets, output, mask=mask)


# ── Python wrapper ─────────────────────────────────────────────────────────────
def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    # Grid = number of program instances to launch
    # Each handles BLOCK_SIZE elements → ceil(n_elements / BLOCK_SIZE) programs
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# ── Correctness check ──────────────────────────────────────────────────────────
print("=== Correctness Check ===")
torch.manual_seed(0)
size = 98432  # not a power of 2 — tests the mask logic
x = torch.rand(size, device='cuda', dtype=torch.float32)
y = torch.rand(size, device='cuda', dtype=torch.float32)

triton_output = triton_add(x, y)
torch_output = x + y

max_diff = torch.max(torch.abs(triton_output - torch_output)).item()
print(f"Max absolute difference vs torch: {max_diff:.2e}")
print(f"allclose: {torch.allclose(triton_output, torch_output)}")


# ── Benchmark ─────────────────────────────────────────────────────────────────
print("\n=== Bandwidth Benchmark ===")

def benchmark(fn, label, n_elements):
    # Warmup
    for _ in range(25):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000

    # 2 reads + 1 write = 3 * n_elements * 4 bytes
    bytes_moved = 3 * n_elements * 4
    bandwidth_gb = bytes_moved / (elapsed_ms / 1000) / 1e9
    print(f"{label:20s} | {elapsed_ms*1000:.1f} us | {bandwidth_gb:.1f} GB/s")

print(f"{'kernel':20s} | {'latency':>10} | {'bandwidth':>12}")
print("-" * 48)

for n in [1<<20, 1<<22, 1<<24]:  # 1M, 4M, 16M elements
    x_b = torch.rand(n, device='cuda', dtype=torch.float32)
    y_b = torch.rand(n, device='cuda', dtype=torch.float32)
    print(f"\n-- n = {n:,} elements --")
    benchmark(lambda: triton_add(x_b, y_b), "Triton", n)
    benchmark(lambda: x_b + y_b,            "PyTorch", n)

print(f"\nRTX 4090 peak GDDR6X bandwidth: ~1,008 GB/s")

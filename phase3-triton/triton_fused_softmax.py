# =============================================================================
# Exercise 3.2 — Fused Softmax in Triton
# =============================================================================
# WHAT THIS FILE DEMONSTRATES:
# Kernel fusion: doing multiple logical operations in a single kernel to
# eliminate intermediate HBM round trips. Naive softmax touches HBM 3x.
# Fused softmax touches HBM once. The difference is pure memory bandwidth.
#
# This is the direct precursor to Flash Attention, which applies the same
# principle to the entire attention computation.
# =============================================================================

import torch
import triton
import triton.language as tl
import time


# =============================================================================
# SECTION 1: THE FUSED TRITON SOFTMAX KERNEL
# =============================================================================
# One program instance handles one entire row of the input matrix.
# All intermediate values (row max, exp, sum) live in registers.
# Only two HBM accesses: one read, one write.
#
# Contrast with naive PyTorch softmax:
#   HBM read  → compute max     → HBM write  (kernel 1)
#   HBM read  → compute exp     → HBM write  (kernel 2)
#   HBM read  → compute sum+div → HBM write  (kernel 3)
#   Total: 3 reads + 3 writes = 6 HBM transactions
#
# Fused kernel:
#   HBM read  → max+exp+sum+div → HBM write
#   Total: 1 read + 1 write = 2 HBM transactions
# =============================================================================

@triton.jit
def softmax_kernel(
    output_ptr,          # pointer to output matrix in HBM
    input_ptr,           # pointer to input matrix in HBM
    input_row_stride,    # how many elements to skip to get to the next row
    output_row_stride,   # same for output
    n_cols,              # number of columns (elements per row)
    BLOCK_SIZE: tl.constexpr,  # must be >= n_cols, power of 2
):
    # -------------------------------------------------------------------------
    # STEP 1: Identify which row this program instance owns
    # -------------------------------------------------------------------------
    # One program instance = one row.
    # If the matrix is [1024 rows x 512 cols], we launch 1024 program instances.
    # pid=0 handles row 0, pid=1 handles row 1, etc.
    row_idx = tl.program_id(0)

    # Compute pointer to the start of this row in HBM
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # -------------------------------------------------------------------------
    # STEP 2: Load the entire row into registers
    # -------------------------------------------------------------------------
    # col_offsets: [0, 1, 2, ..., BLOCK_SIZE-1]
    # mask: True for valid columns, False for padding beyond n_cols
    # This is the ONE read from HBM for this entire kernel.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row — out-of-bounds positions get -inf so they don't affect max/sum
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # -------------------------------------------------------------------------
    # STEP 3: Numerically stable softmax — all in registers, no HBM access
    # -------------------------------------------------------------------------
    # Why subtract the max? exp() overflows for large inputs.
    # softmax(x) = softmax(x - max) mathematically — subtracting a constant
    # from all elements doesn't change the output.
    # This is the same numerical stability trick from your Phase 2 softmax kernel.

    # Find the maximum value in this row (reduction across the block)
    row_max = tl.max(row, axis=0)

    # Subtract max and compute exp — still in registers
    numerator = tl.exp(row - row_max)

    # Sum the exp values (reduction across the block)
    denominator = tl.sum(numerator, axis=0)

    # Normalize — this is the actual softmax output
    softmax_output = numerator / denominator

    # -------------------------------------------------------------------------
    # STEP 4: Write result back to HBM
    # -------------------------------------------------------------------------
    # This is the ONE write to HBM for this entire kernel.
    # Everything between the load and store lived in registers.
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


# =============================================================================
# SECTION 2: PYTHON WRAPPER
# =============================================================================

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    # BLOCK_SIZE must be a power of 2 and >= n_cols
    # triton.next_power_of_2 rounds up to nearest power of 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # One program instance per row
    grid = (n_rows,)

    output = torch.empty_like(x)

    softmax_kernel[grid](
        output,
        x,
        x.stride(0),       # input_row_stride: elements between row starts
        output.stride(0),  # output_row_stride
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# =============================================================================
# SECTION 3: CORRECTNESS CHECK
# =============================================================================

print("=== Correctness Check ===")
torch.manual_seed(0)
x = torch.randn(1024, 512, device='cuda', dtype=torch.float32)

triton_output = triton_softmax(x)
torch_output = torch.softmax(x, dim=1)

max_diff = torch.max(torch.abs(triton_output - torch_output)).item()
print(f"Max absolute difference vs torch: {max_diff:.2e}")
print(f"allclose: {torch.allclose(triton_output, torch_output, atol=1e-5)}")


# =============================================================================
# SECTION 4: BENCHMARK — FUSED vs UNFUSED
# =============================================================================
# Unfused baseline: naive PyTorch softmax written as separate operations
# to make the HBM round trips explicit. This is what happens without fusion.
#
# Note: torch.softmax() is already fused internally — we write the unfused
# version manually to make the comparison honest.
# =============================================================================

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    # Each of these lines is a separate kernel launch — separate HBM round trip
    x_max = x.max(dim=1, keepdim=True).values   # read x, write max
    z = x - x_max                                # read x+max, write z
    numerator = torch.exp(z)                     # read z, write numerator
    denominator = numerator.sum(dim=1, keepdim=True)  # read numerator, write denom
    return numerator / denominator               # read num+denom, write output
    # Total: ~5 separate HBM round trips


def benchmark(fn, label, n_rows, n_cols):
    for _ in range(25):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000

    # Fused: 1 read + 1 write = 2 * n_rows * n_cols * 4 bytes
    # We use fused bandwidth formula for both — shows how much unfused wastes
    bytes_moved = 2 * n_rows * n_cols * 4
    bandwidth_gb = bytes_moved / (elapsed_ms / 1000) / 1e9
    print(f"{label:20s} | {elapsed_ms*1000:.1f} us | {bandwidth_gb:.1f} GB/s (effective)")


print("\n=== Bandwidth Benchmark: Fused vs Unfused Softmax ===")
print(f"{'kernel':20s} | {'latency':>10} | {'bandwidth':>20}")
print("-" * 58)

configs = [
    (1024, 512),    # small rows
    (1024, 2048),   # medium rows
    (1024, 8192),   # large rows — approaches attention head sizes
]

for n_rows, n_cols in configs:
    x_b = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
    print(f"\n-- {n_rows} rows x {n_cols} cols --")
    benchmark(lambda: triton_softmax(x_b),  "Triton fused",   n_rows, n_cols)
    benchmark(lambda: naive_softmax(x_b),   "PyTorch unfused", n_rows, n_cols)
    benchmark(lambda: torch.softmax(x_b, dim=1), "PyTorch softmax", n_rows, n_cols)

print(f"\nRTX 4090 peak GDDR6X bandwidth: ~1,008 GB/s")
print(f"Unfused effective bandwidth is low because it moves more data than reported.")
print(f"Triton fused bandwidth should be highest — fewest HBM transactions.")

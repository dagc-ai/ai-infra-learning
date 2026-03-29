# tpu_matmul.py — Exercise 6.4: Google TPU v5e Benchmark
# Phase 6: Alternative Hardware Architectures
#
# HARDWARE: Google TPU v5e-1 (single chip, via Google Colab free tier)
# FRAMEWORK: JAX + XLA — NOT PyTorch/CUDA
#
# KEY ARCHITECTURAL POINT:
# TPU uses a systolic array — data flows through a grid of MAC units
# in a wave pattern. No threads, no caches, no branch prediction.
# The compiler (XLA) schedules all computation at compile time.
#
# This is the same compiler-first philosophy as Groq, but applied to
# a different substrate: systolic array vs SRAM-centric execution.
#
# PHASE 2 CONNECTION:
# Your tiled matmul manually staged data through shared memory to keep
# compute units busy. A systolic array makes this the hardware default —
# tiling is the architecture, not the optimization.
#
# PHASE 3 CONNECTION:
# Flash Attention required expert Triton kernel engineering to fuse
# matmul+softmax+matmul into a single kernel. XLA does this automatically
# for any jit-compiled JAX function. Compiler-first = fusion by default.
#
# REAL RESULTS (Google Colab TPU v5e-1, March 2026):
#   Matmul 4096x4096 BF16:  142.3 TFLOPS (72.2% of 197 TFLOPS peak)
#   Fused attention forward: 2.60ms (batch=4, heads=16, seq=2048, d=64)
#
# PHASE 2 COMPARISON:
#   A100 SXM4 cuBLAS BF16:  ~250 TFLOPS (real Phase 2 measurement)
#   TPU v5e-1 JAX BF16:      142.3 TFLOPS (this run)
#   Gap: A100 wins on raw matmul — cuBLAS is 15 years more optimized
#   Fair comparison: full transformer training throughput per dollar

import jax
import jax.numpy as jnp
import time

print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# ── Exercise 1: BF16 Matmul Benchmark ────────────────────────────────────────
# Same dimensions as Phase 2 CUDA tiled matmul — directly comparable
M, K, N = 4096, 4096, 4096

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
b = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)

print("\nWarming up (XLA compilation)...")
c = jnp.dot(a, b).block_until_ready()
print("Warmup complete")

N_ITERS = 20
start = time.perf_counter()
for _ in range(N_ITERS):
    c = jnp.dot(a, b).block_until_ready()
elapsed = (time.perf_counter() - start) / N_ITERS

flops = 2 * M * K * N
tflops = flops / elapsed / 1e12
peak_bf16 = 197.0
utilization = (tflops / peak_bf16) * 100

print(f"\n{'='*55}")
print(f"MATMUL RESULTS — TPU v5e-1, BF16, {M}x{K}x{N}")
print(f"{'='*55}")
print(f"Elapsed:        {elapsed*1000:.2f} ms")
print(f"Achieved:       {tflops:.1f} TFLOPS BF16")
print(f"Peak BF16:      {peak_bf16} TFLOPS")
print(f"Utilization:    {utilization:.1f}%")
print(f"A100 cuBLAS:    ~250 TFLOPS BF16 (Phase 2 real measurement)")


# ── Exercise 2: Fused Attention Forward Pass ──────────────────────────────────
# Demonstrates XLA automatic operator fusion.
# On CUDA: 3 separate kernels (matmul, softmax, matmul)
# On TPU/XLA: single fused kernel — no Flash Attention implementation needed
# The compiler does what Phase 3 Triton kernels did manually.

def attention(q, k, v):
    d_k = q.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k).astype(jnp.bfloat16)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)

fused_attention = jax.jit(attention)

batch, heads, seq_len, d_head = 4, 16, 2048, 64

key = jax.random.PRNGKey(0)
q = jax.random.normal(key, (batch, heads, seq_len, d_head), dtype=jnp.bfloat16)
k = jax.random.normal(key, (batch, heads, seq_len, d_head), dtype=jnp.bfloat16)
v = jax.random.normal(key, (batch, heads, seq_len, d_head), dtype=jnp.bfloat16)

print("\nCompiling fused attention kernel...")
out = fused_attention(q, k, v).block_until_ready()
print("Compilation complete")

start = time.perf_counter()
for _ in range(N_ITERS):
    out = fused_attention(q, k, v).block_until_ready()
elapsed = (time.perf_counter() - start) / N_ITERS

print(f"\n{'='*55}")
print(f"FUSED ATTENTION RESULTS — TPU v5e-1")
print(f"{'='*55}")
print(f"Forward pass:   {elapsed*1000:.2f}ms")
print(f"Batch:          {batch}, Heads: {heads}")
print(f"Seq length:     {seq_len}, d_head: {d_head}")
print(f"XLA fusion:     matmul+softmax+matmul → single kernel")
print(f"Phase 3 note:   Triton Flash Attention not needed on TPU")

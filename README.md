# AI Infrastructure Learning Journey

A bottom-up, hands-on journey through the full AI compute stack — from GPU
architecture and kernel programming to distributed training and inference serving.

Every concept has an exercise. Every exercise produces a number. Every number
is committed here.

---

## Progress

| Phase | Topic | Status | Key Result |
|-------|-------|--------|------------|
| 0 | Environment & Tooling | ✅ Complete | RTX 4090 verified, 99.7% peak FP32 utilization on first benchmark |
| 1 | GPU Architecture & Memory Hierarchy | ✅ Complete | Roofline model measured: 947 GB/s bandwidth, 158 TFLOPS compute, ridge at ~327 FLOP/byte |
| 2 | CUDA Kernel Programming | ✅ Complete | Tiled matmul 7.2 TFLOPS vs cuBLAS 54 TFLOPS; 8x HBM traffic reduction confirmed via ncu |
| 3 | Triton Kernel Programming | ✅ Complete | 92% peak bandwidth, 2x fusion win, Flash Attention 6.4x speedup at N=4096 |
| 4 | Distributed Training Primitives | 🔜 Next | DDP, Ring AllReduce, interconnect benchmarking |
| 5 | Inference & Serving Infrastructure | ⬜ Pending | vLLM, quantization, KV cache |
| 6 | Alternative Hardware Architectures | ⬜ Pending | Tenstorrent, AMD ROCm, TPU concepts |
| 7 | Model Architecture & Full Stack View | ⬜ Pending | Transformer from scratch, scaling laws |

---

## Phase 0 — Environment & Tooling

**Hardware:** RTX 4090, 49GB VRAM, CUDA 13.1, PyTorch 2.10, vast.ai ($0.57/hr)

### Key Results

Environment verified via four gates: `nvidia-smi` (driver 580.95.05), `nvcc --version`
(CUDA 13.1), `torch.cuda.is_available()` → True, `torch.__version__` → 2.10.0a.

Baseline matmul benchmark (4096 × 4096, FP32, 100-iteration average):

| Metric | Result |
|--------|--------|
| Elapsed per call | 1.67ms |
| Achieved throughput | 82.4 TFLOPS |
| RTX 4090 FP32 peak | 82.6 TFLOPS |
| Peak utilization | 99.7% |

### What This Means

A GPU is rented hardware you access over the internet and immediately put to work
running math at extraordinary scale. The 82.4 TFLOPS result — 99.7% of theoretical
peak — is achievable because `torch.matmul` calls NVIDIA's hand-tuned cuBLAS library,
not naive code. That ceiling is the baseline everything else in this curriculum gets
measured against. Every kernel written by hand starts below it; the work of optimization
is understanding why and closing the gap.

### Key Insight

`torch.cuda.synchronize()` is the first thing that separates someone who understands
GPU execution from someone who doesn't. GPU operations are asynchronous — without an
explicit sync barrier, you're timing kernel launches (microseconds), not kernel
execution (milliseconds). Every benchmarking script in this repo includes it.

---

## Phase 1 — GPU Architecture & Memory Hierarchy

**Hardware:** RTX 4090, 24GB GDDR6X, CUDA 13.0, vast.ai VM (~$0.40/hr)

### Key Results

**Memory-bound operation** — element-wise multiply, 32M FP32 elements:

| Metric | Result |
|--------|--------|
| Achieved bandwidth | 947.5 GB/s |
| Peak bandwidth | 1,008 GB/s |
| Utilization | 94.0% |
| DRAM throughput (ncu) | 92% |
| SM compute throughput (ncu) | 2.6% |
| Arithmetic intensity | 0.083 FLOP/byte |

**Compute-bound operation** — FP16 matmul, N=8192:

| Metric | Result |
|--------|--------|
| Achieved throughput | 158.1 TFLOPS |
| Peak throughput | 330 TFLOPS |
| Utilization | 47.9% |
| DRAM throughput (ncu) | 13% |
| SM compute throughput (ncu) | 47% |
| Arithmetic intensity | 2,731 FLOP/byte |

**Roofline regime transition (FP16 matmul):**

| Matrix size N | Arithmetic intensity | Regime |
|---------------|---------------------|--------|
| 64 | 21.3 FLOP/byte | Memory-bound |
| 128 | 42.7 FLOP/byte | Memory-bound |
| 256 | 85.3 FLOP/byte | Approaching ridge |
| 8192 | 2,731 FLOP/byte | Compute-bound |

Ridge point on RTX 4090: ~327 FLOP/byte (1,008 GB/s bandwidth ÷ 330 TFLOPS peak × 1000)

### What This Means

Every GPU operation is bottlenecked by one of two things: how fast it can move data
from memory, or how fast it can do arithmetic. Which one limits you depends on the
ratio of math to data movement — arithmetic intensity. We measured this directly:
element-wise multiply spent 92% of its time moving data and 2.6% computing. Large
matmul flipped that entirely. Below the ridge point (~N=256 for matmul), buying a
faster GPU does nothing. You need faster memory or smarter data reuse.

### Key Insight

Arithmetic intensity determines optimization strategy. Kernel fusion and quantization
help memory-bound workloads. More TFLOPS only helps compute-bound ones. Knowing which
regime your workload is in — a 60-second calculation from the spec sheet — should
precede every infrastructure decision.

---

## Phase 2 — CUDA Kernel Programming

**Hardware:** RTX 4090, 49,140MB VRAM, CUDA 12.1, vast.ai VM

### Key Results

**Exercise 2.1 — Vector Addition**

- 1M elements across 4,096 blocks × 256 threads
- Global load bytes (ncu): 8.39MB — exactly 2 arrays × 1M × 4 bytes, zero waste
- Arithmetic intensity: 0.125 FLOP/byte — most memory-bound operation measured across Phases 1–2
- Correctness: h_c[0] = 3.0, h_c[1048575] = 3.0, no CUDA errors

**Exercise 2.2 — Naive vs. Tiled Matmul vs. cuBLAS**

Wall clock (20-iteration average, CUDA events):

| Version | N=1024 | N=2048 | N=4096 | % of cuBLAS |
|---------|--------|--------|--------|-------------|
| Naive | 0.43ms / 5.0 TFLOPS | 3.37ms / 5.1 TFLOPS | — (skipped) | ~9% |
| Tiled (T=16) | 0.33ms / 6.4 TFLOPS | 2.61ms / 6.6 TFLOPS | 19.17ms / 7.2 TFLOPS | ~13% |
| cuBLAS | 0.04ms / 48.7 TFLOPS | 0.32ms / 54.5 TFLOPS | 2.55ms / 53.9 TFLOPS | 100% |

ncu hardware counters (N=1024, single kernel instance):

| Metric | Naive | Tiled (T=16) |
|--------|-------|--------------|
| Global load bytes (HBM) | 4.29 GB | 537 MB |
| Shared memory wavefronts | 0 | 50,339,258 |
| DRAM throughput % of peak | 1.70% | 2.20% |
| SM throughput % of peak | 97.28% | 94.25% |

Tiling reduced HBM traffic 8x — exactly what the arithmetic predicts.

Naive matmul skipped at N=4096: all three matrices total 192MB, blowing past the 72MB
L2 cache. Performance would have collapsed. At N=1024 (12MB total, fits in L2), naive
measured 5.0 TFLOPS — L2 masking the expected collapse, which ncu confirmed via the
8x HBM traffic difference between naive and tiled.

**Exercise 2.3 — Softmax with Warp Shuffle**

- 4,096 rows × 32 columns, one warp (32 threads) per row
- Global load bytes (ncu): 1.57MB — input read exactly once
- Correctness verified: all rows sum to 1.000000
- Warp shuffle reduction: 5 steps of `__shfl_down_sync`, no shared memory, no barriers
- ~1 cycle per step vs. ~5 cycles for shared memory equivalent

### What This Means

A GPU is not a fast calculator — it is a memory system with compute attached. A
straightforwardly written matmul used 0.17% of available compute because the same
data was repeatedly fetched from slow off-chip memory by independent threads. Tiling
fixed this by loading data into fast on-chip SRAM once and reusing it many times,
cutting HBM traffic 8x. The remaining gap to cuBLAS is decades of additional
engineering: three tiling levels instead of one, Tensor Core integration, and
assembly-level inner loop optimization. The warp shuffle exercise showed that even
cross-thread coordination (finding a row maximum) can execute entirely in registers
in 5 instructions — the same primitive that makes Flash Attention's fused kernel
feasible.

### Key Insight

The L2 cache masking effect is a production trap. A kernel benchmarking acceptably at
small problem sizes can collapse at production scale when the working set exceeds L2.
Any kernel evaluation that doesn't profile at actual production problem sizes is
incomplete. SM throughput is not the same as compute utilization — naive kernels
hammering HBM with load instructions register high SM activity while compute units
starve. Wall clock is the honest arbiter.

---

## Phase 3 — Triton Kernel Programming

**Hardware:** RTX 4090, 24GB GDDR6X, RunPod persistent pod, CUDA 12.4, PyTorch 2.4.1, Triton 3.0.0

### Key Results

**Exercise 3.1 — Triton Vector Addition**

- Test size: 98,432 elements (non-power-of-2, stress-tests mask logic)
- Correctness: max absolute difference vs PyTorch = 0.00e+00
- At 16M elements (only size exceeding 72MB L2, measuring real GDDR6X traffic):

| Implementation | Bandwidth | % of Peak |
|---------------|-----------|-----------|
| Triton | 924.0 GB/s | 91.7% |
| PyTorch | 924.5 GB/s | 91.8% |
| RTX 4090 peak | 1,008 GB/s | 100% |

Statistically identical. Triton's compiler produces code on par with hand-tuned CUDA
with no manual thread indexing.

**Exercise 3.2 — Fused Softmax**

Benchmark at 1024 rows × 2048 cols:

| Implementation | Bandwidth | Latency | Notes |
|---------------|-----------|---------|-------|
| Triton fused | 823 GB/s | 20.4 µs | 1 HBM round trip |
| PyTorch unfused (5-op) | 414 GB/s | 40.4 µs | 5 HBM round trips |
| torch.softmax (internally fused) | 1,208 GB/s | 13.9 µs | L2 cache effects at this size |

- Correctness: max absolute difference vs PyTorch = 7.45e-09, allclose = True
- Fusion win: 2x throughput vs explicit unfused implementation
- Unfused bandwidth plateaus at ~411–414 GB/s across all sizes — signature of multiple HBM passes

**Exercise 3.3 — Flash Attention Forward Pass**

- Correctness: max absolute difference vs naive = 1.14e-03 (float32 accumulation across tile rescaling — not a logic bug)
- Benchmark (batch=2, heads=4, d_head=64, float32):

| N | Flash | Naive | Speedup | Naive attn matrix |
|---|-------|-------|---------|-------------------|
| 512 | 0.041ms | 0.072ms | 1.8x | 8.4 MB |
| 1024 | 0.044ms | 0.176ms | 4.0x | 33.6 MB |
| 2048 | 0.155ms | 0.969ms | 6.3x | 134.2 MB |
| 4096 | 0.590ms | 3.750ms | 6.4x | 536.9 MB |

Speedup compounds with N — consistent with O(N) vs O(N²) prediction. Every doubling
of N: Flash cost ~doubles, naive cost ~quadruples.

### What This Means

Triton proved three things experimentally: a Python-level GPU compiler can match
hand-tuned CUDA performance (92% of peak bandwidth with no manual thread management);
kernel fusion cuts memory traffic in half and directly doubles throughput; and Flash
Attention's breakthrough wasn't a faster chip — it was a mathematical restructuring
of attention to avoid writing a giant intermediate matrix to HBM at all. That
restructuring is what made 100K-token context windows physically possible. The
algorithm unlocked the hardware market for long-context inference — not the other
way around.

### Key Insight

Context length is not a free parameter. Without fused tiled attention, memory
requirements scale as O(N²). With it, O(N). At N=4096, naive attention requires
537MB just for the attention matrix. At N=32K, that number exceeds the GPU entirely.
Evaluating any inference framework requires confirming Flash Attention (or equivalent)
is the default attention path, not an optional flag.

---

## Notes

- [GPU Architecture Primer](notes/gpu-architecture-primer.md)
- [Flash Attention Explained](notes/flash-attention-explained.md)

---

## Hardware

| Phase | Hardware | Provider | Cost |
|-------|----------|----------|------|
| 0 | RTX 4090, 49GB VRAM, CUDA 13.1 | vast.ai | $0.57/hr |
| 1 | RTX 4090, 24GB GDDR6X, CUDA 13.0 | vast.ai VM | ~$0.40/hr |
| 2 | RTX 4090, 49GB VRAM, CUDA 12.1 | vast.ai VM | — |
| 3 | RTX 4090, 24GB GDDR6X, CUDA 12.4 | RunPod persistent pod | — |

All RTX 4090 specs: 16,384 CUDA cores, 1,008 GB/s GDDR6X bandwidth, 82.6 TFLOPS FP32,
330 TFLOPS FP16 tensor core peak.

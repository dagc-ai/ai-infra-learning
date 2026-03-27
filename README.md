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
| 4 | Distributed Training Primitives | ✅ Complete | 228 GB/s measured vs 600 GB/s NVLink spec; Ring AllReduce from scratch; 38% of spec in virtualized env |
| 5 | Inference & Serving Infrastructure | ✅ Complete | AWQ Marlin 218.9 tok/s (2.27x BF16); 8x concurrency = 8x throughput flat latency; one flag = 10x perf delta |
| 6 | Alternative Hardware Architectures | 🔜 Next | Tenstorrent, AMD ROCm, TPU concepts |
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

## Phase 4 — Distributed Training Primitives

**Hardware:** 4× A100 SXM 40GB, NVLink NV12 topology, CUDA 12.4.1, PyTorch 2.4.0, RunPod

### Key Results

**Exercise 4.1 — DDP Training**

- 4-process DDP training loop across 4× A100 SXM
- Initial param sum: `20.335490` — identical across all 4 ranks (broadcast at init confirmed)
- Final param sum: `20.307633` — identical across all 4 ranks after 20 steps on different random data (AllReduce confirmed working)
- Step timing: 0.21s (rank 0) to 0.26s (rank 3) — variance demonstrates the synchronization barrier in practice: fastest rank waits for slowest before any rank proceeds

**Exercise 4.2 — Ring AllReduce from Scratch**

- Implemented full Ring AllReduce using only `dist.P2POp` point-to-point primitives — no `dist.all_reduce()`
- Result matched PyTorch `dist.all_reduce()` to `9.54e-07` max absolute difference across all 4 ranks
- Hit deadlock on first implementation using naive `isend` + `recv` — all ranks blocked simultaneously because NCCL `isend` doesn't move data until a matching `recv` is posted on the remote end
- Fixed with `dist.batch_isend_irecv`, which posts sends and receives atomically

**Exercise 4.3 — NVLink Bandwidth Benchmark**

| Tensor size | Time (ms) | Measured BW | % of 600 GB/s spec |
|-------------|-----------|-------------|---------------------|
| 1 MB | 0.051 | 31.0 GB/s | 5.2% |
| 16 MB | 0.202 | 124.3 GB/s | 20.7% |
| 64 MB | 0.596 | 168.8 GB/s | 28.1% |
| 256 MB | 2.092 | 192.5 GB/s | 32.1% |
| 512 MB | 3.817 | 211.0 GB/s | 35.2% |
| 1 GB | 7.244 | 222.3 GB/s | 37.1% |
| 2 GB | 14.152 | 227.6 GB/s | 37.9% |

Peak measured: 228 GB/s — 38% of 600 GB/s A100 SXM NVLink spec. Curve still rising at
2GB. Gap vs. spec attributed to RunPod virtualization overhead, AllReduce synchronization
cost, and NCCL algorithm overhead vs. raw point-to-point benchmark conditions. Bus
bandwidth formula: `2 × (N-1)/N × tensor_size / time` — matches NCCL benchmark
methodology.

### What This Means

When training across multiple GPUs, each GPU sees different data and computes its own
gradient. Before any weight update can happen, all gradients must be averaged across
every GPU so the model stays synchronized. That averaging — AllReduce — runs over the
physical wire connecting GPUs on every single training step. We implemented that
algorithm from scratch, proved it produces the correct result, then measured how fast
the wire actually runs versus its rated spec. The gap between 228 GB/s measured and
600 GB/s rated is not a malfunction — it's what real-world overhead looks like.
Understanding that gap is what separates people who read spec sheets from people who
understand systems.

### Key Insight

Interconnect topology is a first-class architectural decision, not a procurement
detail. PCIe between GPUs delivers ~32 GB/s per link; NVLink delivers 600 GB/s. At
real gradient tensor sizes, PCIe puts you in a regime where communication dominates
compute — and that is not fixable in software. GPU utilization is also the wrong
metric for distributed training: a cluster showing 80% utilization may be spending
40% of that time blocked at AllReduce barriers. The correct metric is Model FLOP
Utilization (MFU). Most production training runs achieve 30–50% MFU. NVLink is
NVIDIA's most durable competitive moat — more defensible than the GPU itself.

## Phase 5 — Inference & Serving Infrastructure

**Hardware:** A100 SXM4 80GB, CUDA 12.4, PyTorch 2.4.0, vLLM 0.18.0, RunPod  
**Model:** Mistral-7B-Instruct-v0.3 (BF16) and Mistral-7B-Instruct-v0.2-AWQ (INT4)

### Key Results

**Exercise 5.1 — vLLM Deployment & Continuous Batching**

vLLM startup telemetry (Mistral 7B BF16):

| Metric | Value |
|--------|-------|
| Model loaded | 13.51 GiB in 5.5 seconds |
| KV cache pool allocated | 57.41 GiB (90% GPU memory utilization) |
| KV cache capacity | 470,288 tokens total |
| Max concurrency at 8,192 token context | 57x |

Single request baseline:

| Metric | Value |
|--------|-------|
| Prompt tokens | 13 |
| Completion tokens | 200 |
| Total time | 2,073ms |
| Throughput | 96.5 tok/s |
| Implied HBM bandwidth utilization | ~1.4 TB/s (~70% of A100 2.0 TB/s peak) |

Concurrent request scaling:

| Concurrency | Throughput | Latency |
|-------------|------------|---------|
| 1 | 94.7 tok/s | 2,110ms |
| 2 | 193.0 tok/s | 2,066ms |
| 4 | 382.7 tok/s | 2,083ms |
| 8 | 752.3 tok/s | 2,110ms |

8x requests produced 8x throughput with flat latency — continuous batching
absorbing concurrent load without queuing penalty.

**Exercise 5.2 — Quantization Benchmarks**

| Format | Model size | Throughput | vs BF16 | Kernel |
|--------|------------|------------|---------|--------|
| BF16 | 13.51 GiB | 96.5 tok/s | 1.0x | FlashAttention v2 |
| AWQ INT4 naive | 3.88 GiB | 20.9 tok/s | 0.2x | naive AWQ |
| AWQ INT4 Marlin | 3.88 GiB | 218.9 tok/s | 2.27x | fused Marlin |

Error encountered and documented: initial AWQ run used `--quantization awq` instead
of `--quantization awq_marlin`. vLLM warned at startup that Marlin was available but
not selected. Result was 20.9 tok/s — 5x slower than BF16. Switching to
`--quantization awq_marlin` recovered full performance (218.9 tok/s). One flag,
10x difference.

AWQ Marlin memory impact: KV cache available 66.92 GiB vs 57.41 GiB for BF16 — 9GB
freed by weight compression, increasing max concurrency from 57x to 67x.

**Exercise 5.3 — Speculative Decoding**

Draft model: nickypro/tinyllama-15M (30.4 MB, LLaMA architecture, Mistral-compatible
tokenizer vocabulary). Target: Mistral 7B AWQ Marlin.

Full configuration comparison:

| Configuration | Throughput | vs BF16 |
|---------------|------------|---------|
| BF16 baseline | 96.5 tok/s | 1.0x |
| AWQ INT4 naive | 20.9 tok/s | 0.2x |
| AWQ INT4 Marlin | 218.9 tok/s | 2.27x |
| AWQ Marlin + speculative (mismatched draft) | 67.5 tok/s | 0.7x |

Result: 67.5 tok/s — 3x slower than AWQ Marlin alone, 30% slower than BF16 baseline.
Cause: TinyLlama shares Mistral's tokenizer vocabulary but was trained independently.
Target model rejection rate was high — paid the cost of running both models with the
benefit of neither. Shared tokenizer vocabulary is necessary but not sufficient for
speculative decoding. Training dynamics must also align.

Note: a correctly matched draft model (Mistral 1B/2B from the same training family)
was not run — Mistral gated models required license approval pending at time of
exercise. The mismatched-draft result is intentional and instructive.

### What This Means

Inference is not a compute problem — it is a memory problem. The A100 has 312 TFLOPS
of compute sitting mostly idle during token generation. What limits speed is how fast
the model's weights can be loaded from HBM on every single decode step. Every
optimization in this phase — batching, quantization, speculative decoding — is a
different strategy for getting more useful work out of each memory access. The same
principle that explained tiled matmul in Phase 2 and Flash Attention in Phase 3
applies here: minimize expensive data movement, maximize work per memory round trip.

The non-obvious finding: optimization method and optimization implementation are
different things. AWQ INT4 with the wrong kernel was 5x slower than BF16. AWQ INT4
with the right kernel was 2.27x faster. Same weights, same precision, one flag.
Speculative decoding with the wrong draft model was 3x slower than no speculative
decoding. The technique is not the result — the implementation is.

### Key Insight

The optimization sequence matters. Fix continuous batching first (free, 8x throughput
gain at concurrency 8). Then quantization kernel (one flag, 10x difference). Then
quantization level (2.27x gain). Then speculative decoding — and only with a draft
model from the same training family as the target. A self-hosted deployment running
naive AWQ at low batch utilization will cost more than a managed API at the same
volume. The gap between a misconfigured and well-configured inference stack on
identical hardware exceeded 10x in measured results.

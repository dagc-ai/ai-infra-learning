# GPU Architecture Primer
*Phase 1 — ai-infra-learning. Written to explain GPU architecture to a technical audience.*

---

## The Execution Model

A GPU is fundamentally a **latency-hiding machine** — not a low-latency machine. CPUs achieve performance through deep pipelines, branch prediction, and out-of-order execution to minimize the latency of any single operation. GPUs instead run thousands of threads simultaneously, and when one group stalls waiting for memory, the scheduler immediately runs another group.

**The hierarchy (small → large):**

| Unit | What it is | Shares |
|------|-----------|--------|
| Thread | Single program instance | Own registers |
| Warp | 32 threads executing in lockstep | Instruction stream |
| Thread Block | Group of warps on one SM | Shared memory (SRAM) |
| Grid | All blocks for one kernel launch | L2 cache, HBM |

**Key constraint:** Threads within a block can communicate via shared memory and synchronize with `__syncthreads()`. Threads in different blocks cannot communicate during execution. This shapes how algorithms must be structured for the GPU.

**Warp divergence:** If threads within a warp take different code paths (an if/else branch), both paths execute and non-participating threads are masked off. Throughput drops proportionally to how divergent the execution is. For regular ML workloads (attention, dense layers), divergence is minimal. For irregular data structures (graphs, sparse ops), divergence is a serious concern — and an architectural opening that alternative designs try to exploit.

---

## Memory Hierarchy

*Numbers for RTX 4090 (Ada Lovelace). H100 SXM numbers noted separately.*

| Level | Latency | Capacity | Bandwidth | Scope |
|-------|---------|----------|-----------|-------|
| Registers | ~1 cycle | ~256 KB/SM | — | Per thread |
| Shared Memory (SRAM) | ~5 cycles | ~228 KB/SM | ~19 TB/s per SM | Per block |
| L1 Cache | ~20 cycles | ~128 KB/SM | shared with SRAM | Per SM |
| L2 Cache | ~50 cycles | ~72 MB total | ~7 TB/s | All SMs |
| GDDR6X (VRAM) | ~500 cycles | 24 GB | **~1008 GB/s** | All SMs |

*H100 comparison: 80 GB HBM3, ~3.35 TB/s bandwidth — 3.3× the 4090's memory bandwidth.*

**The key ratio:** Shared memory is ~100× faster than VRAM. Any algorithm that reads the same data multiple times should stage it in shared memory first — pay the VRAM cost once, amortize it across many uses. This is the foundation of tiled matrix multiply (Phase 2) and Flash Attention (Phase 3).

---

## Bandwidth-Bound vs. Compute-Bound

Every GPU operation has an **arithmetic intensity** — the ratio of floating-point operations to bytes read from memory.
```
Arithmetic Intensity (FLOP/byte) = Total FLOPs / Total bytes from memory
```

The GPU has two performance ceilings:
- **Compute:** ~82.6 TFLOPS FP32, ~330 TFLOPS FP16 (tensor cores)
- **Memory bandwidth:** ~1008 GB/s (GDDR6X)

The **ridge point** (where they intersect) ≈ **327 FLOP/byte** for FP16 on the RTX 4090.

- If intensity < ridge point → **memory-bound**. Adding more TFLOPS does nothing. You need more bandwidth, or restructured code that reuses data (tiling, kernel fusion).
- If intensity > ridge point → **compute-bound**. The memory system keeps up. More TFLOPS → better performance.

### Real measurements (RTX 4090, March 2026)

**Element-wise multiply (32M elements, FP32):**
- Achieved bandwidth: 947.5 GB/s (94.0% of peak)
- Arithmetic intensity: 0.083 FLOP/byte
- ncu DRAM throughput: 92% | ncu SM compute: 2.6%
- Regime: **memory-bound**

**FP16 matmul (N=8192):**
- Achieved TFLOPS: 158.1 (47.9% of FP16 tensor core peak)
- Arithmetic intensity: 2731 FLOP/byte
- ncu DRAM throughput: 13% | ncu SM compute: 47%
- Regime: **compute-bound**

**Transition point:** FP16 matmul flips from memory-bound to compute-bound at **N=256**, where arithmetic intensity (~85 FLOP/byte) crosses the ridge point.

### Transition table (FP16 matmul)

| N | Arithmetic Intensity (FLOP/byte) | Regime |
|---|----------------------------------|--------|
| 64 | 21.3 | memory-bound |
| 128 | 42.7 | memory-bound |
| 256 | 85.3 | **compute-bound** ← flip point |
| 512 | 170.7 | compute-bound |
| 1024 | 341.3 | compute-bound |
| 2048 | 682.7 | compute-bound |
| 4096 | 1365.3 | compute-bound |
| 8192 | 2730.7 | compute-bound |

**Practical implication:** Batch size 1 inference on a large model is almost entirely memory-bound. Weight matrices are large but you process one token at a time — the effective matrix dimension is tiny. You move gigabytes of weights through the memory bus to do a trivially small amount of arithmetic on each one. This is why inference at batch size 1 gets 5–10% of theoretical TFLOPS utilization regardless of chip power.

---

## Why Memory Coalescing Matters

Threads in a warp access memory together. If 32 threads each access consecutive addresses (elements 0–31 of an array), the hardware satisfies all 32 accesses in one memory transaction. If those 32 threads each access scattered addresses (every 64th element), you need 32 separate transactions — 32× the memory traffic for the same data.

Strided or random memory access patterns can reduce effective bandwidth by 10–32× versus the peak spec. Access pattern regularity often matters as much as raw bandwidth numbers when evaluating real workload performance.

---

## ncu Profiler — Key Metrics

Two metrics that tell the full story for any kernel:

| Metric | High means | Low means |
|--------|-----------|-----------|
| `gpu__dram_throughput` | Memory-bound — VRAM bus saturated | Compute-bound or poor occupancy |
| `sm__throughput` | Compute-bound — tensor cores busy | Memory-bound or poor occupancy |

If **both are low**, you have an occupancy or launch overhead problem — not enough parallel work to keep the hardware fed.

---

## The SRAM vs. HBM Architecture Tradeoff

The GPU memory hierarchy reveals a fundamental design tension that different accelerator architectures resolve differently.

**The NVIDIA GPU approach (HBM-centric):** Large off-chip memory (HBM) provides high capacity and reasonable bandwidth. SRAM is present but small relative to total memory. Works well for large models, large batch sizes, and workloads where capacity matters more than latency.

**The SRAM-heavy approach** (seen in dataflow architectures like Tenstorrent's Tensix, and in the scratchpad designs of other custom accelerators): More SRAM per compute unit, smaller or no HBM. The bet is that for inference-dominated workloads with high data reuse — transformer layers, attention — keeping data on-chip eliminates the 100× latency penalty of going off-chip entirely. Wins on efficiency and latency for workloads that fit. Loses on capacity for workloads that don't.

**The in-memory compute approach** (seen in designs like d-Matrix): Move computation closer to where data is stored, rather than moving data to computation. Directly attacks the bandwidth wall by reducing the distance data travels.

**The deterministic latency approach** (seen in Groq's LPU): Eliminate the memory hierarchy variability entirely. Predictable execution time on every token, every time. Trades peak throughput flexibility for latency guarantees.

Understanding where a workload sits on the roofline — and how much data reuse it has — is the starting point for reasoning about which architectural approach fits it best. A workload that is heavily memory-bound with high data reuse is a candidate for SRAM-heavy designs. A workload requiring large model capacity with variable batch sizes favors HBM-centric designs.

---

*Phase 1 complete. Phase 2 builds on this with hand-written CUDA kernels — naive matmul then tiled matmul — directly observing the speedup from explicit SRAM staging.*

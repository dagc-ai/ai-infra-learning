# AI Infrastructure Learning Journey

A bottom-up, hands-on journey through the full AI compute stack — from GPU
architecture and kernel programming to distributed training and inference serving.

Every concept has an exercise. Every exercise produces a number. Every number
is committed here.

---

## Progress

| Phase | Topic | Status | Key Result |
|-------|-------|--------|------------|
| 0 | Environment & Tooling | ✅ Complete | GPU instance, repo, nvitop |
| 1 | GPU Architecture & Memory Hierarchy | ✅ Complete | Roofline model, memory-bound vs compute-bound |
| 2 | CUDA Kernel Programming | ✅ Complete | Tiled matmul 7 TFLOPS, cuBLAS 54 TFLOPS, warp shuffle softmax |
| 3 | Triton Kernel Programming | ✅ Complete | 92% peak bandwidth, 2x fusion win, Flash Attention 6.4x at N=4096 |
| 4 | Distributed Training Primitives | 🔜 Next | DDP, Ring AllReduce, interconnect benchmarking |
| 5 | Inference & Serving Infrastructure | ⬜ Pending | vLLM, quantization, KV cache |
| 6 | Alternative Hardware Architectures | ⬜ Pending | Tenstorrent, AMD ROCm, TPU concepts |
| 7 | Model Architecture & Full Stack View | ⬜ Pending | Transformer from scratch, scaling laws |

---

## Phase 3 Results — Triton Kernel Programming

### Exercise 3.1 — Vector Addition
- Triton kernel matches PyTorch hand-tuned CUDA at 92% of peak GDDR6X bandwidth
- 924 GB/s achieved vs 1,008 GB/s theoretical peak on RTX 4090
- Key insight: Triton's block-level programming model lets the compiler guarantee
  memory coalescing — no manual thread indexing required

### Exercise 3.2 — Fused Softmax
- Fused kernel: 1 HBM round trip. Unfused: 5 HBM round trips.
- 2x throughput advantage at practical sizes (1024 rows × 2048 cols)
- Key insight: fusion eliminates intermediate writes to HBM — the only
  way to win on memory-bandwidth-bound operations

### Exercise 3.3 — Flash Attention Forward Pass
- Implemented online softmax tiling from scratch
- O(N) memory complexity vs O(N²) for naive attention — confirmed empirically

| N    | Flash    | Naive    | Speedup | Naive attn matrix |
|------|----------|----------|---------|-------------------|
| 512  | 0.041ms  | 0.072ms  | 1.8x    | 8.4 MB            |
| 1024 | 0.044ms  | 0.176ms  | 4.0x    | 33.6 MB           |
| 2048 | 0.155ms  | 0.969ms  | 6.3x    | 134.2 MB          |
| 4096 | 0.590ms  | 3.750ms  | 6.4x    | 536.9 MB          |

- Key insight: Flash Attention doesn't just make long context faster —
  at sufficient N it makes it possible at all. The algorithm unlocked
  the hardware market for long-context inference, not the other way around.

---

## Notes

- [GPU Architecture Primer](notes/gpu-architecture-primer.md)
- [Flash Attention Explained](notes/flash-attention-explained.md)

---

## Hardware

All Phase 1–3 benchmarks run on RTX 4090 (24GB GDDR6X, ~1,008 GB/s bandwidth)
via RunPod cloud GPU instances.

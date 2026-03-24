# AI Infrastructure: Kernel to Deployment

A hands-on technical curriculum built to develop genuine credibility at the layer of the stack where almost no enterprise GTM candidate operates — GPU architecture, CUDA kernel programming, distributed training primitives, and alternative hardware stacks.

This is not a reading list. Every phase produces running code, real benchmark numbers, and a publishable artifact.

---

## Background

EE undergrad (Texas A&M) + enterprise infrastructure sales background. The goal is to build the technical depth required to sell at the silicon layer — AI chip vendors, GPU cloud, AI networking — where the gap between "knows the vocabulary" and "has run the code" is immediately visible to a technical buyer.

Primary targets: Tenstorrent, Cerebras, Groq, SambaNova, CoreWeave, Lambda Labs, Arista (AI networking).

---

## What's Here

| Phase | Topic | Status | Key Result |
|-------|-------|--------|------------|
| 1 | GPU Architecture & Memory Hierarchy | ✅ Complete | 947 GB/s measured BW (94% peak); roofline flip at N=256 |
| 2 | CUDA Kernel Programming | 🔄 In progress | — |
| 3 | Triton — Python-Level Kernel Writing | ⬜ Planned | — |
| 4 | Distributed Training Primitives | ⬜ Planned | — |
| 5 | Inference & Serving Infrastructure | ⬜ Planned | — |
| 6 | Alternative Hardware Stacks | ⬜ Planned | — |

---

## Phase 1 — GPU Architecture & Memory Hierarchy

**Core result:** Validated the roofline model experimentally on an RTX 4090.

| Operation | Arithmetic Intensity | Achieved | Regime |
|-----------|---------------------|----------|--------|
| Element-wise multiply (FP32, 32M elements) | 0.083 FLOP/byte | 947.5 GB/s (94% of peak BW) | memory-bound |
| FP16 matmul N=8192 | 2731 FLOP/byte | 158.1 TFLOPS (48% of peak compute) | compute-bound |
| FP16 matmul N=256 | 85.3 FLOP/byte | — | ridge point (flip) |

**ncu hardware counter confirmation:**
- `vectorized_elementwise_kernel`: DRAM 92%, SM compute 2.6% → memory-bound confirmed
- `ampere_fp16_s16816gemm`: DRAM 13%, SM compute 47% → compute-bound confirmed

**Artifacts:**
- [`phase1-gpu-architecture/bandwidth_vs_compute.py`](phase1-gpu-architecture/bandwidth_vs_compute.py) — roofline benchmark
- [`notes/gpu-architecture-primer.md`](notes/gpu-architecture-primer.md) — architecture notes written for a technical sales audience
- [`notes/roofline_model_rtx4090.html`](notes/roofline_model_rtx4090.html) — interactive roofline chart with real benchmark data

**GTM takeaway:** Batch size 1 inference is almost always memory-bound regardless of chip TFLOPS. The optimization levers are bandwidth, quantization (fewer bytes to move), and kernel fusion (data reuse). This is the conversation that separates a hardware sales rep who knows the spec sheet from one who understands the workload.

---

## Setup
```bash
# Rent a VM instance (not Docker) on vast.ai for ncu hardware counter access
# RTX 4090, Ubuntu 22.04 VM, ~$0.40/hr

nvidia-smi
ncu --query-metrics 2>&1 | head -5  # confirms hardware counter access

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install nvtx numpy

git clone https://github.com/dagc-ai/ai-infra-learning.git
```

---

## Notes

Technical write-ups designed for a customer or hiring manager conversation, not an academic audience:

- [`notes/gpu-architecture-primer.md`](notes/gpu-architecture-primer.md) — GPU execution model, memory hierarchy, roofline model, SRAM vs HBM tradeoffs

---

*Curriculum based on a 24-week learning plan covering Layers 3–5 of the AI infrastructure stack. Updated as phases complete.*

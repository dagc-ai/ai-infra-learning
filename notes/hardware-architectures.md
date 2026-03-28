# Hardware Architecture Comparison
## Phase 6 — Alternative Accelerator Architectures

This document compares major AI accelerator architectures through the lens of
the roofline model and memory hierarchy built in Phases 1–5. Each architecture
is an answer to the same question: what is the binding constraint in AI workloads,
and how do we eliminate it?

---

## Architecture 1: Tenstorrent Wormhole (N300S)

### Exercise: tt_matmul.py — BF16 Matmul + Inference Benchmark
**Hardware:** Tenstorrent Wormhole N300S (Koyeb cloud instance)
**Script:** `phase6-alternative-hardware/tt_matmul.py`

---

### Technical Comparison and Takeaways

#### What Was Tested
BF16 matmul at 4096x4096 on Tensix hardware, compared against the same
operation on A100 SXM4 from Phase 2. Goal: observe the NoC tile-routing
execution model vs CUDA shared memory tiling, and measure how far the
N300S gets toward its theoretical BF16 peak.

A fair apples-to-apples comparison requires a real inference workload, not
a raw matmul microbenchmark. The right metric is tokens/sec on the same
model class, single-user, decode-phase inference — which is the actual
memory-bandwidth-bound workload Tenstorrent's architecture is targeting.

#### Benchmark Results

**Raw matmul (4096x4096 BF16):**

| Hardware | BF16 Peak | Memory BW | Achieved | Data Source |
|---|---|---|---|---|
| A100 SXM4 | ~312 TFLOPS | 2.0 TB/s HBM | ~250 TFLOPS | **Real — Phase 2 measurement** |
| RTX 4090 | ~165 TFLOPS | 1.01 TB/s GDDR6X | ~130 TFLOPS | **Real — Phase 1 measurement** |
| N300S Wormhole | ~233 TFLOPS | 576 GB/s GDDR6 | ~140–175 TFLOPS | Community benchmarks† |

†N300S matmul numbers from corsix.org Wormhole hardware series (independent
analysis) and FOSDEM 2025. Koyeb N300S platform had repeated deployment
failures during this session (instances 3f2265e1, 8ebd286b — container never
initialized). Direct measurement was not possible.

Note on the matmul comparison: 4096x4096 favors HBM-backed architectures
because the working set exceeds N300S SRAM and spills to GDDR6. This
microbenchmark understates Tenstorrent's advantage on small working sets
and overstates it vs. a fair inference comparison. See inference numbers below.

**Inference throughput (single-user, 7–8B model class, BF16):**

| Hardware | Model | tok/s (single user) | Data Source |
|---|---|---|---|
| A100 SXM4 | Mistral 7B BF16 | **96.5** | **Real — Phase 5 measurement** |
| N300S Wormhole | Llama 3.1 8B BF16 | **~24** | Tenstorrent official (hand-optimized)† |

†Source: Tenstorrent official announcement, Llama 3.1 support page. This is
their best published number for N300S on their best-supported model using
their optimized inference stack (tt-inference-server). Not a third-party result.

This is the fairest available comparison: same workload type (single-user
decode), same model class (7–8B), same precision (BF16), vendor's own
optimized stack vs. vLLM on A100. The gap is real: ~4x throughput advantage
for the A100 at this model size.

Independent hardware review (The Register, Blackhole QuietBox, Nov 2025)
found decode performance reaching only ~50% of theoretical bandwidth
utilization on Tenstorrent hardware — significantly below the 60–80%
typically seen on NVIDIA or AMD. The pattern holds across generations.

#### Key Technical Takeaways

**1. SRAM bandwidth beats HBM bandwidth — for the right workload.**
In Phase 5, single-request Mistral 7B inference on an A100 measured ~1.4 TB/s
implied HBM bandwidth utilization. Decode is deeply memory-bandwidth bound.
Tensix eliminates the HBM round-trip for workloads that fit in SRAM — latency
drops from ~500 cycles to ~5 cycles. The roofline's memory-bandwidth line gets
steeper, shifting the ridge point left, making low-arithmetic-intensity operations
compute-bound rather than memory-bound.

**2. The SRAM capacity ceiling is the hard constraint.**
192MB total SRAM cannot hold a 7B model in BF16 (~14GB). When the model
spills to GDDR6 at 576 GB/s, the chip runs at less than a third of A100's
HBM bandwidth. The advantage doesn't taper — it inverts. The 4x inference
gap in the table above reflects exactly this: Llama 3.1 8B doesn't fit in
SRAM, so the N300S runs it from GDDR6 while the A100 runs it from HBM.

**3. The BF16 halving is a real procurement consideration.**
Headline spec is 466 TFLOPS FP8. BF16 — what most production inference runs
at for quality-sensitive workloads — is 233 TFLOPS. BF16 requires four
multiplier passes (LoFi + HiFi2 + HiFi3 + HiFi4) vs one for FP8. The number
on the box is not the number you get at your precision. Always compare at
matching precision.

**4. TILE_LAYOUT is a hidden porting cost.**
On CUDA, cuBLAS accepts row-major tensors and handles tiling internally.
On Tensix, ttnn.TILE_LAYOUT is mandatory at the storage level — data must
arrive at each core pre-organized in 32x32 tiles. PyTorch models are row-major
by default. Every model port requires an explicit format conversion step.
Real engineering cost, not a flag flip. This is the same tiling principle
from Phase 2's CUDA matmul, but enforced at the storage level by hardware.

---

### Architecture Overview

**The Bet:**
On-chip SRAM bandwidth matters more than off-chip memory capacity for inference
workloads. If you put enough SRAM next to the compute and route tiles efficiently
between cores, you can serve inference faster and cheaper than designs relying
on HBM round-trips — for workloads that fit in that SRAM.

**Hardware:**
- 128 Tensix cores (dual-chip N300S), each: 5 RISC-V CPUs + matrix unit + ~1.5MB SRAM
- 192MB total on-chip SRAM — no HBM
- 24GB GDDR6 at 576 GB/s (secondary storage)
- 466 TFLOPS FP8 / 233 TFLOPS BF16 peak
- NoC: 2D mesh routing tiles between cores at 3.2 Tbps

**Where It Wins:**
- Sub-1B to ~3B parameter models in quantized precision where working set
  approaches fitting in SRAM — this is the architecture's intended sweet spot
- Power efficiency: 300W TDP vs A100 at 400W, with competitive throughput
  at the right model size
- Unit cost: $1,399 vs $25,000+ for H100

**Where It Loses:**
- 7B+ models: working set exceeds SRAM, spills to GDDR6, throughput drops to
  ~25% of A100 at the same model size
- Training: no HBM means insufficient capacity for weights + gradients +
  optimizer state
- Ecosystem: TILE_LAYOUT conversion, limited model support, SDK still maturing

---

### Business Implications

The total addressable market is narrower than "NVIDIA alternative" suggests —
but it's real. The cost-per-token argument only holds at the right operating
point. The honest version: 4 N300S cards (~$5,600) match one A100's inference
throughput on 7–8B models at roughly half the cost. That's compelling for
validated workloads at scale, but requires 4x the hardware management overhead.

For smaller models where SRAM fits the working set, the per-card argument
improves substantially — this is where Tenstorrent's architecture actually
delivers on its thesis and where the cost story is most defensible.

Ecosystem immaturity is the real barrier. TILE_LAYOUT conversion, limited
model support, and the deployment instability observed in this session are
costs that sophisticated buyers will price into any procurement decision.

---

### GTM Insights

- **Champion profile:** ML infrastructure engineer who has validated performance
  on their specific model — not a procurement manager. Developer credibility
  is the sales motion.
- **Winning conversation:** "For this specific model at this batch size, here
  is cost per token vs H100, here is porting effort, here is break-even."
  Requires knowing model size, precision requirements, and throughput targets
  before walking in.
- **Correct positioning:** Workload-specific complement to NVIDIA, not a
  replacement. Rip-and-replace framing loses — workload-fit framing wins.
- **Main objection:** Ecosystem risk. Apache 2.0 licensing helps but doesn't
  fully answer it. Honest response acknowledges the gap and ties the value
  to specific validated workloads.
- **Jim Keller factor:** Designed AMD K8, Apple M1, Tesla Autopilot chip.
  Credibility shortcut with technical evaluators in the room.
- **Open source differentiation:** TT-Metalium Apache 2.0 — full stack
  auditability from kernel driver to operator library. Real differentiator
  for regulated industries where proprietary black-box SDKs create procurement
  barriers CUDA cannot answer.

---

*Architectures to be appended: AMD MI300X, Groq LPU, Google TPU v5*

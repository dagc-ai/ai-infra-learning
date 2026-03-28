# Hardware Architecture Comparison
## Phase 6 — Alternative Accelerator Architectures

---

## Architecture 1: Tenstorrent Wormhole (N300S)

### Exercise: tt_matmul.py — BF16 Matmul Benchmark
**Hardware:** Tenstorrent Wormhole N300S (Koyeb cloud instance)
**Script:** `phase6-alternative-hardware/tt_matmul.py`

#### What Was Tested
BF16 matmul at 4096x4096 on Tensix hardware, compared against the same
operation on A100 SXM4 from Phase 2. Goal: observe the NoC tile-routing
execution model vs CUDA shared memory tiling, and measure how far the
N300S gets toward its theoretical BF16 peak.

#### Benchmark Results

| Hardware | BF16 peak | Memory BW | Achieved (4096³) | Data Source |
|---|---|---|---|---|
| A100 SXM4 | ~312 TFLOPS | 2.0 TB/s HBM | ~250 TFLOPS | **Real — Phase 2 measurement** |
| RTX 4090 | ~165 TFLOPS | 1.01 TB/s GDDR6X | ~130 TFLOPS | **Real — Phase 1 measurement** |
| N300S Wormhole | ~233 TFLOPS | 576 GB/s GDDR6 | ~140–175 TFLOPS | Community benchmarks* |

*N300S numbers from corsix.org Wormhole hardware series (independent analysis)
and FOSDEM 2025. Koyeb N300S platform had repeated deployment failures during
this session (instances 3f2265e1, 8ebd286b — container never initialized).
Direct measurement was not possible. Numbers represent verified public benchmarks
from developers with physical hardware access.

#### Key Technical Takeaways

**1. SRAM bandwidth beats HBM bandwidth — for the right workload.**
In Phase 5, single-request Mistral 7B inference on an A100 measured ~1.4 TB/s
implied HBM bandwidth utilization. Decode is deeply memory-bandwidth bound —
the compute units are mostly idle, bottlenecked on reading weights from HBM on
every decode step. Tensix eliminates this round-trip for workloads that fit in
SRAM. Effective bandwidth per operation rises sharply; latency drops from ~500
cycles (HBM) to ~5 cycles (SRAM).

**2. The BF16 halving is a real procurement consideration.**
Headline spec is 466 TFLOPS FP8. BF16 — what most production inference runs
at for quality-sensitive workloads — is 233 TFLOPS. BF16 requires four
multiplier passes through the hardware (LoFi + HiFi2 + HiFi3 + HiFi4) vs
one pass for FP8. The number on the box is not the number you get at your
precision. Always compare at matching precision levels.

**3. TILE_LAYOUT is a hidden porting cost.**
On CUDA, cuBLAS accepts row-major tensors and handles tiling internally.
On Tensix, `ttnn.TILE_LAYOUT` is mandatory at the storage level — data must
arrive at each core pre-organized in 32x32 tiles. PyTorch models are row-major
by default. Every model port requires an explicit format conversion step that
CUDA users never think about. Real engineering cost, not a flag flip.

**4. The SRAM capacity ceiling is the hard constraint.**
192MB total SRAM cannot hold a 7B model in BF16 (~14GB). When the model
spills to GDDR6 (576 GB/s), the chip runs at less than a third of A100's
HBM bandwidth. The advantage doesn't taper off — it inverts.

---

## The Bet
On-chip SRAM bandwidth matters more than off-chip memory capacity for inference
workloads — if you put enough SRAM next to the compute and route tiles efficiently
between cores, you can serve inference faster and cheaper than designs that rely
on HBM round-trips.

## Hardware Basics
- 128 Tensix cores (dual-chip N300S), each with 5 RISC-V CPUs + matrix unit + ~1.5MB SRAM
- 192MB total on-chip SRAM — no HBM
- 24GB GDDR6 at 576 GB/s (secondary storage, not primary working memory)
- 466 TFLOPS FP8 / 233 TFLOPS BF16 peak
- NoC: 2D mesh routing tiles between cores at 3.2 Tbps

## How It Maps to the Roofline
Higher SRAM bandwidth makes the roofline's memory-bandwidth line steeper,
shifting the ridge point left. Operations with lower arithmetic intensity can
now be compute-bound rather than memory-bound — which is exactly the regime
inference lives in. The failure mode is capacity: once the model exceeds 192MB
SRAM and spills to GDDR6, the architecture loses its bandwidth advantage
decisively.

## The CUDA vs Tensix Programming Model Difference
In CUDA (Phase 2 tiled matmul), data lives in HBM in row-major format. cuBLAS
handles tiling internally. On Tensix, TILE_LAYOUT is mandatory at the storage
level — the Tensix Unpack unit feeds the Matrix unit in 32x32 tiles and expects
that format on arrival. The tiling requirement is pushed all the way down to
memory layout, not just the access pattern. This is the same principle as Phase
2's tiled matmul, but enforced by the hardware rather than left to the programmer.

## Where It Wins / Where It Loses
**Wins:**
- Small-to-mid model inference in quantized precision where working set fits in SRAM
- Cost per token: N300S is $1,399 vs H100 at $25,000+. TCO argument is compelling
  for validated workloads.

**Loses:**
- Any model exceeding 192MB SRAM — GDDR6 spill drops below A100 HBM bandwidth
- Training: no HBM means insufficient capacity for weights + gradients + optimizer
  state at any meaningful model size
- Ecosystem maturity: TILE_LAYOUT conversion adds porting cost CUDA abstracts away

## Business Implications
The total addressable market is narrower than "NVIDIA alternative" suggests — but
it's real. Small-to-mid model inference at scale, edge deployments, cost-per-token
optimization are legitimate use cases. The price-performance argument at $1,399 vs
$25,000+ is the actual weapon, not TFLOPS comparison. Ecosystem immaturity is the
real barrier: TILE_LAYOUT conversion, limited model support, and SDK stability
are engineering costs that sophisticated buyers will price in.

## GTM Insights
- Champion profile is an ML infrastructure engineer who has validated performance
  on their specific model — not a procurement manager
- Winning motion is workload-specific: "for this model at this batch size, here is
  your cost per token vs H100, here is the porting effort, here is break-even"
- Main objection is ecosystem risk — Apache 2.0 licensing helps but doesn't fully
  answer it; position as a workload-specific complement to NVIDIA, not a replacement
- Jim Keller's hardware pedigree (AMD K8, Apple M1, Tesla Autopilot) is a credibility
  shortcut with technical evaluators

## Open Source Differentiation
TT-Metalium is Apache 2.0 licensed — kernel driver, user-mode driver, and SDK.
Full stack auditability from silicon to operator library. Real differentiator for
regulated industries where proprietary black-box SDKs create procurement barriers
that CUDA cannot answer.

---

*Architectures to be appended: AMD MI300X, Groq LPU, Google TPU v5*

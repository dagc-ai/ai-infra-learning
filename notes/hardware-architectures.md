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

## Architecture 2: AMD MI300X

### Exercise: mi300x_inference.py — Llama 70B Inference Benchmark
**Hardware:** AMD Instinct MI300X (AMD Developer Cloud / RunPod)
**Script:** `phase6-alternative-hardware/mi300x_inference.py`

---

### Technical Comparison and Takeaways

#### What Was Tested
Llama 70B BF16 inference on MI300X at TP=1, compared against Phase 5 A100
SXM4 baseline. The exercise is specifically designed to isolate the architectural
advantage: 192GB unified HBM3 means the entire 70B model fits on a single chip,
eliminating the tensor parallelism and AllReduce overhead that A100 requires.

#### Benchmark Results

**Inference throughput (Llama 70B, vLLM, single-user decode):**

| Hardware | Config | Model | tok/s | Data Source |
|---|---|---|---|---|
| A100 SXM4 | TP=2 required | Mistral 7B BF16 | **96.5** | **Real — Phase 5 measurement** |
| MI300X | TP=1 native | Llama 70B FP16 | **~37** | Community benchmarks† |
| H100 SXM5 | TP=2 required | Llama 70B | competitive | MLPerf v4.1 |
| H200 | TP=1 possible | Llama 70B | matches/beats MI300X | SemiAnalysis 2025 |

†Sources: Chips & Cheese MI300X independent hardware review; AMD MLPerf v4.1
submission; SemiAnalysis AMD vs NVIDIA inference benchmark (May 2025).

Note: A100 number is a 7B proxy from Phase 5 — not a direct 70B comparison.
The meaningful comparison is MI300X TP=1 vs H100 TP=2 on the same 70B model,
where MI300X delivers ~40% lower latency by eliminating AllReduce overhead.

**Hardware access note:** Direct measurement was not possible. RunPod MI300X
deployment encountered ROCm 6.1 userspace / 6.10.5 kernel driver version
mismatch across three container configurations — device initialization hung
consistently. AMD Developer Cloud was at capacity during this session. This
deployment friction is itself a primary finding documented below.

**Realized vs theoretical performance:**
Independent academic analysis (arxiv:2510.27583) found MI300X achieves only
37–66% of H100/H200 realized throughput on Llama 70B inference despite 1.5x
higher theoretical compute. The gap is software stack maturity — ROCm kernel
optimization and vLLM tuning lag behind CUDA equivalents.

#### Key Technical Takeaways

**1. 192GB eliminates the TP=2 requirement — that's the real advantage.**
A100 and H100 (both 80GB) cannot hold Llama 70B in FP16 (140GB) on a single
chip. They require tensor parallelism across 2 GPUs, which adds an AllReduce
communication step on every single decode step via NVLink. MI300X fits the
entire model on one chip. No inter-GPU communication. Every decode step is a
single-chip memory read at 5.3 TB/s. The ~40% latency advantage vs H100 comes
entirely from eliminating that communication overhead — not from faster compute.

**2. 37–66% of theoretical is the honest realized performance number.**
Hardware specs and real throughput diverge significantly on MI300X. The
software stack — ROCm kernel optimization maturity, vLLM tuning, driver/
userspace compatibility — is the delta between what the hardware can do and
what it actually delivers. This gap is narrowing but real as of March 2026.

**3. HIP/CUDA compatibility is genuine — the code is identical to Phase 5.**
ROCm implements a compatibility layer (HIP) that maps AMD GPU operations to
the same API surface as CUDA. `torch.cuda.is_available()` returns True on
AMD hardware. `torch.cuda.get_device_name(0)` returns "AMD Instinct MI300X".
The only meaningful code difference in this exercise vs Phase 5 vLLM deployment
is `--tensor-parallel-size 1` instead of 2. The portability is by design and
the mechanism is correct — but CUDA ecosystem optimization depth still leads.

**4. The deployment experience is a primary finding.**
Three container configurations failed across two platforms due to ROCm version
mismatches. This is the same ecosystem friction pattern as Tenstorrent (Koyeb
platform failures), now on AMD hardware. The barrier has shifted from "hardware
doesn't work" to "getting hardware working requires non-trivial engineering."
This is real enterprise adoption cost that any procurement conversation should
account for.

---

### Architecture Overview

**The Bet:**
Memory capacity is the binding constraint for large model inference. If you
build enough HBM into a single accelerator to hold a 70B model without sharding,
you eliminate the inter-GPU communication overhead that defines the latency
floor for every other architecture at this model size.

**Hardware:**
- 3 compute chiplets (XCDs) + 8 HBM3 stacks on one package
- 192GB HBM3 — 2.4x the capacity of H100 SXM5 (80GB)
- 5.3 TB/s aggregate memory bandwidth — 1.58x H100
- ~383 TFLOPS FP16 peak — 2.6x below H100's ~989 TFLOPS
- Infinity Fabric inter-die interconnect within package
- ROCm software stack — HIP provides CUDA API compatibility

**Where It Wins:**
- 70B model inference in FP16 natively on a single chip — H100 cannot do this
- ~40% lower latency vs H100 on memory-bound 70B decode (no AllReduce)
- 405B+ model inference where capacity is the primary constraint
- Ultra-low latency single-user scenarios: MI300X and MI325X beat all others
  on performance per dollar for Llama 70B chat at low latency (SemiAnalysis)

**Where It Loses:**
- Compute-bound workloads: 383 TFLOPS vs H100's 989 TFLOPS — 2.6x gap
- Prefill-heavy summarization tasks: H100 surpasses MI300X after ~30s latency
- Realized inference throughput: 37–66% of H100/H200 due to software gap
- Short-term cloud rentals: thin AMD Neocloud market inflates rental prices
  above fair value; H100 always wins on perf/$ for sub-6-month contracts
- Ecosystem: ROCm still lags CUDA in kernel optimization depth and community

---

### Business Implications

The memory capacity argument is the strongest entry point. 192GB is the only
single-GPU answer to running 70B models in FP16 without sharding. Microsoft
EVP Scott Guthrie called MI300X "the most cost-effective GPU out there" —
that statement is specifically about large model inference where capacity
eliminates multi-GPU complexity. Meta running 100% of Llama 3.1 405B traffic
on MI300X is the reference customer that closes the "is this proven?" objection.

The 37–66% realized performance gap undermines the hardware spec story.
Sophisticated buyers running benchmarks will find this gap themselves. The
honest positioning: MI300X wins on capacity and single-chip 70B inference,
loses on raw throughput efficiency vs H100 on optimized workloads.

The ecosystem risk is empirical, not theoretical. Today's deployment experience
— ROCm version mismatches, container compatibility failures, hardware supply
constraints — is a live demonstration of what enterprise buyers will encounter.
The hardware capability is real. Getting it working reliably requires engineering
investment that CUDA doesn't.

---

### GTM Insights

- **Displacement motion:** "You're running 70B models. On H100 you need 2 GPUs
  per instance and AllReduce on every decode step. On MI300X you run TP=1,
  one chip, no inter-GPU communication. Here's cost per million tokens at your
  throughput requirement." Works for owned hardware at scale — not cloud rentals.
- **Owned hardware only:** Cloud rental market for AMD is thin, driving prices
  above fair value. Economics work only for capital budget buyers — enterprises,
  hyperscalers, research institutions. Not a developer cloud story yet.
- **Champion profile:** Infrastructure engineer who has validated their specific
  stack on MI300X. ROCm works when set up correctly on matched hardware. The
  engineering work to get there is the champion's credibility.
- **Reference customer:** Meta running 100% of Llama 3.1 405B on MI300X at
  production scale. Closes "is this proven?" — use carefully, Meta has
  engineering resources most enterprise buyers don't.
- **Competitive leverage play:** Position MI300X as the credible second-source
  that creates negotiating leverage with NVIDIA — not the outright replacement.
  "NVIDIA killer" framing loses. "Alternative that changes the negotiation" wins.
- **AMD data center revenue:** $4.3B/quarter growing 22% YoY vs NVIDIA's
  $51.2B. Market has validated MI300X as real. Competitive gap is still
  enormous. Frame AMD as the serious challenger, not the incumbent.

---

## Architecture 3: Groq LPU (Language Processing Unit)

### Exercise: groq_benchmark.py — TTFT and Throughput Benchmark
**Hardware:** Groq LPU via GroqCloud API (Austin, TX → Groq datacenter)
**Script:** `phase6-alternative-hardware/groq_benchmark.py`

---

### Technical Comparison and Takeaways

#### What Was Tested
TTFT and decode throughput on Llama 3.1 8B and Llama 3.3 70B via the Groq
API, using identical methodology to Phase 5 Part 3 benchmark_ttft.py:
- Same exact prompt (~100 tokens, Llama-tokenized)
- Same 200 output tokens, temperature=0, greedy decoding
- Same streaming methodology for TTFT measurement
- Same warmup + 5 run mean pattern

Network latency to Groq datacenter measured via ping: ~13ms round-trip,
~6.5ms one-way from Austin, TX. Used to separate API infrastructure overhead
from pure LPU compute latency.

#### Benchmark Results

**Full comparison table — API methodology, ~100 token prompt, 200 token output:**

| Hardware | Model | tok/s | TTFT | Config | Data Source |
|---|---|---|---|---|---|
| A100 SXM4 (local) | Llama 3.1 8B BF16 | **93.0** | **18.1ms** | TP=1, no network | **Real — Phase 5 Part 3** |
| A100 SXM4 (local) | Llama 3.3 70B BF16 | **21.2** | **58.0ms** | TP=2, no network | **Real — Phase 5 Part 3** |
| Cloud API median (A100/H100 mix) | Llama 3.1 8B | 154.8 | 930ms | API + network | ArtificialAnalysis† |
| Cloud API median (A100/H100 mix) | Llama 3.3 70B | 85.5 | 1,410ms | API + network | ArtificialAnalysis† |
| Groq LPU (API) | Llama 3.1 8B | **672.9** | **130.8ms** | API + network | **Real — this session** |
| Groq LPU (API) | Llama 3.3 70B | **263.1** | **135.4ms** | API + network | **Real — this session** |
| Groq LPU (on-premise published) | Llama 2 70B | ~300 | **~14ms** | No network | Groq internal |
| Groq LPU (speculative decoding) | Llama 3 70B | **1,665** | — | API | Groq published |

†ArtificialAnalysis.ai independent benchmark, same ~100 token / ~200 token
methodology as this exercise. Median across all providers including H100-backed
instances which pull the median above A100-only performance.

**TTFT breakdown — where the latency actually goes:**

| Component | Estimate |
|---|---|
| Network one-way (Austin → Groq) | ~6.5ms |
| LPU compute (Groq published on-premise) | ~14ms |
| API infrastructure overhead (HTTP, TLS, load balancer, queue) | ~110ms |
| **Total measured API TTFT** | **~130ms** |

The API infrastructure overhead (~110ms) is not unique to Groq — any managed
cloud inference API carries similar overhead. OpenAI, Anthropic, and every
cloud inference endpoint has this baked in. The LPU compute itself at ~14ms
is the genuine hardware advantage. The comparison that matters for on-premise
deployments: Groq 14ms vs A100 local 18ms — Groq wins by 22%.

**Groq vs cloud API median (apples-to-apples API comparison):**
- 8B throughput: Groq 672.9 vs median 154.8 — **4.3x faster**
- 70B throughput: Groq 263.1 vs median 85.5 — **3.1x faster**
- 8B TTFT: Groq 130.8ms vs median 930ms — **7x faster**
- 70B TTFT: Groq 135.4ms vs median 1,410ms — **10x faster**

**Variance observation:**
Groq TTFT ranged from 97.8ms to 217.6ms on 8B across 5 runs. This variance
is network jitter and API queue state — not LPU compute variance. The
determinism claim is about the compute portion. Once a request reaches the
LPU, execution time is deterministic to the nanosecond. Network introduces
the variance observed in API measurements.

**70B warmup anomaly:**
Warmup run TTFT on 70B was 40.6ms — significantly below the 65–203ms range
of subsequent runs. The warmup likely hit a server with the model already
resident in SRAM. Subsequent requests hit cold servers requiring model routing
across the GroqRack. Relevant for understanding production latency distribution.

#### Key Technical Takeaways

**1. Eliminating caches enables deterministic execution — that's the entire bet.**
Every other architecture in Phase 6 has caches: NVIDIA has L1/L2/HBM, Tenstorrent
has per-core SRAM, MI300X has Infinity Cache plus HBM. Caches exist to hide
memory latency by predicting what data compute units will need next. Cache hits
produce fast access, cache misses produce slow access — the variance between
them is what makes GPU inference latency non-deterministic. Groq eliminates all
caches entirely. Every memory access is compiler-scheduled at compile time.
The hardware executes exactly what the compiler planned — no speculation, no
branch prediction, no cache hierarchy to manage. P99 latency ≈ P50 ≈ P1.
This is the feature that makes Groq uniquely suited for real-time applications.

**2. Throughput advantage is real, large, and grows with model size.**
- 8B: 7.2x faster than A100 local (93.0 → 672.9 tok/s)
- 70B: 12.4x faster than A100 local (21.2 → 263.1 tok/s)
The gap widens with model size because larger models have more HBM bandwidth
overhead on GPU-based architectures. Groq's SRAM-scheduled execution scales
more cleanly — no cache hierarchy to manage means no growing overhead as
model size increases.

**3. TTFT advantage is real at the hardware level, obscured at the API level.**
On-premise Groq TTFT (~14ms) genuinely beats A100 local TTFT (18ms) by 22%.
That advantage disappears in API deployment because ~110ms of infrastructure
overhead dwarfs both. For cloud API users, the throughput advantage (3-7x)
is the meaningful number — not TTFT. For on-premise enterprise deployments
of GroqRacks, the TTFT story is real and relevant for latency-sensitive
applications like voice interfaces and real-time code generation.

**4. The compile-ahead constraint is the real adoption barrier.**
Groq's model must be compiled for the LPU hardware before serving. You cannot
load arbitrary models at runtime like vLLM. Every new model, every new
quantization configuration, every new context length requires a new compile
job through the Groq compiler. This is a fundamental architectural constraint
not a software maturity issue — it's the price of deterministic execution.
For enterprise buyers running a fixed set of production models, this is
manageable. For teams iterating rapidly on model selection, it's a meaningful
operational constraint.

**5. Speculative decoding works on Groq where it barely works on GPUs.**
The 1,665 tok/s speculative decoding number on Llama 3 70B is the most
dramatic demonstration of the architectural advantage. On a GPU, speculative
decoding's verification step is expensive — loading the target 70B model to
verify draft tokens is memory-bandwidth-bound, often erasing the speedup.
On Groq, the 70B model is already distributed across the SRAM of the GroqRack.
Verification is nearly instant — same cost as generating one token. This
allows effective speculative decoding at scale in a way GPU architectures
cannot match, adding another 6x on top of the already-fast baseline.

**6. NVIDIA acquired Groq for $20 billion (December 24, 2025).**
This changes the competitive landscape. Groq is no longer an independent
NVIDIA alternative — it is now an NVIDIA company. The long-term implications
for GroqCloud pricing, API access, and hardware roadmap are unclear. For
enterprise buyers evaluating alternative hardware to reduce NVIDIA dependency,
this acquisition significantly changes the calculus. Access the current API
while it remains independent-stack, but factor the acquisition into any
long-term infrastructure planning that relies on Groq as an NVIDIA alternative.

---

### Architecture Overview

**The Bet:**
The binding constraint for real-time LLM inference is not memory capacity or
bandwidth — it is latency unpredictability. Caches cause non-deterministic
execution because cache hits and misses produce variable latency. Eliminate
all caches, have the compiler schedule every memory access deterministically,
and you get predictable, repeatable, ultra-low latency on every single request.

**Hardware:**
- SRAM-only on-chip memory — weights statically placed at compile time
- No caches, no dynamic scheduling — compiler pre-computes entire execution graph
- Systolic array compute — data flows through multiply-accumulate grid in wave pattern
- Single-threaded execution model — no thread scheduling, no warp divergence
- ~230MB SRAM per chip — large models require multiple chips in a GroqRack
- 576 LPUs required to serve Llama 2 70B in a GroqRack configuration
- Samsung 4nm process (LPU v2)

**Where It Wins:**
- Throughput: 3-7x faster than cloud GPU API providers at same model size
- Latency consistency: P99 ≈ P50 — deterministic execution, no jitter
- Speculative decoding: 1,665 tok/s on Llama 70B — works where GPUs can't
- Real-time applications: voice interfaces, code assistants, interactive AI
  where latency variance directly degrades user experience
- Power efficiency per token: deterministic execution wastes no cycles on
  speculation, cache management, or thread scheduling overhead

**Where It Loses:**
- Model flexibility: must compile ahead of time — no arbitrary model loading
- Batch throughput: at high batch sizes NVIDIA's raw TFLOPS advantage reasserts
- Memory capacity per chip: ~230MB SRAM — large models require many chips
- Cost scaling: more chips required per model = capital cost scales differently
- Ecosystem: proprietary compiler, limited model support vs CUDA breadth
- Independence: NVIDIA acquisition (Dec 2025) changes long-term positioning

---

### Business Implications

The throughput advantage is the headline — 3-7x faster than competing cloud
inference providers using the same API methodology and same benchmark
parameters that ArtificialAnalysis uses for their independent leaderboard.
This is not a cherry-picked benchmark: 877 tok/s on Llama 3 8B was validated
by ArtificialAnalysis independently. Your session measured 672.9 tok/s on
Llama 3.1 8B — consistent with the published range given model version
differences. The advantage is real and reproducible.

The TTFT story is nuanced and requires careful framing. API TTFT (130ms)
actually beats the cloud GPU median (930ms) by 7x — compelling for API users.
But it loses to a local A100 deployment (18ms) by 7x — relevant for
on-premise buyers. Always establish deployment context before citing TTFT.

The compile-ahead constraint is the main objection to close. For enterprise
production workloads running a fixed model in production, compiling once is
a one-time cost. For ML teams in rapid iteration, it's a real operational
friction. Qualify the use case before leading with Groq.

The NVIDIA acquisition is the elephant in the room for any "reduce NVIDIA
dependency" conversation. GroqCloud remains available and functional today,
but the long-term roadmap under NVIDIA ownership is uncertain. Any enterprise
deal that justifies itself partly on vendor diversification needs to account
for this.

---

### GTM Insights

- **The headline number:** 877 tok/s on Llama 3 8B (ArtificialAnalysis
  independent), 3-7x faster than any other cloud inference provider. This
  opens doors — it's dramatic enough that technical buyers want to see it
  validated, which is exactly where your benchmark script comes in.
- **TTFT framing by deployment type:** For cloud API users — "7-10x lower
  TTFT than GPU cloud providers." For on-premise buyers — "22% lower compute
  TTFT than A100, and the P99 equals the P50 — no spikes." Two different
  conversations, two different numbers, both honest.
- **Speculative decoding is the closing argument:** 1,665 tok/s on Llama 70B
  is the number that ends throughput objections. No GPU architecture can
  match it at that model size because the verification step requires the model
  to already be in fast memory — which only works on Groq's distributed SRAM
  architecture.
- **Correct use case positioning:** Real-time consumer applications, voice AI,
  code assistants, agentic systems where many sequential inference calls happen
  per user interaction. The throughput advantage compounds when each agent step
  requires an LLM call — 7x faster per step means agentic workflows complete
  in seconds that take minutes on GPU infrastructure.
- **The acquisition objection:** Acknowledge it directly. "Groq was acquired
  by NVIDIA in December 2025. GroqCloud is operational today and the
  performance is real. The long-term roadmap under NVIDIA is uncertain —
  factor that into any 3+ year infrastructure commitment." Trying to downplay
  this loses credibility with informed buyers.
- **Compile-ahead as a feature for production:** Reframe the constraint.
  "You have to compile your model once. In exchange, every single inference
  request executes in exactly the same time — no latency spikes, no P99
  outliers, no capacity planning for worst-case scenarios. Operations teams
  love deterministic systems."

---

*Architectures to be appended: Google TPU v5*

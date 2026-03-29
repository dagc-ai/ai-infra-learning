# Inference Infrastructure Notes
## Phase 5 — A100 SXM 80GB, Mistral 7B

---

## Core Principle (Applies to Every Exercise)
Inference is a memory problem, not a compute problem. The GPU has hundreds of TFLOPS
sitting mostly idle during decode. What limits token generation speed is how fast model
weights and KV cache can be loaded from HBM. Every optimization in this phase is
a variation of the same principle from Phases 1-4: minimize expensive data movement,
maximize useful work per memory access.

---

## Exercise 5.1 — vLLM Deployment & Continuous Batching

### What It Is In Simple Terms
vLLM is the engine that loads a model and serves it to users efficiently. Its core
innovation (PagedAttention) manages KV cache memory like an OS manages RAM — on demand,
no fragmentation, no waste. Continuous batching combines multiple user requests into
one GPU forward pass, amortizing the cost of loading weights across all requests
simultaneously.

### KV Cache Explained
Every token the model generates requires attention over every previous token — prompt
and output combined. The Key and Value matrices for all those tokens are cached in VRAM
to avoid recomputation. This cache grows with every token generated and every concurrent
request. At 80GB VRAM with Mistral 7B BF16:
- Model weights: 13.51 GiB
- KV cache pool: 57.41 GiB (90% utilization minus weights)
- Max concurrent capacity: 470,288 tokens (~57 full-length requests)

### Benchmark Results
Single request baseline: 96.5 tok/s, 2073ms
Implied HBM bandwidth utilization: ~1.4 TB/s (~70% of A100 peak 2.0 TB/s)

Concurrent requests (continuous batching):
- Concurrency  1:   94.7 tok/s | 2110ms latency
- Concurrency  2:  193.0 tok/s | 2066ms latency
- Concurrency  4:  382.7 tok/s | 2083ms latency
- Concurrency  8:  752.3 tok/s | 2110ms latency

8x requests = 8x throughput, flat latency. The GPU was underutilized at concurrency=1.
Batching filled those gaps without touching per-user latency.

### Business Insights
Cost at 8x concurrency on $1.49/hr A100: ~2.7M tokens/hour = $0.00055 per 1K tokens
raw compute before markup. GPU utilization is the lever that controls margin.

Latency and throughput are a product decision. Chatbots need low TTFT. Batch pipelines
need high throughput. Same hardware, different configuration, different optimization target.

### GTM Implications
Self-host vs managed API inflection point is roughly 5-10M tokens/day. Below that,
simplicity wins. Above that, cost wins. Every inference vendor (vLLM, Anyscale,
Together AI, Fireworks, Groq) is selling solved utilization. Understanding the
underlying constraints lets you evaluate those claims rather than take them at face value.

---

## Exercise 5.2 — Quantization Benchmarks

### What It Is In Simple Terms
Quantization shrinks the numbers that represent model weights — from 16-bit floats
down to 4-bit integers. Smaller numbers = smaller model = more fits through the
memory bandwidth bus per second = faster token generation. The tradeoff is precision
loss. The non-obvious part: the quantization kernel matters as much as the
quantization level. The wrong kernel makes INT4 slower than FP16.

### Benchmark Results — A100 SXM 80GB, Mistral 7B

Format           | Model Size  | Throughput   | vs BF16 | Kernel
-----------------|-------------|--------------|---------|-------------
BF16             | 13.51 GiB   |  96.5 tok/s  |  1.0x   | FlashAttn
AWQ INT4 naive   |  3.88 GiB   |  20.9 tok/s  |  0.2x   | naive AWQ
AWQ INT4 Marlin  |  3.88 GiB   | 218.9 tok/s  |  2.27x  | fused Marlin

### Why Kernel Choice Is Everything
Naive AWQ dequantizes INT4 weights back to FP16 before the matrix multiply — a
separate expensive step. Marlin fuses dequantization directly into the matrix multiply
kernel. Same weights, same precision, 10x different throughput. One flag difference
in the launch command.

### Why Not Full 4x Theoretical Speedup
Attention computation and KV cache stay in FP16 — only linear layer weights are
quantized. As weight loading gets faster, unquantized operations become a larger
fraction of total time. The roofline bottleneck shifts.

### Memory Benefit Beyond Throughput
AWQ INT4 freed 9GB of VRAM for KV cache vs BF16, increasing max concurrency from
57x to 67x at the same context length. Quantization changes what models and context
lengths are possible within a fixed hardware budget — not just how fast they run.

### Business Insights
Misconfiguration is more common than hardware shortage. The AWQ naive vs Marlin gap
is 10x on identical hardware and weights. A customer running naive quantization is
leaving 10x performance on the table from a config change, not a hardware purchase.

Quantization unlocks larger models on existing hardware. A customer who cannot run
70B in BF16 on a single A100 can run it in INT4 for the same hardware cost. Model
capability and infrastructure cost are directly connected.

Quality degradation is workload-dependent. AWQ calibrated quantization minimizes
quality loss vs naive INT4. For summarization, classification, structured extraction
the delta is often negligible. For legal, medical, code generation it warrants
evaluation. Not one-size-fits-all.

### GTM Implications
Audit before upsell. A customer running naive AWQ gets 10x improvement from a config
change, not a hardware purchase. That conversation builds credibility and earns the
right to the hardware conversation later — grounded in real numbers.

Every inference API is making quantization decisions on your behalf. Self-hosted
inference gives enterprises control over the quality/speed tradeoff. For workloads
with quality SLAs that control has real value.

---

## Exercise 5.3 — Speculative Decoding

### What It Is In Simple Terms
Instead of asking the expensive expert model to generate every token sequentially,
you hire a tiny cheap draft model to guess the next several tokens first. The large
model then reviews all the guesses at once and corrects what is wrong. If the draft
model guesses correctly most of the time, you get multiple tokens from one large model
forward pass instead of one. If it guesses wrong constantly, you have paid the cost
of both models and gotten slower results than using the large model alone.

### What We Ran
- Target: Mistral 7B AWQ Marlin (218.9 tok/s baseline)
- Draft: TinyLlama 15M (30MB, same tokenizer vocabulary, different training)
- Result: 67.5 tok/s — 3x slower than target alone

### Why It Was Slower
TinyLlama shares Mistral's tokenizer vocabulary but was trained on different data
with different dynamics. The target model rejected most draft tokens, paying the cost
of both models with the benefit of neither. Async scheduling was also disabled by
vLLM when speculative decoding is active, adding independent overhead.

### The Non-Obvious Requirement
Shared vocabulary is necessary but not sufficient. The draft model must come from
the same training run as the target — same model family, same data distribution,
same output style. A Mistral 1B or 2B draft for a Mistral 7B target. Not a
LLaMA-family model drafting for a Mistral-family target, even if the vocabulary
is identical.

### Full Benchmark Summary — All Configurations

Configuration                  | Throughput  | vs BF16
-------------------------------|-------------|--------
BF16 baseline                  |  96.5 tok/s |  1.0x
AWQ INT4 naive                 |  20.9 tok/s |  0.2x
AWQ INT4 Marlin                | 218.9 tok/s |  2.27x
AWQ Marlin + speculative (bad) |  67.5 tok/s |  0.7x

### The Optimization Hierarchy
Fix in this order before buying hardware:

1. Continuous batching — 8x throughput at same latency, free
2. Quantization kernel — 10x difference between naive AWQ and Marlin, one flag
3. Quantization level — 2.27x BF16 to Marlin INT4, 3.5x smaller model
4. Speculative decoding — only helps with correct draft model family at low batch sizes

### Business Insights
"We use speculative decoding" means nothing without context. Outcomes range from
2x faster to 30% slower depending entirely on draft model selection. A customer
running the wrong draft model is paying for additional complexity and getting worse
performance. They likely do not know this.

Memory is the real constraint throughout. BF16 was memory-bandwidth bound.
Quantization helped by shrinking what moved through the bandwidth bus. Speculative
decoding hurt because it added overhead to an already-optimized memory pattern.
The roofline model from Phase 1 predicted every single one of these results.

### GTM Implications
Diagnostic sequence when a customer says inference is slow or expensive:
1. What precision? (BF16 vs INT4)
2. What quantization kernel? (naive AWQ vs Marlin — 10x difference)
3. What batch sizes? (continuous batching opportunity)
4. Speculative decoding? If so, what draft model and from what training family?

That sequence alone could uncover 10-20x performance improvement on existing hardware
before a single dollar of new spend is justified.

Speculative decoding is a latency play, not a throughput play. At low batch sizes
with the right draft model it reduces TTFT for interactive use cases — chatbots,
coding assistants, developer tools. It does not help batch pipelines. Knowing which
use case a customer has determines whether it is even the right conversation.

---

## Phase 5 Part 2 — Llama 3.1 8B Repeat Study

### Purpose
Repeat all Phase 5 exercises on Meta Llama 3.1 8B Instruct to validate whether
findings from Mistral 7B generalize across model families, and to run speculative
decoding with a correctly matched draft model (Llama 3.2 1B from the same training
family).

Hardware: A100 SXM4 80GB. vLLM 0.18.0. Same benchmark scripts as Part 1.

---

### Exercise 5.1 Repeat — Llama 3.1 8B BF16 Baseline

vLLM startup telemetry:
- Model loaded: 14.99 GiB (vs 13.51 GiB for Mistral — 1.5 GiB larger)
- KV cache pool: 55.20 GiB (vs 57.41 GiB — less room due to larger weights)
- KV cache capacity: 452,192 tokens (vs 470,288 for Mistral)
- Max concurrency: 55x (vs 57x)

Both models use GQA with 8 KV heads — the capacity difference is purely weight size,
not attention architecture. The extra 1.5 GiB of weights directly reduces KV cache
pool.

Single request: 93.1 tok/s, 2149ms

Concurrent benchmark:
- Concurrency  1 |  91.2 tok/s | 2192ms
- Concurrency  2 | 186.5 tok/s | 2138ms
- Concurrency  4 | 367.6 tok/s | 2169ms
- Concurrency  8 | 731.1 tok/s | 2179ms

Continuous batching behavior identical to Mistral — near-linear throughput scaling
with flat latency. Architecture-agnostic property of vLLM's scheduler.

**Cross-model BF16 comparison:**
- Mistral 7B:   96.5 tok/s — 13.51 GiB weights
- Llama 3.1 8B: 93.1 tok/s — 14.99 GiB weights

3% gap tracks directly with 11% weight size difference. Memory bandwidth bound
result predicted by roofline model before running.

---

### Exercise 5.2 Repeat — Llama 3.1 8B Quantization

AWQ model: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 (5.37 GiB)

Benchmark results:

Format           | Model Size  | Throughput   | vs BF16 | Kernel
-----------------|-------------|--------------|---------|-------------
BF16             | 14.99 GiB   |  93.1 tok/s  |  1.0x   | FlashAttn v2
AWQ INT4 naive   |  5.37 GiB   |  20.7 tok/s  |  0.2x   | naive AWQ
AWQ INT4 Marlin  |  5.37 GiB   | 199.1 tok/s  |  2.14x  | fused Marlin

**Cross-model quantization comparison:**

Model           | BF16       | AWQ naive  | AWQ Marlin | Marlin/BF16
----------------|------------|------------|------------|------------
Mistral 7B      |  96.5 tok/s|  20.9 tok/s| 218.9 tok/s|  2.27x
Llama 3.1 8B    |  93.1 tok/s|  20.7 tok/s| 199.1 tok/s|  2.14x

AWQ naive penalty is identical across both models (~20.8 tok/s). Dequantization
overhead completely dominates — the underlying model is irrelevant. Pure kernel
property confirmed across two model families.

Marlin speedup slightly lower for Llama (2.14x vs 2.27x) because Llama AWQ weights
are 1.49 GiB larger (5.37 vs 3.88 GiB). Larger model = attention and unquantized
operations are a larger fraction of total time = quantization benefit marginally
diluted.

---

### Exercise 5.3 Repeat — Speculative Decoding with Correct Draft Model

**Setup:**
- Target: Llama 3.1 8B AWQ Marlin (199.1 tok/s baseline)
- Draft: Llama 3.2 1B Instruct (2.47 GiB, 16 layers, vocab_size 128256)
- Pairing rationale: same Meta training family, identical tokenizer vocabulary,
  same RLHF process — correct draft model selection

**Unsloth mirror vs official weights validation:**
Downloaded unsloth/Llama-3.2-1B-Instruct first due to Meta gating delay.
After official access granted, ran identical benchmark on meta-llama/Llama-3.2-1B-Instruct.
vLLM reused compiled kernel cache — same architecture confirmed.

Results were statistically identical:
- Unsloth: 107.3 tok/s single request, 60.0% acceptance rate
- Official: 106.4 tok/s single request, 59.5% acceptance rate

Unsloth mirror produced valid benchmark data. Noted in repo for transparency.

**Speculative decoding benchmark results:**

Single request:
- AWQ Marlin alone:       199.1 tok/s
- AWQ Marlin + spec dec:  106.4 tok/s
- Draft acceptance rate:  59.5%
- Mean acceptance length: 3.98 tokens per round

Concurrent benchmark:
Concurrency  1 | 106.9 tok/s | 1869ms  (vs 91.2 tok/s Marlin alone — spec wins)
Concurrency  2 | 136.6 tok/s | 2909ms  (scheduling anomaly — head-of-line blocking)
Concurrency  4 | 383.6 tok/s | 2070ms  (spec roughly matches Marlin alone)
Concurrency  8 | 720.9 tok/s | 2203ms  (spec roughly matches Marlin alone)

**Why single-request throughput is lower despite correct draft model:**
Async scheduling is disabled by vLLM when speculative decoding is active. This
forces synchronous CPU/GPU handoffs between every draft and verify round. The
scheduling overhead alone accounts for roughly 2x throughput reduction on isolated
single-request measurements, independent of draft quality.

**Why concurrency=1 in concurrent benchmark beats Marlin alone:**
No explicit warmup in benchmark_concurrent.py. The spec decoding benefit from
accepting ~4 tokens per verify pass outweighs the synchronous scheduling overhead
when measuring wall-clock time across a complete request.

**Why concurrency=2 anomaly:**
Two requests sharing synchronous draft/verify rounds create head-of-line blocking.
One request's draft phase blocks the other's verify phase. Worst-case interaction
between speculative decoding's synchronous execution model and vLLM's scheduler.
Resolves at concurrency=4 where the scheduler has enough work to fill gaps.

**The correct interpretation:**
Speculative decoding with the right draft model family improves latency (time to
complete a response) at low batch sizes. The 60% acceptance rate and 4-token mean
acceptance length are genuine — the 1B Llama is correctly predicting most tokens.
The throughput measurement penalty is a vLLM 0.18.0 scheduling limitation, not
a fundamental property of speculative decoding.

---

### Complete Phase 5 Benchmark Summary — Both Model Families

Configuration                              | Throughput   | vs own BF16
-------------------------------------------|--------------|------------
Mistral 7B BF16                            |   96.5 tok/s |  1.0x
Mistral 7B AWQ naive                       |   20.9 tok/s |  0.2x
Mistral 7B AWQ Marlin                      |  218.9 tok/s |  2.27x
Mistral 7B AWQ Marlin + spec (wrong family)|   67.5 tok/s |  0.7x
Llama 3.1 8B BF16                          |   93.1 tok/s |  1.0x
Llama 3.1 8B AWQ naive                     |   20.7 tok/s |  0.2x
Llama 3.1 8B AWQ Marlin                    |  199.1 tok/s |  2.14x
Llama 3.1 8B AWQ Marlin + spec (correct)   |  106.4 tok/s |  1.14x

### Key Findings That Generalize Across Both Model Families

1. Inference is memory-bandwidth bound. Throughput tracks weight size, not
   parameter quality. Roofline model predicts results before running.

2. AWQ naive penalty is model-agnostic. Both models hit ~20.8 tok/s. Kernel
   property, not model property.

3. Marlin speedup scales with weight size. Larger models see slightly lower
   relative speedup because unquantized operations (attention, layernorm) become
   a larger fraction of total time.

4. Speculative decoding requires same training family, not just same vocabulary.
   Wrong family: 0.7x BF16. Correct family: 1.14x BF16 at low concurrency.

5. Draft acceptance rate is stable across both runs (~60%). The 1B/8B Llama
   pairing is a legitimate production configuration.

```markdown
---

## Phase 5 Part Appendix — TTFT Benchmark & Hardware Comparison Setup

### Purpose
Measure Time to First Token (TTFT) and decode throughput using a streaming benchmark
methodology designed to produce results directly comparable to published Groq LPU
benchmarks. Establishes the A100 SXM4 baseline for the Phase 6 alternative hardware
architecture comparison.

Hardware: 2x A100 SXM4 80GB (RunPod), vLLM 0.18.0.
Script: phase5-inference/benchmark_ttft.py

### Methodology
- Fixed prompt: ~100 input tokens (same text across all runs)
- Fixed output: 200 tokens (temperature=0, greedy decoding)
- Streaming enabled: measures TTFT as time to first chunk received
- 1 warmup run (excluded), 5 timed runs, report mean
- Single request, no batching — matches Groq's published benchmark methodology

### Why TTFT Requires Streaming
Non-streaming benchmarks measure total wall-clock time only — prefill + decode
collapsed into one number. Streaming sends tokens as they are generated, allowing
measurement of the exact moment the first token arrives. TTFT = time from request
sent to first token received. Everything after is decode throughput.

### Benchmark Results

| Model         | Precision | TTFT (ms) | tok/s | Total (ms) | Hardware             | Config |
|---------------|-----------|-----------|-------|------------|----------------------|--------|
| Llama 3.1 8B  | BF16      | 18.1      | 93.0  | 2,157      | A100 SXM4 80GB       | TP=1   |
| Llama 3.3 70B | BF16      | 58.0      | 21.2  | 9,436      | 2x A100 SXM4 80GB    | TP=2   |

### Key Observations

**Warmup effect on TTFT:**
Warmup run TTFT was 42ms for 8B vs 18ms steady state. Gap caused by CUDA graph
initialization — first request builds execution graph, subsequent requests reuse it.
Warmup runs matter. The 18.1ms figure reflects real steady-state performance.

**Prefix cache effect:**
vLLM's prefix cache hit rate climbed to 74% by run 5 — the same 100-token prompt
repeated across runs caused KV cache reuse. TTFT on runs 2-5 slightly benefits from
this. In production with unique prompts per request, cold TTFT would be marginally
higher. Noted for methodology transparency.

**70B TTFT vs 8B:**
TTFT scales with model size — 80 layers vs 32 layers means more compute to process
the prompt before producing the first token. 58ms vs 18ms is a 3.2x ratio, roughly
proportional to layer count difference.

**70B KV cache constraint:**
With 134GB of weights split across 2x 80GB GPUs, only 3.93 GiB remained for KV
cache (25,728 token capacity). Adequate for single-request benchmarking (300 tokens
total) but severely limits concurrent serving capacity. 70B BF16 on 2x A100 is a
viable single-user configuration, not a production multi-user configuration.

**Decode throughput gap:**
70B decode throughput (21.2 tok/s) vs 8B (93.0 tok/s) is a 4.4x gap — not 8.75x
as raw parameter ratio would suggest. TP=2 distributes weight loading across both
GPUs, partially recovering bandwidth. Each GPU only needs to load half the model
weights per decode step. TP=1 on a single GPU would be slower still.

### Groq Comparison Preview (Full Analysis in Phase 6)

Published Groq LPU benchmarks (Artificial Analysis, independent):

| Model         | tok/s (standard) | tok/s (spec dec) | TTFT (API)  |
|---------------|------------------|------------------|-------------|
| Llama 3.1 8B  | 877              | --               | ~200ms      |
| Llama 3.3 70B | 276              | 1,665            | ~450ms      |

A100 vs Groq throughput gap:
- 8B: Groq 877 vs A100 93.0 — 9.4x faster on Groq
- 70B: Groq 276 vs A100 21.2 — 13x faster on Groq (standard mode)

A100 vs Groq TTFT:
- 8B: A100 local 18ms vs Groq API ~200ms — A100 wins by 11x
- 70B: A100 local 58ms vs Groq API ~450ms — A100 wins by 7.8x

The TTFT reversal is explained by network round-trip. Groq's API TTFT includes
client-to-datacenter latency. Local vLLM has none. For on-premises deployment,
A100 TTFT is competitive. For cloud API users, Groq's throughput advantage is real
and the TTFT difference shrinks with longer outputs.

The architectural reason for Groq's throughput advantage is covered in Phase 6:
SRAM-centric execution eliminates the HBM bandwidth bottleneck that makes A100
inference memory-bound. Full analysis there.
```

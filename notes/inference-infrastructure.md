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

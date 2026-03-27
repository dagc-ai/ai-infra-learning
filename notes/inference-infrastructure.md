# Inference Infrastructure

## Exercise 5.1 — vLLM Deployment & Continuous Batching

### Core Technical Findings

**Inference is a memory problem, not a compute problem.**
The GPU has 312 TFLOPS of compute sitting mostly idle during inference. What limits 
token generation speed is how fast weights can be loaded from VRAM. Every token 
requires reading the entire model from HBM. More compute wouldn't help. More memory 
bandwidth would.

**Continuous batching is free throughput.**
Continuous batching allows vLLM to combine multiple concurrent requests into a single 
forward pass. Because inference is memory-bandwidth bound — the bottleneck is loading 
model weights from HBM, not compute — serving 8 requests in one pass costs nearly the 
same time as serving 1. Throughput scales linearly with concurrency while latency stays 
flat, as long as the GPU is not yet saturated.

**Memory determines everything about what you can serve.**
Mistral 7B BF16 consumed 13.5GB of 80GB VRAM. The remaining ~57GB became KV cache — 
which determines concurrent user capacity and maximum context length. Every deployment 
decision flows from this memory budget.

### Benchmark Results — A100 SXM 80GB, Mistral 7B BF16

Single request baseline: 96.5 tok/s, 2073ms

Concurrent requests:
- Concurrency 1:  94.7 tok/s | 2110ms latency
- Concurrency 2: 193.0 tok/s | 2066ms latency
- Concurrency 4: 382.7 tok/s | 2083ms latency
- Concurrency 8: 752.3 tok/s | 2110ms latency

Implied HBM bandwidth utilization at concurrency=1: ~1.4 TB/s (~70% of A100 peak 2.0 TB/s)

### Business Insights

**Cost per token is the unit economics of inference.**
At 752 tok/s on a $1.49/hr A100, you generate ~2.7M tokens/hour. Raw compute cost: 
~$0.00055 per 1K tokens before markup. GPU utilization is the lever that controls margin.

**Memory capacity is the binding constraint for enterprise deployment.**
The question enterprises face is not "can we run AI" but "can we run this model at this 
context length at this concurrency within our hardware budget." Memory hierarchy 
constraints from Phase 1 directly determine enterprise infrastructure spend.

**Latency and throughput are a product decision, not just a technical one.**
Chatbots need low TTFT. Batch pipelines need high throughput. These are different 
deployment configurations of the same model on the same hardware. Product teams who 
understand this tradeoff make better decisions about SLAs, pricing tiers, and 
infrastructure sizing.

### GTM Implications

**Self-host vs. managed API inflection point.**
At roughly 5-10M tokens/day, self-hosting economics start to dominate over managed 
APIs. Below that, simplicity wins. Above that, cost wins. Knowing where a customer 
sits on that curve determines which conversation to have.

**Every inference vendor is selling solved utilization.**
vLLM, Anyscale, Together AI, Fireworks, Groq — all selling a version of "we solved 
the utilization problem better than you can yourself." Differentiation is hardware 
(Groq LPU), software (PagedAttention), or operational expertise. Understanding the 
underlying constraints lets you evaluate those claims rather than take them at face value.

## Exercise 5.2 — Quantization Benchmarks

### Core Technical Findings

**Quantization format matters as much as quantization level.**
INT4 with the wrong kernel (naive AWQ) was 5x slower than BF16. Same weights, wrong
execution path. Dequantization overhead exceeded bandwidth savings. INT4 with the right
kernel (AWQ Marlin) was 2.27x faster. The difference was one flag.

**Why Marlin wins.**
Naive AWQ dequantizes INT4 weights back to FP16 before the matrix multiply — a separate
expensive step. Marlin fuses dequantization directly into the matrix multiply kernel,
eliminating the overhead. Same math, fewer memory round trips. Same tiling principle
from Phase 2 CUDA kernels applied to quantization.

**Why you don't get the full 4x theoretical speedup.**
Attention and KV cache stay in FP16 — only linear layer weights are quantized. As weight
loading gets faster, those unquantized operations become a larger fraction of total time.
The roofline bottleneck shifts.

**The memory benefit is real regardless of throughput.**
AWQ INT4 is 3.5x smaller — 3.88GB vs 13.51GB. Freed 9GB of VRAM for KV cache,
increasing max concurrency from 57x to 67x at the same context length.

### Benchmark Results — A100 SXM 80GB, Mistral 7B

Format           | Model Size  | Throughput   | vs BF16 | Kernel
-----------------|-------------|--------------|---------|-------------
BF16             | 13.51 GiB   |  96.5 tok/s  |  1.0x   | FlashAttn
AWQ INT4 naive   |  3.88 GiB   |  20.9 tok/s  |  0.2x   | naive AWQ
AWQ INT4 Marlin  |  3.88 GiB   | 218.9 tok/s  |  2.27x  | fused Marlin

### Business Insights

**Misconfiguration is more common than hardware shortage.**
The AWQ naive vs Marlin gap is 10x on identical hardware and identical weights.
Customers running naive quantization are leaving 10x performance on the table from
a misconfigured inference stack, not insufficient hardware.

**Quantization is a memory budget decision, not just a speed decision.**
INT4 doesn't just make generation faster — it changes what models are possible within
a fixed hardware budget. Fitting a 70B model on a single A100 80GB requires INT4.
In BF16 it doesn't fit.

**Quality degradation is workload-dependent.**
AWQ calibrated quantization minimizes quality loss vs naive INT4 rounding. For
summarization, classification, structured extraction — quality delta is often
negligible. For legal, medical, code generation — warrants evaluation. Not
one-size-fits-all.

### GTM Implications

**Audit before upsell.**
A customer running naive AWQ gets 10x improvement from a config change, not a
hardware purchase. Leading with that conversation builds credibility and earns
the right to the hardware conversation later — grounded in real numbers.

**Quantization unlocks larger models on existing hardware.**
A customer who can't run 70B in BF16 on an A100 can run it in INT4 for the same
hardware cost. The model capability conversation and infrastructure cost conversation
are directly connected.

**Every inference API is making quantization decisions on your behalf.**
Self-hosted inference gives enterprises control over the quality/speed tradeoff.
For workloads with quality SLAs, that control has real value.

# The Full Stack View
## Tracing a Single Token Through the Entire AI Compute Stack

**Author:** Daniel Guerrero
**Repo:** github.com/dagc-ai/ai-infra-learning
**Status:** Phase 7 capstone, synthesizes Phases 0-7

---

## The Central Thesis

Every layer of the AI compute stack has a binding constraint and an optimization
regime. The engineer who can identify which regime they are in, before deciding
how to optimize, makes correct decisions fast. The engineer who cannot wastes
resources optimizing the wrong variable.

At the hardware level the tension is between memory bandwidth and compute
throughput. At the distributed training level it is between compute and
communication. At the inference level it is between latency and throughput.
At the model training level it is between model capacity and data volume.
The variables change at every layer. The structure of the tension does not.

This document traces a single token, the letter "R" at the start of "ROMEO:",
from raw text through every layer of the stack, connecting each transformation
to the constraint that governs it and the phase where we measured it.

---

## Layer 0 - Silicon: Transistors, SRAM, HBM

### What It Does
Moves bits between memory and compute units. Every operation in every layer
above this one ultimately reduces to: read data from memory, perform arithmetic,
write result back to memory.

### The Constraint: Memory Bandwidth
The binding constraint at this layer is not how fast arithmetic can be performed.
Modern GPUs have enormous compute capacity. The constraint is how fast data can
move between memory and the arithmetic units. An A100 can perform 312 TFLOPS of
BF16 arithmetic but can only feed those units at 2.0 TB/s from HBM. For
operations with low arithmetic intensity (few FLOPs per byte read), the compute
units sit idle waiting for data.

### The Memory Hierarchy
```
Registers     ~1 cycle    ~256KB per SM     private to thread
SRAM          ~5 cycles   ~228KB per SM     shared within block
L2 Cache      ~50 cycles  ~50MB             shared across SMs
HBM           ~500 cycles ~80GB (A100)      off-chip, high bandwidth
NVLink        ~1us        600 GB/s (A100)   between GPUs
InfiniBand    ~5us        400 Gb/s          between nodes
```

The cost of a memory access increases by roughly 100x at every level of this
hierarchy. Every optimization technique across all seven phases is, at its core,
an attempt to keep data in faster memory longer.

### Phase 0-1 Measurements
RTX 4090 baseline matmul achieved 99.7% of FP32 peak, confirming compute-bound
behavior at large matrix sizes. Memory bandwidth measurement established the
roofline model as the primary analytical tool for all subsequent phases.

### Connection Up
The token "R" is stored somewhere in this hierarchy. Where it lives, whether
register, SRAM, or HBM, determines how fast it can be accessed by every kernel
above.

---

## Layer 1 - Kernels: CUDA and Triton

### What It Does
Implements the mathematical operations of the model: matrix multiplications,
softmax, attention, as programs that run directly on GPU hardware. The kernel
programmer's job is to orchestrate data movement through the memory hierarchy
to maximize arithmetic intensity.

### The Constraint: Arithmetic Intensity
Every kernel operates in one of two regimes:

- Memory-bandwidth bound: FLOPs/byte ratio is low. The GPU finishes its
  arithmetic and waits for the next data transfer. Optimization target: reduce
  memory round trips.
- Compute bound: FLOPs/byte ratio is high. Data arrives faster than it can
  be processed. Optimization target: increase parallelism and utilization.

Element-wise operations such as ReLU, dropout, and elementwise add are
memory-bandwidth bound. Matrix multiplications are compute bound at large sizes.
Softmax sits in between and depends on sequence length.

### The Tiling Pattern
The universal kernel optimization: load a tile of data from HBM into SRAM,
perform as much arithmetic as possible on it without going back to HBM, then
write results. The tiled matmul in Phase 2 demonstrated this concretely. Naive
matmul made O(N^3) HBM accesses; tiled matmul reduced this dramatically by
reusing data loaded into SRAM across multiple operations.

### Flash Attention
Flash Attention is the tiling pattern applied to the attention operation.
Standard attention materializes the full NxN attention matrix in HBM, requiring
O(N^2) memory. Flash Attention tiles the computation: loads Q, K, V chunks into
SRAM, computes a partial softmax result using the online softmax algorithm with
a running max and sum, accumulates into an output tile, and never materializes
the full NxN matrix. Memory complexity drops from O(N^2) to O(N).

Phase 3 measurement: Flash Attention Triton kernel 6.4x faster than naive at
N=4096. The speedup is entirely from reduced memory bandwidth usage. Same FLOPs,
dramatically fewer HBM round trips.

### Phase 2-3 Measurements
- Triton vector add: 924 GB/s (92% of RTX 4090 peak bandwidth)
- Fused softmax: 2x over unfused PyTorch
- Flash Attention: 6.4x over naive at N=4096
- Tiled matmul: 5-10x over naive, approaching cuBLAS

### Connection Up
The token "R" has been converted to an integer ID by the tokenizer. That integer
will be used as an index into the embedding table, a simple lookup that is
completely memory-bandwidth bound. The embedding lookup is the first kernel that
touches our token.

---

## Layer 2 - Distributed Training: AllReduce and Collective Communications

### What It Does
Coordinates gradient synchronization across multiple GPUs during training.
When a model is too large to fit on one GPU, or when training requires more
throughput than one GPU provides, multiple GPUs must work together and their
gradients must be averaged after every backward pass.

### The Constraint: Interconnect Bandwidth
At this layer the binding constraint shifts from HBM bandwidth to interconnect
bandwidth, the speed at which data moves between GPUs and between nodes.
```
NVLink (within node):       600 GB/s bidirectional (A100)
PCIe (within node):          64 GB/s bidirectional
InfiniBand (between nodes): 400 Gb/s
```

The ratio of interconnect bandwidth to compute throughput determines whether
a distributed training job is compute-bound or communication-bound. At 8 GPUs
within a node connected by NVLink, the ratio is favorable and training stays
compute-bound. Across hundreds of nodes connected by InfiniBand, communication
overhead dominates and AllReduce becomes the bottleneck, not GPU utilization.

### Ring AllReduce
The standard algorithm for gradient synchronization. Each GPU sends its chunk
to the next GPU in a ring, accumulates a partial sum from the previous GPU, and
passes it along. After N-1 steps, every GPU has the full sum. Bus bandwidth
utilization approaches 100% of available bandwidth as N grows.

### Phase 4 Measurements
- DDP AllReduce verified empirically on 4x A100 SXM4
- Ring AllReduce implemented from scratch, matching NCCL throughput
- NVLink bandwidth measured at 228 GB/s (38% of spec in virtualized environment)
- Gap between measured and spec attributed to virtualization overhead

### Connection Up
Our token "R" does not directly touch this layer during inference. But during
the training run that produced the model weights which will process "R", every
gradient update to every parameter was synchronized through AllReduce. The
quality of the model that processes our token is a direct function of how cleanly
gradients were synchronized across the training cluster.

---

## Layer 3 - Inference Serving: vLLM, KV Cache, Quantization

### What It Does
Takes a trained model and serves it efficiently to handle concurrent requests.
The inference layer manages memory, batches requests, and optimizes the
autoregressive generation loop.

### The Constraint: Memory Capacity and Bandwidth
Inference has two tensions that training does not:

1. Latency vs. throughput: minimizing time-to-first-token pulls against
   maximizing tokens-per-second-per-GPU. Larger batches improve throughput
   but increase latency.
2. Model weights vs. KV cache: GPU memory must accommodate both the model
   parameters and the growing KV cache for active requests.

The KV cache is the central memory management problem. During autoregressive
generation, every new token attends to all previous tokens. The Key and Value
matrices for those tokens must be stored and they grow linearly with sequence
length and batch size. For a 70B model at 100K context with batch=1, the KV
cache alone exceeds 50GB.

### PagedAttention
vLLM's core innovation: manage KV cache like OS virtual memory. Instead of
allocating contiguous blocks per sequence (which causes fragmentation), allocate
fixed-size pages and maintain a page table mapping logical to physical addresses.
Sequences can grow without fragmentation; pages can be shared across requests
using the same prompt prefix. The result: higher batch sizes, better GPU
utilization, same model quality.

### Quantization
Reducing numerical precision to reduce memory footprint and increase throughput:
- BF16: 2 bytes/param, negligible quality loss, native A100 tensor core support
- INT8: 1 byte/param, small quality loss, 2-3x throughput gain
- INT4: 0.5 bytes/param, noticeable quality loss without careful implementation

The critical nuance: INT4 throughput depends entirely on the dequantization
kernel. Naive INT4 (dequantize then multiply in FP16) is slower than BF16
because dequantization adds overhead. Marlin kernel (fused dequantize + matmul)
achieves 2.27x over BF16 baseline.

### Phase 5 Measurements
- vLLM on A100 SXM4 80GB: Llama 3.1 8B BF16 at ~96.5 tok/s
- AWQ INT4 naive: 5x slower than BF16
- AWQ Marlin INT4: 2.27x faster than BF16
- Speculative decoding failure case documented with root cause

### Connection Up
When "ROMEO:" is submitted as a prompt, the inference server tokenizes it,
allocates KV cache pages, runs the prefill pass (processing all prompt tokens
in parallel), then enters the autoregressive generation loop: one forward pass
per output token, KV cache growing by one entry per step.

---

## Layer 4 - Alternative Hardware: Different Answers to the Same Question

### What It Does
Every accelerator architecture is an engineering answer to one question:
where is the binding constraint in AI workloads?

### The Constraint: Architecture-Dependent
Different hardware bets on different bottlenecks:

| Architecture | Primary Bet | Key Spec |
|-------------|-------------|----------|
| NVIDIA H100 | Flexibility across training and inference | 80GB HBM3, NVLink 900 GB/s |
| Tenstorrent Wormhole | On-chip bandwidth beats off-chip capacity | SRAM-centric, NoC mesh |
| AMD MI300X | Memory capacity is the binding constraint | 192GB HBM3, 5.3 TB/s |
| Google TPU v5 | General-purpose overhead is waste | Systolic array, XLA compiler |
| Groq LPU | Latency variance is the real problem | Deterministic execution, no cache |

The MI300X unified memory architecture (192GB shared CPU+GPU) directly addresses
the model capacity constraint. Large models that require offloading on NVIDIA
hardware fit entirely on-chip. The Tenstorrent bet, that SRAM bandwidth matters
more than HBM capacity for inference workloads, mirrors the Flash Attention
insight: keep computation near data, minimize off-chip movement.

### Phase 6 Measurements
- Groq LPU: Llama 3.1 8B at 672.9 tok/s (7.2x over A100 baseline)
- TPU v5e: 4096x4096 BF16 matmul at 142.3 TFLOPS (72.2% utilization)
- XLA auto-fused matmul+softmax+matmul: what Triton Flash Attention did
  manually is the default behavior on TPU
- Meta-finding: NVIDIA's deepest moat is ecosystem maturity, not peak specs.
  Every alternative hardware deployment failed on access friction alone.
  A100 deployed and benchmarking in under 20 minutes every time.

### Connection Up
Our token "R" could in principle be processed by any of these architectures.
The hardware choice determines throughput, latency, and cost per token, not
the mathematical operations, which are identical across all of them. The same
attention computation runs everywhere; only the physical implementation of
data movement differs.

---

## Layer 5 - Model Architecture: What the Numbers Mean

### What It Does
Defines the mathematical operations that transform token representations into
predictions. The transformer architecture is a sequence of learned transformations
that progressively build up a contextual representation of each token.

### Tracing "R" Through the Model

**Step 1 - Tokenization**

"ROMEO:" gets encoded by the GPT-2 BPE tokenizer. In GPT-2's vocabulary,
"ROMEO" may encode as one or more subword tokens depending on frequency.
The first character "R" is part of the first token ID in that sequence.

Each token ID is an integer index into the embedding table.

**Step 2 - Embedding Lookup**

Token ID for "R" (or the subword containing it) maps to a 384-dimensional
vector via the token embedding table (wte).
Position 0 maps to a separate 384-dimensional vector via the position
embedding table (wpe).
These two vectors are summed: one 384-dimensional point in embedding space.

This operation is memory-bandwidth bound. It is a lookup into a table of
50,257 x 384 = 19.3M parameters. No arithmetic intensity, pure data fetch.

**Step 3 - Transformer Block (x6)**

Each of the 6 blocks applies two transformations with residual connections:

*Attention:*

The representation of "R" generates three vectors via learned linear projections:
- Query (Q): "what information am I looking for from other tokens?"
- Key (K): "what information do I offer to tokens querying me?"
- Value (V): "what content will I contribute if another token attends to me?"

Q is compared against the K of every other token via dot product, scaled by
1/sqrt(head_dim) to prevent magnitude growth, then passed through softmax to
produce attention weights. Those weights sum to 1.0 across all positions.
The weighted sum of all V vectors produces an updated representation of "R"
that has absorbed context from every token it chose to attend to.

This runs in parallel across 6 independent heads (n_head=6), each operating
in a 64-dimensional subspace (head_dim = 384/6 = 64). Each head can learn
to track different relationship types simultaneously. The 6 outputs are
concatenated and projected back to 384 dimensions.

The causal mask ensures "R" can only attend to itself and tokens before it,
never tokens that follow. This is enforced by setting future positions to
negative infinity before the softmax, producing exactly 0.0 attention weight.

*Residual connection:*

x = x + attention(LayerNorm(x))

The block learns only the delta, what to add to the current representation.
At initialization, the attention output is near zero and the block is nearly
an identity function. Gradients flow backward through the addition directly
to earlier layers without passing through the attention computation. This is
the gradient highway that makes 6-layer (and 96-layer) networks trainable.

*MLP:*

x = x + mlp(LayerNorm(x))

Linear(384 -> 1536) -> GELU -> Linear(1536 -> 384).
Expands to 4x dimension, applies a smooth nonlinearity, projects back.
Applied independently at every token position. Where factual associations
are encoded and retrieved.

After 6 blocks, "R" has been transformed from a static lookup vector into
a context-aware representation that encodes its relationship to every other
token in the sequence.

**Step 4 - Output Head**

Final 384-dimensional representation -> Linear(384 -> 50257) -> logits.

The logits are raw scores over the full vocabulary. The highest score is the
model's prediction for what token comes next after the full prompt context.
For "ROMEO:" a well-trained model assigns high probability to a newline or
space token, because that character pattern is almost always followed by
spoken dialogue in Shakespeare.

During training: softmax(logits) is compared to the actual next token via
cross-entropy loss. The gradient of that loss flows backward through all
6 blocks, updating every parameter in the direction that reduces surprise.

During inference: sample from softmax(logits / temperature), append the
sampled token to the context, repeat.

### The Constraint: Capacity vs. Data

The binding constraint for model quality is not architecture. It is the ratio
of model capacity (non-embedding parameters) to training data volume (tokens).

Too many parameters for available data: the model memorizes rather than
generalizes. Train loss approaches zero while val loss climbs. Generated text
is verbatim recitation of training data.

Too few parameters for task complexity: the model underfits. Both losses
plateau at a high value. The model has learned frequency statistics but
not structure.

The Chinchilla-optimal ratio: approximately 20 tokens per parameter. Below
this ratio, adding more training steps hurts rather than helps.

### Phase 7 Measurements

| Config | Params | Val Loss | Behavior |
|--------|--------|----------|----------|
| 30M model | 30.02M | 9.58 | Memorization, verbatim recitation |
| 7.25M model | 7.25M | 5.70 | Partial generalization |
| Scaling law fit | 4 models | alpha=0.0321 | Power law holds for 3 of 4 |
| Chinchilla effect | small model | worse than mini | 174x data-starved |

Training stability finding: AdamW, residual connections, LayerNorm, and
gradient clipping are collectively robust at small scale. Individual failure
modes require removing multiple components simultaneously. At 70B+ parameters
on trillion-token datasets, each mechanism becomes individually critical.

---

## The Full Stack in One Table

| Layer | What It Does | Binding Constraint | Key Metric |
|-------|-------------|-------------------|------------|
| Silicon | Move bits between memory and compute | Memory bandwidth | FLOP/byte (arithmetic intensity) |
| Kernels | Implement math operations on GPU | HBM round trips | GB/s achieved vs. peak |
| Distributed | Synchronize gradients across GPUs | Interconnect bandwidth | AllReduce bus bandwidth |
| Inference | Serve model to concurrent requests | Memory capacity | KV cache utilization |
| Alt Hardware | Different memory/compute tradeoffs | Architecture-dependent | Tokens/sec/dollar |
| Model | Transform token representations | Capacity/data ratio | Train vs. val loss gap |
| Application | Prompts, evals, agents | Reliability, latency | Task-specific eval scores |

---

## The Recurring Principle

At every layer, the same structure repeats:

**1. There is a binding constraint.** The one resource that, if doubled, would
most improve the system. Identifying it correctly is the primary analytical skill.

**2. There are two regimes.** Defined by which side of the constraint you are
on. Optimizing the non-binding resource wastes effort.

**3. The constraint moves up the hierarchy as scale increases.** A single GPU
is memory-bandwidth bound. A 1000-GPU cluster is communication-bound. A
trillion-parameter model is data-bound. The same principles apply; the binding
variable changes.

**4. Techniques at each layer answer the same question.** How do I get more
useful work from the binding resource? Flash Attention reduces HBM round trips.
Ring AllReduce maximizes interconnect utilization. PagedAttention eliminates KV
cache fragmentation. Chinchilla-optimal training maximizes information extracted
per parameter.

The roofline model, introduced in Phase 1 as a tool for analyzing GPU kernel
performance, is not a GPU-specific concept. It is a general framework for
identifying binding constraints in any compute system. It applies to kernel
optimization, distributed training topology design, inference serving
architecture, and model training budget allocation.

Everything else is a specific instantiation of this principle at a specific
layer of the stack.

---

## What This Curriculum Built

Six months. Seven phases. One mental model.

The goal was never to become an ML engineer. It was to build accurate intuition
about how AI infrastructure actually works, from the physical movement of bits
through memory hierarchies to the mathematical operations that give language
models their capabilities, so that conversations about AI infrastructure are
grounded in first principles rather than marketing claims.

The mental model is complete. A token enters. Hardware moves it through a memory
hierarchy optimized for arithmetic intensity. Kernels compute attention using
tiling algorithms that minimize HBM round trips. Gradients were synchronized
across GPUs during training using AllReduce over high-bandwidth interconnects.
The model was trained to a Chinchilla-optimal parameter/data ratio. The inference
server manages KV cache pages to maximize concurrent throughput.

Every number in every benchmark in this repo is a measurement of one of these
constraints. Every optimization technique is an answer to one of these tensions.
The stack is deep. The principle is simple.

---

*Phase 7 complete. Phases 0-7 complete.*
*github.com/dagc-ai/ai-infra-learning*

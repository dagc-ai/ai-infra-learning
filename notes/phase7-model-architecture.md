# Phase 7 — Model Architecture & The Full Stack View
## Notes: Exercises, Findings, and Insights

**Hardware:** A100 SXM4 80GB (RunPod)
**Stack:** PyTorch 2.4.0, CUDA 12.4, Python 3.11, Ubuntu 22.04
**Repo:** github.com/dagc-ai/ai-infra-learning

---

## Exercise 7.1 — Data Preparation

### Goal
Convert raw text into a format a neural network can consume. Establish the
dataset that all subsequent training runs will use so results are directly
comparable across exercises.

### Experiment
Downloaded the TinyShakespeare corpus (~1.1MB of concatenated Shakespeare plays).
Tokenized using the GPT-2 BPE tokenizer (tiktoken). Split 90/10 into train and
validation sets. Saved as binary uint16 arrays for fast memory-mapped loading
during training.

### Key Data
| Metric | Value |
|--------|-------|
| Raw characters | 1,115,394 |
| Total tokens | 338,025 |
| Vocabulary size | 50,257 (GPT-2 BPE) |
| Compression ratio | 3.30 chars/token |
| Train tokens | 304,222 |
| Val tokens | 33,803 |
| train.bin size | 0.6 MB |
| val.bin size | 0.1 MB |

### Technical Insights
BPE tokenization compresses English text at ~3.3 characters per token by
iteratively merging the most frequent adjacent character pairs until reaching
the target vocabulary size. Common words become single tokens; rare words split
into subword pieces. This compression ratio directly determines sequence length
for a given input — shorter sequences mean cheaper attention computation, since
attention cost scales as O(N²) with sequence length.

Using uint16 (not int32) halves storage and memory bandwidth requirements. uint16
covers 0–65,535, which comfortably holds GPT-2's 50,257 token vocabulary.
Memory-mapped I/O (np.memmap) means the OS loads only requested chunks from disk
rather than the entire dataset into RAM — the right pattern for large datasets
even when the dataset itself is small.

### Developer Insights
The train/val split is a contract: the model never sees validation tokens during
training. Validation loss is only meaningful if this contract holds. Any leakage
— even indirect, through hyperparameter selection on val loss — inflates apparent
generalization performance.

For large corpora, data preparation pipelines are often more complex than the
training loop itself. Deduplication, quality filtering, tokenization at scale,
and domain mixing ratios are active research areas at frontier labs. Our pipeline
is the minimal correct version.

### Business Insights
The tokenized dataset size — 304,222 training tokens — is the actual information
budget the model has to learn from. Not characters, not words, not documents.
Tokens. When evaluating an enterprise's capacity to train a custom model, this is
the number that matters. Most companies dramatically overestimate their effective
token count before deduplication and quality filtering.

### GTM Implications
"We have millions of documents" is not a data size claim. Tokenized, deduplicated,
quality-filtered token count is. A company with 500,000 internal documents averaging
500 words each has roughly 500M tokens before filtering — enough for a Chinchilla-
optimal ~25M parameter model. Establishing this number early in a customer
conversation prevents misaligned expectations about custom model feasibility.

---

## Exercise 7.2 — Model Implementation (nanoGPT)

### Goal
Build a complete GPT-style transformer from scratch in PyTorch. Every design
decision explicitly annotated. Every tensor shape tracked. Goal is to hold the
entire model in working memory — not just know that transformers exist.

### Experiment
Implemented the full model in ~250 lines of PyTorch: token and position
embeddings, causal multi-head self-attention with masking, position-wise MLP
with GELU activation, residual connections, pre-norm LayerNorm, and an
autoregressive generation function. Ran a sanity check with random inputs to
verify output shapes and initialization loss.

### Key Data
| Component | Parameters (n_embd=384, n_layer=6, n_head=6) |
|-----------|----------------------------------------------|
| Token embedding (wte) | 50,257 × 384 = 19.3M |
| Position embedding (wpe) | 256 × 384 = 98K |
| Per transformer block | ~1.77M |
| 6 transformer blocks | ~10.6M |
| **Total** | **30.02M** |

**Sanity check results:**
| Metric | Expected | Actual |
|--------|----------|--------|
| Logits shape | (4, 64, 50257) | (4, 64, 50257) ✓ |
| Initial loss | ~10.8249 (ln 50257) | 10.9261 |
| Loss gap from random | ~0.0 | 0.1012 ✓ |

### Technical Insights
**The embedding table dominates at small n_embd.** With vocab_size=50,257 and
n_embd=384, the token embedding table alone consumes 19.3M of 30M total parameters
— 64% of the model before a single transformer layer runs. This is the silent
budget killer at small model sizes: you think you have a 30M parameter reasoning
system; you actually have a 10.6M parameter reasoning system with a 19.3M parameter
lookup table attached.

**Weight tying reduces parameters and improves performance.** The token embedding
matrix (vocab→vectors) and the output head (vectors→vocab logits) are transposes
of each other conceptually — one encodes tokens into space, the other decodes back.
Tying their weights (sharing the same matrix) reduces parameters by 19.3M and
empirically improves language modeling performance. Standard since Press & Wolf (2017).

**The causal mask is a hard constraint, not a soft regularizer.** Future tokens
are set to -inf before softmax, which maps to exactly 0.0 attention weight. This
isn't learned — it's enforced by construction. The model cannot attend to future
tokens under any circumstances. This is what makes the same architecture work for
both training (parallel, all positions at once) and inference (sequential,
one token at a time).

**Initial loss of 10.926 vs expected 10.825 (gap: 0.10).** A randomly initialized
model should assign equal probability to all 50,257 tokens, giving loss = ln(50257)
= 10.825. Slight deviation is expected from non-uniform random initialization.
Gap > 0.5 would indicate a bug in initialization. Gap of 0.10 is clean.

### Developer Insights
Annotating every tensor shape is not optional when implementing attention from
scratch. The reshape/transpose sequence that converts (B, T, C) → (B, n_head, T,
head_dim) → back to (B, T, C) is where most implementation bugs appear. Tracking
shapes explicitly catches dimension mismatches before they become silent numerical
errors.

The `contiguous()` call before `view()` is a common CUDA-specific requirement.
Transpose operations create non-contiguous memory layouts; view() requires
contiguous memory. Forgetting this produces a cryptic RuntimeError at the reshape
step — not at the transpose. The fix is always `.contiguous().view(...)`.

Pre-norm vs post-norm matters for training stability at depth. The original
"Attention Is All You Need" paper applied LayerNorm after the residual connection
(post-norm). GPT-2 and essentially all subsequent models use pre-norm — LayerNorm
applied before the sublayer, not after. Pre-norm training is more stable,
especially beyond 12 layers.

### Business Insights
Understanding the model architecture at this level makes parameter count claims
legible. When a vendor says "our 70B model" — you can now reason that approximately
6.5% of those parameters are the embedding table (128K vocab × 4096 dims = 524M),
leaving ~65.5B parameters of actual transformer computation. This is the number
that drives inference cost, not the headline parameter count.

### GTM Implications
The generation function in this implementation is the exact process that every
production LLM uses — forward pass, sample next token, append, repeat. The KV
cache optimization built in Phase 5 (vLLM, PagedAttention) exists entirely because
this loop runs once per output token. A 500-token response requires 500 sequential
forward passes through the full model. That compute cost is architectural and
non-negotiable — it can only be managed, not eliminated.

---

## Exercise 7.3 — Overfitting: The Capacity vs. Data Mismatch

### Goal
Train the 30M parameter model on TinyShakespeare and observe what happens when
model capacity dramatically exceeds dataset size. Make the overfitting failure mode
viscerally concrete rather than abstractly understood.

### Experiment
Trained the 30M parameter model for 5,000 iterations (batch=64, block=256,
81.9M total tokens processed). Logged train and validation loss every 500 steps.
Evaluated generated text qualitatively. Then repeated with a reduced model
(7.25M parameters, n_layer=4, n_head=4, n_embd=128, dropout=0.2) to observe
partial remediation.

### Key Data
**30M model training run:**
| Iteration | Train Loss | Val Loss | Notes |
|-----------|------------|----------|-------|
| 0 | 10.91 | 10.91 | Random init baseline |
| 500 | 2.49 | 5.19 | Val already diverging |
| 1000 | 0.46 | 6.94 | Severe overfit |
| 2000 | 0.14 | 8.70 | Near memorization |
| 5000 | 0.07 | 9.58 | Complete memorization |

**7.25M model training run:**
| Iteration | Train Loss | Val Loss |
|-----------|------------|----------|
| 0 | 10.84 | 10.84 |
| 500 | 4.04 | 4.98 |
| 1000 | 3.33 | 4.94 |
| 5000 | 1.92 | 5.70 |

**Throughput comparison:**
| Model | Parameters | Tok/s |
|-------|------------|-------|
| 30M | 30.02M | 203,747 |
| 7.25M | 7.25M | 389,150 |

**Data/parameter ratio:**
| Model | Ratio |
|-------|-------|
| 30M | 0.01 tokens/param |
| 7.25M | 0.042 tokens/param |

**Generated text — 30M model (verbatim Shakespeare):**
> "And so did I:--Well, we were born to die. 'Tis very late,
> she'll not come down to-night: I promise you, but for your
> company, I would have been a-bed an hour ago."

**Generated text — 7.25M model (novel generation):**
> "'Tis time, and no more unlikely / Than was a worser than a
> beggar'st man; / For a little more milder sees for a groan-
> govern'd spirit"

The 30M model output is verbatim Romeo and Juliet Act IV.
The 7.25M model output is not in the training set.

### Technical Insights
**Near-zero training loss is a red flag, not a success.** Training loss of 0.07
on next-token prediction means the model is assigning very high probability to
the correct next token at every position. On a 300K token dataset, this indicates
memorization — the model has stored the specific sequences, not the underlying
patterns. The proof is in the generated text: exact lines from the source material.

**Val loss rising above the midpoint of the random baseline indicates active harm.**
The 30M model's val loss of 9.58 is close to the random initialization baseline
of 10.82. This means the learned representations are nearly useless for predicting
unseen text — worse than a model that has simply learned token frequency statistics.
The memorized representations interfere with generalization.

**The throughput delta is the roofline model reappearing at the architecture level.**
The 7.25M model runs at 389K tok/s vs 203K tok/s for the 30M model — nearly 2x
faster. The cause is not clock speed or CUDA version. It's activation tensor size:
smaller n_embd and fewer layers means attention matrices and MLP activations fit
better in L2 cache, reducing HBM round trips. The same memory hierarchy principle
from Phase 1 applies at every level of the stack.

**The embedding table problem persists even at smaller scale.** The 7.25M model
has 50,257 × 128 = 6.4M parameters in the token embedding table, leaving only
~0.8M parameters in the transformer blocks. The model that learned to generate
novel Shakespeare did so with less reasoning capacity than a small MLP. The 7.25M
parameter claim is technically accurate and practically misleading.

### Developer Insights
The train/val loss curve shape is the primary diagnostic tool. Read the curve
before reading the final number:
- Train down, val up from early: capacity/data mismatch. Reduce model or get more data.
- Train down, val plateau: approaching capacity ceiling. More data or larger model.
- Both down in parallel: healthy. Continue training.
- Val loss rising above random baseline: extreme memorization. Stop immediately.

Generated text quality is a secondary diagnostic. Verbatim recitation means
memorization. Novel but domain-appropriate output means generalization. This
distinction is visible to the naked eye without any quantitative evaluation.

Early stopping should always be implemented. The optimal checkpoint for the 30M
model was around iteration 300 — before val loss began its terminal climb. Running
to 5,000 iterations wasted compute and produced a worse model. Production training
loops track best val loss and checkpoint accordingly.

### Business Insights
Fine-tuning a large model on a small enterprise dataset replicates this exact
experiment. A vendor fine-tuning a 70B model on 50,000 customer support tickets
is running the same capacity/data mismatch we demonstrated. The resulting model
will perform well on examples resembling the fine-tuning set and poorly on novel
inputs from the same domain — exactly the failure mode customers discover 90 days
after deployment when edge cases start surfacing.

The correct technical alternatives — LoRA (fine-tune 0.1% of parameters), RAG
(retrieval-augmented generation), or few-shot prompting — exist specifically to
avoid this failure mode. They keep the base model's generalization intact while
adapting to domain-specific content.

### GTM Implications
When a vendor says "we fine-tuned on your data," the diagnostic questions are:
1. What was your train/val loss trajectory during fine-tuning?
2. What was your parameter count vs. fine-tuning token count ratio?
3. Did you evaluate on a held-out set that was never used for hyperparameter selection?

Most vendors cannot answer question 1 cleanly. Asking it reframes the conversation
from "trust our benchmark" to "show me your loss curves." Vendors with rigorous
fine-tuning workflows will have this data. Vendors who don't will deflect to
qualitative examples — which, as this exercise demonstrated, can look convincing
even when the model is reciting memorized content.

---

## Exercise 7.4 — Scaling Law Experiment

### Goal
Train four models at increasing scale and verify empirically that validation loss
follows a power-law relationship with non-embedding parameter count. Make scaling
laws concrete rather than theoretical. Observe the Chinchilla effect when a model
is overtrained relative to its data budget.

### Experiment
Trained four models (nano, micro, mini, small) with identical architecture families
but increasing depth and width. Fixed: same dataset (TinyShakespeare), same batch
size (32), same block size (256), dropout=0 (to measure pure capacity effect
without regularization confounding). Varying: n_layer, n_head, n_embd, max_iters.
Fit a power law to the resulting val loss vs. non-embedding parameter count.

### Key Data
**Model configurations:**
| Model | n_layer | n_head | n_embd | Total Params | Non-embed Params | Max Iters |
|-------|---------|--------|--------|-------------|-----------------|-----------|
| nano  | 2 | 2 | 64  | 3.33M  | 98,944    | 2000 |
| micro | 3 | 3 | 96  | 5.18M  | 333,120   | 2500 |
| mini  | 4 | 4 | 128 | 7.25M  | 788,736   | 3000 |
| small | 6 | 6 | 192 | 12.36M | 2,659,200 | 4000 |

**Results:**
| Model | Best Val Loss | Best at Iter | Final Val Loss |
|-------|--------------|--------------|----------------|
| nano  | 5.3410 | 500  | 5.8221 |
| micro | 4.9774 | 750  | 5.6838 |
| mini  | 4.7920 | 750  | 6.7995 |
| small | 4.8181 | 500  | 8.9108 |

**Power law fit (first 3 models):**
| Metric | Value |
|--------|-------|
| Equation | L(N) = 7.5929 × N^(-0.0321) |
| Scaling exponent α | 0.0321 |
| Kaplan et al. α | 0.076 |
| Our α as fraction of Kaplan | 0.42× |
| Fit error (nano) | 1.8% |
| Fit error (micro) | 1.4% |
| Fit error (mini) | 2.4% |
| small (outlier — overtrained) | law breaks down |

**Throughput by model size:**
| Model | Non-embed Params | Tok/s |
|-------|-----------------|-------|
| nano  | 98,944    | 440,153 |
| micro | 333,120   | 380,754 |
| mini  | 788,736   | 336,571 |
| small | 2,659,200 | 247,234 |

**Chinchilla-optimal token count vs. actual:**
| Model | Non-embed Params | Optimal Tokens (20×) | Actual Tokens | Starvation Factor |
|-------|-----------------|---------------------|---------------|-------------------|
| nano  | 98,944    | 1.98M  | 304K | 6.5× underfed |
| micro | 333,120   | 6.66M  | 304K | 21.9× underfed |
| mini  | 788,736   | 15.77M | 304K | 51.8× underfed |
| small | 2,659,200 | 53.18M | 304K | 174.8× underfed |

### Technical Insights
**The scaling law held for three of four models with 1-2% prediction error.**
A two-parameter power law equation predicted validation loss from parameter count
alone, before training. This is the core Kaplan et al. finding made concrete: model
performance is predictable from scale. The relationship is not approximate — it is
precise enough to use for resource planning.

**The fourth model broke the law — and breaking it is the more important finding.**
The small model's best val loss (4.8181) was worse than the mini model (4.7920)
despite having 3.4× more non-embedding parameters. The cause: small was overtrained
by 174.8× relative to its Chinchilla-optimal token count. The scaling law assumes
each model is trained to its optimal point. Violate that assumption and the
prediction fails. The law has a hidden precondition that most citations omit.

**Our scaling exponent α = 0.0321 vs. Kaplan's 0.076.** The direction is correct
— loss decreases with scale — but the rate is slower. Two causes: (1) every model
in our experiment was data-starved to some degree, compressing the loss range and
flattening the curve; (2) BPE tokenization on 300K tokens is a harder generalization
problem than the billion-token corpora Kaplan used. The exponent is dataset-dependent,
not universal.

**Throughput scales sub-linearly with parameter count.** A 27× increase in
non-embedding parameters produced only a 1.78× slowdown (440K → 247K tok/s).
This is because the embedding table — which dominates total parameter count —
is a simple lookup with minimal compute. The transformer blocks are where matrix
multiplications happen. Adding parameters to the embedding table adds memory
without adding proportional compute. Parameter count and FLOPs are correlated
but not the same metric.

**Optimal stopping point was early for every model.** None of the four models
achieved best val loss at their final iteration. Best checkpoints occurred at
25-50% of max training steps. The loss curves peaked and then began climbing —
the signature of a model that has exhausted novel training signal and started
fitting noise. Early stopping based on val loss would have saved 50-75% of
training compute on every run.

### Developer Insights
**A scaling study costs ~5% of the full training run.** The standard workflow
before a large training commitment: train 4-6 models at 1-10% of intended scale,
fit the power law, project to target scale. If the curve is clean (1-3% fit error,
monotonically decreasing), the full run is likely to behave predictably. If it's
noisy or non-monotonic, something is wrong with the setup — data quality, training
instability, or architecture inconsistency — and fixing it at small scale is
orders of magnitude cheaper than discovering it mid-run at full scale.

**The first number to compute before any training run is the Chinchilla ratio.**
Tokens available ÷ (parameters × 20) = Chinchilla allocation fraction. If this
number is below 1.0, you are going to overfit regardless of regularization. The
choices are: reduce model size, acquire more data, or accept that you are measuring
capacity ceiling rather than optimal performance. Running a large model on small
data and reporting the results as a capability benchmark is misleading by construction.

**Dropout=0 for scaling experiments is the correct choice.** Dropout is
regularization — it artificially improves val loss relative to what the model's
raw capacity would achieve. Including dropout confounds the measurement of the
capacity-loss relationship. Kaplan et al. used dropout=0 for the same reason.
When the goal is measuring architecture scaling, remove all regularization so the
capacity signal is clean.

### Business Insights
**Chinchilla-optimal smaller models beat larger undertrained models at lower
inference cost — permanently.** Mistral 7B outperforming Llama 2 13B is a direct
consequence of this. A well-trained 7B model costs roughly half the inference
compute of a 13B model at every single API call, forever. Over millions of
inference requests, this compounds into a significant cost differential. The
decision to train smaller and more optimally is not just a training-time decision
— it's an inference cost decision that persists for the model's entire deployment
lifetime.

**To halve model loss through parameter scaling alone requires ~8,000× more
parameters** (at Kaplan's α = 0.076, 2^(1/α) ≈ 8,192). This is why the industry
is hitting a scaling wall and pivoting toward inference-time compute (chain-of-
thought reasoning, o1-style models) — the returns from adding parameters are
diminishing at a rate that makes pure scaling economically unsustainable beyond
the current frontier.

**Benchmark comparisons are only valid between Chinchilla-optimal models.**
Comparing a well-trained 7B model against an undertrained 70B model at a specific
benchmark may favor the 7B model while giving a misleading impression of the
70B model's actual capability ceiling. The right comparison requires matching
training token counts relative to parameter counts, not just parameter counts.

### GTM Implications
**"How many training tokens did this model see?" is the most underused question
in enterprise AI vendor evaluation.** Parameter count is a headline number.
Token count is the engineering number. The ratio of the two determines whether
the model was trained in the Chinchilla-optimal regime — which determines whether
its benchmark performance reflects its actual capability or its training data coverage.

**Custom model build scoping requires a data audit before an architecture decision.**
Chinchilla-optimal training requires ~20 tokens per parameter. A customer who wants
a 70B custom model needs 1.4 trillion tokens of proprietary data. If they have
500M tokens, the right model is ~25M parameters — not 70B. Building to the data,
not to the vanity parameter count, produces a better model that is permanently
cheaper to serve. This is the technically correct answer and the commercially
correct answer simultaneously.

**The scaling study workflow is a consultative differentiator.** Proposing a
4-run scaling study before committing to a training budget demonstrates that you
understand the economics of model development at a level most enterprise sales
counterparts do not. It also creates a natural checkpoint: if the scaling curve
is clean, proceed; if it's noisy, diagnose before spending. This framing positions
you as a risk-reduction partner rather than a vendor pushing a solution.

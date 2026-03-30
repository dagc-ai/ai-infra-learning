"""
train.py — Training loop for nanoGPT on Shakespeare
=====================================================
This script implements the core training loop for a GPT-style language model.

Key concepts implemented here:
  - Mini-batch gradient descent with AdamW optimizer
  - Learning rate warmup + cosine decay schedule
  - Gradient clipping (prevents training instability)
  - Periodic validation loss evaluation
  - Checkpoint saving
  - Throughput measurement (tokens/sec)

The training objective is next-token prediction (causal language modeling):
  Given a sequence of T tokens, predict the next token at every position.
  Loss = mean cross-entropy over all positions in all sequences in the batch.
"""

import os
import math
import time
import numpy as np
import torch
from model import GPT, GPTConfig


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Data
data_dir     = os.path.dirname(os.path.abspath(__file__))  # same dir as this script
block_size   = 256    # sequence length — must match GPTConfig.block_size

# Training
batch_size   = 64     # sequences per step
max_iters    = 5000   # total training steps
eval_interval= 500    # evaluate val loss every N steps
eval_iters   = 50     # number of val batches to average for stable estimate

# Optimizer
learning_rate = 1e-3  # peak learning rate (AdamW)
weight_decay  = 0.1   # L2 regularization on weights (not biases/layernorms)
beta1, beta2  = 0.9, 0.95  # AdamW momentum parameters (GPT-2 values)
grad_clip     = 1.0   # gradient clipping threshold — gradients are rescaled
                      # if their global norm exceeds this value

# Learning rate schedule
# Warmup: linearly ramp LR from 0 → learning_rate over warmup_iters steps
# Why warmup? At the start of training, weights are random. Large LR with
# random gradients causes chaotic updates. Warmup lets the model find a
# reasonable region before full LR kicks in.
warmup_iters  = 200
# Cosine decay: after warmup, LR decays following a cosine curve to min_lr
# This smoothly reduces step size as training converges.
lr_decay_iters= max_iters
min_lr        = 1e-4  # minimum LR at end of cosine decay (10% of peak)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype  = torch.bfloat16  # BF16 for A100 — better training stability than FP16
                          # (larger exponent range, less overflow risk)
                          # The A100 has native BF16 tensor cores

print(f"Device: {device}, dtype: {dtype}")


# =============================================================================
# DATA LOADING
# =============================================================================

def get_batch(split):
    """
    Sample a random batch of (input, target) sequences from the dataset.

    Uses memory-mapped I/O (np.memmap) — the OS loads only the requested
    chunks from disk rather than loading the entire dataset into RAM.
    For Shakespeare this doesn't matter much (~0.6MB), but it's the right
    pattern for large datasets.

    Target is input shifted by 1 position:
      input:  [t0, t1, t2, ..., t_{T-1}]
      target: [t1, t2, t3, ..., t_T]

    This implements next-token prediction: at every position i,
    the model sees tokens 0..i and must predict token i+1.
    The causal mask in the attention layer enforces this constraint.

    Returns:
      x: (batch_size, block_size) int64 — input token IDs
      y: (batch_size, block_size) int64 — target token IDs (x shifted by 1)
    """
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')

    # Sample batch_size random starting positions
    # Each sequence is block_size tokens; we need block_size+1 tokens total
    # (block_size for input, block_size for target, overlapping by block_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack into tensors
    # x: tokens at positions [i : i+block_size]
    # y: tokens at positions [i+1 : i+block_size+1]  (shifted by 1)
    x = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(iteration):
    """
    Warmup + cosine decay learning rate schedule.

    Three regimes:
    1. Warmup (0 → warmup_iters): linear ramp from 0 to learning_rate
    2. Decay  (warmup_iters → lr_decay_iters): cosine decay to min_lr
    3. After  (> lr_decay_iters): constant min_lr

    Why cosine decay instead of linear or step?
    Cosine decay has a smooth, gradual slowdown that tends to work better
    in practice than abrupt step drops. The slow start of the decay gives
    the model time to explore; the rapid end helps it settle into a minimum.
    """
    # Phase 1: linear warmup
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters

    # Phase 3: minimum LR after decay period
    if iteration > lr_decay_iters:
        return min_lr

    # Phase 2: cosine decay between warmup and lr_decay_iters
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1.0 → 0.0
    return min_lr + coeff * (learning_rate - min_lr)


# =============================================================================
# VALIDATION LOSS ESTIMATION
# =============================================================================

@torch.no_grad()
def estimate_loss(model):
    """
    Estimate loss on train and val splits by averaging over eval_iters batches.

    Why average multiple batches instead of using one?
    A single batch gives a noisy estimate of the true loss. Averaging over
    eval_iters=50 batches gives a much more stable number for logging.

    @torch.no_grad() disables gradient tracking — we don't need gradients
    for evaluation and this saves memory + compute.

    model.eval() switches BatchNorm and Dropout to inference mode
    (Dropout becomes identity; BatchNorm uses running stats).
    model.train() switches them back.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=dtype):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# =============================================================================
# MODEL + OPTIMIZER SETUP
# =============================================================================

config = GPTConfig(
    block_size = block_size,
    vocab_size  = 50257,
    n_layer     = 6,
    n_head      = 6,
    n_embd      = 384,
    dropout     = 0.1,
)

model = GPT(config).to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.2f}M")

# AdamW optimizer
# Weight decay is applied to weight matrices but NOT to:
#   - Biases (1D tensors)
#   - LayerNorm parameters (gain and bias)
#   - Embeddings
# This is the standard GPT-2 setup. Decaying these 1D params can hurt training.
decay_params   = [p for n, p in model.named_parameters()
                  if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters()
                  if p.dim() < 2]

optim_groups = [
    {'params': decay_params,   'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0},
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                               betas=(beta1, beta2))

n_decay   = sum(p.numel() for p in decay_params)
n_nodecay = sum(p.numel() for p in nodecay_params)
print(f"Decayed params: {n_decay/1e6:.2f}M | Non-decayed: {n_nodecay/1e6:.2f}M")


# =============================================================================
# TRAINING LOOP
# =============================================================================

print(f"\nStarting training: {max_iters} iterations, batch_size={batch_size}, "
      f"block_size={block_size}")
print(f"Tokens per iter: {batch_size * block_size:,}")
print("-" * 70)

# Track total tokens processed — useful for comparing efficiency across runs
tokens_processed = 0
t0 = time.time()

for iter in range(max_iters + 1):

    # -------------------------------------------------------------------------
    # Evaluation step
    # -------------------------------------------------------------------------
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        t1 = time.time()
        dt = t1 - t0

        # Tokens per second across the interval
        # (eval_interval steps × batch_size sequences × block_size tokens)
        if iter > 0:
            tps = (eval_interval * batch_size * block_size) / dt
        else:
            tps = 0

        print(f"iter {iter:5d} | train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f} | "
              f"lr {get_lr(iter):.6f} | "
              f"{tps:,.0f} tok/s")
        t0 = time.time()

        # Save checkpoint at each eval point
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'iter': iter,
            'val_loss': losses['val'],
        }
        torch.save(checkpoint, 'checkpoint.pt')

    if iter == max_iters:
        break

    # -------------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------------

    # Update learning rate for this iteration
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get a batch
    X, Y = get_batch('train')

    # Forward pass with automatic mixed precision
    # torch.autocast casts operations to BF16 where safe (matmuls, attention)
    # while keeping numerically sensitive ops (softmax, loss) in FP32.
    # This gives ~2x speedup on A100 with negligible quality impact.
    with torch.autocast(device_type='cuda', dtype=dtype):
        logits, loss = model(X, Y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster than zero_grad()
    loss.backward()

    # Gradient clipping
    # Clips the global L2 norm of all gradients to grad_clip=1.0.
    # If ||gradients|| > 1.0, rescale all gradients proportionally.
    # Prevents "exploding gradients" — occasional very large gradient updates
    # that can destabilize training, especially early on.
    # We'll deliberately disable this in the instability exercise (Session 3)
    # to see what happens.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    tokens_processed += batch_size * block_size


# =============================================================================
# FINAL GENERATION SAMPLE
# =============================================================================

print("\n" + "="*70)
print("Training complete. Generating sample text...")
print("="*70 + "\n")

import tiktoken
enc = tiktoken.get_encoding("gpt2")

model.eval()
# Encode a prompt and generate
prompt = "ROMEO:"
prompt_tokens = enc.encode(prompt)
context = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    generated = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=40)

decoded = enc.decode(generated[0].tolist())
print(decoded)

print(f"\nTotal tokens processed during training: {tokens_processed:,}")

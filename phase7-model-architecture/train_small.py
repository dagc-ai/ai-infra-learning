"""
train_small.py — Correctly-sized nanoGPT for TinyShakespeare
=============================================================
Same training loop as train.py with two changes:
  1. Smaller model config (3M params vs 30M) — forces generalization
  2. Higher dropout (0.2 vs 0.1) — additional regularization

Everything else identical so results are directly comparable.
"""

import os, math, time
import numpy as np
import torch
from model import GPT, GPTConfig

data_dir      = os.path.dirname(os.path.abspath(__file__))
block_size    = 256
batch_size    = 64
max_iters     = 5000
eval_interval = 500
eval_iters    = 50
learning_rate = 1e-3
weight_decay  = 0.1
beta1, beta2  = 0.9, 0.95
grad_clip     = 1.0
warmup_iters  = 200
lr_decay_iters= max_iters
min_lr        = 1e-4
device        = 'cuda'
dtype         = torch.bfloat16

def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - block_size, (batch_size,))
    x     = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y     = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=dtype):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------
# KEY CHANGE: smaller config, higher dropout
# 30M → ~3M parameters
# dropout 0.1 → 0.2
# -----------------------------------------------------------------------
config = GPTConfig(
    block_size = block_size,
    vocab_size  = 50257,
    n_layer     = 4,    # was 6
    n_head      = 4,    # was 6
    n_embd      = 128,  # was 384
    dropout     = 0.2,  # was 0.1
)

model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.2f}M  (was 30.02M)")
print(f"Data/param ratio: {304222/n_params:.1f}x  (was 0.01x)")

decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{'params': decay_params,   'weight_decay': weight_decay},
     {'params': nodecay_params, 'weight_decay': 0.0}],
    lr=learning_rate, betas=(beta1, beta2)
)

print(f"\nStarting training: {max_iters} iters, batch={batch_size}, block={block_size}")
print(f"{'iter':>6} | {'train':>10} | {'val':>10} | {'lr':>10} | {'tok/s':>10}")
print("-" * 60)

t0 = time.time()
for iter in range(max_iters + 1):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        dt = time.time() - t0
        tps = (eval_interval * batch_size * block_size) / dt if iter > 0 else 0
        print(f"{iter:6d} | {losses['train']:10.4f} | {losses['val']:10.4f} | "
              f"{get_lr(iter):10.6f} | {tps:10,.0f}")
        t0 = time.time()

    if iter == max_iters:
        break

    lr = get_lr(iter)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    X, Y = get_batch('train')
    with torch.autocast(device_type='cuda', dtype=dtype):
        logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# -----------------------------------------------------------------------
# Generation sample
# -----------------------------------------------------------------------
import tiktoken
enc = tiktoken.get_encoding("gpt2")
model.eval()
prompt_tokens = torch.tensor(enc.encode("ROMEO:"), dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    out = model.generate(prompt_tokens, max_new_tokens=300, temperature=0.8, top_k=40)
print("\n" + "="*60)
print("Generated text:")
print("="*60)
print(enc.decode(out[0].tolist()))

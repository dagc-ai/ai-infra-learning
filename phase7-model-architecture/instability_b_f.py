"""
instability_b_f.py — Experiment B: Warmup Sensitivity
======================================================
Why warmup exists:
  At initialization, weights are random. The first gradient update
  is computed from a model that has no structure — it's pure noise.
  With a large batch size, that first update is computed over many
  tokens simultaneously and carries a lot of gradient signal.
  If the LR is at full peak, this large, noisy update can push
  weights into a bad region of the loss landscape that the rest
  of training cannot recover from.

  Warmup linearly ramps LR from 0 → peak over N steps, giving
  the model time to develop basic structure before full-magnitude
  updates begin.

Engineered failure conditions:
  - batch_size=512  (16x larger than our normal 32)
    Each update covers 131,072 tokens — amplifies early chaos
  - warmup_iters=0  (jump straight to peak LR)
  - LR=1e-3         (normal peak — not elevated)
  - grad_clip=1.0   (active — isolates warmup as the variable)

Three configurations:
  1. BROKEN:   batch=512, no warmup
  2. PARTIAL:  batch=512, short warmup (50 iters instead of 200)
  3. CONTROL:  batch=512, full warmup (200 iters)

Comparing 1 vs 3 isolates warmup.
Comparing 2 vs 3 shows how much warmup length matters.
"""

import os
import math
import time
import numpy as np
import torch
from model_f import GPT, GPTConfig

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
DEVICE     = 'cuda'
DTYPE      = torch.bfloat16
BLOCK_SIZE = 256
MAX_ITERS  = 500
PEAK_LR    = 1e-3
MIN_LR     = 1e-4
GRAD_CLIP  = 1.0

# Large batch — this is the amplifier
BATCH_SIZE = 128


def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(DATA_DIR, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x     = torch.stack([torch.from_numpy(data[i  :i+BLOCK_SIZE  ].astype(np.int64)) for i in ix])
    y     = torch.stack([torch.from_numpy(data[i+1:i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def get_lr(it, warmup_iters):
    if warmup_iters > 0 and it < warmup_iters:
        # Linear ramp from 0 → peak
        return PEAK_LR * (it / warmup_iters)
    decay_ratio = (it - warmup_iters) / max(MAX_ITERS - warmup_iters, 1)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (PEAK_LR - MIN_LR)


def run(label, warmup_iters):
    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"  batch={BATCH_SIZE} | warmup={warmup_iters} iters | "
          f"peak_lr={PEAK_LR} | grad_clip={GRAD_CLIP}")
    print(f"  tokens per step: {BATCH_SIZE * BLOCK_SIZE:,}")
    print(f"{'='*65}")

    torch.manual_seed(42)
    config = GPTConfig(
        n_layer       = 6,
        n_head        = 6,
        n_embd        = 192,
        use_residual  = True,
        
    )
    model = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    print(f"\n  {'iter':>5} | {'loss':>10} | {'grad_norm':>10} | {'lr':>10} | status")
    print(f"  {'-'*60}")

    prev_loss    = None
    loss_spikes  = 0

    for it in range(MAX_ITERS + 1):
        if it == MAX_ITERS:
            break

        lr = get_lr(it, warmup_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        X, Y = get_batch('train')
        with torch.autocast(device_type='cuda', dtype=DTYPE):
            _, loss = model(X, Y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  {it:5d} | {'NaN/Inf':>10} | {'---':>10} | "
                  f"{lr:>10.6f} | *** COLLAPSED ***")
            break

        loss_val = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        ).item()
        optimizer.step()

        # Detect loss spikes — sudden increases after initial descent
        # This is the warmup failure signature: early chaotic updates
        # that push weights into bad regions, visible as loss increasing
        # after it has already started decreasing
        if prev_loss is not None and it > 20:
            if loss_val > prev_loss * 1.15:  # 15% jump
                loss_spikes += 1
        prev_loss = loss_val

        if it % 25 == 0:
            # Status assessment
            if math.isnan(grad_norm):
                status = "!! GRAD NaN"
            elif grad_norm >= GRAD_CLIP * 0.99:
                status = "! CLIPPED"   # gradient was actually clipped
            else:
                status = "OK"

            print(f"  {it:5d} | {loss_val:>10.4f} | {grad_norm:>10.4f} | "
                  f"{lr:>10.6f} | {status}")

    print(f"\n  Loss spikes (>15% jump after iter 20): {loss_spikes}")
    print(f"  Final LR: {get_lr(MAX_ITERS, warmup_iters):.6f}")

    del model, optimizer
    torch.cuda.empty_cache()


# =============================================================================
# THREE WARMUP CONFIGURATIONS
# =============================================================================

run("1. BROKEN:  batch=512, NO warmup (warmup=0)",
    warmup_iters=0)

run("2. PARTIAL: batch=512, short warmup (warmup=50)",
    warmup_iters=50)

run("3. CONTROL: batch=512, full warmup (warmup=200)",
    warmup_iters=200)

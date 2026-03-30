"""
instability_c_f.py — Experiment C: Learning Rate Too High
==========================================================
The failure mode: LR so high that every parameter update overshoots
the loss minimum. The model oscillates around a good solution without
ever settling into it — or diverges entirely.

Two distinct regimes to engineer:
  1. Moderate overshoot (LR=0.01, 10x):
     Model learns fast, then oscillates. Loss bounces rather than
     smoothly decreasing. May still reach a reasonable solution
     but inefficiently.

  2. Severe overshoot (LR=0.1, 100x) WITHOUT gradient clipping:
     Updates are so large that weights move far outside any
     reasonable range. Loss should diverge or NaN within
     the first 20-50 iterations.

  3. Severe LR WITH gradient clipping (LR=0.1, grad_clip=1.0):
     Clipping caps the update magnitude even when LR is high.
     This is the "high LR + clipping" regime from Experiment C
     in our original instability run — which outperformed baseline.
     Here we isolate why.

  4. CONTROL: LR=0.001, grad_clip=1.0

Key diagnostic: loss variance between steps.
A well-tuned LR produces smooth monotonic descent.
A too-high LR produces oscillation — loss goes down then up then down.
We measure this as the standard deviation of loss over a rolling window.
"""

import os
import math
import numpy as np
import torch
from model_f import GPTConfig, GPT

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
DEVICE     = 'cuda'
DTYPE      = torch.bfloat16
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS  = 300
WARMUP     = 50    # short warmup so LR reaches peak quickly


def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(DATA_DIR, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x     = torch.stack([torch.from_numpy(data[i  :i+BLOCK_SIZE  ].astype(np.int64)) for i in ix])
    y     = torch.stack([torch.from_numpy(data[i+1:i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def get_lr(it, peak_lr):
    if it < WARMUP:
        return peak_lr * (it / WARMUP)
    decay_ratio = (it - WARMUP) / (MAX_ITERS - WARMUP)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (peak_lr * 0.1) + coeff * (peak_lr - peak_lr * 0.1)


def run(label, peak_lr, grad_clip):
    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"  peak_lr={peak_lr} | grad_clip={grad_clip} | warmup={WARMUP}")
    print(f"{'='*65}")

    torch.manual_seed(42)
    config = GPTConfig(
        n_layer      = 6,
        n_head       = 6,
        n_embd       = 192,
        use_residual = True,
    )
    model = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    print(f"\n  {'iter':>5} | {'loss':>10} | {'grad_norm':>10} | "
          f"{'lr':>10} | {'loss_std':>10} | status")
    print(f"  {'-'*70}")

    loss_history  = []
    collapsed     = False

    for it in range(MAX_ITERS + 1):
        if it == MAX_ITERS:
            break

        lr = get_lr(it, peak_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        X, Y = get_batch('train')
        with torch.autocast(device_type='cuda', dtype=DTYPE):
            _, loss = model(X, Y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  {it:5d} | {'NaN/Inf':>10} | {'---':>10} | "
                  f"{lr:>10.6f} | {'---':>10} | *** COLLAPSED ***")
            collapsed = True
            break

        loss_val = loss.item()
        loss_history.append(loss_val)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip if grad_clip is not None else float('inf')
        ).item()

        optimizer.step()

        if it % 25 == 0:
            # Rolling std over last 20 steps — measures oscillation
            # High std = model is bouncing around, not converging
            # Low std  = smooth descent
            window = loss_history[-20:] if len(loss_history) >= 20 else loss_history
            loss_std = float(np.std(window)) if len(window) > 1 else 0.0

            if math.isnan(grad_norm) or grad_norm > 100:
                status = "!! EXPLODING"
            elif grad_clip and grad_norm >= grad_clip * 0.99:
                status = "! CLIPPED"
            elif loss_std > 0.5:
                status = "~ OSCILLATING"
            else:
                status = "OK"

            print(f"  {it:5d} | {loss_val:>10.4f} | {grad_norm:>10.4f} | "
                  f"{lr:>10.6f} | {loss_std:>10.4f} | {status}")

    if not collapsed:
        window = loss_history[-50:]
        final_std = float(np.std(window))
        print(f"\n  Final loss:     {loss_history[-1]:.4f}")
        print(f"  Loss std (last 50 steps): {final_std:.4f}")
        print(f"  Note: high std = persistent oscillation, low std = converged")

    del model, optimizer
    torch.cuda.empty_cache()


# =============================================================================
# FOUR LR CONFIGURATIONS
# =============================================================================

run("1. MODERATE: LR=0.01 (10x), with clipping",
    peak_lr=0.01,  grad_clip=1.0)

run("2. SEVERE:   LR=0.1 (100x), NO clipping",
    peak_lr=0.1,   grad_clip=None)

run("3. SEVERE:   LR=0.1 (100x), WITH clipping",
    peak_lr=0.1,   grad_clip=1.0)

run("4. CONTROL:  LR=0.001 (baseline), with clipping",
    peak_lr=0.001, grad_clip=1.0)

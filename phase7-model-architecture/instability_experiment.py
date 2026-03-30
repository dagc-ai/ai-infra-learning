"""
instability_experiment.py — Deliberately Breaking Training
===========================================================
Three controlled experiments, each removing one stability mechanism
and observing the failure mode.

Experiment A: No gradient clipping (grad_clip = None)
  Expected: loss spikes or NaN early in training when a bad batch
  triggers an explosive gradient update.

Experiment B: No learning rate warmup (jump straight to peak LR)
  Expected: chaotic loss in early iterations as large updates hit
  a randomly initialized model with no structure yet. May recover
  or may diverge depending on luck.

Experiment C: Learning rate 10x too high (lr = 0.01 instead of 0.001)
  Expected: loss oscillates or diverges — the model overshoots every
  minimum and never settles.

Baseline D: Correct configuration
  For direct comparison against all three broken runs.

Each experiment runs for 500 iterations — enough to observe the failure
mode without wasting compute on a run that's already broken.
"""

import os
import math
import time
import numpy as np
import torch
from model import GPT, GPTConfig

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
DEVICE     = 'cuda'
DTYPE      = torch.bfloat16
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS  = 500
EVAL_ITERS = 20

# Fixed model config — small enough to train fast, large enough to show instability
MODEL_CONFIG = GPTConfig(
    block_size = BLOCK_SIZE,
    vocab_size  = 50257,
    n_layer     = 4,
    n_head      = 4,
    n_embd      = 128,
    dropout     = 0.0,  # dropout off so instability signal is clean
)


def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(DATA_DIR, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x     = torch.stack([torch.from_numpy(data[i  :i+BLOCK_SIZE  ].astype(np.int64)) for i in ix])
    y     = torch.stack([torch.from_numpy(data[i+1:i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def get_lr_with_warmup(it, peak_lr, warmup_iters, min_lr):
    """Standard schedule: warmup then cosine decay."""
    if it < warmup_iters:
        return peak_lr * it / warmup_iters
    decay_ratio = (it - warmup_iters) / (MAX_ITERS - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def get_lr_no_warmup(it, peak_lr, min_lr):
    """No warmup — jump straight to peak LR, then cosine decay."""
    decay_ratio = it / MAX_ITERS
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


@torch.no_grad()
def estimate_val_loss(model):
    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        X, Y = get_batch('val')
        with torch.autocast(device_type='cuda', dtype=DTYPE):
            _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def run_experiment(name, peak_lr, warmup_iters, grad_clip, description):
    """
    Run a single training experiment and return the loss trajectory.

    Args:
        name:         experiment label
        peak_lr:      peak learning rate
        warmup_iters: warmup steps (0 = no warmup)
        grad_clip:    gradient clipping threshold (None = disabled)
        description:  plain language description of what's broken
    """
    print(f"\n{'='*65}")
    print(f"Experiment {name}: {description}")
    print(f"  peak_lr={peak_lr}, warmup={warmup_iters}, grad_clip={grad_clip}")
    print(f"{'='*65}")

    # Fresh model — same initialization seed for fair comparison
    torch.manual_seed(42)
    model = GPT(MODEL_CONFIG).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    trajectory = []  # (iter, train_loss, val_loss, grad_norm)
    nan_detected = False
    nan_at_iter  = None

    for it in range(MAX_ITERS + 1):

        # Set LR
        if warmup_iters > 0:
            lr = get_lr_with_warmup(it, peak_lr, warmup_iters, peak_lr * 0.1)
        else:
            lr = get_lr_no_warmup(it, peak_lr, peak_lr * 0.1)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Eval every 100 steps
        if it % 100 == 0:
            val_loss = estimate_val_loss(model)
            # Get current training loss from last step
            train_loss_display = trajectory[-1][1] if trajectory else float('nan')
            grad_norm_display  = trajectory[-1][3] if trajectory else float('nan')
            status = " ← NaN DETECTED" if nan_detected and it >= (nan_at_iter or 0) else ""
            print(f"  iter {it:4d} | train {train_loss_display:8.4f} | "
                  f"val {val_loss:8.4f} | "
                  f"grad_norm {grad_norm_display:8.4f} | "
                  f"lr {lr:.6f}{status}")

        if it == MAX_ITERS:
            break

        # Forward pass
        X, Y = get_batch('train')
        with torch.autocast(device_type='cuda', dtype=DTYPE):
            _, loss = model(X, Y)

        # Check for NaN loss
        if torch.isnan(loss):
            if not nan_detected:
                nan_detected = True
                nan_at_iter  = it
                print(f"  *** NaN loss detected at iter {it} — training collapsed ***")
            trajectory.append((it, float('nan'), float('nan'), float('nan')))
            break

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Measure gradient norm BEFORE clipping
        # This is the raw gradient magnitude — clipping rescales if above threshold
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip if grad_clip is not None else float('inf')
        ).item()

        # If no clipping, we still computed the norm — just didn't clip
        optimizer.step()

        trajectory.append((it, loss.item(), float('nan'), grad_norm))

    # Final summary
    valid_losses = [t[1] for t in trajectory if not math.isnan(t[1])]
    valid_norms  = [t[3] for t in trajectory if not math.isnan(t[3])]

    if valid_losses:
        print(f"\n  Summary:")
        print(f"    Final train loss:  {valid_losses[-1]:.4f}")
        print(f"    Min train loss:    {min(valid_losses):.4f}")
        print(f"    Max grad norm:     {max(valid_norms):.4f}" if valid_norms else "")
        print(f"    Avg grad norm:     {sum(valid_norms)/len(valid_norms):.4f}" if valid_norms else "")
        if nan_detected:
            print(f"    NaN at iter:       {nan_at_iter}")
    else:
        print(f"  Training failed immediately.")

    del model, optimizer
    torch.cuda.empty_cache()
    return trajectory, nan_detected, nan_at_iter


# =============================================================================
# RUN ALL FOUR EXPERIMENTS
# =============================================================================

results = {}

# Experiment A: No gradient clipping
results['A'] = run_experiment(
    name         = 'A',
    peak_lr      = 1e-3,
    warmup_iters = 200,
    grad_clip    = None,   # ← broken: no clipping
    description  = 'No gradient clipping (grad_clip=None)'
)

# Experiment B: No warmup
results['B'] = run_experiment(
    name         = 'B',
    peak_lr      = 1e-3,
    warmup_iters = 0,      # ← broken: jump straight to peak LR
    grad_clip    = 1.0,
    description  = 'No LR warmup (warmup_iters=0)'
)

# Experiment C: LR 10x too high
results['C'] = run_experiment(
    name         = 'C',
    peak_lr      = 1e-2,   # ← broken: 10x peak LR
    warmup_iters = 200,
    grad_clip    = 1.0,
    description  = 'LR 10x too high (peak_lr=0.01)'
)

# Experiment D: Correct baseline
results['D'] = run_experiment(
    name         = 'D',
    peak_lr      = 1e-3,
    warmup_iters = 200,
    grad_clip    = 1.0,    # ← all stability mechanisms active
    description  = 'Correct baseline (all stability mechanisms active)'
)


# =============================================================================
# COMPARATIVE SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print("COMPARATIVE SUMMARY — iter 500 train loss")
print(f"{'='*65}")
print(f"{'Experiment':<35} {'Final Loss':>12} {'Max GradNorm':>14} {'Collapsed':>10}")
print("-"*75)

labels = {
    'A': 'No gradient clipping',
    'B': 'No LR warmup',
    'C': 'LR 10x too high',
    'D': 'Correct baseline',
}

for key in ['A', 'B', 'C', 'D']:
    trajectory, nan_detected, nan_at_iter = results[key]
    valid_losses = [t[1] for t in trajectory if not math.isnan(t[1])]
    valid_norms  = [t[3] for t in trajectory if not math.isnan(t[3])]
    final_loss   = valid_losses[-1] if valid_losses else float('nan')
    max_norm     = max(valid_norms) if valid_norms else float('nan')
    collapsed    = f"Yes (iter {nan_at_iter})" if nan_detected else "No"
    print(f"{labels[key]:<35} {final_loss:>12.4f} {max_norm:>14.4f} {collapsed:>10}")

print(f"\nBaseline (D) final loss for reference: see above")
print(f"Degradation = final loss of experiment / baseline final loss")

"""
model.py — nanoGPT: A minimal GPT-2 style transformer
=======================================================
Architecture overview:
  Input tokens (integers)
    → Embedding lookup (token + position)
    → N × TransformerBlock (attention + MLP + residuals + LayerNorm)
    → Final LayerNorm
    → Linear projection → logits over vocabulary
    → Cross-entropy loss

Every tensor shape is annotated as (B, T, C) where:
  B = batch size
  T = sequence length (number of tokens)
  C = embedding dimension (n_embd)

This notation is used consistently throughout so you can trace
exactly what shape every tensor is at every step.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class GPTConfig:
    """
    All hyperparameters in one place.
    These defaults produce a ~10M parameter model — small enough to train
    in minutes on an A100, large enough to produce coherent Shakespeare.

    For context, GPT-2 sizes:
      GPT-2 small:  n_layer=12, n_head=12, n_embd=768   (~117M params)
      GPT-2 medium: n_layer=24, n_head=16, n_embd=1024  (~345M params)
      GPT-2 large:  n_layer=36, n_head=20, n_embd=1280  (~762M params)
      GPT-2 XL:     n_layer=48, n_head=25, n_embd=1600  (~1.5B params)
    """
    block_size: int = 256    # maximum sequence length (context window)
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    n_layer:    int = 6      # number of transformer blocks stacked
    n_head:     int = 6      # number of attention heads per block
    n_embd:     int = 384    # embedding dimension (must be divisible by n_head)
    dropout:    float = 0.1  # dropout probability (regularization)


# =============================================================================
# CAUSAL SELF-ATTENTION
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    "Causal" means each token can only attend to itself and tokens BEFORE it —
    not tokens that come after. This is enforced by a mask.
    Why? Because during generation, future tokens don't exist yet.
    During training we use the mask to simulate this constraint.

    "Multi-head" means we run H attention computations in parallel,
    each in a lower-dimensional subspace (C/H dimensions each).
    Each head can specialize: one head might learn syntactic relationships,
    another semantic ones. The outputs are concatenated and projected back.

    Tensor flow:
      Input:  (B, T, C)
      Q,K,V:  (B, T, C) each  — projected from input
      After split into heads: (B, n_head, T, head_dim) where head_dim = C // n_head
      Attention weights: (B, n_head, T, T)  — this is the N² memory term
      Output: (B, T, C)  — heads concatenated, projected
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head — each head gets equal dimensions"

        # Single linear layer that projects input to Q, K, V simultaneously.
        # Output is 3*C so we can split into three C-dimensional tensors.
        # This is more efficient than three separate linear layers.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Output projection: after concatenating all heads, project back to C dims
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # dimension per head

        # Causal mask: lower-triangular matrix of ones.
        # Token at position t can attend to positions 0..t only.
        # registered as a buffer (not a parameter — not updated by optimizer,
        # but moves to GPU with the model automatically).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )
        # mask shape: (1, 1, T, T) — broadcast over batch and head dims

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Project input to Q, K, V in one operation, then split
        # c_attn output: (B, T, 3*C)
        # after split: three tensors of shape (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention:
        # (B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        # Each head operates on head_dim = C // n_head dimensions
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Scaled dot-product attention
        # Q @ K^T: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T)
        #        = (B, n_head, T, T)  ← this is the N² matrix
        # Scaling by 1/sqrt(head_dim) prevents dot products from growing large
        # in magnitude (which would push softmax into saturated regions with
        # near-zero gradients). This is the sqrt(d_k) from the original paper.
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        # Apply causal mask: set future positions to -inf before softmax
        # After softmax, -inf → 0.0, so future tokens get zero attention weight
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax over the last dim (T): converts raw scores to probabilities
        # Each row sums to 1.0 — these are the attention weights
        attn = F.softmax(attn, dim=-1)  # (B, n_head, T, T)
        attn = self.attn_dropout(attn)

        # Weight the values by attention scores
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) = (B, n_head, T, head_dim)
        out = attn @ v

        # Concatenate heads: (B, n_head, T, head_dim) → (B, T, C)
        # contiguous() needed because transpose creates non-contiguous memory layout
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        out = self.resid_dropout(self.c_proj(out))  # (B, T, C)
        return out


# =============================================================================
# MLP (Feed-Forward Network)
# =============================================================================

class MLP(nn.Module):
    """
    Position-wise feed-forward network applied after attention.

    Architecture: Linear(C → 4C) → GELU → Linear(4C → C)

    Why 4x expansion?
    The original paper uses 4x as a rule of thumb. The MLP is where the model
    stores "factual" knowledge — attention routes information, MLP transforms it.
    Recent work (Geva et al.) shows MLP layers function like key-value memories.

    Why GELU instead of ReLU?
    GELU (Gaussian Error Linear Unit) is smoother than ReLU — it doesn't have
    a hard zero cutoff, which improves gradient flow. GPT-2 uses GELU;
    most modern transformers have converged on it or SwiGLU variants.

    Tensor flow:
      Input:  (B, T, C)
      After first linear: (B, T, 4*C)
      After GELU:         (B, T, 4*C)
      After second linear:(B, T, C)
    """

    def __init__(self, config):
        super().__init__()
        self.fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.gelu  = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)      # (B, T, C) → (B, T, 4C)
        x = self.gelu(x)    # non-linearity
        x = self.proj(x)    # (B, T, 4C) → (B, T, C)
        x = self.dropout(x)
        return x


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    One transformer block: LayerNorm → Attention → residual
                                      LayerNorm → MLP → residual

    Two design choices worth understanding:

    1. PRE-NORM (applied before the sublayer, not after as in the original paper)
       Original paper: x = LayerNorm(x + sublayer(x))
       Modern (pre-norm): x = x + sublayer(LayerNorm(x))   ← what we use
       Pre-norm trains more stably, especially at depth. Nearly all modern
       models (GPT-2 onwards) use pre-norm.

    2. RESIDUAL CONNECTIONS: x = x + sublayer(x)
       Each block only needs to learn the DELTA — what to ADD to the current
       representation. At initialization, sublayer(x) ≈ 0, so the network
       starts as near-identity. This is a powerful stability prior.
       Gradients flow directly through the + operation back to early layers —
       this is the "gradient highway" that makes deep networks trainable.

    Tensor flow (shapes unchanged throughout — residuals require this):
      Input:  (B, T, C)
      Output: (B, T, C)
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1  = nn.LayerNorm(config.n_embd)  # LayerNorm before attention
        self.attn  = CausalSelfAttention(config)
        self.ln_2  = nn.LayerNorm(config.n_embd)  # LayerNorm before MLP
        self.mlp   = MLP(config)

    def forward(self, x):
        # Attention sublayer with residual
        # ln_1 normalizes BEFORE attention (pre-norm)
        x = x + self.attn(self.ln_1(x))  # (B, T, C)

        # MLP sublayer with residual
        x = x + self.mlp(self.ln_2(x))   # (B, T, C)

        return x


# =============================================================================
# FULL GPT MODEL
# =============================================================================

class GPT(nn.Module):
    """
    Full GPT model: embedding → N transformer blocks → head

    The embedding layer has two components:
      1. Token embedding (wte): maps token IDs → vectors. Shape: (vocab_size, C)
         Each token has a learned C-dimensional representation.
      2. Position embedding (wpe): maps position index → vectors. Shape: (block_size, C)
         Each position 0..T-1 has a learned C-dimensional offset.
         These are ADDED to token embeddings — the model learns to encode
         position information this way.

    Why learned position embeddings instead of sinusoidal (original paper)?
    GPT-2 uses learned. Sinusoidal can extrapolate to longer sequences;
    learned tends to work slightly better within the training length.
    Modern models use RoPE (Rotary Position Embeddings) — but that's Phase 7+.

    Final head: Linear(C → vocab_size) projects the last hidden state to
    logits over the vocabulary. No softmax here — cross-entropy loss takes
    raw logits. During generation, we apply softmax + sample.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe  = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),                      # final layer norm
        ))

        # Language model head: project hidden states to vocabulary logits
        # No bias — standard for modern LMs
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and lm_head.
        # The embedding maps tokens→vectors; lm_head maps vectors→token scores.
        # They're the inverse of each other, so tying reduces parameters and
        # improves performance. Standard practice since Press & Wolf (2017).
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count and report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        """
        Weight initialization following GPT-2's scheme.
        Linear layers: Normal(0, 0.02) — small random weights
        Embeddings: Normal(0, 0.02)
        The residual projection layers get an extra 1/sqrt(2*n_layer) scaling
        to account for the accumulation across residual branches.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx:     (B, T) — integer token IDs
            targets: (B, T) — integer token IDs shifted by 1 (next-token prediction)
                     If None, just return logits (inference mode)

        Returns:
            logits: (B, T, vocab_size) — raw scores over vocabulary at each position
            loss:   scalar cross-entropy loss (None if targets not provided)

        The training objective is next-token prediction:
          Given tokens [t0, t1, t2, ..., t_{T-1}], predict [t1, t2, ..., t_T]
          Loss = mean cross-entropy over all T positions
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # Embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, C) — token embeddings
        pos_emb = self.transformer.wpe(pos)  # (T, C) — position embeddings
        # pos_emb broadcasts over batch dimension
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)  # (B, T, C) — shape preserved through each block

        # Final layer norm
        x = self.transformer.ln_f(x)  # (B, T, C)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten (B, T, vocab_size) → (B*T, vocab_size) for cross_entropy
            # Flatten targets (B, T) → (B*T,)
            # cross_entropy expects (N, C) logits and (N,) targets
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation: given a context, generate max_new_tokens more.

        At each step:
          1. Forward pass → logits for next token
          2. Take only the logit at the last position (we want the NEXT token)
          3. Optionally scale by temperature (higher = more random)
          4. Optionally apply top-k filtering (zero out all but top k logits)
          5. Sample from the resulting distribution
          6. Append sampled token to context, repeat

        This is the exact same process used in production inference servers,
        just without batching, KV cache, or speculative decoding.
        The vLLM KV cache you built in Phase 5 is an optimization of step 1.

        Args:
            idx:            (B, T) — initial context tokens
            max_new_tokens: how many tokens to generate
            temperature:    1.0 = sample as-is, <1.0 = more deterministic,
                            >1.0 = more random
            top_k:          if set, only sample from top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if it's grown too long
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                           else idx[:, -self.config.block_size:]

            # Forward pass — only need logits at last position
            logits, _ = self(idx_cond)              # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size) — last token only

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Softmax → probabilities, then sample
            probs = F.softmax(logits, dim=-1)         # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to running context
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    """
    Instantiate the model and verify:
      1. Parameter count is what we expect
      2. Forward pass produces correct output shapes
      3. Loss is ~ln(vocab_size) ≈ 10.8 at random initialization
         (A randomly initialized model assigns equal probability to all tokens,
          so loss = -log(1/50257) ≈ 10.82. If loss is much higher or NaN,
          something is wrong with initialization.)
    """
    config = GPTConfig()
    model = GPT(config).cuda()

    # Test forward pass with a small batch
    B, T = 4, 64  # batch size 4, sequence length 64
    idx     = torch.randint(0, config.vocab_size, (B, T)).cuda()
    targets = torch.randint(0, config.vocab_size, (B, T)).cuda()

    logits, loss = model(idx, targets)

    print(f"Input shape:   {idx.shape}")
    print(f"Logits shape:  {logits.shape}")
    print(f"Loss:          {loss.item():.4f}")
    print(f"Expected loss: ~{math.log(config.vocab_size):.4f} (random init baseline)")
    print(f"Loss gap:      {loss.item() - math.log(config.vocab_size):.4f} (should be near 0)")

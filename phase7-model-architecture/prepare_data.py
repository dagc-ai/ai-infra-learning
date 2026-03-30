"""
prepare_data.py — Shakespeare Corpus Preparation for nanoGPT
=============================================================
This script does three things:
  1. Downloads the tiny Shakespeare dataset (~1MB of raw text)
  2. Tokenizes it using the GPT-2 tokenizer (tiktoken)
  3. Splits it into train/val sets and saves as binary arrays

Why these choices matter:
- We use GPT-2's tokenizer (not character-level) so our model architecture
  is directly comparable to real GPT-2. This matters in the scaling experiment
  when we want to reason about parameter counts vs. real models.
- We save as binary uint16 arrays (not text) because during training we'll
  be loading millions of tokens per second. Binary is ~10x faster to load
  than parsing text.
- uint16 can hold values 0–65535, which covers GPT-2's vocab of 50,257 tokens.
"""

import os
import requests
import numpy as np
import tiktoken

# =============================================================================
# STEP 1: Download the corpus
# =============================================================================
# Tiny Shakespeare is ~1MB of concatenated Shakespeare plays.
# It's the canonical toy dataset for language model tutorials because:
#   - Small enough to train on in minutes
#   - Has enough structure (iambic pentameter, dialogue patterns) that a model
#     can visibly "learn" it — you'll see coherent-looking output after training
#   - Long enough (~300K tokens) to demonstrate real loss curves

data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
print("Downloading Shakespeare corpus...")
response = requests.get(data_url)
text = response.text  # raw string, ~1.1M characters
print(f"Corpus size: {len(text):,} characters")

# =============================================================================
# STEP 2: Tokenize with tiktoken (GPT-2 BPE tokenizer)
# =============================================================================
# What is tokenization?
# Raw text can't go into a neural network directly. We need to convert it to
# integers that index into an embedding table.
#
# Two approaches:
#   Character-level: each character is one token. Simple, but "running" = 7 tokens.
#   Subword BPE:     common subwords get single tokens. "running" might be 1-2 tokens.
#
# GPT-2 uses Byte Pair Encoding (BPE) with a vocabulary of 50,257 tokens.
# BPE works by iteratively merging the most frequent pairs of bytes/characters
# until the vocab size is reached. Common words become single tokens;
# rare words get split into subword pieces.
#
# encode_ordinary() encodes text without adding any special tokens
# (no <|endoftext|> markers between documents — fine for our purposes).

enc = tiktoken.get_encoding("gpt2")  # loads GPT-2's vocab and merge rules
tokens = enc.encode_ordinary(text)   # list of integers, one per token
tokens = np.array(tokens, dtype=np.uint16)  # convert to numpy for fast I/O

print(f"Tokenized: {len(tokens):,} tokens")
print(f"Vocabulary size: {enc.n_vocab:,}")

# Compression ratio tells us how efficiently BPE compresses the text.
# ~3-4 chars/token is typical for English with GPT-2's tokenizer.
# Higher ratio = fewer tokens = shorter sequences = cheaper attention computation.
print(f"Compression ratio: {len(text)/len(tokens):.2f} chars/token")

# =============================================================================
# STEP 3: Train/validation split
# =============================================================================
# We hold out 10% of the data for validation.
# The model NEVER trains on validation tokens — it's used purely to measure
# how well the model generalizes to text it hasn't seen.
#
# Why does this matter?
# Training loss measures how well the model fits the training data.
# Validation loss measures whether the model has learned general patterns
# vs. just memorized the training set (overfitting).
# For our purposes with Shakespeare, overfitting is unlikely — the dataset
# is large relative to our small model. But we keep val split for completeness
# and because the scaling experiment needs it to compare runs fairly.

split = int(0.9 * len(tokens))      # index of the 90% mark
train_tokens = tokens[:split]        # first 90% → training
val_tokens   = tokens[split:]        # last 10%  → validation

print(f"\nTrain tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")

# =============================================================================
# STEP 4: Save as binary files
# =============================================================================
# .tofile() writes the raw bytes of the numpy array to disk.
# To load during training: np.memmap('train.bin', dtype=np.uint16, mode='r')
# memmap is memory-mapped I/O — the OS loads chunks on demand rather than
# reading the entire file into RAM. Critical for large datasets; fine for ours.

train_tokens.tofile('train.bin')
val_tokens.tofile('val.bin')
print("\nSaved train.bin and val.bin")

# =============================================================================
# STEP 5: Sanity check — decode first 100 tokens back to text
# =============================================================================
# If the tokenizer round-trips correctly, this should look like the opening
# lines of the Shakespeare file. If it's garbage, something went wrong.

sample = enc.decode(tokens[:100].tolist())
print(f"\nFirst 100 tokens decoded:\n{'='*40}\n{sample}\n{'='*40}")

# Final summary of what we produced
print(f"\nSummary:")
print(f"  train.bin: {os.path.getsize('train.bin') / 1e6:.1f} MB")
print(f"  val.bin:   {os.path.getsize('val.bin') / 1e6:.1f} MB")
print(f"  Token dtype: uint16 (range 0–65535, vocab fits in 50,257)")

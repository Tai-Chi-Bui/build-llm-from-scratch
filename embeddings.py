# embeddings.py
#
# This module converts token IDs (integers) into rich numerical vectors that
# the transformer model can process. This is the first thing that happens
# inside the model — before attention, before any layers.
#
# WHY DO WE NEED EMBEDDINGS?
#   The model received token IDs like [6109, 3626, 6100, 345] from the
#   dataloader. But a single integer doesn't carry much meaning — the model
#   needs a richer representation. An embedding converts each integer into a
#   vector of 768 numbers (for GPT-2 small). These 768 numbers together
#   capture the "meaning" of the token in a way the model can learn from.
#
#   Think of it like this:
#     Token ID 6109 ("Every") → just a label, like a locker number
#     Embedding of 6109       → the contents of the locker: 768 numbers that
#                                encode meaning, context, grammar, etc.
#
# THERE ARE TWO TYPES OF EMBEDDINGS:
#
#   1. Token embeddings — "WHAT is this token?"
#      Each token ID maps to a unique 768-dim vector. The model learns these
#      during training so that similar words end up with similar vectors.
#
#   2. Positional embeddings — "WHERE is this token in the sequence?"
#      Position 0 has its own 768-dim vector, position 1 has another, etc.
#      Without these, the model couldn't tell the difference between
#      "dog bites man" and "man bites dog" — the same tokens, different order.
#
#   The final embedding = token_embedding + positional_embedding
#   This single vector now encodes BOTH what the token is AND where it sits.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  THE EMBEDDING FLOW                                                    │
# │                                                                        │
# │  Token IDs:  [6109, 3626, 6100, 345]                                  │
# │                ↓      ↓      ↓     ↓       (look up in token table)   │
# │  Token vecs: [v0,    v1,    v2,   v3]      each is 768 numbers        │
# │               +      +      +     +                                    │
# │  Pos vecs:   [p0,    p1,    p2,   p3]      (look up in position table)│
# │               =      =      =     =                                    │
# │  Final:      [e0,    e1,    e2,   e3]      what + where combined      │
# │                                                                        │
# │  Shape: (4 tokens, 768 dims) → this goes into the transformer layers  │
# └─────────────────────────────────────────────────────────────────────────┘

import torch
import torch.nn as nn
import tiktoken
from dataloader import create_dataloader

# ── Configuration ─────────────────────────────────────────────────────────────
#
# These are the exact settings for GPT-2 small (124M parameters).
# We'll use them throughout the entire project.
#
GPT_CONFIG = {

    # VOCAB_SIZE: The model's dictionary size — how many unique tokens it knows.
    # This determines the number of rows in the token embedding table.
    # GPT-2's BPE tokenizer has 50,257 tokens:
    #   256 byte tokens + 50,000 learned merges + 1 special <|endoftext|> token
    "vocab_size"     : 50257,

    # CONTEXT_LENGTH: The model's "reading window" — how many tokens it can
    # look at simultaneously. Anything beyond this window is invisible.
    # GPT-2 uses 1024; we use 256 to save memory during training.
    # Longer = can understand longer passages, but uses quadratically more
    # memory in the attention layers (256² = 65K vs 1024² = 1M attention scores).
    "context_length" : 256,

    # EMB_DIM: How many numbers describe each token. Each token ID gets
    # converted into a vector of this many numbers. Think of it like
    # describing a person — 5 numbers (height, weight, age...) gives a rough
    # picture, but 768 numbers captures far more nuance and detail.
    # Bigger = more expressive but slower and more memory.
    "emb_dim"        : 768,

    # N_HEADS: Attention figures out which tokens are related to each other.
    # Instead of doing this once with one big calculation, we split it into
    # 12 parallel "heads". Each head can focus on a different type of
    # relationship — one might learn grammar, another meaning, another
    # word proximity. Each head works on emb_dim / n_heads = 768/12 = 64
    # dimensions independently.
    "n_heads"        : 12,

    # N_LAYERS: How many times the data passes through a transformer block
    # (attention + feed-forward network). Each layer refines understanding:
    #   Early layers  → basic patterns ("adjective before noun")
    #   Middle layers → sentence structure and grammar
    #   Later layers  → complex reasoning and long-range dependencies
    # More layers = deeper understanding, but more parameters and slower.
    # GPT-2 small has 12 layers, contributing most of its 124M parameters.
    "n_layers"       : 12,

    # DROP_RATE: During training, randomly zero out 10% of values.
    # This prevents the model from memorizing the training data by forcing
    # it to not rely on any single neuron — like studying with random
    # flashcards removed, you're forced to truly learn, not just memorize.
    # Set to 0.0 during text generation (inference) so nothing is dropped.
    "drop_rate"      : 0.1,

    # QKV_BIAS: In attention, each token is transformed into 3 vectors:
    #   Q (Query) — "what am I looking for?"
    #   K (Key)   — "what do I contain?"
    #   V (Value) — "what information do I offer?"
    # These are computed as W*x + b. Setting bias=False skips the "+ b" part.
    # GPT-2 doesn't use bias here — slightly simpler and works fine.
    "qkv_bias"       : False,
}


# ── 1. Token Embedding Layer ──────────────────────────────────────────────────
#
# nn.Embedding(num_embeddings, embedding_dim) creates a LOOKUP TABLE — a big
# matrix where each row corresponds to one token in the vocabulary.
#
#   shape: [vocab_size, emb_dim]  →  [50257, 768]
#
# Visually, the table looks like this:
#
#   Token ID 0     → [0.12, -0.34, 0.56, ..., 0.78]   (768 numbers)
#   Token ID 1     → [0.91, -0.23, 0.45, ..., 0.67]
#   Token ID 2     → [-0.11, 0.88, 0.33, ..., -0.54]
#   ...
#   Token ID 50256 → [0.44, 0.22, -0.67, ..., 0.19]
#
# When you pass in token ID 6109, it simply returns row 6109 from this table.
# That's all an embedding is — a fancy dictionary lookup from integer → vector.
#
# The values start RANDOM and get updated during training (via backpropagation)
# so that tokens with similar meanings end up with similar vectors.
#
token_embedding_layer = nn.Embedding(
    GPT_CONFIG["vocab_size"],    # 50,257 rows (one per possible token)
    GPT_CONFIG["emb_dim"],       # 768 columns (the size of each vector)
)
print("Token embedding table shape:", token_embedding_layer.weight.shape)
# → torch.Size([50257, 768])
# This table has 50,257 × 768 = ~38.6 million learnable numbers!


# ── 2. Positional Embedding Layer ────────────────────────────────────────────
#
# Another lookup table, but this one maps POSITIONS (not tokens) to vectors.
#
#   shape: [context_length, emb_dim]  →  [256, 768]
#
#   Position 0   → [0.05, -0.12, 0.33, ..., 0.41]   (768 numbers)
#   Position 1   → [0.22, 0.08, -0.19, ..., 0.55]
#   Position 2   → [-0.33, 0.67, 0.11, ..., -0.28]
#   ...
#   Position 255 → [0.17, -0.44, 0.82, ..., 0.06]
#
# WHY DO WE NEED THIS?
#   Without positional info, the model sees "dog bites man" and "man bites dog"
#   as identical — same tokens, same token embeddings, just different order.
#   Positional embeddings let the model know: "this token is at position 0,
#   that token is at position 2" — so word ORDER matters.
#
# HOW IT WORKS:
#   The positional vector for position i is ADDED to the token vector at
#   position i. So the final embedding carries both meaning AND position.
#
# These values also start random and are learned during training. The model
# figures out on its own what "being at position 5" should mean.
#
positional_embedding_layer = nn.Embedding(
    GPT_CONFIG["context_length"],   # 256 rows (one per possible position)
    GPT_CONFIG["emb_dim"],          # 768 columns (same size as token embeddings
                                    # so we can add them together)
)
print("Positional embedding table shape:", positional_embedding_layer.weight.shape)
# → torch.Size([256, 768])


# ── 3. Load a real batch from our dataloader ──────────────────────────────────
#
# Now let's get some actual data to embed. The dataloader (from dataloader.py)
# gives us batches of token IDs. Each batch has shape (batch_size, max_length).
#
# Reminder of the pipeline so far:
#   "I had always thought..."  →  tokenizer  →  [40, 550, 1464, ...]
#   →  sliding window  →  (input, target) pairs  →  DataLoader  →  batches
#
# We're now at the next step: batches → embeddings
#
tokenizer = tiktoken.get_encoding("gpt2")

with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader(
    raw_text,
    tokenizer,
    batch_size=4,
    max_length=GPT_CONFIG["context_length"],   # 256 tokens per sample
    stride=GPT_CONFIG["context_length"],       # no overlap between samples
    shuffle=False,
)

# Grab the first batch — a 2D grid of token IDs
input_batch, target_batch = next(iter(dataloader))
print(f"\nInput batch shape : {input_batch.shape}")
# → torch.Size([4, 256])
# meaning: 4 sequences, each 256 tokens long
#
# It looks like this (with made-up IDs for illustration):
#   [[  40,  550, 1464, 1807, ..., 3155],   ← sequence 0 (256 token IDs)
#    [3155,  287,  262, 1627, ...,  466],   ← sequence 1
#    [ 466,  11,  257,  1627, ..., 2472],   ← sequence 2
#    [2472,  373,  257,  8996, ..., 4920]]  ← sequence 3
#
# Right now these are just integers. The model can't learn from integers.
# Next step: turn each integer into a 768-dim vector via embedding lookup.


# ── 4. Look up token embeddings ───────────────────────────────────────────────
#
# This is the magic step: we feed integer token IDs into the embedding layer,
# and each integer gets replaced by its 768-dimensional vector.
#
# What happens inside:
#   input_batch = [[40, 550, 1464, ...],    shape: (4, 256) — integers
#                  [3155, 287, 262, ...],
#                  ...]
#
#   token_embedding_layer looks up each ID in its table:
#     40   → row 40   of the table → [0.12, -0.34, ..., 0.78]  (768 nums)
#     550  → row 550  of the table → [0.91, -0.23, ..., 0.67]  (768 nums)
#     1464 → row 1464 of the table → [-0.11, 0.88, ..., -0.54] (768 nums)
#
#   Result: each 2D grid of integers becomes a 3D block of vectors.
#
token_embeddings = token_embedding_layer(input_batch)
print(f"Token embeddings shape: {token_embeddings.shape}")
# → torch.Size([4, 256, 768])
#
# The shape went from (4, 256) → (4, 256, 768)
#   4   = batch_size  (still 4 sequences)
#   256 = seq_len     (still 256 tokens per sequence)
#   768 = emb_dim     (NEW! each token is now a 768-number vector)
#
# Each integer has been replaced by a rich vector of 768 learned numbers.


# ── 5. Build positional indices and look up positional embeddings ─────────────
#
# Now we need to add position information. First, we create a simple list of
# position numbers: [0, 1, 2, 3, ..., 255]
#
# Then we look up each position in the positional embedding table, just like
# we looked up token IDs in the token embedding table.
#
# Why don't we need a separate position list per sequence?
#   Every sequence has tokens at positions 0, 1, 2, ..., 255 — the positions
#   are always the same. So one set of position indices works for the whole batch.
#
seq_len = input_batch.shape[1]                       # 256
position_indices = torch.arange(seq_len)             # tensor([0, 1, 2, ..., 255])
print(f"\nPosition indices : {position_indices}")

# Look up each position number in the positional embedding table:
#   0   → row 0   → [0.05, -0.12, ..., 0.41]  (768 nums)
#   1   → row 1   → [0.22, 0.08, ..., 0.55]   (768 nums)
#   ...
#   255 → row 255 → [0.17, -0.44, ..., 0.06]  (768 nums)
positional_embeddings = positional_embedding_layer(position_indices)
print(f"Positional embeddings shape: {positional_embeddings.shape}")
# → torch.Size([256, 768])
# Note: no batch dimension — it's (256, 768) not (4, 256, 768).
# PyTorch broadcasting will handle this in the next step.


# ── 6. Add them together ──────────────────────────────────────────────────────
#
# This is where WHAT + WHERE merge into a single vector per token.
#
# token_embeddings      shape: [4, 256, 768]   (what each token IS)
# positional_embeddings shape:    [256, 768]   (where each token SITS)
#
# We simply add them element-wise. But wait — the shapes don't match!
# token_embeddings has 3 dimensions, positional_embeddings has only 2.
#
# PyTorch "broadcasting" handles this automatically:
#   [256, 768] gets treated as [1, 256, 768], then repeated 4 times →
#   [4, 256, 768]. So every sequence in the batch gets the SAME position
#   vectors added, which makes sense — position 0 means the same thing
#   regardless of which sequence we're looking at.
#
# After addition, each token's vector now encodes:
#   - Its identity (from the token embedding)
#   - Its position (from the positional embedding)
#
input_embeddings = token_embeddings + positional_embeddings
print(f"\nFinal input embeddings shape: {input_embeddings.shape}")
# → torch.Size([4, 256, 768])
# Same shape as token_embeddings — adding doesn't change dimensions.


# ── 7. Dropout ────────────────────────────────────────────────────────────────
#
# Dropout is a regularization trick. During training, it randomly sets 10%
# of the values in the tensor to zero.
#
# WHY?
#   Without dropout, the model might "memorize" the training data by relying
#   on very specific combinations of numbers. By randomly zeroing some values,
#   we force the model to spread knowledge across many neurons — making it
#   more robust and better at handling text it hasn't seen before.
#
# Example (simplified, with 5 values instead of 768):
#   Before dropout: [0.52, -0.34, 0.78, 0.12, -0.91]
#   After dropout:  [0.52,  0.00, 0.78, 0.00, -0.91]  ← 2 values zeroed
#
# IMPORTANT: dropout is only active during training (model.train()).
# During text generation (model.eval()), dropout is automatically disabled
# so you get the full, unmasked embeddings.
#
dropout = nn.Dropout(p=GPT_CONFIG["drop_rate"])   # p=0.1 means 10% chance
input_embeddings = dropout(input_embeddings)
print(f"After dropout shape : {input_embeddings.shape}")
# Shape unchanged — dropout only zeroes values, never changes dimensions.
# Still (4, 256, 768).


# ── 8. Inspect a single token's journey ──────────────────────────────────────
#
# Let's trace ONE token through the entire embedding pipeline to make it
# concrete. We'll take the very first token of the first sequence.
#
print("\n" + "="*55)
print("SINGLE TOKEN JOURNEY")
print("="*55)

# input_batch[0, 0] = first sequence, first position
# .item() converts a single-element tensor to a plain Python int
example_token_id = input_batch[0, 0].item()
example_text     = tokenizer.decode([example_token_id])

# Step A: Look up the token embedding (WHAT is this token?)
tok_vec = token_embedding_layer(torch.tensor([example_token_id]))
# Step B: Look up the positional embedding (it's at position 0)
pos_vec = positional_embedding_layer(torch.tensor([0]))
# Step C: Add them together (WHAT + WHERE)
final   = tok_vec + pos_vec

# We only print the first 5 of the 768 values — just to see they're real numbers
print(f"\nToken ID  : {example_token_id}  ('{example_text}')")
print(f"Token vec (first 5 values) : {tok_vec[0, :5].tolist()}")
print(f"Pos   vec (first 5 values) : {pos_vec[0, :5].tolist()}")
print(f"Sum       (first 5 values) : {final[0, :5].tolist()}")
print(f"\nFull embedding vector length: {final.shape[1]}")
# → 768: this single vector is what the transformer layers will process
# for this one token at this one position.
print("\nEmbeddings working correctly!")
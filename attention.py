# attention.py
import torch
import torch.nn as nn


# ── 1. Tiny worked example — build intuition first ────────────────────────────
#
# We use a 6-token sentence with 3-dimensional embeddings.
# (Real GPT-2 uses 768 dimensions — same math, just bigger.)
#
print("=" * 55)
print("PART 1: INTUITION WITH A TINY EXAMPLE")
print("=" * 55)

torch.manual_seed(42)

# Simulate 6 token embeddings, each 3-dimensional
# Think of this as "Your journey starts with one step"
seq_len = 6
emb_dim = 3
inputs = torch.randn(seq_len, emb_dim)
print(f"\nInput shape: {inputs.shape}  [seq_len=6, emb_dim=3]")
print(f"Input tensor:\n{inputs}\n")

# ── Step 1: Define weight matrices ────────────────────────────────────────────
#
# Wq, Wk, Wv are learned during training.
# They project the input embedding (dim=3) into Q, K, V vectors (dim=2).
# Using d_out=2 here just to keep the numbers readable.
#
d_out = 2

torch.manual_seed(42)
Wq = nn.Parameter(torch.randn(emb_dim, d_out))   # [3, 2]
Wk = nn.Parameter(torch.randn(emb_dim, d_out))   # [3, 2]
Wv = nn.Parameter(torch.randn(emb_dim, d_out))   # [3, 2]

# ── Step 2: Compute Q, K, V ───────────────────────────────────────────────────
#
# Matrix multiplication projects each token's embedding into Q/K/V space.
# @ is Python's matrix multiplication operator.
#
Q = inputs @ Wq    # [6, 3] @ [3, 2] = [6, 2]
K = inputs @ Wk    # [6, 3] @ [3, 2] = [6, 2]
V = inputs @ Wv    # [6, 3] @ [3, 2] = [6, 2]

print(f"Q shape: {Q.shape}   (one query vector per token)")
print(f"K shape: {K.shape}   (one key vector per token)")
print(f"V shape: {V.shape}   (one value vector per token)")

# ── Step 3: Compute raw attention scores ──────────────────────────────────────
#
# Q @ K.T  →  [6, 2] @ [2, 6] = [6, 6]
# scores[i][j] = dot product of token i's query with token j's key
# = how much token i wants to attend to token j
#
scores = Q @ K.T
print(f"\nRaw attention scores shape: {scores.shape}  [6×6 matrix]")
print(f"scores[i][j] = how much token i attends to token j")
print(f"\nRaw scores:\n{scores.detach()}\n")

# ── Step 4: Scale the scores ──────────────────────────────────────────────────
#
# WHY SCALE? Without scaling, dot products grow large as d_out increases.
# Large values → softmax saturates → gradients vanish → training stalls.
# Dividing by sqrt(d_out) keeps scores in a healthy range.
# This is why the mechanism is called "scaled dot-product attention".
#
scale  = d_out ** 0.5              # sqrt(2) ≈ 1.414
scores = scores / scale
print(f"After scaling by sqrt({d_out})={scale:.3f}:\n{scores.detach()}\n")

# ── Step 5: Softmax → attention weights ───────────────────────────────────────
#
# Softmax converts raw scores into probabilities.
# dim=-1 means we softmax across the last dimension (columns),
# so each ROW sums to 1.0
# weights[i] tells us: for token i, how much attention to pay to each token
#
weights = torch.softmax(scores, dim=-1)
print(f"Attention weights (each row sums to 1.0):\n{weights.detach()}\n")

# Verify rows sum to 1
row_sums = weights.sum(dim=-1)
print(f"Row sums (should all be 1.0): {row_sums.detach()}\n")

# ── Step 6: Weighted sum of values → output ───────────────────────────────────
#
# For each token i, we compute a weighted blend of ALL value vectors.
# weights[i] @ V  →  scalar weights × value vectors, summed up
# Output[i] is NOT just token i's own value — it's a blend of everyone's values,
# weighted by how much attention token i paid to each.
#
output = weights @ V   # [6, 6] @ [6, 2] = [6, 2]
print(f"Output shape: {output.shape}  (same seq_len, new representation)")
print(f"Output:\n{output.detach()}\n")


# ── 2. Clean SelfAttention class ──────────────────────────────────────────────
print("=" * 55)
print("PART 2: CLEAN SELF-ATTENTION CLASS")
print("=" * 55)

class SelfAttention(nn.Module):
    def __init__(self, emb_dim, d_out, qkv_bias=False):
        """
        emb_dim  : dimension of input embeddings (e.g. 768)
        d_out    : dimension of output (Q, K, V vectors)
        qkv_bias : whether to add a bias term to the linear projections
                   GPT-2 does NOT use bias here
        """
        super().__init__()

        # nn.Linear(in, out, bias) is equivalent to x @ W.T (+ bias)
        # It is preferred over raw nn.Parameter because:
        #   - uses better weight initialisation (Xavier uniform)
        #   - handles bias cleanly
        #   - integrates with PyTorch's optimiser tracking
        self.W_query = nn.Linear(emb_dim, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(emb_dim, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(emb_dim, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        x: input tensor of shape [batch_size, seq_len, emb_dim]
        returns: context vectors of shape [batch_size, seq_len, d_out]
        """
        # Project input into Q, K, V spaces
        # nn.Linear applies the projection to the last dimension automatically
        Q = self.W_query(x)   # [batch, seq_len, d_out]
        K = self.W_key(x)     # [batch, seq_len, d_out]
        V = self.W_value(x)   # [batch, seq_len, d_out]

        # Compute attention scores
        # We need Q @ K^T but K has shape [batch, seq_len, d_out]
        # K.transpose(-2, -1) swaps the last two dims: [batch, d_out, seq_len]
        # Result: [batch, seq_len, seq_len]
        d_k     = Q.shape[-1]
        scores  = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

        # Softmax over last dimension (attend across all tokens)
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        output = weights @ V   # [batch, seq_len, d_out]
        return output


# ── 3. Test the class with a real batch ───────────────────────────────────────
torch.manual_seed(42)

# Simulate a real-sized batch
batch_size = 4
seq_len    = 8
emb_dim    = 768
d_out      = 768   # in GPT-2, d_out = emb_dim / n_heads per head
                   # for now we treat the whole thing as one head

x = torch.randn(batch_size, seq_len, emb_dim)
print(f"\nInput  shape: {x.shape}  [batch=4, seq=8, emb=768]")

attn = SelfAttention(emb_dim=768, d_out=768)
out  = attn(x)
print(f"Output shape: {out.shape}  [batch=4, seq=8, d_out=768]")
print("\nShape unchanged — each token now has context from all other tokens")


# ── 4. Visualise what the attention weights look like ─────────────────────────
print("\n" + "=" * 55)
print("PART 3: ATTENTION WEIGHT VISUALISATION")
print("=" * 55)

# Use the small 6-token example so numbers are readable
words   = ["Your", "journey", "starts", "with", "one", "step"]
x_small = torch.randn(1, 6, 3)    # batch=1, seq=6, emb=3

attn_small = SelfAttention(emb_dim=3, d_out=3)

with torch.no_grad():
    Q = attn_small.W_query(x_small)
    K = attn_small.W_key(x_small)
    scores  = Q @ K.transpose(-2, -1) / (3 ** 0.5)
    weights = torch.softmax(scores, dim=-1)   # [1, 6, 6]

print("\nAttention weights matrix (row = token attending, col = token attended to):")
print(f"{'':10}", end="")
for w in words:
    print(f"{w:10}", end="")
print()

for i, word_i in enumerate(words):
    print(f"{word_i:10}", end="")
    for j in range(len(words)):
        print(f"{weights[0, i, j].item():.3f}     ", end="")
    print()

print("\nEach ROW sums to 1.0")
print("High value → token in that row pays a lot of attention to token in that column")

print("\n✅ Self-attention working correctly!")

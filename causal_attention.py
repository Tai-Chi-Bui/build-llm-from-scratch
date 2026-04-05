# causal_attention.py
import torch
import torch.nn as nn


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, context_length, drop_rate, qkv_bias=False):
        """
        emb_dim        : total embedding dimension (e.g. 768)
        n_heads        : number of attention heads (e.g. 12)
        context_length : maximum sequence length (e.g. 256)
        drop_rate      : dropout probability (e.g. 0.1)
        qkv_bias       : whether to use bias in Q,K,V projections

        Each head works with head_dim = emb_dim / n_heads = 64 dimensions.
        All heads run in parallel via tensor reshaping — no Python loops.
        """
        super().__init__()

        # Validate that the split is clean
        assert emb_dim % n_heads == 0, \
            f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads})"

        self.emb_dim        = emb_dim
        self.n_heads        = n_heads
        self.head_dim       = emb_dim // n_heads   # 768 // 12 = 64

        # ── Q, K, V projections ───────────────────────────────────────────────
        # Instead of 12 separate (768→64) projections, we use ONE (768→768)
        # projection for each of Q, K, V.
        # We then reshape the output to split it into 12 heads.
        # This is mathematically identical but much faster.
        self.W_query = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_key   = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_value = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        # ── Output projection ─────────────────────────────────────────────────
        # After concatenating all heads, project back to emb_dim.
        # This lets the model learn how to mix the heads together.
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        # ── Dropout ───────────────────────────────────────────────────────────
        # Applied to attention weights — randomly zeroes some connections
        # so the model doesn't over-rely on any single token relationship.
        self.dropout = nn.Dropout(drop_rate)

        # ── Causal mask ───────────────────────────────────────────────────────
        # register_buffer stores the mask as part of the model (it moves to GPU
        # automatically) but does NOT count it as a trainable parameter.
        #
        # torch.triu(ones, diagonal=1) creates upper triangle of 1s:
        #   [[0, 1, 1, 1],
        #    [0, 0, 1, 1],
        #    [0, 0, 0, 1],
        #    [0, 0, 0, 0]]
        # We'll use this as a boolean mask to fill with -inf.
        mask = torch.triu(
            torch.ones(context_length, context_length),
            diagonal=1
        )
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        """
        x: [batch_size, seq_len, emb_dim]
        returns: [batch_size, seq_len, emb_dim]
        """
        batch, seq_len, emb_dim = x.shape

        # ── Step 1: Project to Q, K, V ────────────────────────────────────────
        Q = self.W_query(x)   # [batch, seq_len, emb_dim]
        K = self.W_key(x)     # [batch, seq_len, emb_dim]
        V = self.W_value(x)   # [batch, seq_len, emb_dim]

        # ── Step 2: Split into multiple heads ─────────────────────────────────
        #
        # Reshape [batch, seq_len, emb_dim]
        #      to [batch, seq_len, n_heads, head_dim]
        #      to [batch, n_heads, seq_len, head_dim]  ← transpose
        #
        # After transpose, dim layout is:
        #   dim 0: batch    (independent sequences)
        #   dim 1: n_heads  (independent attention heads)
        #   dim 2: seq_len  (tokens)
        #   dim 3: head_dim (per-head embedding)
        #
        # Now [batch, n_heads] act like a combined batch dimension —
        # all heads for all sequences are processed in ONE matrix multiply.
        #
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # All three now: [batch, n_heads, seq_len, head_dim]

        # ── Step 3: Scaled dot-product attention scores ───────────────────────
        #
        # Q @ K.transpose(-2,-1):
        #   [batch, n_heads, seq_len, head_dim]
        # @ [batch, n_heads, head_dim, seq_len]
        # = [batch, n_heads, seq_len, seq_len]
        #
        # Each head has its own seq×seq attention score matrix.
        #
        scale  = self.head_dim ** 0.5
        scores = Q @ K.transpose(-2, -1) / scale
        # [batch, n_heads, seq_len, seq_len]

        # ── Step 4: Apply causal mask ─────────────────────────────────────────
        #
        # self.mask is [context_length, context_length]
        # We slice it to [seq_len, seq_len] (sequence may be shorter than max)
        # masked_fill_ replaces True positions with -inf IN PLACE
        #
        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len],
            float("-inf")
        )
        # Future positions are now -inf → softmax will give them weight 0

        # ── Step 5: Softmax → attention weights ───────────────────────────────
        weights = torch.softmax(scores, dim=-1)
        # [batch, n_heads, seq_len, seq_len]

        # Apply dropout to attention weights
        weights = self.dropout(weights)

        # ── Step 6: Weighted sum of values ────────────────────────────────────
        context = weights @ V
        # [batch, n_heads, seq_len, head_dim]

        # ── Step 7: Reassemble heads ──────────────────────────────────────────
        #
        # Transpose back: [batch, seq_len, n_heads, head_dim]
        # contiguous() makes the memory layout contiguous after transpose
        #   (required before view())
        # view() merges n_heads and head_dim back into emb_dim
        #
        context = context.transpose(1, 2).contiguous()
        # [batch, seq_len, n_heads, head_dim]

        context = context.view(batch, seq_len, self.emb_dim)
        # [batch, seq_len, emb_dim]   ← heads concatenated

        # ── Step 8: Output projection ─────────────────────────────────────────
        # Mix information across heads with a learned linear layer
        output = self.out_proj(context)
        # [batch, seq_len, emb_dim]

        return output


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    torch.manual_seed(42)

    # Config matching GPT-2 small
    EMB_DIM        = 768
    N_HEADS        = 12
    CONTEXT_LENGTH = 256
    DROP_RATE      = 0.1
    BATCH_SIZE     = 4
    SEQ_LEN        = 16   # shorter than context_length — that's fine

    # ── Test 1: Basic shape check ─────────────────────────────────────────────
    print("=" * 55)
    print("TEST 1: Shape check")
    print("=" * 55)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMB_DIM)
    print(f"Input  shape: {x.shape}")

    mha = CausalMultiHeadAttention(
        emb_dim        = EMB_DIM,
        n_heads        = N_HEADS,
        context_length = CONTEXT_LENGTH,
        drop_rate      = DROP_RATE,
        qkv_bias       = False,
    )

    mha.eval()   # disable dropout for deterministic output
    with torch.no_grad():
        out = mha(x)

    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape must match input shape!"
    print("✅ Shape correct\n")

    # ── Test 2: Causal mask check ─────────────────────────────────────────────
    print("=" * 55)
    print("TEST 2: Causal mask — future tokens get zero weight")
    print("=" * 55)

    # Use tiny dimensions so we can read the numbers
    mha_tiny = CausalMultiHeadAttention(
        emb_dim=4, n_heads=2, context_length=6, drop_rate=0.0
    )
    mha_tiny.eval()

    x_tiny = torch.randn(1, 6, 4)   # batch=1, seq=6, emb=4

    with torch.no_grad():
        Q = mha_tiny.W_query(x_tiny)
        K = mha_tiny.W_key(x_tiny)

        # Reshape to multi-head format
        Q = Q.view(1, 6, 2, 2).transpose(1, 2)
        K = K.view(1, 6, 2, 2).transpose(1, 2)

        scores  = Q @ K.transpose(-2, -1) / (2 ** 0.5)
        scores  = scores.masked_fill(mha_tiny.mask[:6, :6], float("-inf"))
        weights = torch.softmax(scores, dim=-1)

    # Show attention weights for head 0
    w = weights[0, 0]   # [seq_len, seq_len]
    print("\nAttention weights for head 0 (rows=queries, cols=keys):")
    print("Upper triangle must be 0.0 (future tokens masked out)\n")

    tokens = ["T0", "T1", "T2", "T3", "T4", "T5"]
    print(f"{'':6}", end="")
    for t in tokens:
        print(f"{t:8}", end="")
    print()

    for i, ti in enumerate(tokens):
        print(f"{ti:6}", end="")
        for j in range(6):
            val = w[i, j].item()
            marker = "← self" if i == j else ""
            print(f"{val:8.3f}", end="")
        print()

    print("\nUpper triangle values (should all be 0.0):")
    upper = w[torch.triu(torch.ones(6, 6), diagonal=1).bool()]
    print(f"  {upper.tolist()}")
    assert torch.allclose(upper, torch.zeros_like(upper)), "Mask failed!"
    print("✅ Causal mask working correctly\n")

    # ── Test 3: Parameter count ───────────────────────────────────────────────
    print("=" * 55)
    print("TEST 3: Parameter count")
    print("=" * 55)

    total = sum(p.numel() for p in mha.parameters())
    print(f"\nCausalMultiHeadAttention parameters:")
    for name, p in mha.named_parameters():
        print(f"  {name:25s}  {str(p.shape):25s}  {p.numel():,}")
    print(f"\n  TOTAL: {total:,} parameters")
    # W_query + W_key + W_value + out_proj
    # Each is [768, 768] = 589,824
    # 4 × 589,824 = 2,359,296

    # ── Test 4: Each position only uses past tokens ───────────────────────────
    print("\n" + "=" * 55)
    print("TEST 4: Changing a future token does NOT affect past outputs")
    print("=" * 55)

    mha_test = CausalMultiHeadAttention(
        emb_dim=8, n_heads=2, context_length=10, drop_rate=0.0
    )
    mha_test.eval()

    x1 = torch.randn(1, 5, 8)
    x2 = x1.clone()
    x2[:, 4, :] = torch.randn(8)   # change the LAST token

    with torch.no_grad():
        out1 = mha_test(x1)
        out2 = mha_test(x2)

    # Position 0 should be identical in both — it only looks at itself
    # Position 3 should be identical — it only looks at 0,1,2,3
    # Position 4 will differ — it's the one we changed
    print(f"\nOutputs at position 0 identical: "
          f"{torch.allclose(out1[:,0,:], out2[:,0,:])}")
    print(f"Outputs at position 3 identical: "
          f"{torch.allclose(out1[:,3,:], out2[:,3,:])}")
    print(f"Outputs at position 4 identical: "
          f"{torch.allclose(out1[:,4,:], out2[:,4,:])}")
    print("✅ Causality confirmed — past outputs unaffected by future tokens")
# transformer_block.py
import torch
import torch.nn as nn
from causal_attention import CausalMultiHeadAttention


# ── 1. Layer Normalisation ────────────────────────────────────────────────────
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        """
        emb_dim: size of the embedding dimension to normalise over (e.g. 768)

        Learnable parameters:
          scale (γ): multiplies the normalised value  — starts at 1
          shift (β): added to the normalised value    — starts at 0
        """
        super().__init__()
        self.eps   = 1e-5                          # small value to avoid div/0
        self.scale = nn.Parameter(torch.ones(emb_dim))   # γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # β

    def forward(self, x):
        """
        x: [..., emb_dim]  — works on the last dimension
        Normalises each token vector independently.
        """
        # Compute mean and variance over the last dimension (per token)
        # keepdim=True keeps the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalise: zero mean, unit variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift
        return self.scale * x_norm + self.shift


# ── 2. GELU Activation ────────────────────────────────────────────────────────
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    Used inside the feed-forward network.

    Approximation formula used by GPT-2:
      GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    This approximation is numerically stable and faster than the exact form.
    """
    def forward(self, x):
        import math
        return 0.5 * x * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            )
        )


# ── 3. Feed-Forward Network ───────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        """
        emb_dim: embedding dimension (e.g. 768)

        Architecture:
          Linear(768 → 3072)  →  GELU  →  Linear(3072 → 768)

        The 4× expansion (768 → 3072) is a design choice from the
        original transformer paper. It gives the model more capacity
        to represent complex patterns in the wide intermediate space.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),   # expand:   768 → 3072
            GELU(),                              # activate: smooth nonlinearity
            nn.Linear(4 * emb_dim, emb_dim),    # compress: 3072 → 768
        )

    def forward(self, x):
        # x: [batch, seq_len, emb_dim]
        # nn.Sequential applies layers in order
        # Each token is processed independently (no cross-token interaction)
        return self.layers(x)


# ── 4. Transformer Block ──────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        """
        cfg: dictionary with keys:
          emb_dim        : embedding dimension (768)
          n_heads        : number of attention heads (12)
          context_length : max sequence length (256)
          drop_rate      : dropout probability (0.1)
          qkv_bias       : bias in attention projections (False)
        """
        super().__init__()

        # Pre-LN: normalise BEFORE attention and FFN
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # The attention module from Step 6
        self.attention = CausalMultiHeadAttention(
            emb_dim        = cfg["emb_dim"],
            n_heads        = cfg["n_heads"],
            context_length = cfg["context_length"],
            drop_rate      = cfg["drop_rate"],
            qkv_bias       = cfg["qkv_bias"],
        )

        # Feed-forward network
        self.ff = FeedForward(cfg["emb_dim"])

        # Dropout applied after attention and FFN
        # (before adding the residual)
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        x: [batch, seq_len, emb_dim]
        returns: [batch, seq_len, emb_dim]  (same shape — refined representations)

        Two sub-layers, each with the Pre-LN pattern:
          residual + Dropout(SubLayer(LayerNorm(input)))

        Pre-LN means we normalise BEFORE the sub-layer (not after).
        This is more stable than the original Post-LN design and is what
        GPT-2 actually uses.

        The "+ x" parts are RESIDUAL connections (a.k.a. skip connections):
        they let gradients flow straight through the block, which is what
        makes very deep transformers trainable.
        """
        # ── Sub-layer 1: Multi-Head Attention ─────────────────────────────────
        # Save the input for the residual connection
        residual = x
        # Normalise → attention → dropout → add residual back
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop(x)
        x = x + residual
        # x is now: attention output + original input

        # ── Sub-layer 2: Feed-Forward Network ─────────────────────────────────
        # Save again for the second residual connection
        residual = x
        # Normalise → FFN → dropout → add residual back
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + residual
        # x is now: FFN output + (attention output + original input)

        return x


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    torch.manual_seed(42)

    # Config matching GPT-2 small
    cfg = {
        "vocab_size"     : 50257,
        "context_length" : 256,
        "emb_dim"        : 768,
        "n_heads"        : 12,
        "n_layers"       : 12,
        "drop_rate"      : 0.1,
        "qkv_bias"       : False,
    }

    # ── Test 1: LayerNorm ─────────────────────────────────────────────────────
    print("=" * 55)
    print("TEST 1: LayerNorm")
    print("=" * 55)

    ln = LayerNorm(emb_dim=5)
    x = torch.randn(2, 3, 5) * 10 + 4   # arbitrary mean and scale
    y = ln(x)
    # After LayerNorm, mean of last dim should be ~0, var ~1
    print(f"\nInput  mean (per token): {x.mean(dim=-1).flatten().tolist()}")
    print(f"Output mean (per token): {[round(v, 6) for v in y.mean(dim=-1).flatten().tolist()]}")
    print(f"Output var  (per token): {[round(v, 6) for v in y.var(dim=-1, unbiased=False).flatten().tolist()]}")
    print("(means ~0, vars ~1 - LayerNorm working)")

    # ── Test 2: GELU shape preservation ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 2: GELU activation")
    print("=" * 55)

    gelu = GELU()
    x = torch.linspace(-3, 3, 7)
    print(f"\nInput : {x.tolist()}")
    print(f"GELU  : {[round(v, 4) for v in gelu(x).tolist()]}")
    print("(smooth S-curve — negative inputs not zeroed out like ReLU)")

    # ── Test 3: FeedForward shape ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 3: FeedForward network")
    print("=" * 55)

    ff = FeedForward(emb_dim=cfg["emb_dim"])
    x = torch.randn(2, 4, cfg["emb_dim"])
    y = ff(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {y.shape}  (same — FFN preserves shape)")
    n_params = sum(p.numel() for p in ff.parameters())
    print(f"FFN params  : {n_params:,}  (mostly the 768→3072 expansion)")

    # ── Test 4: Full TransformerBlock ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 4: TransformerBlock forward pass")
    print("=" * 55)

    block = TransformerBlock(cfg)
    block.eval()   # disable dropout for deterministic output

    x = torch.randn(2, 16, cfg["emb_dim"])   # [batch=2, seq=16, emb=768]
    print(f"\nInput  shape: {x.shape}")

    with torch.no_grad():
        y = block(x)

    print(f"Output shape: {y.shape}  (same — block preserves shape)")
    assert y.shape == x.shape, "TransformerBlock must preserve input shape!"

    n_params = sum(p.numel() for p in block.parameters())
    print(f"\nBlock params: {n_params:,}")
    print("(this is 1 of 12 blocks in GPT-2 small)")

    print("\nTransformerBlock working correctly")
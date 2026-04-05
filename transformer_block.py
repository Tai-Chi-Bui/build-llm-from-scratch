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

        Two sub-layers, each with:
          Pre-LayerNorm → Sub-layer → Dropout → Residual add
        """
        # ── Sub-layer 1: Multi-Head Attention ────────
# gpt_model.py
import torch
import torch.nn as nn
from transformer_block import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        """
        cfg: configuration dictionary with keys:
          vocab_size     : number of tokens in vocabulary (50257)
          context_length : maximum sequence length (256)
          emb_dim        : embedding dimension (768)
          n_heads        : attention heads per block (12)
          n_layers       : number of transformer blocks (12)
          drop_rate      : dropout probability (0.1)
          qkv_bias       : bias in attention projections (False)
        """
        super().__init__()

        # ── Embedding layers ──────────────────────────────────────────────────
        # Token embedding: maps each token ID to a learned vector
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Positional embedding: maps each position index to a learned vector
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Embedding dropout
        self.emb_drop = nn.Dropout(cfg["drop_rate"])

        # ── Transformer blocks ────────────────────────────────────────────────
        # nn.ModuleList holds a list of modules and registers them properly
        # so their parameters are tracked by PyTorch's optimiser.
        # (A plain Python list would NOT track parameters correctly.)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # ── Final layer normalisation ─────────────────────────────────────────
        # Applied once after all transformer blocks, before the output head.
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # ── Output head ───────────────────────────────────────────────────────
        # Projects from emb_dim → vocab_size to produce logits.
        # bias=False is standard for GPT-style output heads.
        # No softmax here — we return raw logits.
        # (Loss function and sampling apply softmax themselves.)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # ── Weight tying ──────────────────────────────────────────────────────
        # The output head and token embedding share the same weight matrix.
        # This works because both map between the same two spaces:
        #   tok_emb  : vocab_size → emb_dim  (lookup)
        #   out_head : emb_dim → vocab_size  (projection)
        # They are transposes of each other conceptually.
        # Sharing saves ~38M parameters and improves performance.
        self.out_head.weight = self.tok_emb.weight

        # ── Weight initialisation ─────────────────────────────────────────────
        # Apply our custom init to all submodules
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialise weights following GPT-2's scheme:
          - Linear and Embedding layers: normal distribution, std=0.02
          - Deeper layers in the residual stream get scaled down by
            1/sqrt(n_layers) to prevent the residual stream from
            growing too large as depth increases.
          - Biases: zero initialised
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        """
        token_ids: [batch_size, seq_len]  — integer token IDs

        Returns logits: [batch_size, seq_len, vocab_size]
          logits[b, t, v] = score for token v being next after position t
                            in sequence b
        """
        batch, seq_len = token_ids.shape

        # ── Step 1: Token embeddings ──────────────────────────────────────────
        # Look up the embedding vector for each token ID
        tok_embeddings = self.tok_emb(token_ids)
        # [batch, seq_len, emb_dim]

        # ── Step 2: Positional embeddings ─────────────────────────────────────
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # .to(token_ids.device) ensures positions are on the same device
        # (CPU or GPU) as the input — important when using GPU
        positions     = torch.arange(seq_len, device=token_ids.device)
        pos_embeddings = self.pos_emb(positions)
        # [seq_len, emb_dim] — broadcasts across batch dimension

        # ── Step 3: Combine and apply dropout ─────────────────────────────────
        x = self.emb_drop(tok_embeddings + pos_embeddings)
        # [batch, seq_len, emb_dim]

        # ── Step 4: Pass through all transformer blocks ───────────────────────
        # Each block refines the representations.
        # The output of one block feeds directly into the next.
        for block in self.trf_blocks:
            x = block(x)
        # [batch, seq_len, emb_dim]

        # ── Step 5: Final layer normalisation ─────────────────────────────────
        x = self.final_norm(x)
        # [batch, seq_len, emb_dim]

        # ── Step 6: Project to vocabulary logits ──────────────────────────────
        logits = self.out_head(x)
        # [batch, seq_len, vocab_size]
        # logits[b, t, :] = scores over all 50257 tokens for position t

        return logits


# ── Text generation function ──────────────────────────────────────────────────
def generate_text_simple(model, token_ids, max_new_tokens, context_size):
    """
    Greedy text generation — always picks the highest-scoring next token.

    model          : GPTModel instance (in eval mode)
    token_ids      : [1, seq_len]  — starting token IDs (batch size 1)
    max_new_tokens : how many new tokens to generate
    context_size   : model's maximum context window

    Returns: [1, seq_len + max_new_tokens]
    """
    for _ in range(max_new_tokens):
        # Trim to the last context_size tokens if sequence is too long
        # (the model cannot handle sequences longer than context_size)
        idx_cond = token_ids[:, -context_size:]

        # Forward pass — no gradients needed during generation
        with torch.no_grad():
            logits = model(idx_cond)
            # [1, seq_len, vocab_size]

        # We only care about the LAST position's logits
        # — that's the prediction for the next token
        logits = logits[:, -1, :]
        # [1, vocab_size]

        # Pick the token with the highest score (greedy decoding)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        # [1, 1]

        # Append the new token to the sequence and loop
        token_ids = torch.cat([token_ids, next_token], dim=1)
        # [1, seq_len + 1]

    return token_ids


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tiktoken

    GPT_CONFIG = {
        "vocab_size"     : 50257,
        "context_length" : 256,
        "emb_dim"        : 768,
        "n_heads"        : 12,
        "n_layers"       : 12,
        "drop_rate"      : 0.1,
        "qkv_bias"       : False,
    }

    torch.manual_seed(42)
    tokenizer = tiktoken.get_encoding("gpt2")

    # ── Test 1: Model creation and parameter count ────────────────────────────
    print("=" * 55)
    print("TEST 1: Model creation")
    print("=" * 55)

    model = GPTModel(GPT_CONFIG)
    model.eval()

    # Count parameters
    # Note: weight tying means tok_emb and out_head share weights,
    # so we count unique parameters only
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(
        p.numel() for p in set(model.parameters())
    )

    print(f"\nParameter counts:")
    print(f"  Total (with sharing counted twice): {total_params:,}")
    print(f"  Unique (actual memory used)       : {unique_params:,}")
    print(f"\nBreakdown:")
    print(f"  Token embedding  : {model.tok_emb.weight.numel():>12,}")
    print(f"  Pos embedding    : {model.pos_emb.weight.numel():>12,}")
    print(f"  12 × Transformer : {sum(p.numel() for b in model.trf_blocks for p in b.parameters()):>12,}")
    print(f"  Final LayerNorm  : {sum(p.numel() for p in model.final_norm.parameters()):>12,}")
    print(f"  Output head      : (shared with tok_emb)")

    # ── Test 2: Forward pass shape ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 2: Forward pass")
    print("=" * 55)

    batch_input = torch.randint(0, GPT_CONFIG["vocab_size"], (4, 16))
    print(f"\nInput  shape : {batch_input.shape}  [batch=4, seq=16]")

    with torch.no_grad():
        logits = model(batch_input)

    print(f"Output shape : {logits.shape}  [batch=4, seq=16, vocab=50257]")
    print(f"\nlogits[0, 0, :5] = {logits[0, 0, :5].tolist()}")
    print("(raw scores — highest score = predicted next token)")

    predicted_ids = torch.argmax(logits, dim=-1)
    print(f"\nPredicted token IDs shape: {predicted_ids.shape}")
    print(f"First sequence predictions: {predicted_ids[0].tolist()}")

    # ── Test 3: Text generation ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 3: Text generation (before training)")
    print("=" * 55)

    prompt = "Every effort moves you"
    encoded = tokenizer.encode(prompt)
    token_ids = torch.tensor([encoded])   # [1, seq_len]

    print(f"\nPrompt       : '{prompt}'")
    print(f"Prompt tokens: {encoded}")

    generated = generate_text_simple(
        model          = model,
        token_ids      = token_ids,
        max_new_tokens = 10,
        context_size   = GPT_CONFIG["context_length"],
    )

    decoded = tokenizer.decode(generated[0].tolist())
    print(f"\nGenerated    : '{decoded}'")
    print("\n(Gibberish is expected — the model has random weights)")
    print("(After training in Step 9 it will generate coherent text)")

    # ── Test 4: Weight tying verification ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 4: Weight tying verification")
    print("=" * 55)

    same_object = model.tok_emb.weight is model.out_head.weight
    print(f"\ntok_emb.weight is out_head.weight : {same_object}")
    print(f"tok_emb.weight shape              : {model.tok_emb.weight.shape}")
    print(f"out_head.weight shape             : {model.out_head.weight.shape}")
    print("✅ Weights are shared — same tensor object in memory")

    # ── Test 5: Memory footprint ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 5: Memory footprint")
    print("=" * 55)

    # Each parameter is a 32-bit float = 4 bytes
    mem_bytes  = unique_params * 4
    mem_mb     = mem_bytes / (1024 ** 2)
    mem_gb     = mem_bytes / (1024 ** 3)

    print(f"\nUnique parameters : {unique_params:,}")
    print(f"Memory (float32)  : {mem_mb:.1f} MB  ({mem_gb:.2f} GB)")
    print(f"\nThis fits comfortably in CPU RAM.")
    print(f"A GPU with 4GB+ VRAM can train this model.")

    # ── Test 6: Full architecture summary ────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 6: Architecture summary")
    print("=" * 55)
    print()
    print("GPTModel")
    print("├── tok_emb      Embedding(50257, 768)")
    print("├── pos_emb      Embedding(256, 768)")
    print("├── emb_drop     Dropout(p=0.1)")
    print("├── trf_blocks   ModuleList(")
    print("│   ├── [0]  TransformerBlock(")
    print("│   │        ├── norm1      LayerNorm(768)")
    print("│   │        ├── attention  CausalMultiHeadAttention(768, heads=12)")
    print("│   │        ├── norm2      LayerNorm(768)")
    print("│   │        ├── ff         FeedForward(768→3072→768)")
    print("│   │        └── drop       Dropout(p=0.1)")
    print("│   ├── [1]  TransformerBlock(...")
    print("│   └── ... × 12 total")
    print("│   )")
    print("├── final_norm   LayerNorm(768)")
    print("└── out_head     Linear(768, 50257, bias=False)  [shared with tok_emb]")

    print("\n✅ Full GPT model assembled and working!")
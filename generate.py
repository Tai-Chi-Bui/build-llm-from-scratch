# generate.py
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gpt_model import GPTModel


# ── Core generation function ──────────────────────────────────────────────────
def generate(
    model,
    token_ids,
    max_new_tokens,
    context_size,
    temperature=1.0,
    top_k=None,
    eos_id=None,
):
    """
    Generate text with controlled randomness.

    model          : trained GPTModel in eval mode
    token_ids      : [1, seq_len] starting token IDs
    max_new_tokens : how many tokens to generate
    context_size   : model's maximum context window
    temperature    : float > 0
                       < 1.0 → sharper, more deterministic
                       = 1.0 → use raw model probabilities
                       > 1.0 → flatter, more random
    top_k          : int or None
                       None → sample from full vocabulary
                       k    → restrict to k highest-prob tokens
    eos_id         : optional token ID to stop generation early
                     (GPT-2 uses token 50256 = <|endoftext|>)
    """
    for _ in range(max_new_tokens):

        # Trim to context window
        idx_cond = token_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on last position only — predicting the NEXT token
        logits = logits[:, -1, :]   # [1, vocab_size]

        # ── Top-k filtering ───────────────────────────────────────────────────
        # Keep only the k tokens with the highest logits.
        # Set all others to -infinity so softmax gives them 0 probability.
        if top_k is not None:
            # torch.topk returns (values, indices) of top k elements
            top_logits, _ = torch.topk(logits, top_k)

            # The minimum value in the top-k set
            min_val = top_logits[:, -1]   # [1]

            # Replace everything below this threshold with -inf
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # ── Temperature scaling ───────────────────────────────────────────────
        # Divide logits by temperature BEFORE softmax.
        # temp < 1 → logits grow larger → distribution sharpens
        # temp > 1 → logits shrink  → distribution flattens
        if temperature > 0.0:
            logits = logits / temperature

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)   # [1, vocab_size]

            # Sample one token from the distribution
            # torch.multinomial draws one sample proportional to probs
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        else:
            # temperature = 0 → pure greedy (deterministic)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop early if end-of-sequence token is generated
        if eos_id is not None and next_token.item() == eos_id:
            break

        # Append new token and continue
        token_ids = torch.cat([token_ids, next_token], dim=1)

    return token_ids


# ── Helper: encode prompt and decode output ───────────────────────────────────
def text_to_ids(text, tokenizer, device):
    ids = tokenizer.encode(text)
    return torch.tensor([ids]).to(device)

def ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)


# ── Load trained model ─────────────────────────────────────────────────────────
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    config     = checkpoint["config"]
    model      = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


# ── Demonstrations ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load the model we trained in Step 9
    print("Loading trained model...")
    model, config = load_model("gpt_model_trained.pth", device)
    print("✅ Model loaded\n")

    PROMPT       = "Every effort moves you"
    CONTEXT_SIZE = config["context_length"]


    # ── Demo 1: Greedy vs sampling ────────────────────────────────────────────
    print("=" * 60)
    print("DEMO 1: Greedy decoding (temperature=0)")
    print("=" * 60)

    torch.manual_seed(42)
    ids  = text_to_ids(PROMPT, tokenizer, device)
    out  = generate(model, ids, max_new_tokens=25,
                    context_size=CONTEXT_SIZE, temperature=0.0)
    print(f"\n'{ids_to_text(out, tokenizer)}'\n")
    print("Note: run this 3 times — output is always identical")


    # ── Demo 2: Temperature effect ────────────────────────────────────────────
    print("=" * 60)
    print("DEMO 2: Temperature effect (top_k=10 fixed)")
    print("=" * 60)

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

    for temp in temperatures:
        torch.manual_seed(42)
        ids = text_to_ids(PROMPT, tokenizer, device)
        out = generate(
            model, ids,
            max_new_tokens = 20,
            context_size   = CONTEXT_SIZE,
            temperature    = temp,
            top_k          = 10,
        )
        text = ids_to_text(out, tokenizer)
        print(f"\ntemp={temp:.1f}: '{text}'")

    print("\n→ Low temp: safe, repetitive")
    print("→ High temp: creative but risking nonsense")


    # ── Demo 3: Top-k effect ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 3: Top-k effect (temperature=1.0 fixed)")
    print("=" * 60)

    top_ks = [1, 5, 20, 50, None]

    for k in top_ks:
        torch.manual_seed(42)
        ids = text_to_ids(PROMPT, tokenizer, device)
        out = generate(
            model, ids,
            max_new_tokens = 20,
            context_size   = CONTEXT_SIZE,
            temperature    = 1.0,
            top_k          = k,
        )
        text  = ids_to_text(out, tokenizer)
        label = f"k={k}" if k else "k=None (full vocab)"
        print(f"\n{label:20s}: '{text}'")

    print("\n→ k=1: always picks top token (greedy)")
    print("→ k=None: can pick any of 50,257 tokens (risky)")


    # ── Demo 4: Sweet spot combinations ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 4: Recommended setting combinations")
    print("=" * 60)

    combos = [
        (0.0,  None, "Greedy — deterministic, good for factual tasks"),
        (0.7,  20,   "Conservative — coherent, low repetition"),
        (1.0,  50,   "Balanced  — GPT-2 default territory"),
        (1.4,  100,  "Creative  — diverse, occasional errors"),
        (2.0,  200,  "Wild      — very diverse, often nonsensical"),
    ]

    for temp, k, label in combos:
        torch.manual_seed(42)
        ids = text_to_ids(PROMPT, tokenizer, device)
        out = generate(
            model, ids,
            max_new_tokens = 25,
            context_size   = CONTEXT_SIZE,
            temperature    = temp,
            top_k          = k,
        )
        text = ids_to_text(out, tokenizer)
        print(f"\n{label}")
        print(f"  temp={temp}, k={k}")
        print(f"  '{text}'")


    # ── Demo 5: EOS token — early stopping ───────────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 5: Early stopping with EOS token")
    print("=" * 60)

    # Token 50256 = <|endoftext|> — tells the model a document ended
    # If the model generates this, we stop (it thinks it's done)
    torch.manual_seed(123)
    ids = text_to_ids(PROMPT, tokenizer, device)
    out = generate(
        model, ids,
        max_new_tokens = 50,
        context_size   = CONTEXT_SIZE,
        temperature    = 1.0,
        top_k          = 40,
        eos_id         = 50256,
    )
    text = ids_to_text(out, tokenizer)
    tokens_generated = out.shape[1] - ids.shape[1]
    print(f"\nGenerated {tokens_generated} tokens before stopping")
    print(f"'{text}'")


    # ── Demo 6: Visualise probability distributions ───────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 6: Visualising temperature effect on probabilities")
    print("=" * 60)

    # Get logits for one position
    ids_viz = text_to_ids(PROMPT, tokenizer, device)
    with torch.no_grad():
        logits_viz = model(ids_viz)[:, -1, :]   # [1, 50257]

    # Top 10 tokens by probability (under temperature=1.0)
    probs_raw = torch.softmax(logits_viz, dim=-1)
    top10_probs, top10_ids = torch.topk(probs_raw[0], 10)
    top10_words = [tokenizer.decode([i.item()]) for i in top10_ids]

    # Pre-compute the full softmax distributions ONCE per temperature
    # (instead of recomputing inside the loop for every word)
    probs_t05 = torch.softmax(logits_viz / 0.5, dim=-1)[0]
    probs_t10 = probs_raw[0]
    probs_t20 = torch.softmax(logits_viz / 2.0, dim=-1)[0]

    print("\nTop 10 tokens after prompt 'Every effort moves you':")
    print(f"\n{'Token':15s}  {'p(temp=0.5)':>12}  {'p(temp=1.0)':>12}  {'p(temp=2.0)':>12}")
    print("-" * 55)

    # Use enumerate to get the correct index — the previous version used
    # top10_words.index(word) which returned the WRONG index when BPE produced
    # duplicate decoded strings (rare but possible).
    for rank, (word, token_id) in enumerate(zip(top10_words, top10_ids)):
        p05 = probs_t05[token_id].item()
        p10 = probs_t10[token_id].item()
        p20 = probs_t20[token_id].item()

        # Use a fixed-width format spec for the token label so long strings
        # don't break the alignment (the old `' '*(12-len(word))` crashes
        # when len(word) > 12).
        label = f"'{word}'"
        print(f"{label:15s}  {p05:>12.4f}  {p10:>12.4f}  {p20:>12.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    temps_plot = [0.5, 1.0, 2.0]

    for ax, temp in zip(axes, temps_plot):
        scaled_probs = torch.softmax(logits_viz / temp, dim=-1)
        top_p, top_i = torch.topk(scaled_probs[0], 10)
        words_plot   = [tokenizer.decode([i.item()]) for i in top_i]

        bars = ax.bar(range(10), top_p.tolist(), color="steelblue", alpha=0.8)
        ax.set_xticks(range(10))
        ax.set_xticklabels(
            [f"'{w}'" for w in words_plot],
            rotation=45, ha="right", fontsize=8
        )
        ax.set_title(f"Temperature = {temp}")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.suptitle(
        f"Prompt: '{PROMPT}' → next token distribution",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig("temperature_distributions.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to temperature_distributions.png")

    print("\n✅ All decoding demos complete!")
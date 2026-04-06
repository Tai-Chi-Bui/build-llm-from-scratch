# train.py
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import math
import os
from gpt_model import GPTModel, generate_text_simple
from dataloader import create_dataloader


# ── Configuration ─────────────────────────────────────────────────────────────
GPT_CONFIG = {
    "vocab_size"     : 50257,
    "context_length" : 256,
    "emb_dim"        : 768,
    "n_heads"        : 12,
    "n_layers"       : 12,
    "drop_rate"      : 0.1,
    "qkv_bias"       : False,
}

TRAIN_CONFIG = {
    "learning_rate"  : 5e-4,
    "weight_decay"   : 0.1,
    "num_epochs"     : 10,
    "batch_size"     : 2,      # small — we have a tiny dataset
    "eval_freq"      : 5,      # evaluate every N batches
    "eval_iter"      : 1,      # batches to average when evaluating
    "sample_freq"    : 2,      # generate sample text every N epochs
}


# ── Loss functions ─────────────────────────────────────────────────────────────
def compute_batch_loss(model, input_batch, target_batch, device):
    """
    Compute cross-entropy loss for one batch.

    input_batch  : [batch, seq_len]  — token IDs fed to model
    target_batch : [batch, seq_len]  — correct next token IDs

    Cross-entropy internally:
      1. Applies softmax to logits → probabilities
      2. Indexes the probability of the correct token
      3. Takes negative log
      4. Averages over all positions and batch items
    """
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass
    logits = model(input_batch)
    # logits : [batch, seq_len, vocab_size]

    # PyTorch's cross_entropy expects:
    #   input : [N, vocab_size]
    #   target: [N]
    # So we flatten the batch and sequence dimensions together.
    #
    # .flatten(0, 1) merges dims 0 and 1:
    #   [batch, seq_len, vocab_size] → [batch*seq_len, vocab_size]
    #
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),       # [batch*seq_len, vocab_size]
        target_batch.flatten(),     # [batch*seq_len]
    )
    return loss


def compute_loader_loss(model, dataloader, device, num_batches):
    """
    Compute average loss over num_batches from a dataloader.
    Used to get stable train/val loss estimates.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(dataloader):
            if i >= num_batches:
                break
            loss = compute_batch_loss(model, input_batch, target_batch, device)
            total_loss += loss.item()

    model.train()
    return total_loss / max(num_batches, 1)


# ── Sample text generation ─────────────────────────────────────────────────────
def generate_sample(model, tokenizer, device, prompt, max_new_tokens=30):
    """
    Generate a text sample from the model during training.
    Lets us qualitatively monitor whether the model is learning.
    """
    model.eval()

    encoded     = tokenizer.encode(prompt)
    token_ids   = torch.tensor([encoded]).to(device)

    with torch.no_grad():
        generated = generate_text_simple(
            model          = model,
            token_ids      = token_ids,
            max_new_tokens = max_new_tokens,
            context_size   = GPT_CONFIG["context_length"],
        )

    decoded = tokenizer.decode(generated[0].tolist())
    model.train()
    return decoded


# ── Training loop ──────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, optimiser, device, cfg, tokenizer):
    """
    Main training loop.

    Each epoch: iterate over every batch in train_loader once.
    Every eval_freq batches: compute and log train + val loss.
    Every sample_freq epochs: generate a text sample.
    """
    train_losses = []
    val_losses   = []
    steps        = []
    global_step  = 0

    print(f"\nTraining on: {device}")
    print(f"Epochs      : {cfg['num_epochs']}")
    print(f"Batches/ep  : {len(train_loader)}")
    print(f"Total steps : {cfg['num_epochs'] * len(train_loader)}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):

            # ── Forward pass ──────────────────────────────────────────────────
            optimiser.zero_grad()   # clear gradients from previous step
                                    # (PyTorch accumulates by default)

            loss = compute_batch_loss(model, input_batch, target_batch, device)

            # ── Backward pass ─────────────────────────────────────────────────
            # Compute gradients for every parameter via backpropagation
            loss.backward()

            # ── Gradient clipping ──────────────────────────────────────────────
            # Caps the global gradient norm at 1.0.
            # Prevents "exploding gradients" — occasional huge gradient spikes
            # that would completely derail training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── Optimiser step ────────────────────────────────────────────────
            # Update all parameters in the direction that reduces loss
            optimiser.step()

            epoch_loss  += loss.item()
            global_step += 1

            # ── Periodic evaluation ───────────────────────────────────────────
            if global_step % cfg["eval_freq"] == 0:
                train_loss = compute_loader_loss(
                    model, train_loader, device, cfg["eval_iter"]
                )
                val_loss = compute_loader_loss(
                    model, val_loader, device, cfg["eval_iter"]
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                steps.append(global_step)

                print(
                    f"  Step {global_step:4d} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f}"
                )

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch:2d}/{cfg['num_epochs']} | "
              f"Avg loss: {avg_loss:.4f}")

        # Generate a text sample to qualitatively check progress
        if epoch % cfg["sample_freq"] == 0:
            sample = generate_sample(
                model, tokenizer, device,
                prompt="Every effort moves you",
                max_new_tokens=20,
            )
            print(f"  Sample: \"{sample}\"\n")

    return train_losses, val_losses, steps


# ── Plot losses ────────────────────────────────────────────────────────────────
def plot_losses(train_losses, val_losses, steps):
    """Save a loss curve plot to disk."""
    plt.figure(figsize=(9, 4))
    plt.plot(steps, train_losses, label="Train loss", linewidth=2)
    plt.plot(steps, val_losses,   label="Val loss",   linewidth=2, linestyle="--")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("GPT Pretraining Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)
    print("Loss curve saved to loss_curve.png")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Device selection
    # Uses GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reproducibility
    torch.manual_seed(42)

    # Load text
    with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # ── Train / validation split ──────────────────────────────────────────────
    # 90% train, 10% validation
    # We split on characters BEFORE tokenising to avoid data leakage
    split      = int(0.9 * len(raw_text))
    train_text = raw_text[:split]
    val_text   = raw_text[split:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_text):,} chars")
    print(f"  Val  : {len(val_text):,} chars")

    tokenizer = tiktoken.get_encoding("gpt2")

    # ── Create data loaders ───────────────────────────────────────────────────
    train_loader = create_dataloader(
        train_text,
        tokenizer,
        batch_size  = TRAIN_CONFIG["batch_size"],
        max_length  = GPT_CONFIG["context_length"],
        stride      = GPT_CONFIG["context_length"],
        shuffle     = True,
    )

    val_loader = create_dataloader(
        val_text,
        tokenizer,
        batch_size  = TRAIN_CONFIG["batch_size"],
        max_length  = GPT_CONFIG["context_length"],
        stride      = GPT_CONFIG["context_length"],
        shuffle     = False,
    )

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches  : {len(val_loader)}")

    # ── Initialise model ──────────────────────────────────────────────────────
    model = GPTModel(GPT_CONFIG).to(device)
    model.train()

    # ── Verify starting loss ──────────────────────────────────────────────────
    # A freshly initialised model with 50,257 tokens should give:
    # loss ≈ -log(1/50257) ≈ 10.82
    # If we see something close to this, initialisation is correct.
    print("\n" + "=" * 55)
    print("SANITY CHECK: Initial loss")
    print("=" * 55)

    initial_train_loss = compute_loader_loss(
        model, train_loader, device, num_batches=2
    )
    initial_val_loss = compute_loader_loss(
        model, val_loader, device, num_batches=2
    )
    print(f"\nExpected initial loss : ~{-math.log(1/50257):.2f}")
    print(f"Actual train loss     : {initial_train_loss:.4f}")
    print(f"Actual val loss       : {initial_val_loss:.4f}")

    # ── Initial sample (before training) ─────────────────────────────────────
    tokenizer_for_sample = tiktoken.get_encoding("gpt2")
    sample_before = generate_sample(
        model, tokenizer_for_sample, device,
        prompt="Every effort moves you",
        max_new_tokens=20,
    )
    print(f"\nBefore training: \"{sample_before}\"")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = TRAIN_CONFIG["learning_rate"],
        weight_decay = TRAIN_CONFIG["weight_decay"],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TRAINING")
    print("=" * 55)

    train_losses, val_losses, steps = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimiser    = optimiser,
        device       = device,
        cfg          = TRAIN_CONFIG,
        tokenizer    = tokenizer_for_sample,
    )

    # ── Final sample (after training) ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)

    sample_after = generate_sample(
        model, tokenizer_for_sample, device,
        prompt="Every effort moves you",
        max_new_tokens=30,
    )
    print(f"\nAfter training : \"{sample_after}\"")
    print(f"\nFinal train loss : {train_losses[-1]:.4f}")
    print(f"Final val loss   : {val_losses[-1]:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict"    : model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "train_losses"        : train_losses,
        "val_losses"          : val_losses,
        "config"              : GPT_CONFIG,
    }, "gpt_model_trained.pth")
    print("\nModel saved to gpt_model_trained.pth")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_losses(train_losses, val_losses, steps)
    print("\n✅ Training complete!")
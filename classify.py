# classify.py
import torch
import torch.nn as nn
import tiktoken
import pandas as pd
import urllib.request
import zipfile
import os
from torch.utils.data import Dataset, DataLoader
from gpt_model import GPTModel
from generate import generate, text_to_ids, ids_to_text


# ── 1. Download and prepare the SMS spam dataset ──────────────────────────────
def download_spam_dataset(target_dir="data"):
    """
    Downloads the SMS Spam Collection dataset.
    Returns path to the TSV file.
    """
    os.makedirs(target_dir, exist_ok=True)
    tsv_path = os.path.join(target_dir, "SMSSpamCollection")

    if os.path.exists(tsv_path):
        print("  Dataset already downloaded")
        return tsv_path

    url      = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = os.path.join(target_dir, "spam.zip")

    print("  Downloading SMS Spam dataset...", end=" ", flush=True)
    urllib.request.urlretrieve(url, zip_path)
    print("done")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    os.remove(zip_path)
    return tsv_path


def load_and_balance_dataset(tsv_path):
    """
    Load dataset and balance classes.

    The dataset is imbalanced: ~87% ham, ~13% spam.
    We balance by downsampling ham to match spam count.
    This prevents the model from learning to always predict ham.

    Returns a DataFrame with columns: [label, text, label_id]
      label_id: 0 = ham (not spam), 1 = spam
    """
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["label", "text"],
    )

    print(f"\n  Raw dataset:")
    print(f"    Total   : {len(df):,}")
    print(f"    Ham     : {(df.label == 'ham').sum():,}")
    print(f"    Spam    : {(df.label == 'spam').sum():,}")

    # Balance: downsample ham to match spam count
    n_spam = (df.label == "spam").sum()
    df_ham  = df[df.label == "ham"].sample(n_spam, random_state=42)
    df_spam = df[df.label == "spam"]
    df      = pd.concat([df_ham, df_spam]).sample(frac=1, random_state=42)
    df      = df.reset_index(drop=True)

    # Convert string labels to integers
    df["label_id"] = df["label"].map({"ham": 0, "spam": 1})

    print(f"\n  Balanced dataset:")
    print(f"    Total   : {len(df):,}")
    print(f"    Ham (0) : {(df.label_id == 0).sum():,}")
    print(f"    Spam(1) : {(df.label_id == 1).sum():,}")

    return df


def split_dataset(df, train_frac=0.7, val_frac=0.1):
    """
    Split into train / validation / test sets.
    test_frac = 1 - train_frac - val_frac = 0.2
    """
    n       = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val :]

    return train_df, val_df, test_df


# ── 2. Dataset class ──────────────────────────────────────────────────────────
class SpamDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        """
        df         : DataFrame with columns [text, label_id]
        tokenizer  : tiktoken tokenizer
        max_length : maximum token length (pad/truncate to this)
        """
        self.texts     = df["text"].tolist()
        self.labels    = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize and pad all texts
        self.encoded = [self._encode(t) for t in self.texts]

    def _encode(self, text):
        """
        Tokenize text and pad or truncate to max_length.

        Padding: we extend short sequences with the <|endoftext|> token (50256).
        This is GPT-2's convention for padding.

        Truncating: long sequences are cut to max_length.
        We keep the LAST max_length tokens (not the first) because
        the last token is what we classify — we want the most text
        before that final position.
        """
        ids = self.tokenizer.encode(text)

        # Truncate if too long (keep last max_length tokens)
        if len(ids) > self.max_length:
            ids = ids[-self.max_length:]

        # Pad if too short
        pad_id  = self.tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]   # = 50256
        padding = [pad_id] * (self.max_length - len(ids))
        ids     = ids + padding

        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],  dtype=torch.long),
        )


# ── 3. Classification model ───────────────────────────────────────────────────
class GPTClassifier(nn.Module):
    def __init__(self, gpt_model, n_classes, freeze_layers=True):
        """
        gpt_model    : a pretrained GPTModel instance
        n_classes    : number of output classes (2 for spam/ham)
        freeze_layers: whether to freeze most layers

        Architecture:
          GPT backbone (frozen except last block + final norm)
          → final hidden state at the LAST token position
          → Linear(emb_dim, n_classes)
          → logits over classes

        IMPLEMENTATION NOTE — replacing out_head:
          The original GPTModel ends with `out_head: Linear(emb_dim → vocab_size)`
          which produces next-token logits. For classification, we REPLACE that
          head with a new `Linear(emb_dim → n_classes)`. This means:

            - The full GPTModel.forward() now produces [batch, seq, n_classes]
              (instead of [batch, seq, 50257])
            - Weight tying with tok_emb is broken on the new head, which is
              correct — the classifier output space is unrelated to the vocab
            - tok_emb itself is unchanged
        """
        super().__init__()
        self.gpt = gpt_model

        # Replace the language-modeling head with a classification head.
        # This unties the head from tok_emb (it was previously weight-tied)
        # and changes the model's final output dimension from vocab_size
        # to n_classes.
        emb_dim = gpt_model.tok_emb.embedding_dim
        self.gpt.out_head = nn.Linear(emb_dim, n_classes, bias=True)

        if freeze_layers:
            self._freeze()

    def _freeze(self):
        """
        Freeze all parameters except:
          - Last transformer block (index -1)
          - Final layer norm
          - The new classification head
        """
        # Freeze everything first
        for param in self.gpt.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for param in self.gpt.trf_blocks[-1].parameters():
            param.requires_grad = True

        # Unfreeze final layer norm
        for param in self.gpt.final_norm.parameters():
            param.requires_grad = True

        # Unfreeze the replaced classification head — it was just frozen above
        # along with the rest of self.gpt, but it's the most important thing
        # to train, so we explicitly turn its gradients back on.
        for param in self.gpt.out_head.parameters():
            param.requires_grad = True

    def forward(self, token_ids):
        """
        token_ids: [batch, seq_len]
        returns  : [batch, n_classes]  — logit scores for each class
        """
        # Pass through the modified GPT (out_head now produces n_classes logits)
        # output shape: [batch, seq_len, n_classes]
        per_position_logits = self.gpt(token_ids)

        # Take ONLY the LAST token's logits as the classification "summary".
        # This works because of causal attention: the last position has seen
        # the entire sequence and can therefore make a sequence-level decision.
        # shape: [batch, n_classes]
        return per_position_logits[:, -1, :]


# ── 4. Training and evaluation functions ──────────────────────────────────────
def compute_accuracy(model, dataloader, device):
    """
    Compute classification accuracy over a dataloader.
    Returns fraction of correctly classified samples.
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for token_ids, labels in dataloader:
            token_ids = token_ids.to(device)
            labels    = labels.to(device)

            logits    = model(token_ids)
            # Pick the class with the highest logit
            predicted = torch.argmax(logits, dim=-1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    model.train()
    return correct / total


def train_classifier(model, train_loader, val_loader, optimiser, device, n_epochs):
    """
    Fine-tuning loop for classification.

    Very similar to pretraining loop (Step 9) but:
      - Loss: cross_entropy on 2 classes, not 50,257
      - Metric: accuracy (%), not perplexity
      - Input shape: [batch, seq_len] → [batch, 2] logits
    """
    train_losses  = []
    val_losses    = []
    train_accs    = []
    val_accs      = []

    print(f"\nFine-tuning on: {device}")
    print(f"Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (token_ids, labels) in enumerate(train_loader):
            token_ids = token_ids.to(device)
            labels    = labels.to(device)

            optimiser.zero_grad()

            # Forward pass
            logits = model(token_ids)   # [batch, 2]

            # Cross-entropy loss for classification
            # logits: [batch, n_classes]
            # labels: [batch]  — integer class indices
            loss = nn.functional.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()

        # Compute metrics at end of each epoch
        avg_loss  = epoch_loss / len(train_loader)
        train_acc = compute_accuracy(model, train_loader, device)
        val_acc   = compute_accuracy(model, val_loader,   device)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for token_ids, labels in val_loader:
                token_ids = token_ids.to(device)
                labels    = labels.to(device)
                logits    = model(token_ids)
                val_loss += nn.functional.cross_entropy(logits, labels).item()
        val_loss /= len(val_loader)
        model.train()

        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:2d}/{n_epochs} | "
            f"Train loss: {avg_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Train acc: {train_acc*100:.1f}% | "
            f"Val acc: {val_acc*100:.1f}%"
        )

    return train_losses, val_losses, train_accs, val_accs


# ── 5. Inference function ─────────────────────────────────────────────────────
def classify_text(text, model, tokenizer, max_length, device):
    """
    Classify a single text string as spam or ham.
    Returns: (label_string, confidence)
    """
    model.eval()

    # Tokenize
    ids = tokenizer.encode(text)
    if len(ids) > max_length:
        ids = ids[-max_length:]
    pad_id  = 50256
    padding = [pad_id] * (max_length - len(ids))
    ids     = ids + padding

    token_ids = torch.tensor([ids]).to(device)   # [1, max_length]

    with torch.no_grad():
        logits = model(token_ids)   # [1, 2]

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)[0]   # [2]

    predicted_id  = torch.argmax(probs).item()
    confidence    = probs[predicted_id].item()
    label         = "SPAM" if predicted_id == 1 else "HAM (not spam)"

    return label, confidence, probs.tolist()


# ── 6. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading SMS Spam dataset")
    print("=" * 60)

    tsv_path = download_spam_dataset()
    df       = load_and_balance_dataset(tsv_path)
    train_df, val_df, test_df = split_dataset(df)

    print(f"\n  Train: {len(train_df):,} samples")
    print(f"  Val  : {len(val_df):,} samples")
    print(f"  Test : {len(test_df):,} samples")

    # ── Find max token length ─────────────────────────────────────────────────
    # Use the 95th percentile length to avoid one very long message
    # blowing up memory for everyone else
    all_lengths = [
        len(tokenizer.encode(t)) for t in df["text"].tolist()
    ]
    max_length  = int(sorted(all_lengths)[int(0.95 * len(all_lengths))])
    max_length  = min(max_length, 128)   # cap at 128
    print(f"\n  Max token length (95th percentile, capped 128): {max_length}")

    # ── Create datasets and loaders ───────────────────────────────────────────
    train_dataset = SpamDataset(train_df, tokenizer, max_length)
    val_dataset   = SpamDataset(val_df,   tokenizer, max_length)
    test_dataset  = SpamDataset(test_df,  tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=8,  shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=8,  shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=8,  shuffle=False)

    # ── Load pretrained GPT-2 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading pretrained GPT-2 backbone")
    print("=" * 60)

    GPT_CONFIG = {
        "vocab_size"     : 50257,
        "context_length" : 1024,
        "emb_dim"        : 768,
        "n_heads"        : 12,
        "n_layers"       : 12,
        "drop_rate"      : 0.1,
        "qkv_bias"       : True,
    }

    checkpoint = torch.load("gpt2_pretrained.pth", map_location=device)
    gpt_model  = GPTModel(GPT_CONFIG)
    gpt_model.load_state_dict(checkpoint["model_state_dict"])
    print("  ✅ GPT-2 weights loaded")

    # ── Build classifier ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Building classifier")
    print("=" * 60)

    model = GPTClassifier(
        gpt_model    = gpt_model,
        n_classes    = 2,
        freeze_layers = True,
    )
    model.to(device)

    # Show what is frozen vs trainable
    print("\n  Parameter status:")
    total      = 0
    trainable  = 0
    for name, param in model.named_parameters():
        status = "TRAIN" if param.requires_grad else "frozen"
        total     += param.numel()
        trainable += param.numel() if param.requires_grad else 0
        if param.requires_grad:
            print(f"    {status}  {name:55s}  {param.numel():>10,}")

    print(f"\n  Frozen    : {total - trainable:>10,} params")
    print(f"  Trainable : {trainable:>10,} params")
    print(f"  Total     : {total:>10,} params")
    print(f"  Training  : {trainable/total*100:.1f}% of all parameters")

    # ── Baseline accuracy (before fine-tuning) ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Baseline accuracy (before fine-tuning)")
    print("=" * 60)

    baseline_acc = compute_accuracy(model, val_loader, device)
    print(f"\n  Val accuracy (untrained head): {baseline_acc*100:.1f}%")
    print(f"  Random baseline (2 classes)  : 50.0%")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Fine-tuning")
    print("=" * 60)

    # Only pass trainable parameters to the optimiser
    # Frozen parameters are skipped automatically when requires_grad=False
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = 5e-5,    # much smaller lr than pretraining
                                # we're fine-tuning, not learning from scratch
        weight_decay = 0.01,
    )

    train_losses, val_losses, train_accs, val_accs = train_classifier(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimiser    = optimiser,
        device       = device,
        n_epochs     = 5,
    )

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Final evaluation on held-out test set")
    print("=" * 60)

    test_acc = compute_accuracy(model, test_loader, device)
    print(f"\n  Test accuracy: {test_acc*100:.2f}%")
    print(f"  (Test set was never seen during training or validation)")

    # ── Live inference examples ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Live inference on new messages")
    print("=" * 60)

    test_messages = [
        # Clear spam
        "WINNER!! You have been selected to receive a $1000 cash prize! "
        "Call now to claim your reward!",

        # Clear ham
        "Hey, are you coming to dinner tonight? Let me know by 6pm.",

        # Subtle spam
        "Congrats! Your mobile number has won a £2000 prize. "
        "To claim call 09061743810",

        # Subtle ham
        "Can we reschedule our meeting to Thursday? I have a conflict.",

        # Tricky: legitimate-sounding but spam
        "Dear customer, your account has been suspended. "
        "Click here immediately to verify your details.",

        # Very short ham
        "Ok sounds good see you then",
    ]

    print()
    for msg in test_messages:
        label, confidence, probs = classify_text(
            msg, model, tokenizer, max_length, device
        )
        display_msg = msg[:60] + "..." if len(msg) > 60 else msg
        print(f"  Message  : '{display_msg}'")
        print(f"  Prediction: {label}  (confidence: {confidence*100:.1f}%)")
        print(f"  Probs     : ham={probs[0]*100:.1f}%  spam={probs[1]*100:.1f}%")
        print()

    # ── Save fine-tuned model ─────────────────────────────────────────────────
    torch.save({
        "model_state_dict" : model.state_dict(),
        "config"           : GPT_CONFIG,
        "max_length"       : max_length,
        "n_classes"        : 2,
        "class_names"      : ["ham", "spam"],
    }, "gpt2_spam_classifier.pth")

    print(f"Model saved to gpt2_spam_classifier.pth")
    print("\n✅ Step 12 complete!")
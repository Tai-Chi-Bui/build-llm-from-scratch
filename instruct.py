# instruct.py
import torch
import torch.nn as nn
import tiktoken
import json
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader
from gpt_model import GPTModel
from generate import generate


# ── 1. Download instruction dataset ──────────────────────────────────────────
def download_instruction_data(target_path="data/instruction_data.json"):
    """
    Downloads a small Alpaca-style instruction dataset.
    1,000 instruction/input/output triples.
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists(target_path):
        print("  Dataset already downloaded")
    else:
        url = (
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
            "/main/ch07/01_main-chapter-code/instruction-data.json"
        )
        print("  Downloading instruction dataset...", end=" ", flush=True)
        urllib.request.urlretrieve(url, target_path)
        print("done")

    with open(target_path, "r") as f:
        data = json.load(f)

    print(f"  Loaded {len(data):,} instruction examples")
    return data


# ── 2. Prompt formatting ──────────────────────────────────────────────────────
def format_prompt(example, include_response=True):
    """
    Format a single example into the Alpaca prompt template.

    example: dict with keys "instruction", "input", "output"

    If include_response=True  → full prompt including the response
                                 (used during training)
    If include_response=False → prompt without response
                                 (used during inference — model fills this in)
    """
    instruction = example["instruction"]
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    # Build the instruction + optional input section
    if inp:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    if include_response:
        return prompt + output
    else:
        return prompt


# ── 3. Dataset class ──────────────────────────────────────────────────────────
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        data       : list of dicts with instruction/input/output keys
        tokenizer  : tiktoken tokenizer
        max_length : truncate sequences longer than this
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = data

        # Pre-tokenize everything
        self.encoded_pairs = [
            self._encode(example) for example in data
        ]

    def _encode(self, example):
        """
        Returns (input_ids, target_ids) where target is input SHIFTED RIGHT BY 1
        and prompt positions are masked with -100 so loss is only computed
        on response tokens.

        Why the shift? For next-token prediction, the model's logits at position i
        are meant to predict the token at position i+1. So target[i] must equal
        input[i+1], not input[i]. Without the shift, the model would just learn
        to copy the current token (which is trivially encoded in the input).

        Why -100 on the prompt positions? PyTorch's cross_entropy ignores any
        position whose target is -100 (ignore_index=-100). We want the model
        to learn to GENERATE the response, not memorise the prompt.

        Example: full_ids = [P0, P1, P2, R0, R1, R2] (3 prompt + 3 response tokens)
          input_ids  = [P0, P1, P2, R0, R1]            (drop last token)
          target_ids = [-100, -100, R0, R1, R2]         (shift by 1, mask prompt)
            - position 0: model sees P0   → target -100 (ignored: predicting P1 is prompt)
            - position 1: model sees P0,P1 → target -100 (ignored: predicting P2 is prompt)
            - position 2: model sees P0,P1,P2 → target R0 (LOSS: first response token)
            - position 3: model sees ...,R0 → target R1 (LOSS)
            - position 4: model sees ...,R1 → target R2 (LOSS)
        """
        full_text    = format_prompt(example, include_response=True)
        prompt_only  = format_prompt(example, include_response=False)

        full_ids   = self.tokenizer.encode(full_text)
        prompt_ids = self.tokenizer.encode(prompt_only)

        # Truncate if too long
        full_ids = full_ids[:self.max_length]

        # ── Shift right by 1 for next-token prediction ─────────────────────────
        # input_ids  : everything except the last token
        # target_ids : everything except the first token
        input_ids = full_ids[:-1]

        n_prompt_tokens = len(prompt_ids)

        # Build target with -100 for prompt positions and real IDs for response
        # positions. The number of prompt tokens to mask in the SHIFTED target
        # is (n_prompt_tokens - 1) because the shift already drops one prompt
        # position from the start.
        target_ids = (
            [-100] * max(0, n_prompt_tokens - 1)   # mask prompt positions
            + full_ids[n_prompt_tokens:]            # real response token IDs
        )

        # Ensure target length matches input length (truncate if prompt was
        # so long it ate into the response after truncation)
        target_ids = target_ids[:len(input_ids)]

        # If target ended up shorter than input (rare edge case from truncation),
        # pad the tail with -100 so they stay the same length
        if len(target_ids) < len(input_ids):
            target_ids = target_ids + [-100] * (len(input_ids) - len(target_ids))

        return input_ids, target_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, target_ids = self.encoded_pairs[idx]
        return (
            torch.tensor(input_ids,  dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


# ── 4. Collate function — handling variable length sequences ──────────────────
def collate_batch(batch, pad_token_id=50256):
    """
    Custom collate function for the DataLoader.

    Unlike classification (fixed length), instruction responses vary in length.
    We pad all sequences in a batch to the length of the longest one.

    Padding strategy:
      input_ids  : pad with pad_token_id (50256 = <|endoftext|>)
      target_ids : pad with -100 (ignored by cross_entropy)

    This way padding tokens never contribute to the loss.
    """
    input_batch  = []
    target_batch = []

    # Find the longest sequence in this batch
    max_len = max(len(item[0]) for item in batch)

    for input_ids, target_ids in batch:
        # How much padding does this sequence need?
        pad_len = max_len - len(input_ids)

        # Pad inputs with the pad token
        padded_input  = input_ids.tolist()  + [pad_token_id] * pad_len

        # Pad targets with -100 (ignored in loss)
        padded_target = target_ids.tolist() + [-100] * pad_len

        input_batch.append(torch.tensor(padded_input,  dtype=torch.long))
        target_batch.append(torch.tensor(padded_target, dtype=torch.long))

    return (
        torch.stack(input_batch),    # [batch, max_len]
        torch.stack(target_batch),   # [batch, max_len]
    )


# ── 5. Loss function ──────────────────────────────────────────────────────────
def compute_instruction_loss(model, input_ids, target_ids, device):
    """
    Compute cross-entropy loss on response tokens only.

    The -100 positions in target_ids are automatically skipped by
    PyTorch's cross_entropy when ignore_index=-100 is set.
    """
    input_ids  = input_ids.to(device)
    target_ids = target_ids.to(device)

    logits = model(input_ids)
    # logits : [batch, seq_len, vocab_size]

    # Flatten for cross_entropy
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),     # [batch*seq_len, vocab_size]
        target_ids.flatten(),     # [batch*seq_len]
        ignore_index=-100,        # ← skip all -100 positions
    )
    return loss


# ── 6. Training loop ──────────────────────────────────────────────────────────
def train_instruction(
    model, train_loader, val_loader, optimiser, device, n_epochs, tokenizer
):
    """
    Instruction fine-tuning loop.
    Generates a sample response every epoch to track qualitative progress.
    """
    train_losses = []
    val_losses   = []

    # Sample prompt to track progress qualitatively
    sample_prompt = {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": ""
    }

    print(f"\nFine-tuning on : {device}")
    print(f"Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for input_ids, target_ids in train_loader:
            optimiser.zero_grad()

            loss = compute_instruction_loss(
                model, input_ids, target_ids, device
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                val_loss += compute_instruction_loss(
                    model, input_ids, target_ids, device
                ).item()
        val_loss /= len(val_loader)
        model.train()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:2d}/{n_epochs} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {val_loss:.4f}")

        # Generate a sample response to track qualitative progress
        sample_response = generate_response(
            model, tokenizer, sample_prompt, device,
            max_new_tokens=40,
        )
        print(f"  Sample → '{sample_response}'\n")

    return train_losses, val_losses


# ── 7. Response generation ────────────────────────────────────────────────────
def generate_response(
    model, tokenizer, example, device,
    max_new_tokens=100,
    temperature=0.7,
    top_k=40,
):
    """
    Generate a response to an instruction example.

    Builds the prompt (without response), feeds it to the model,
    and extracts only the generated response text.
    """
    model.eval()

    # Build prompt without response
    prompt    = format_prompt(example, include_response=False)
    input_ids = tokenizer.encode(prompt)
    ids_tensor = torch.tensor([input_ids]).to(device)

    # Generate
    with torch.no_grad():
        output_ids = generate(
            model          = model,
            token_ids      = ids_tensor,
            max_new_tokens = max_new_tokens,
            context_size   = 1024,
            temperature    = temperature,
            top_k          = top_k,
            eos_id         = 50256,    # stop at <|endoftext|>
        )

    # Decode full output
    full_text = tokenizer.decode(output_ids[0].tolist())

    # Extract only the response — everything after "### Response:\n"
    marker = "### Response:\n"
    if marker in full_text:
        response = full_text.split(marker)[-1].strip()
    else:
        response = full_text[len(prompt):].strip()

    model.train()
    return response


# ── 8. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(42)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading instruction dataset")
    print("=" * 60)
    print()

    data = download_instruction_data()

    # Split: 85% train, 10% val, 5% test
    n        = len(data)
    n_train  = int(0.85 * n)
    n_val    = int(0.10 * n)

    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val:]

    print(f"\n  Train : {len(train_data):,} examples")
    print(f"  Val   : {len(val_data):,} examples")
    print(f"  Test  : {len(test_data):,} examples")

    # ── Show a formatted example ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Data format inspection")
    print("=" * 60)

    example = data[0]
    print(f"\nRaw example:")
    print(f"  instruction: '{example['instruction']}'")
    print(f"  input      : '{example.get('input','')}'")
    print(f"  output     : '{example['output']}'")

    full_prompt  = format_prompt(example, include_response=True)
    print(f"\nFormatted (full training text):")
    print("-" * 40)
    print(full_prompt)
    print("-" * 40)

    # Show which tokens get -100 vs real loss
    prompt_only = format_prompt(example, include_response=False)
    full_ids    = tokenizer.encode(full_prompt)
    prompt_ids  = tokenizer.encode(prompt_only)
    n_prompt    = len(prompt_ids)
    n_response  = len(full_ids) - n_prompt

    # After the right-shift in InstructionDataset:
    #   input_ids has length len(full_ids) - 1
    #   target_ids has (n_prompt - 1) leading -100 positions, then n_response real IDs
    n_input    = len(full_ids) - 1
    n_ignored  = max(0, n_prompt - 1)
    n_loss     = n_response

    print(f"\nToken breakdown (after shift-right-by-1 for next-token prediction):")
    print(f"  Total tokens     : {len(full_ids):3d}")
    print(f"  Input length     : {n_input:3d}  (full_ids[:-1])")
    print(f"  Ignored positions: {n_ignored:3d}  (target=-100, predicting prompt tokens)")
    print(f"  Loss positions   : {n_loss:3d}  (target=real IDs, predicting response tokens)")

    # ── Create datasets ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Creating datasets and loaders")
    print("=" * 60)

    train_dataset = InstructionDataset(train_data, tokenizer, max_length=512)
    val_dataset   = InstructionDataset(val_data,   tokenizer, max_length=512)
    test_dataset  = InstructionDataset(test_data,  tokenizer, max_length=512)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = 4,
        shuffle     = True,
        collate_fn  = collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 4,
        shuffle     = False,
        collate_fn  = collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = 4,
        shuffle     = False,
        collate_fn  = collate_batch,
    )

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── Load pretrained model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Loading pretrained GPT-2")
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
    model      = GPTModel(GPT_CONFIG)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print("  ✅ GPT-2 pretrained weights loaded")

    # For instruction fine-tuning we train the FULL model
    # (unlike classification where we froze most layers)
    # because we want the model to change how it generates text,
    # not just add a classification head on top
    trainable = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} (all layers)")

    # ── Before fine-tuning: show baseline responses ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Baseline responses (before fine-tuning)")
    print("=" * 60)

    test_examples = [
        {"instruction": "What is the capital of France?",
         "input": "", "output": ""},
        {"instruction": "Convert 100 Fahrenheit to Celsius.",
         "input": "", "output": ""},
        {"instruction": "Write a haiku about the ocean.",
         "input": "", "output": ""},
    ]

    print()
    for ex in test_examples:
        response = generate_response(model, tokenizer, ex, device,
                                     max_new_tokens=50)
        print(f"  Instruction: '{ex['instruction']}'")
        print(f"  Response   : '{response[:120]}'")
        print()

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 6: Instruction fine-tuning")
    print("=" * 60)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = 3e-5,   # very small — full model fine-tuning
        weight_decay = 0.01,
    )

    train_losses, val_losses = train_instruction(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimiser    = optimiser,
        device       = device,
        n_epochs     = 3,
        tokenizer    = tokenizer,
    )

    # ── After fine-tuning: show improved responses ────────────────────────────
    print("=" * 60)
    print("STEP 7: Responses after fine-tuning")
    print("=" * 60)

    print()
    for ex in test_examples:
        response = generate_response(model, tokenizer, ex, device,
                                     max_new_tokens=60)
        print(f"  Instruction: '{ex['instruction']}'")
        print(f"  Response   : '{response}'")
        print()

    # ── Test on unseen instructions ───────────────────────────────────────────
    print("=" * 60)
    print("STEP 8: Test on held-out examples")
    print("=" * 60)
    print()

    for example in test_data[:5]:
        response = generate_response(
            model, tokenizer, example, device, max_new_tokens=80
        )
        print(f"  Instruction: '{example['instruction']}'")
        if example.get("input"):
            print(f"  Input      : '{example['input'][:60]}'")
        print(f"  Expected   : '{example['output']}'")
        print(f"  Generated  : '{response}'")
        print()

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 9: Interactive mode — try it yourself")
    print("=" * 60)
    print("\nType your instructions below.")
    print("Press Enter twice to submit. Type 'quit' to exit.\n")

    while True:
        print("Instruction: ", end="")
        instruction = input().strip()

        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        print("Input (optional, press Enter to skip): ", end="")
        user_input = input().strip()

        example = {
            "instruction": instruction,
            "input"      : user_input,
            "output"     : "",
        }

        print("\nGenerating response...")
        response = generate_response(
            model, tokenizer, example, device,
            max_new_tokens = 150,
            temperature    = 0.7,
            top_k          = 40,
        )
        print(f"\nResponse: {response}\n")
        print("-" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict" : model.state_dict(),
        "config"           : GPT_CONFIG,
        "train_losses"     : train_losses,
        "val_losses"       : val_losses,
    }, "gpt2_instruct.pth")

    print("\nModel saved to gpt2_instruct.pth")
    print("\n✅ Step 13 complete — instruction fine-tuning done!")
    print("\n🎉 You have built a complete LLM from scratch!")
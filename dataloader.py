# dataloader.py
#
# This module prepares raw text for GPT training by creating (input, target)
# pairs using a sliding-window approach.
#
# The core idea behind language-model training:
#   Given a sequence of tokens [A, B, C, D, E], the model learns to predict
#   the NEXT token at every position:
#
#     Input:  [A, B, C, D]   →   Target: [B, C, D, E]
#
#   So for each position i, the model sees tokens[0..i] and must predict
#   tokens[i+1]. The target is simply the input shifted right by one token.
#   This is called "causal language modeling" or "next-token prediction".
#
# The sliding window creates many overlapping (or non-overlapping) training
# examples from one long text. Two key parameters control this:
#
#   max_length (context window): how many tokens per sample — this matches
#       the model's context size. The model sees this many tokens at once.
#
#   stride (step size): how far the window moves between consecutive samples.
#       - stride == max_length → no overlap, each token appears in exactly
#         one sample (most data-efficient, no redundancy)
#       - stride < max_length  → overlapping windows, the same token appears
#         in multiple samples from different contexts (can improve learning
#         but increases dataset size)
#       - stride == 1 → maximum overlap (every possible window is a sample)

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# ── 1. Dataset class ──────────────────────────────────────────────────────────
#
# A PyTorch Dataset tells the DataLoader two things:
#   __len__     → how many samples exist total
#   __getitem__ → how to fetch sample number [idx]
#
# The DataLoader then uses these to serve batches during training.
#
class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        """
        Tokenizes the full text once, then creates all (input, target) pairs
        using a sliding window.

        Args:
            text       : raw string of training text
            tokenizer  : tiktoken tokenizer (converts text → token IDs)
            max_length : how many tokens per input window (= model context size)
            stride     : how many tokens to advance the window each step.
                         stride < max_length → overlapping samples.
                         stride == max_length → no overlap.
        """
        self.input_ids  = []
        self.target_ids = []

        # Tokenize the entire text once up front — this is cheap and avoids
        # re-tokenizing every time __getitem__ is called during training.
        token_ids = tokenizer.encode(text)

        # Slide the window across the token stream.
        # We need at least (max_length + 1) tokens from each start position
        # to form one (input, target) pair, because the target is shifted
        # right by 1.
        #
        # Example with max_length=4, stride=2 on tokens [0,1,2,3,4,5,6,7,8]:
        #   Window 0: input=[0,1,2,3] target=[1,2,3,4]  (start=0)
        #   Window 1: input=[2,3,4,5] target=[3,4,5,6]  (start=2)
        #   Window 2: input=[4,5,6,7] target=[5,6,7,8]  (start=4)
        for start in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[start : start + max_length]
            target_chunk = token_ids[start + 1 : start + max_length + 1]

            # Convert Python lists to PyTorch tensors so they can be batched
            # and moved to GPU later. Each tensor has shape (max_length,).
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Total number of (input, target) pairs we created
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return the idx-th pair — DataLoader calls this repeatedly to
        # build batches of shape (batch_size, max_length)
        return self.input_ids[idx], self.target_ids[idx]


# ── 2. DataLoader factory ─────────────────────────────────────────────────────
#
# The DataLoader wraps a Dataset and handles:
#   - Batching   : groups N samples into one tensor for GPU efficiency.
#                  Instead of processing samples one-by-one, we stack them
#                  into a (batch_size, max_length) tensor so matrix operations
#                  can process all samples in parallel.
#   - Shuffling  : randomises the order of samples each epoch so the model
#                  doesn't memorise the sequence order of the training text.
#   - drop_last  : discards the final incomplete batch if it has fewer than
#                  batch_size samples. This avoids shape mismatches and keeps
#                  gradient magnitudes consistent across all training steps.
#
def create_dataloader(
    text,
    tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
):
    dataset = GPTDataset(text, tokenizer, max_length, stride)

    # DataLoader is a PyTorch utility that wraps a Dataset and turns it into
    # an iterable that yields batches. Without it, you'd have to manually:
    #   1. Generate random indices for shuffling
    #   2. Slice the dataset into groups of batch_size
    #   3. Stack individual tensors into batched tensors
    #   4. Handle the last incomplete batch
    # DataLoader does all of this for you.
    #
    # Usage in the training loop:
    #   for input_batch, target_batch in dataloader:
    #       # input_batch shape:  (batch_size, max_length)
    #       # target_batch shape: (batch_size, max_length)
    #       loss = model(input_batch, target_batch)
    #       ...
    #
    # Each full pass through the dataloader = one epoch (all samples seen once).
    dataloader = DataLoader(
        dataset,             # the GPTDataset containing all (input, target) pairs
        batch_size=batch_size,  # how many samples per batch (stacked into one tensor)
        shuffle=shuffle,     # if True, re-randomize sample order every epoch
        drop_last=drop_last, # if True, discard the final batch when it has < batch_size samples
        # num_workers=0 means data loading happens in the main process.
        # On Windows, num_workers>0 can cause multiprocessing issues, so we
        # keep it at 0. On Linux, you could increase this for faster loading.
        num_workers=0,
    )
    return dataloader


# ── 3. Inspect the dataloader ─────────────────────────────────────────────────
#
# The code below runs only when this file is executed directly (not imported).
# It provides three inspections to build intuition about how the data pipeline
# works before we plug it into the training loop.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │              THE FULL DATA FLOW (step by step)                         │
# │                                                                        │
# │  STEP 1: Raw Text (a plain string)                                     │
# │  ──────                                                                │
# │    "I had always thought Jack Donaghue..."                             │
# │    This is just a big string sitting in a .txt file.                   │
# │    The model can't read text — it only understands numbers.            │
# │                                                                        │
# │                          ↓                                             │
# │                                                                        │
# │  STEP 2: Tokenization (text → list of integers)                       │
# │  ──────                                                                │
# │    tokenizer.encode("I had always thought")                            │
# │    → [40, 550, 1464, 1807]                                            │
# │    Each word (or sub-word) becomes an integer ID from a vocabulary     │
# │    of 50,257 tokens. Now the text is a long list of numbers.          │
# │                                                                        │
# │                          ↓                                             │
# │                                                                        │
# │  STEP 3: Sliding Window (list → many (input, target) pairs)           │
# │  ──────                                                                │
# │    We chop the long list into fixed-size windows. Each window          │
# │    becomes one training example:                                       │
# │                                                                        │
# │    Full token list: [40, 550, 1464, 1807, 890, 11, 257, 3155, ...]    │
# │                                                                        │
# │    Window 1:  input  = [40, 550, 1464, 1807]   (4 tokens)             │
# │              target = [550, 1464, 1807, 890]   (shifted right by 1)   │
# │                                                                        │
# │    Window 2:  input  = [550, 1464, 1807, 890]  (moved by stride)      │
# │              target = [1464, 1807, 890, 11]                            │
# │                                                                        │
# │    Why shifted by 1? Because the model's job is to PREDICT THE NEXT   │
# │    TOKEN. So at position 0 it sees token 40 and should predict 550,   │
# │    at position 1 it sees [40,550] and should predict 1464, etc.       │
# │                                                                        │
# │                          ↓                                             │
# │                                                                        │
# │  STEP 4: Convert to Tensors (Python lists → PyTorch tensors)          │
# │  ──────                                                                │
# │    torch.tensor([40, 550, 1464, 1807])                                │
# │    Tensors are like numpy arrays — they can live on the GPU and        │
# │    support fast math operations. This step just wraps the numbers.     │
# │                                                                        │
# │    At this point we have a Dataset: a big collection of                │
# │    (input_tensor, target_tensor) pairs.                                │
# │                                                                        │
# │                          ↓                                             │
# │                                                                        │
# │  STEP 5: DataLoader (Dataset → batches)                               │
# │  ──────                                                                │
# │    The DataLoader groups samples into batches. If batch_size=4:        │
# │                                                                        │
# │    Sample 0: [40,  550,  1464, 1807]  ┐                               │
# │    Sample 1: [550, 1464, 1807, 890 ]  │→ stacked into one tensor      │
# │    Sample 2: [1464,1807, 890,  11  ]  │   shape: (4, 4)               │
# │    Sample 3: [1807, 890, 11,   257 ]  ┘   (batch_size, max_length)    │
# │                                                                        │
# │    This 2D tensor is what the model receives each training step.       │
# │    The GPU processes all 4 samples simultaneously (in parallel).       │
# │                                                                        │
# │                          ↓                                             │
# │                                                                        │
# │  STEP 6: Training Loop (not in this file, but here's the idea)        │
# │  ──────                                                                │
# │    for input_batch, target_batch in dataloader:                        │
# │        predictions = model(input_batch)   # model guesses next tokens  │
# │        loss = compare(predictions, target_batch)  # how wrong was it?  │
# │        loss.backward()   # compute gradients                           │
# │        optimizer.step()  # update model weights to be less wrong       │
# │                                                                        │
# │    Repeat for many epochs until the model gets good at predicting.     │
# └─────────────────────────────────────────────────────────────────────────┘
#
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # --- 3a. Single-sample inspection (batch_size=1, small window) ----------
    # We use a tiny window (4 tokens) and stride=1 so we can see exactly
    # how the sliding window creates consecutive (input, target) pairs.
    # With stride=1, each sample is just 1 token shifted from the previous.
    #
    # For text "A B C D E F" with max_length=4, stride=1:
    #   Sample 1: input=[A,B,C,D]  target=[B,C,D,E]
    #   Sample 2: input=[B,C,D,E]  target=[C,D,E,F]
    print("=" * 55)
    print("SINGLE SAMPLE INSPECTION  (max_length=4, stride=1)")
    print("=" * 55)

    loader_small = create_dataloader(
        raw_text,
        tokenizer,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False,    # keep original order so output is predictable
        drop_last=False,
    )

    # iter() creates an iterator, and next() pulls one batch at a time.
    # This lets us inspect individual samples sequentially.
    iterator = iter(loader_small)

    for i in range(3):
        input_batch, target_batch = next(iterator)

        # DataLoader always returns tensors with a batch dimension, even
        # when batch_size=1. squeeze(0) removes that leading dimension:
        #   (1, 4) → (4,)  — makes the output easier to read.
        # .tolist() converts the tensor back to a Python list for decoding.
        input_ids  = input_batch.squeeze(0).tolist()
        target_ids = target_batch.squeeze(0).tolist()

        input_text  = tokenizer.decode(input_ids)
        target_text = tokenizer.decode(target_ids)

        # Notice: the target is the input shifted right by 1 token.
        # The last token of input is NOT in target's last position — instead,
        # target ends with the NEXT token from the text.
        print(f"\nSample {i+1}")
        print(f"  Input IDs  : {input_ids}")
        print(f"  Target IDs : {target_ids}")
        print(f"  Input text : '{input_text}'")
        print(f"  Target text: '{target_text}'")

    # --- 3b. Real training batch (batch_size=8) -----------------------------
    # Now we use a realistic batch: 8 samples of 8 tokens each, with
    # stride=max_length (no overlap). This shows what the model actually
    # receives during training — a 2D tensor of shape (batch_size, seq_len).
    #
    # Each row in the batch is an independent training sample. The model
    # processes all 8 samples in parallel on the GPU.
    print("\n" + "=" * 55)
    print("REAL TRAINING BATCH  (max_length=8, stride=8, batch=8)")
    print("=" * 55)

    loader_real = create_dataloader(
        raw_text,
        tokenizer,
        batch_size=8,
        max_length=8,
        stride=8,        # no overlap: stride = max_length
        shuffle=False,
        drop_last=True,
    )

    # next(iter(...)) grabs just the first batch for inspection
    input_batch, target_batch = next(iter(loader_real))

    # Shape is (8, 8) = (batch_size, seq_len) — this is the shape that
    # gets fed into the model's embedding layer.
    print(f"\nInput batch shape : {input_batch.shape}")
    print(f"  → [batch_size=8, seq_len=8]")
    print(f"\nInput batch tensor:")
    print(input_batch)
    print(f"\nTarget batch tensor:")
    print(target_batch)

    # Decoding row 0 shows the actual text that one sample represents.
    # The target is the same text shifted by one token.
    print(f"\nRow 0 decoded:")
    print(f"  Input  → '{tokenizer.decode(input_batch[0].tolist())}'")
    print(f"  Target → '{tokenizer.decode(target_batch[0].tolist())}'")

    # --- 3c. Dataset statistics ---------------------------------------------
    # Shows how many training samples and batches we get from our text
    # with production-like settings (max_length=256, no overlap).
    # This determines how many gradient updates happen per epoch.
    print("\n" + "=" * 55)
    print("DATASET STATISTICS")
    print("=" * 55)

    # Using stride=256 (=max_length) for no overlap — this is typical for
    # pretraining where we want to cover the full text without redundancy.
    loader_stats = create_dataloader(
        raw_text,
        tokenizer,
        batch_size=4,
        max_length=256,
        stride=256,
        shuffle=True,
    )

    total_tokens = len(raw_text.encode("utf-8"))   # rough count
    dataset_size = len(loader_stats.dataset)
    # batches_per_epoch = dataset_size // batch_size (since drop_last=True)
    # This is how many optimizer.step() calls happen each epoch.
    print(f"\nmax_length (context window) : 256 tokens")
    print(f"Total training samples      : {dataset_size}")
    print(f"Batches per epoch           : {len(loader_stats)}")
    print(f"Batch size                  : 4")
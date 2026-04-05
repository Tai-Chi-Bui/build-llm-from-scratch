# tokenizer.py
#
# This module handles tokenization — the process of converting raw text into
# a sequence of integer IDs that an LLM can process. LLMs don't understand
# characters or words directly; they operate on "tokens", which are sub-word
# units learned from a large corpus.
#
# We use OpenAI's `tiktoken` library with the GPT-2 Byte-Pair Encoding (BPE)
# tokenizer. BPE works by:
#   1. Starting with individual bytes/characters as the initial vocabulary.
#   2. Iteratively merging the most frequent adjacent pair of tokens into a
#      new token (e.g., "t" + "h" → "th", then "th" + "e" → "the").
#   3. Repeating until the vocabulary reaches a target size (50,257 for GPT-2).
#
# This gives BPE a nice property: common words like "the" become single tokens,
# while rare or unknown words are broken into smaller subword pieces — so the
# model never encounters a truly "unknown" word.

import tiktoken


def get_tokenizer():
    """Returns the GPT-2 BPE tokenizer.

    The GPT-2 tokenizer has a vocabulary of 50,257 tokens:
      - 256 raw byte tokens (one per possible byte value)
      - 50,000 merge tokens (learned via BPE from a large text corpus)
      - 1 special token: <|endoftext|> (used as a document separator)

    Returns:
        tiktoken.Encoding: A tokenizer that can encode text → token IDs
                           and decode token IDs → text.
    """
    return tiktoken.get_encoding("gpt2")


def demonstrate_tokenizer():
    """Walks through the key concepts of BPE tokenization with examples.

    This function demonstrates:
      1. Encoding (text → token IDs) and decoding (token IDs → text)
      2. How special tokens like <|endoftext|> are handled
      3. The vocabulary size of the GPT-2 tokenizer
      4. How individual words map to token IDs
      5. How BPE splits unknown/rare words into subword pieces
      6. Compression ratio when tokenizing a real training text
    """
    tokenizer = get_tokenizer()

    # ── Basic encode / decode ──────────────────────────────────────────────
    # Tokenization is a reversible mapping: text ↔ list of integer IDs.
    # The model's embedding layer will later map each ID to a learned vector.
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces."

    # encode() converts text → list of integer token IDs.
    # `allowed_special` tells the tokenizer to treat <|endoftext|> as a single
    # special token (ID 50256) rather than splitting it into characters.
    # Without this flag, tiktoken raises an error on special tokens to prevent
    # accidental injection.
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("Original text:")
    print(f"  {repr(text)}\n")

    print("Token IDs:")
    print(f"  {token_ids}\n")

    # decode() is the inverse: it maps token IDs back to the original string.
    # encode(decode(ids)) == ids and decode(encode(text)) == text always hold,
    # making tokenization a lossless, bijective transformation.
    decoded = tokenizer.decode(token_ids)
    print("Decoded back:")
    print(f"  {repr(decoded)}\n")

    # ── Vocabulary facts ───────────────────────────────────────────────────
    # The vocabulary size determines the number of rows in the model's token
    # embedding matrix. GPT-2 uses 50,257 tokens (256 bytes + 50,000 merges
    # + 1 special <|endoftext|> token).
    print(f"Vocabulary size: {tokenizer.n_vocab:,} tokens")
    print(f"<|endoftext|> token ID: {tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]}\n")

    # ── See individual token ↔ text mappings ───────────────────────────────
    # Common English words often map to a single token because BPE learned
    # them as frequent byte-pair merges. This makes the representation compact.
    sample = "Every effort moves you"
    ids = tokenizer.encode(sample)
    print(f"Token breakdown of: {repr(sample)}")
    for tid in ids:
        piece = tokenizer.decode([tid])
        print(f"  ID {tid:>6}  →  {repr(piece)}")
    print()

    # ── BPE handles unknown words gracefully ───────────────────────────────
    # Words not seen during BPE training are decomposed into known subword
    # pieces. This means the tokenizer never produces an "UNK" (unknown)
    # token — it always falls back to smaller pieces it does know, down to
    # individual bytes if necessary.
    unknown = "Supercalifragilistic"
    ids = tokenizer.encode(unknown)
    print(f"BPE splits {repr(unknown)} into {len(ids)} subword tokens:")
    for tid in ids:
        piece = tokenizer.decode([tid])
        print(f"  ID {tid:>6}  →  {repr(piece)}")
    print()

    # ── Token count of our training text ──────────────────────────────────
    # The compression ratio (chars per token) tells us how efficient the
    # tokenizer is for this particular text. English text typically achieves
    # ~3-4 characters per token with GPT-2's BPE. A higher ratio means fewer
    # tokens to process, which directly reduces the sequence length the model
    # needs to handle (and thus training/inference cost).
    with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    all_ids = tokenizer.encode(raw_text)
    print(f"Training text → {len(all_ids):,} tokens from {len(raw_text):,} characters")
    print(f"Compression ratio: {len(raw_text)/len(all_ids):.2f} chars per token")


if __name__ == "__main__":
    demonstrate_tokenizer()
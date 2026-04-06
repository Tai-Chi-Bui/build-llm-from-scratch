# Build an LLM From Scratch — Beginner Guide

A guided, top-to-bottom tour of this project for someone who has **never done AI engineering before**.

You don't need to know what an "embedding" is. You don't need to know what "attention" means. You don't even need to know what a tensor is. You just need basic Python and the ability to think of "a list of numbers" as a thing. We'll explain every concept the moment we use it.

Read this top-to-bottom in one sitting, or one Step at a time. Each Step ends with a short **Checkpoint** — three questions you should be able to answer before moving on. If you can't answer them, re-read the Step.

---

## Table of Contents

- [Part 0 — Welcome and Setup](#part-0--welcome-and-setup)
- [Part 1 — The 30-second Mental Model of an LLM](#part-1--the-30-second-mental-model-of-an-llm)
- [Part 2 — The Foundations](#part-2--the-foundations)
  - [Step 1 — Tokenization: text → numbers](#step-1--tokenization-text--numbers)
  - [Step 2 — Data Loading: numbers → training pairs](#step-2--data-loading-numbers--training-pairs)
  - [Step 3 — Embeddings: numbers → rich vectors](#step-3--embeddings-numbers--rich-vectors)
  - [Step 4 — Attention: how tokens "talk" to each other](#step-4--attention-how-tokens-talk-to-each-other)
  - [Step 5 — Causal Multi-Head Attention: 12 specialists, no peeking](#step-5--causal-multi-head-attention-12-specialists-no-peeking)
  - [Step 6 — The Transformer Block: assembling one repeating unit](#step-6--the-transformer-block-assembling-one-repeating-unit)
  - [Step 7 — The Full GPT Model: stacking 12 blocks](#step-7--the-full-gpt-model-stacking-12-blocks)
- [Part 3 — Training and Generation](#part-3--training-and-generation)
  - [Step 8 — Training From Scratch](#step-8--training-from-scratch)
  - [Step 9 — Smarter Text Generation](#step-9--smarter-text-generation)
  - [Step 10 — Loading Real GPT-2 Weights](#step-10--loading-real-gpt-2-weights)
- [Part 4 — Fine-Tuning: Specializing the Model](#part-4--fine-tuning-specializing-the-model)
  - [Step 11 — Spam Classification](#step-11--spam-classification)
  - [Step 12 — Instruction Following](#step-12--instruction-following)
- [Part 5 — Putting It All Together](#part-5--putting-it-all-together)
- [Appendix A — Glossary](#appendix-a--glossary)
- [Appendix B — Troubleshooting](#appendix-b--troubleshooting)

---

# Part 0 — Welcome and Setup

## What is this project?

This project is an **educational reimplementation of GPT-2**, the same family of models that powers ChatGPT (just much smaller). The goal isn't to build something competitive with ChatGPT — it's to **understand**, line by line, what an LLM actually does.

You'll see:
- How text is converted into numbers a computer can process
- How those numbers get turned into "rich vectors" that capture meaning
- How the famous "attention mechanism" works under the hood
- How the model is trained to predict the next word
- How real OpenAI GPT-2 weights can be loaded into our code
- How to fine-tune the model for new tasks (spam detection, instruction following)

By the end, the box that says "LLM" will not be a black box anymore.

## What you need to know first

Just three things:
1. **Basic Python.** You can write a `for` loop and call a function. That's enough.
2. **The idea of a "list of numbers".** That's literally what a "vector" is.
3. **Patience to read carefully.** Most of the difficulty is in the words, not the math.

You do **not** need:
- Calculus (we'll never compute a derivative by hand)
- Linear algebra (we'll talk about matrix multiplication once, conceptually)
- Prior ML experience
- A GPU (everything runs on CPU; just slower)

## Setup

The project uses Python 3.13 and a virtual environment that's already created at [.venv/](.venv/).

```bash
# Activate the virtual environment (Windows bash)
source .venv/Scripts/activate

# Verify dependencies are installed
python -c "import torch, tiktoken; print('OK')"
```

If you're starting fresh:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

If you're on Windows and see strange `UnicodeEncodeError` errors when running the scripts, set the encoding once per terminal:

```bash
export PYTHONIOENCODING=utf-8
```

That's it. You're ready.

## How to read this guide

Each Step has the same structure:

1. **The question** — what problem are we solving?
2. **The intuition** — an analogy or mental model first
3. **The code** — a guided tour of the relevant file with line links
4. **Try it yourself** — run the file and see real output
5. **Checkpoint** — three self-check questions

**Don't skip the "Try it yourself" parts.** Reading code is one thing; watching it actually print things in your terminal is what makes it real.

---

# Part 1 — The 30-Second Mental Model of an LLM

Before we look at any code, here's the entire idea of an LLM in one sentence:

> **An LLM is a function that takes a sequence of words and predicts the most likely next word. To generate longer text, it just keeps doing that over and over.**

That's it. Everything else — embeddings, attention, transformer blocks — is engineering details that make this prediction more accurate.

Here's the full pipeline at a glance:

```
   "The cat sat on the"
            │
            ▼
   ┌─────────────────┐
   │   Tokenizer     │  "The cat sat on the"  →  [464, 3797, 3332, 319, 262]
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   Embeddings    │  Each integer  →  list of 768 numbers
   └─────────────────┘   ("rich vectors" the model can work with)
            │
            ▼
   ┌─────────────────┐
   │ 12 × Transformer│  The numbers flow through 12 stacked
   │     Blocks      │  blocks. Each block lets tokens "talk to
   └─────────────────┘   each other" and refine their meaning.
            │
            ▼
   ┌─────────────────┐
   │   Output Head   │  For every position, output a score for
   └─────────────────┘   each of the 50,257 possible next tokens.
            │
            ▼
       Pick a token
            │
            ▼
   "The cat sat on the mat"
            │
            └────► append, repeat
```

The training objective is just: **make the predicted next word match the real next word in the training data**. That's it. From this single objective, the model learns grammar, facts, reasoning patterns, even how to write code. Wild.

## File map

Here's every file in the project and what it teaches, in the recommended reading order:

| # | File | Teaches |
|---|---|---|
| 1 | [tokenizer.py](tokenizer.py) | Converting text to integers (BPE tokenization) |
| 2 | [dataloader.py](dataloader.py) | Building (input, target) training pairs from raw text |
| 3 | [embeddings.py](embeddings.py) | Turning integers into rich numerical vectors |
| 4 | [attention.py](attention.py) | The core of the transformer: how tokens "attend" to each other |
| 5 | [causal_attention.py](causal_attention.py) | The full multi-head, "no peeking at the future" version |
| 6 | [transformer_block.py](transformer_block.py) | Assembling one block: attention + feed-forward + tricks |
| 7 | [gpt_model.py](gpt_model.py) | Stacking 12 blocks into a complete GPT model |
| 8 | [train.py](train.py) | Training the model from scratch |
| 9 | [generate.py](generate.py) | Smarter text generation (temperature, top-k) |
| 10 | [load_gpt2_weights.py](load_gpt2_weights.py) | Loading real OpenAI GPT-2 weights |
| 11 | [classify.py](classify.py) | Fine-tuning for spam detection |
| 12 | [instruct.py](instruct.py) | Fine-tuning for instruction following |

Plus one data file: [data/the_verdict.txt](data/the_verdict.txt) — a small story we use as toy training data.

Ready? Let's start.

---

# Part 2 — The Foundations

## Step 1 — Tokenization: text → numbers

**The question.** Computers don't understand letters. They understand numbers. So how do we turn `"Every effort moves you"` into something a computer can process?

**The intuition.** Imagine you're building a dictionary where every "word fragment" gets its own ID number:

```
"Every"   → 6109
" effort" → 3626
" moves"  → 6100
" you"    → 345
```

So `"Every effort moves you"` becomes the list `[6109, 3626, 6100, 345]`. That's tokenization in one sentence: **a reversible mapping from text to a list of integers.**

The catch: a vocabulary of every possible English word would be huge (and useless for typos, code, foreign words, etc). The trick the GPT-2 tokenizer uses is called **Byte-Pair Encoding (BPE)**:

1. Start with every individual byte as its own token (256 tokens).
2. Look at the training corpus. Find the most common pair of adjacent tokens. Merge them into a new token. Repeat.
3. Stop after 50,000 merges.

After this, common chunks like `"the"` and `" effort"` become single tokens, while rare or weird strings get split into smaller pieces. The final vocabulary has:
- 256 raw byte tokens (one for every possible byte)
- 50,000 learned merge tokens
- 1 special token: `<|endoftext|>` (used to mark document boundaries)
- **Total: 50,257 tokens**

That number — **50,257** — will appear over and over in the codebase. Now you know where it comes from.

**Important property: BPE never has "unknown words".** If you give it a word it has never seen, it just splits it into smaller and smaller pieces until it finds tokens it knows. Worst case, it uses individual bytes. So the model never receives an "unknown" symbol.

**The code.** Open [tokenizer.py](tokenizer.py). It's the simplest file in the project. Key sections:

- [tokenizer.py:32-35](tokenizer.py#L32-L35) — `get_tokenizer()` returns OpenAI's pre-trained GPT-2 BPE tokenizer via the `tiktoken` library. We don't train our own tokenizer; we use the same one OpenAI used.
- [tokenizer.py:64-66](tokenizer.py#L64-L66) — Encoding text: `tokenizer.encode("Hello, do you like tea?")` returns a list of integers.
- [tokenizer.py:73](tokenizer.py#L73) — Decoding back: `tokenizer.decode([15496, 11, 466, 345, 588, 8887, 30])` gives you the original string back.
- [tokenizer.py:97-105](tokenizer.py#L97-L105) — A demo of BPE splitting an unknown word `"Supercalifragilistic"` into 6 subword pieces.

**Try it yourself.**

```bash
python tokenizer.py
```

You'll see the script encode some text, decode it back, show how BPE handles unknown words, and report the **compression ratio** of English text (about 4 characters per token — meaning 1,000 characters of text becomes about 250 tokens).

**Checkpoint:**
1. Why does the model need a tokenizer at all?
2. What is BPE and what's the size of its vocabulary in this project?
3. What does it mean that `encode(decode(ids)) == ids` always?

---

## Step 2 — Data Loading: numbers → training pairs

**The question.** OK so we have a long list of token IDs from our text file. How do we turn that into something we can use for training?

**The intuition.** Remember the 30-second model: an LLM is trained to **predict the next token**. So a single training "example" is a pair:
- **Input**: a chunk of tokens
- **Target**: the same chunk shifted right by one position

That's it. If the input is `[A, B, C, D]`, the target is `[B, C, D, E]`. Why shifted by one? Because the model's job at position `i` is to predict `target[i]`, which is the token that comes **after** `input[i]`. The "+1 shift" IS the next-token prediction.

To turn one long list of tokens into many training examples, we use a **sliding window**:

```
Tokens: [40, 550, 1464, 1807, 890, 11, 257, 3155, ...]

Window of size 4, stride (step) 2:

  Sample 0: input=[40, 550, 1464, 1807]   target=[550, 1464, 1807, 890]
  Sample 1: input=[1464, 1807, 890, 11]   target=[1807, 890, 11, 257]
  Sample 2: input=[890, 11, 257, 3155]    target=[11, 257, 3155, ...]
  ...
```

Two parameters control this:
- **`max_length`** — how many tokens per window. This is the model's "context window" — how far back it can look.
- **`stride`** — how many tokens to slide forward between windows. If `stride == max_length`, no overlap. If `stride < max_length`, samples overlap.

For training GPT, we typically use `stride == max_length` so every token is seen exactly once per epoch.

**Why batch?** Modern hardware (GPUs especially) can process many sequences in parallel. Instead of passing one sample to the model at a time, we **batch** them — stack `batch_size` samples into a single 2D tensor of shape `(batch_size, max_length)`. The model processes all of them at once. PyTorch's `DataLoader` handles the batching for us.

**The code.** Open [dataloader.py](dataloader.py).

- [dataloader.py:43-89](dataloader.py#L43-L89) — `GPTDataset` class. Tokenizes the entire text once (cheap), then slides a window across it to build all `(input, target)` pairs.
- [dataloader.py:73-80](dataloader.py#L73-L80) — The actual sliding-window loop. For each window position, slice out `input_chunk` and `target_chunk` (which is `input_chunk` shifted by 1).
- [dataloader.py:92-141](dataloader.py#L92-L141) — `create_dataloader()` factory. Wraps the dataset in a PyTorch `DataLoader` that handles batching, shuffling, and dropping the incomplete last batch.
- [dataloader.py:148-227](dataloader.py#L148-L227) — A long ASCII flow diagram explaining the entire pipeline. Read this carefully — it's the visual you want in your head for the rest of the project.

**Try it yourself.**

```bash
python dataloader.py
```

You'll see three demos:
1. **Single sample inspection** with a tiny window (size 4, stride 1) so you can see exactly how the window slides. Notice that target is always input shifted by one token.
2. **Real training batch** showing the actual `(8, 8)` 2D tensor that the model would receive.
3. **Dataset statistics** for production-like settings (`max_length=256`, `stride=256`).

**Checkpoint:**
1. Why is `target` the input shifted right by 1 token?
2. What's the difference between `stride < max_length` and `stride == max_length`?
3. What shape does a single batch have? (Answer: `(batch_size, max_length)` of integer token IDs.)

---

## Step 3 — Embeddings: numbers → rich vectors

**The question.** We've got integers like `6109`. But an integer is a poor representation of a word. The number `6109` has no relationship to `6110`. There's nothing in the integer itself to capture that "cat" and "kitten" are similar. How do we give the model something richer?

**The intuition.** Use the integer as a **lookup key** into a big table where every row is a vector of 768 numbers. That row IS the meaning of the token, encoded as 768 numbers that the model will learn during training.

Imagine a giant filing cabinet with 50,257 drawers. Each drawer contains a folder with 768 numbers in it. You give it a token ID, it pulls out the corresponding folder. That folder is the token's "embedding".

```
Token ID 6109 ("Every") → row 6109 → [0.12, -0.34, 0.56, ..., 0.78]   (768 nums)
Token ID 3626 (" effort") → row 3626 → [0.91, -0.23, 0.45, ..., 0.67]  (768 nums)
```

The 768 numbers start out **random**. During training, they get adjusted via gradient descent so that words with similar meaning end up with similar vectors.

**But there's a catch: the model needs to know the ORDER of words.** Tokens go into the model as a set of vectors, and the attention mechanism (we'll see in Step 4) doesn't naturally know that token 0 came before token 1. So we need to encode position information too.

The trick: **a second lookup table**, this one indexed by position:

```
Position 0   → row 0   → [0.05, -0.12, 0.33, ..., 0.41]   (768 nums)
Position 1   → row 1   → [0.22, 0.08, -0.19, ..., 0.55]
...
Position 255 → row 255 → [0.17, -0.44, 0.82, ..., 0.06]
```

For every token at position `i`, we look up two vectors:
- The **token embedding** (what the token IS)
- The **positional embedding** (where it SITS in the sequence)

Then we **add them together**. Element-wise. The result is a single 768-dim vector that encodes both meaning AND position. That's what feeds into the rest of the model.

Why addition? Because it's the simplest way to combine two pieces of information without growing the vector size, and because it works in practice. The model figures out from training how to disentangle them.

**The code.** Open [embeddings.py](embeddings.py).

- [embeddings.py:57-105](embeddings.py#L57-L105) — `GPT_CONFIG`. The hyperparameters of GPT-2 small. Read every comment here — these constants will appear in every later file. Especially understand `vocab_size` (50,257), `context_length` (256), `emb_dim` (768).
- [embeddings.py:108-136](embeddings.py#L108-L136) — Building the **token embedding table** with `nn.Embedding(50257, 768)`. Note the shape: 50,257 rows × 768 columns ≈ 38.6 million numbers.
- [embeddings.py:139-170](embeddings.py#L139-L170) — Building the **positional embedding table** with `nn.Embedding(256, 768)`. 256 rows (one per possible position).
- [embeddings.py:194-235](embeddings.py#L194-L235) — The actual lookup. `token_embedding_layer(input_batch)` takes integer IDs and replaces each one with its 768-dim vector. The result has shape `(batch, seq_len, 768)`.
- [embeddings.py:255-275](embeddings.py#L255-L275) — Adding token + positional embeddings together via PyTorch broadcasting.
- [embeddings.py:283-303](embeddings.py#L283-L303) — **Dropout**. During training, we randomly zero out 10% of the values in the embedding tensor. See the explanation: it prevents the model from memorizing the training data by forcing it to spread information across many neurons.

> **One small jargon item that helps:** `torch.nn` is PyTorch's "neural network toolbox". `nn.Embedding`, `nn.Linear`, `nn.Dropout`, etc. are pre-built layers you can compose. Think of them like Lego bricks for neural networks.

**Try it yourself.**

```bash
python embeddings.py
```

You'll see the full pipeline from token IDs (shape `(4, 256)` — 4 sequences of 256 tokens each) to final embeddings (shape `(4, 256, 768)` — 4 sequences × 256 positions × 768 numbers per position). The script also traces a single token through the pipeline and prints the first 5 values of its vector.

**Checkpoint:**
1. What's the difference between a token embedding and a positional embedding?
2. What shape does the output have? Why does the last dimension have 768 entries?
3. Why do we add the two embeddings together instead of concatenating them?

---

## Step 4 — Attention: how tokens "talk" to each other

This is **the** key idea behind transformers. Take it slow.

**The question.** Each token now has its own 768-dim vector. But the meaning of a word depends on context: "bank" means different things in "river bank" vs "savings bank". How can each token's vector be **updated based on the other tokens around it**?

**The intuition.** Think of it as a library lookup with three things per token:

- **Query (Q):** "Here's what I'm looking for."
- **Key (K):** "Here's what I'm about."
- **Value (V):** "Here's the information I can share."

For every token, we compute a Q, K, and V vector. Then for each token (the "querier"), we compare its Q against the K of every other token in the sequence. The closer the match, the more "attention" we pay to that token. We then take a **weighted average of all the V vectors** based on those scores. The result becomes the new representation of the querying token.

That's it. That's attention.

Concretely, for a sentence with 6 tokens:

```
1. For each token, compute Q, K, V vectors (each 64-dim, say).
2. Compute the attention "scores" matrix: scores[i, j] = Q[i] · K[j]
   (This is a 6x6 matrix. scores[i, j] tells us how much token i wants
    to attend to token j.)
3. Scale the scores: scores / sqrt(d_k)   ← prevents big numbers blowing up softmax
4. Apply softmax row-wise so each row sums to 1.
   Now each row is a probability distribution: "of all 6 tokens,
   token i pays this much attention to each."
5. The new representation of token i is:
       new_rep[i] = sum over j of: weights[i, j] * V[j]
   In other words, a weighted blend of all the V vectors.
```

Why scale by `sqrt(d_k)`? Because raw dot products grow large as `d_k` gets bigger. Large numbers passed to softmax produce extreme outputs (one number gets 1.0 and the rest get 0.0), which kills the gradients. Dividing by `sqrt(d_k)` keeps the scores in a healthy range. This is why the formal name is **scaled dot-product attention**.

**The code.** Open [attention.py](attention.py). This file is structured as a learning journey:

- [attention.py:6-96](attention.py#L6-L96) — **Part 1: A tiny worked example**. We use a 6-token sentence with 3-dimensional embeddings (instead of 768) so the numbers fit on screen. The script:
  - Builds the Q, K, V weight matrices ([attention.py:34-36](attention.py#L34-L36))
  - Projects input → Q, K, V ([attention.py:43-45](attention.py#L43-L45))
  - Computes raw scores `Q @ K.T` ([attention.py:57](attention.py#L57))
  - Scales by `sqrt(d_out)` ([attention.py:69-70](attention.py#L69-L70))
  - Applies softmax to get attention weights ([attention.py:80](attention.py#L80))
  - Computes the weighted sum `weights @ V` ([attention.py:94](attention.py#L94))
- [attention.py:99-146](attention.py#L99-L146) — **Part 2: A clean `SelfAttention` class** that wraps all the steps into one PyTorch module.
- [attention.py:168-200](attention.py#L168-L200) — **Part 3: Visualization** of the attention weight matrix on a 6-token sentence. Each row sums to 1.0; each cell tells you how much token-in-row attends to token-in-column.

**Try it yourself.**

```bash
python attention.py
```

You'll see the actual numbers at every step. Run it twice and read the output side-by-side with the code. The math will click.

> **Common confusion:** "Why are Q, K, V three different things if they all come from the same input?"
> Because each one is multiplied by a **different learned weight matrix**. Think of it as projecting the same vector into 3 different "modes":
> - Q mode: "what does this token want to find?"
> - K mode: "what does this token offer to be matched against?"
> - V mode: "what does this token pass along if matched?"
> These three views of the same token give the model a lot of flexibility.

**Checkpoint:**
1. In one sentence: what does attention compute?
2. Why do we scale the scores by `sqrt(d_k)` before softmax?
3. After attention, does each token's vector still mean only that token, or does it now contain information from the other tokens? (Answer: it's a blend.)

---

## Step 5 — Causal Multi-Head Attention: 12 specialists, no peeking

Now we add two important upgrades to plain attention.

**Upgrade 1: Multi-head.** Instead of doing attention once with one big set of Q/K/V matrices (size 768), we split it into **12 parallel "heads"**, each working on a smaller chunk of 64 dimensions (`768 / 12 = 64`).

Why? Because different heads can learn different types of relationships:
- One head might learn "this verb attends to its subject"
- Another might learn "this pronoun attends to the noun it refers to"
- Another might learn "this word attends to the start of the sentence"

By having multiple heads, the model captures multiple kinds of dependencies in parallel. We then concatenate the outputs of all heads back into a single 768-dim vector and project it once more.

**Implementation trick:** Instead of running 12 separate (768→64) projections, we run ONE big (768→768) projection and then **reshape** the output to look like 12 heads of 64. Mathematically equivalent, much faster.

**Upgrade 2: Causal masking — "no peeking at the future".** During training, we feed the model entire sentences at once. But the model's job is to predict the **next** token. If a token at position 3 could see token at position 5, it would be cheating: the model could just copy the answer.

So we **mask** the attention scores: any cell `scores[i, j]` where `j > i` is set to `-infinity` before softmax. After softmax, those cells become exactly zero, so token `i` pays zero attention to anything that comes after it.

```
Before masking (full 6×6 score matrix):
            T0    T1    T2    T3    T4    T5
       T0 [ ##    ##    ##    ##    ##    ## ]
       T1 [ ##    ##    ##    ##    ##    ## ]
       T2 [ ##    ##    ##    ##    ##    ## ]
       T3 [ ##    ##    ##    ##    ##    ## ]
       T4 [ ##    ##    ##    ##    ##    ## ]
       T5 [ ##    ##    ##    ##    ##    ## ]

After causal masking (everything in upper triangle becomes -inf):
            T0    T1    T2    T3    T4    T5
       T0 [ ##   -inf  -inf  -inf  -inf  -inf ]
       T1 [ ##    ##   -inf  -inf  -inf  -inf ]
       T2 [ ##    ##    ##   -inf  -inf  -inf ]
       T3 [ ##    ##    ##    ##   -inf  -inf ]
       T4 [ ##    ##    ##    ##    ##   -inf ]
       T5 [ ##    ##    ##    ##    ##    ##  ]

After softmax: -inf becomes 0, the lower-triangle stays as a probability distribution.
```

This is what makes the attention "causal" — every position can only attend to itself and the past. It's also why GPT models are called **causal language models**.

**The code.** Open [causal_attention.py](causal_attention.py).

- [causal_attention.py:6-61](causal_attention.py#L6-L61) — `__init__`: builds the Q/K/V projections (`Linear(768, 768)`), the output projection, the dropout, and registers a triangular **mask** as a buffer (a non-trainable tensor that lives on the model).
- [causal_attention.py:63-93](causal_attention.py#L63-L93) — `forward()` Steps 1-2: project the input to Q, K, V, then **reshape** to split into 12 heads. The result has shape `(batch, n_heads, seq_len, head_dim)`.
- [causal_attention.py:95-118](causal_attention.py#L95-L118) — Step 3-4: compute scaled scores `Q @ K.T / sqrt(head_dim)`, then **mask out the future** with `masked_fill(mask, -inf)`.
- [causal_attention.py:120-130](causal_attention.py#L120-L130) — Step 5-6: softmax → weights, then `weights @ V` to get the per-head context vectors.
- [causal_attention.py:131-149](causal_attention.py#L131-L149) — Step 7-8: concatenate the 12 heads back into a single 768-dim vector and apply the final output projection.

The `__main__` block at the bottom runs four tests to convince you the implementation is right:
1. **Shape preservation**: input shape `(4, 16, 768)` → output shape `(4, 16, 768)`.
2. **Causal mask check**: prints the actual attention weights for a 6-token example, showing the upper triangle is all zeros.
3. **Parameter count**: 4 × `(768 × 768)` = 2,359,296 parameters per attention layer.
4. **Causality test**: changes a future token in the input and verifies that earlier outputs are unchanged. Past outputs are not affected by future inputs.

**Try it yourself.**

```bash
python causal_attention.py
```

Pay close attention to Test 2 — you'll see the actual attention weight matrix and verify with your own eyes that the upper triangle is zeros.

**Checkpoint:**
1. Why split 768 dims into 12 heads of 64 instead of using one big head?
2. What does "causal masking" prevent the model from doing during training?
3. What's the shape of the score matrix for one head, given a sequence of length `S`? (Answer: `(S, S)`.)

---

## Step 6 — The Transformer Block: assembling one repeating unit

A "transformer block" is the unit that gets stacked 12 times to build the full GPT-2 small. Each block has the same structure, and the data flows through them sequentially. We're now going to understand what's inside one block.

A block contains:

1. **Multi-head causal attention** (we just built it in Step 5)
2. **Feed-forward network** (a small two-layer MLP applied to each token independently)
3. **Layer normalization** before each of the above (twice per block)
4. **Residual connections** ("skip connections") around each
5. **Dropout** for regularization

Let's understand each piece.

### LayerNorm — "the thermostat"

**The question.** As data flows through many layers, the numbers can drift to very large or very small magnitudes, which destabilizes training. How do we keep them in a healthy range?

**The intuition.** For each token's vector independently, compute its mean and variance, then **normalize**: subtract the mean, divide by standard deviation. Now the vector has mean 0 and variance 1. Then apply a learnable scale and shift so the model can choose to undo the normalization if it wants.

It's a thermostat. It doesn't matter what temperature the room was at — it brings it back to a known reference point, then lets the model decide how far to deviate.

### GELU — "a smoother on/off switch"

The feed-forward network needs an **activation function** — a nonlinear "bend" between linear layers. Without it, stacking linear layers would just be... another linear layer. Boring and not expressive.

The classic activation is ReLU (`max(0, x)`): it lets positive numbers through and zeros out negatives. Hard cliff at zero.

GELU (Gaussian Error Linear Unit) is a smoother version: it lets most positive numbers through, smoothly suppresses small ones, and lets a tiny bit of negative through. GPT-2 uses GELU because the smoother transition makes gradients flow better.

```
ReLU:          GELU:
   |              |
   |     /        |     /
   |    /         |    /
   |   /          |  /
   |  /           |/'
___|/____      ___|______
   |              |
```

### FeedForward — "each token thinks for itself"

A simple two-layer MLP applied **independently to each token's vector**:

```
Linear(768 → 3072)  →  GELU  →  Linear(3072 → 768)
```

The 4× expansion (768 → 3072) gives the model a wide intermediate space to do nonlinear processing. Then it compresses back to 768 to keep the dimensions consistent.

Importantly, the FFN does **no cross-token interaction**. Token 5 is processed independently from token 6. (All cross-token interaction happens in the attention layer.)

### Residual connections — "the express lane"

A residual connection is a simple but huge idea:

```python
x = sublayer(x) + x   # add the input back to the output
```

That `+ x` at the end is the residual. Why does it matter? Because gradients can flow **directly** through it during backpropagation, bypassing the sublayer if needed. This makes very deep networks (12+ layers) trainable. Without residuals, gradients shrink as they flow back through layers, and training stalls.

Think of it as an express lane on a highway. Even if traffic is slow inside the layer, the gradient has a fast lane straight through.

### Pre-LN vs Post-LN

There are two ways to combine LayerNorm with the sublayer:

**Post-LN (original, used in the original transformer paper):**
```python
x = LayerNorm(SubLayer(x) + x)
```

**Pre-LN (used by GPT-2 and most modern models):**
```python
x = SubLayer(LayerNorm(x)) + x
```

Pre-LN is more stable and is what we use here. The intuition: normalize the input to the sublayer, but leave the residual stream untouched.

### Putting it all together

A complete transformer block looks like this:

```
                  ┌──────────────────────────────────────┐
   x (input)  ───►│                                      │
                  │     residual = x                     │
                  │     x = LayerNorm(x)                 │
                  │     x = MultiHeadAttention(x)        │
                  │     x = Dropout(x)                   │
                  │     x = x + residual                 │ ← attention sub-layer
                  │                                      │
                  │     residual = x                     │
                  │     x = LayerNorm(x)                 │
                  │     x = FeedForward(x)               │
                  │     x = Dropout(x)                   │
                  │     x = x + residual                 │ ← feed-forward sub-layer
                  │                                      │
                  └──────────────────────────────────────┘──► x (output, same shape)
```

**The code.** Open [transformer_block.py](transformer_block.py).

- [transformer_block.py:8-36](transformer_block.py#L8-L36) — `LayerNorm`. Computes mean and variance over the last dimension, normalizes, then applies learnable scale (γ) and shift (β).
- [transformer_block.py:40-56](transformer_block.py#L40-L56) — `GELU`. The math approximation used by GPT-2.
- [transformer_block.py:60-83](transformer_block.py#L60-L83) — `FeedForward`. The two-layer MLP with GELU in the middle.
- [transformer_block.py:87-117](transformer_block.py#L87-L117) — `TransformerBlock.__init__`. Builds two LayerNorms, the attention module, the FFN, and a dropout.
- [transformer_block.py:119-156](transformer_block.py#L119-L156) — `TransformerBlock.forward`. The full Pre-LN pattern with two sub-layers and two residual connections.

The `__main__` block tests each component individually: LayerNorm gives mean ≈ 0 and variance ≈ 1, GELU produces a smooth S-curve, FeedForward preserves shape, and the full block also preserves shape.

**Try it yourself.**

```bash
python transformer_block.py
```

You'll see numerical proof that LayerNorm normalizes correctly, GELU has the expected smooth behavior, and the full block produces an output of the same shape as the input (this is critical — every block must preserve shape so we can stack them).

**Checkpoint:**
1. What does LayerNorm do, and why is it useful?
2. Why does the feed-forward layer expand to 4× the dimension and then compress back?
3. What's the purpose of residual connections, and where are they in the block?

---

## Step 7 — The Full GPT Model: stacking 12 blocks

We now have all the pieces. Time to assemble the full model.

**The question.** How do we go from one transformer block to a complete GPT-2 model that can produce next-token predictions?

**The intuition.** Stack 12 blocks. The output of block 1 is the input to block 2, and so on. Then add an output head that projects the final hidden state into a probability distribution over the 50,257 vocabulary tokens.

The full pipeline:

```
Token IDs                          shape: (batch, seq_len)
     │
     ▼
Token Embedding lookup             shape: (batch, seq_len, 768)
     +
Positional Embedding lookup        shape: (256, 768)  → broadcasted
     │
     ▼
Embedding Dropout
     │
     ▼
Block 1 (attn + FFN)               shape unchanged
     │
     ▼
Block 2                            shape unchanged
     │
     ▼
... 10 more blocks ...
     │
     ▼
Block 12                           shape: (batch, seq_len, 768)
     │
     ▼
Final LayerNorm                    shape: (batch, seq_len, 768)
     │
     ▼
Output Head: Linear(768 → 50257)   shape: (batch, seq_len, 50257)
     │
     ▼
Logits — raw scores for each
possible next token at each position
```

**Two important details:**

### Weight tying

The token embedding table has shape `(50257, 768)`: each row is a token's vector. The output head has shape `(768, 50257)`: each column projects a hidden state to a token's score. These are **transposes of each other** — they represent the same mapping (between token IDs and their 768-dim representations) in opposite directions.

Modern LLMs **share** these two weight matrices: instead of learning two separate sets of parameters, they use one tensor that serves both purposes. This is called **weight tying**. It saves about 38 million parameters and slightly improves performance.

In code: `self.out_head.weight = self.tok_emb.weight`. Same tensor, same memory.

### Greedy generation

Once the model produces logits at the last position, we need to pick a token. The simplest method is **greedy**: just pick the token with the highest score.

```python
next_token = torch.argmax(logits, dim=-1)
```

Then append it to the sequence and run the model again. Repeat for as many tokens as you want to generate. This is what `generate_text_simple()` does. (We'll see fancier sampling methods in Step 9.)

**The code.** Open [gpt_model.py](gpt_model.py).

- [gpt_model.py:7-61](gpt_model.py#L7-L61) — `GPTModel.__init__`. Builds the token embedding, positional embedding, dropout, 12 transformer blocks (in an `nn.ModuleList`), final LayerNorm, and output head.
- [gpt_model.py:50-57](gpt_model.py#L50-L57) — Weight tying: `self.out_head.weight = self.tok_emb.weight`.
- [gpt_model.py:63-75](gpt_model.py#L63-L75) — `_init_weights`: GPT-2 initializes Linear and Embedding weights from a normal distribution with standard deviation 0.02. This matters for stability at the start of training.
- [gpt_model.py:77-120](gpt_model.py#L77-L120) — `forward()`. Reads token IDs, computes token + positional embeddings, applies dropout, passes through all 12 blocks, applies final LayerNorm, and projects to logits.
- [gpt_model.py:124-158](gpt_model.py#L124-L158) — `generate_text_simple()`. The greedy generation loop. For each new token: trim to the last `context_size` tokens, run forward pass, take the last position's logits, argmax to pick the next token, append, repeat.

The `__main__` block runs 6 tests:
1. **Model creation** and parameter count (~124 million unique parameters — that's the "124M" in "GPT-2 small 124M").
2. **Forward pass shape**: input `(4, 16)` → logits `(4, 16, 50257)`.
3. **Generation**: produces gibberish (because the model is randomly initialized).
4. **Weight tying verification**: checks that `tok_emb.weight is out_head.weight`.
5. **Memory footprint**: ~472 MB in float32.
6. **Architecture summary**: a tree view of the whole model.

**Try it yourself.**

```bash
python gpt_model.py
```

You'll see the model assemble, forward a fake batch, generate gibberish (expected — random weights), and report the parameter count. Yes, it really has 124 million parameters.

**Checkpoint:**
1. Why do we apply LayerNorm one more time after all the blocks?
2. What's weight tying, and what does it save?
3. What does `generate_text_simple()` actually do step by step?

---

# Part 3 — Training and Generation

## Step 8 — Training From Scratch

**The question.** We have a model with 124 million randomly initialized parameters. How do we make it actually learn to predict the next token?

**The intuition.** This is supervised learning — the most basic kind. The recipe:

1. Pull a batch of `(input, target)` pairs from the dataloader.
2. Run the input through the model to get predictions (logits).
3. Compare the predictions to the targets using a **loss function**.
4. Use **backpropagation** to compute gradients (how should each parameter change to reduce the loss?).
5. Use an **optimizer** to nudge the parameters in the right direction.
6. Repeat for many batches and many epochs.

That's it. That's the whole training loop. In pseudocode:

```python
for epoch in range(num_epochs):
    for input_batch, target_batch in dataloader:
        logits = model(input_batch)
        loss   = cross_entropy(logits, target_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Six lines. That's all of training.

### Cross-entropy loss explained without math

Cross-entropy loss measures: **how surprised was the model by the correct answer?**

- If the model predicted the correct token with probability 1.0 → loss = 0 (no surprise).
- If the model predicted the correct token with probability 0.5 → loss ≈ 0.69 (a bit surprised).
- If the model predicted the correct token with probability 0.0001 → loss ≈ 9.2 (very surprised).

Mathematically: `loss = -log(probability the model assigned to the correct token)`. The lower the probability, the higher the loss. Training reduces the loss, which means the model assigns higher probabilities to the correct answers over time.

For a fresh untrained GPT-2 with 50,257 tokens, the initial loss is about `-log(1/50257) ≈ 10.82`. That's exactly the "I have no idea, all 50K tokens are equally likely" baseline. As training progresses, the loss drops below this number — the model is getting less confused.

### Gradient clipping

Sometimes gradients spike — a huge gradient on a single step would yank the parameters way off course and ruin training. The solution: **clip** the gradient norm at a maximum value (we use 1.0). If the global gradient norm exceeds 1.0, scale all gradients down so the norm equals 1.0. This prevents single bad steps from wrecking everything.

### What's an "epoch", "batch", "learning rate"?

- **Epoch:** one full pass through the training data. If your dataset has 100 batches, one epoch is 100 batches.
- **Batch:** the chunk of `(input, target)` pairs processed together. Bigger batch = more stable gradients but more memory.
- **Learning rate:** how big a step the optimizer takes per parameter update. Too small = slow learning. Too big = training unstable. We use `5e-4` here (0.0005), which is typical for from-scratch training.

**The code.** Open [train.py](train.py).

- [train.py:13-31](train.py#L13-L31) — `GPT_CONFIG` (the model architecture) and `TRAIN_CONFIG` (training hyperparameters).
- [train.py:35-67](train.py#L35-L67) — `compute_batch_loss()`. Runs forward pass, flattens logits and targets to 2D/1D, calls `nn.functional.cross_entropy`. Note the `.flatten(0, 1)` — this merges batch and sequence dims so cross-entropy can compute one loss per `(batch_idx, seq_idx)` cell.
- [train.py:70-86](train.py#L70-L86) — `compute_loader_loss()`. Averages loss over a few batches for stable train/val metrics.
- [train.py:114-193](train.py#L114-L193) — `train()`. The actual training loop. For each epoch, for each batch:
  1. `optimiser.zero_grad()` — clear leftover gradients from previous step
  2. `compute_batch_loss(...)` — forward pass + loss
  3. `loss.backward()` — backpropagation
  4. `clip_grad_norm_` — gradient clipping
  5. `optimiser.step()` — apply gradients
- [train.py:213-341](train.py#L213-L341) — The `__main__` block: load text, split into train/val, create dataloaders, build model, sanity-check the initial loss, train, generate sample text, save the checkpoint, plot losses.

**Try it yourself.**

```bash
python train.py
```

This actually trains a GPT-2 from scratch on `the_verdict.txt`! Things to watch:
1. The initial loss should be around 10.82 (the "no idea" baseline).
2. The loss should drop steadily — within a few epochs you'll see it under 5.
3. Sample text generated every 2 epochs gets less random over time. After 10 epochs, the model has mostly memorized the training text (because the dataset is tiny).
4. A `loss_curve.png` is saved at the end so you can see the training/validation loss over time.

The final model is saved to `gpt_model_trained.pth`.

> **Important:** The model is being trained on a tiny 20KB story. It will overfit (memorize) rather than learn general English. That's expected — this Step is about understanding the training mechanics, not building a real LLM. We use real OpenAI weights in Step 10.

**Checkpoint:**
1. What does cross-entropy loss measure conceptually?
2. Why is the initial loss approximately `10.82` for an untrained model?
3. What are the 6 steps of one training iteration (in pseudocode)?

---

## Step 9 — Smarter Text Generation

**The question.** Greedy decoding always picks the highest-probability token. That's deterministic and boring — the same prompt always gives the same output, and the output tends to be repetitive ("The cat sat on the mat. The mat was on the cat. The cat...").

How do we generate more varied, more interesting text?

**The intuition.** Don't always pick the top token. Sometimes pick a less-likely one. But how do we control "how often" we pick a less-likely one? With two parameters: **temperature** and **top-k**.

### Temperature — the "creativity dial"

Temperature is a number you divide the logits by **before** softmax:
- **temperature < 1** (e.g., 0.5): logits get larger → softmax outputs are sharper → the top token gets even more probability → more deterministic.
- **temperature = 1**: do nothing, use raw probabilities.
- **temperature > 1** (e.g., 1.5): logits get smaller → softmax outputs are flatter → all tokens become more equal → more random, more "creative".

Visualization:

```
Raw logits:        [3.0, 2.0, 1.0]
Probabilities:     [0.66, 0.24, 0.09]   ← raw (temp=1.0)

Temp=0.5:          logits become [6.0, 4.0, 2.0]
Probabilities:     [0.87, 0.12, 0.02]   ← sharper, top dominates more

Temp=2.0:          logits become [1.5, 1.0, 0.5]
Probabilities:     [0.43, 0.31, 0.26]   ← flatter, all options closer
```

### Top-k — the "shortlist filter"

Top-k restricts sampling to only the **k tokens with the highest scores**. Everything else is set to `-inf` so its probability becomes 0.

- `top_k=1` → only the best token is allowed → equivalent to greedy.
- `top_k=10` → sample from the top 10 tokens.
- `top_k=None` → no filtering, can sample from all 50,257 tokens (risky: even very low-probability tokens can be picked).

The combination of temperature + top-k is what people usually mean by "creative but controlled" sampling. Typical settings:
- **Factual tasks:** `temperature=0` (greedy). Always the most likely answer.
- **Conversational:** `temperature=0.7, top_k=20`. Balanced.
- **Creative writing:** `temperature=1.0, top_k=50`. More diverse.

### Stopping early with EOS

The model has a special token, `<|endoftext|>` (ID 50256), that means "this document ends here". If you let the model generate until it produces this token, you can stop early. We pass `eos_id=50256` to the generator.

**The code.** Open [generate.py](generate.py).

- [generate.py:11-90](generate.py#L11-L90) — `generate()`. The main function. Per step:
  1. Trim to context window
  2. Forward pass → take last position's logits
  3. Apply top-k filter (set non-top-k to `-inf`)
  4. Apply temperature: `logits / temperature`
  5. Softmax → probabilities
  6. Sample one token via `torch.multinomial`
  7. Append, optionally stop on EOS
- [generate.py:94-100](generate.py#L94-L100) — Helpers to encode prompts and decode outputs.
- [generate.py:104-111](generate.py#L104-L111) — `load_model()` to load a saved checkpoint.
- [generate.py:115-301](generate.py#L115-L301) — Six demonstrations:
  1. Pure greedy (always identical output)
  2. Effect of varying temperature (with top-k fixed)
  3. Effect of varying top-k (with temperature fixed)
  4. Recommended setting combinations
  5. Early stopping with EOS
  6. A matplotlib plot showing how temperature reshapes the probability distribution

**Try it yourself.**

After running `train.py` (which produces `gpt_model_trained.pth`):

```bash
python generate.py
```

Watch how the same prompt produces wildly different outputs depending on temperature. Try editing the script to use your own prompts.

**Checkpoint:**
1. What does temperature control, and what's the difference between `temperature=0.5` and `temperature=2.0`?
2. What does `top_k=1` produce, and why?
3. Why does the script always reset the random seed (`torch.manual_seed(42)`) before each demo?

---

## Step 10 — Loading Real GPT-2 Weights

**The question.** Our model has 124 million parameters, but if we train it on a 20 KB text file we get gibberish. Real GPT-2 was trained on 40 GB of internet text for weeks. Can we just **load** OpenAI's actual GPT-2 weights into our model and get a real LLM?

Yes. That's exactly what this Step does.

**The intuition.** OpenAI released GPT-2's weights publicly in 2019. They're stored in TensorFlow's checkpoint format. Our GPTModel has the same architecture (same layer sizes, same number of layers, same number of heads), so in principle we just need to copy the numbers from OpenAI's tensors into our PyTorch tensors.

In practice, there are five fiddly mapping issues to handle:

1. **Q, K, V splitting.** OpenAI stores the three projections (`W_query`, `W_key`, `W_value`) as ONE big combined matrix of shape `(768, 2304)`. We need to split it along the last axis into three `(768, 768)` matrices.

2. **Transposes.** OpenAI uses `(in_dim, out_dim)` convention; PyTorch's `nn.Linear` uses `(out_dim, in_dim)`. So every weight matrix needs `.T` (transpose).

3. **Bias terms.** GPT-2 uses bias vectors in the attention projections (`qkv_bias=True`). Our default training config in Step 8 used `qkv_bias=False`. When loading OpenAI weights we must build the model with `qkv_bias=True` to match.

4. **Naming differences.** OpenAI calls layer norms `ln_1` and `ln_2`; we call them `norm1` and `norm2`. The code maps them manually.

5. **Weight tying.** Our `out_head` shares its weight tensor with `tok_emb`. So when we set `tok_emb.weight`, the output head is automatically correct — no separate copy needed.

**The code.** Open [load_gpt2_weights.py](load_gpt2_weights.py).

- [load_gpt2_weights.py:11-56](load_gpt2_weights.py#L11-L56) — `download_gpt2_files()`. Downloads 7 files from OpenAI's public Azure blob: the actual weight tensors, the tokenizer files, and an `hparams.json` describing the architecture.
- [load_gpt2_weights.py:60-113](load_gpt2_weights.py#L60-L113) — `load_gpt2_params()`. Reads the TensorFlow checkpoint and parses the variable names like `model/h0/attn/c_attn/w` into a nested Python dict structure (`params["blocks"][0]["attn"]["c_attn"]["w"]`).
- [load_gpt2_weights.py:117-132](load_gpt2_weights.py#L117-L132) — `assign()`. Validates that our PyTorch tensor and OpenAI's NumPy tensor have the same shape, then wraps the NumPy array as an `nn.Parameter`. Catches mapping bugs early with a clear error.
- [load_gpt2_weights.py:135-242](load_gpt2_weights.py#L135-L242) — `load_weights_into_gpt()`. The big mapping function. For each transformer block, it:
  - Splits the combined Q/K/V weight ([load_gpt2_weights.py:158-169](load_gpt2_weights.py#L158-L169))
  - Splits the combined Q/K/V bias ([load_gpt2_weights.py:172-183](load_gpt2_weights.py#L172-L183))
  - Copies the output projection
  - Copies the feed-forward layers (with transposes)
  - Copies the layer norm parameters
- [load_gpt2_weights.py:246-371](load_gpt2_weights.py#L246-L371) — `__main__`: download → load hparams → parse weights → build model → copy weights → verify with text generation → save in our checkpoint format.

**Try it yourself.**

> ⚠️ This downloads ~500 MB of data and requires `tensorflow` to read the checkpoint format. The script will install TensorFlow if missing.

```bash
python load_gpt2_weights.py
```

After loading, the script generates text from a few prompts. Now you'll see **real, coherent text**:

```
Prompt : 'The capital of France is'
Output : 'The capital of France is Paris, and the city is also one of the
          largest in the world.'
```

This is the same model that powered ChatGPT's early ancestor. The output is saved as `gpt2_pretrained.pth` and used in the next two Steps as a starting point for fine-tuning.

**Checkpoint:**
1. Why does loading OpenAI's weights require `qkv_bias=True` while training from scratch in Step 8 used `qkv_bias=False`?
2. What does `assign()` do, and why is shape validation important?
3. Why don't we need to separately copy the output head weights?

---

# Part 4 — Fine-Tuning: Specializing the Model

## Step 11 — Spam Classification

**The question.** Now that we have a real pretrained GPT-2, can we adapt it to a specific task — like classifying SMS messages as spam or not spam?

**The intuition.** Yes, with **fine-tuning**. The general recipe:
1. Take the pretrained model (it already knows English).
2. Replace the output head (which predicts 50,257 vocab tokens) with a new, smaller head that predicts just 2 classes (spam vs ham).
3. Freeze most of the model so we don't accidentally destroy the language knowledge.
4. Train only the new head + the last few layers on labeled spam/ham examples.

This works because the lower layers of GPT-2 have learned generic language features (grammar, word meanings, sentence structure) that transfer to many tasks. We only need to adapt the top of the model to the new task.

### Dataset balancing — a critical detail

The SMS Spam dataset has ~5,500 messages: about 87% ham (legit) and 13% spam. If you train a classifier on this naively, it learns to **always predict ham** and gets 87% accuracy without learning anything about spam.

The fix: **balance** the dataset by downsampling ham to match the spam count. Now both classes have equal representation, and the model has to actually learn the difference.

### Replacing the head

GPT-2 ends with `out_head: Linear(768 → 50257)` — a projection from hidden state to vocabulary scores. For classification, we replace this with `Linear(768 → 2)` — a projection from hidden state to (ham_score, spam_score).

The new head's parameters start random and get trained from scratch. The rest of the model (everything except the last transformer block, the final LayerNorm, and the new head) is **frozen** — its `requires_grad` is set to `False` so the optimizer doesn't update those parameters. This makes training much faster and uses much less memory.

Result: instead of training 124M parameters, we train about 7M (the last block + the new head). The model fine-tunes in minutes.

**The code.** Open [classify.py](classify.py).

- [classify.py:15-79](classify.py#L15-L79) — Download the SMS Spam Collection, load it, and balance ham vs spam.
- [classify.py:99-148](classify.py#L99-L148) — `SpamDataset`. Tokenizes each SMS, **truncates** if too long (keeping the LAST tokens, since the last position is what we classify), and **pads** with `<|endoftext|>` (token 50256) if too short.
- [classify.py:152-225](classify.py#L152-L225) — `GPTClassifier`. **Replaces the existing `out_head`** with a new `Linear(emb_dim, n_classes)`. The full GPTModel forward now produces classification logits directly. The `_freeze()` method freezes everything except the last block, the final LayerNorm, and the new output head.
- [classify.py:198-225](classify.py#L198-L225) — `forward()`. Just calls the modified GPT and takes the **last token's logits** (the last position has seen the entire sequence and can make a sequence-level decision).
- [classify.py:243-315](classify.py#L243-L315) — `train_classifier()`. A standard fine-tuning loop with cross-entropy loss on 2 classes and accuracy metric.
- [classify.py:319-346](classify.py#L319-L346) — `classify_text()`. Inference on a single message. Returns label, confidence, and the full probability vector.
- [classify.py:349-525](classify.py#L349-L525) — The `__main__` pipeline: download, balance, split, load pretrained GPT-2, build classifier, fine-tune, evaluate on test set, and run live inference on 6 hand-picked example messages.

**Try it yourself.**

> ⚠️ Requires `gpt2_pretrained.pth` from Step 10.

```bash
python classify.py
```

You should see:
- Initial baseline accuracy ≈ 50% (random — the new head is untrained)
- After 5 epochs of fine-tuning, val accuracy climbs to ~95%
- Final test accuracy on held-out data ≈ 95%
- The 6 example messages get classified correctly

**Checkpoint:**
1. Why is balancing the dataset important when 87% is ham?
2. What does "freezing" a layer do and why is it useful here?
3. Why do we use the LAST token's hidden state for classification instead of the first?

---

## Step 12 — Instruction Following

**The question.** GPT-2 just predicts next tokens. It doesn't naturally answer questions or follow instructions. How do we teach it to behave like an instruction-following assistant?

**The intuition.** Show it examples. Lots of examples of (instruction, response) pairs, formatted in a consistent way, and fine-tune the model on them. Over time the model learns: "when I see this format ending with `### Response:`, my job is to generate a helpful answer."

This is exactly how the original ChatGPT was created: pretrain on internet text, then fine-tune on human-written instruction examples. Same idea, much smaller scale.

### The Alpaca prompt format

We format every example consistently:

```
### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

If there's an additional input (like for translation tasks):

```
### Instruction:
Translate the following sentence to French.

### Input:
Hello, how are you?

### Response:
Bonjour, comment allez-vous?
```

This particular format is called **Alpaca** (after the Stanford Alpaca dataset). It's just one of many possible formats — the important thing is that you pick one and stick with it.

### The `-100` masking trick

Here's a subtle but important detail. We want the model to learn to **generate the response**, not to memorize the prompt. So when computing the loss:

- For positions in the **response**: compute cross-entropy normally.
- For positions in the **prompt**: ignore them. Do not compute loss.

PyTorch's `cross_entropy` has a special `ignore_index` parameter (default `-100`). Any position whose target equals `-100` is silently skipped. So we set the target IDs to `-100` for all prompt positions and to the real token IDs for all response positions.

But there's another subtlety: **next-token prediction needs the target shifted by 1.** The model's output at position `i` is meant to predict the token at position `i+1`. So:

```
Suppose full_ids = [P0, P1, P2, R0, R1, R2]   (3 prompt tokens, 3 response tokens)

input_ids  = [P0, P1, P2, R0, R1]              (drop the last token)
target_ids = [-100, -100, R0, R1, R2]          (shift by 1, mask prompt positions)

  Position 0: model sees P0, target=-100 → ignored (still in prompt)
  Position 1: model sees P0,P1, target=-100 → ignored (still in prompt)
  Position 2: model sees P0,P1,P2, target=R0 → loss (first response token!)
  Position 3: model sees ...,R0, target=R1 → loss
  Position 4: model sees ...,R1, target=R2 → loss
```

The model is trained to produce the right response token at each position, given everything before it.

### Full-model fine-tuning

Unlike classification (where we froze most layers), instruction tuning **trains all 124M parameters**. The reason: we're teaching the model to change *how it generates text*, not just adding a small task-specific head. We need every layer to adapt.

The learning rate is much smaller than from-scratch training (`3e-5` vs `5e-4`) because we're nudging an already-good model, not building one from scratch.

**The code.** Open [instruct.py](instruct.py).

- [instruct.py:14-37](instruct.py#L14-L37) — Download a 1,000-example instruction dataset from Sebastian Raschka's repo.
- [instruct.py:40-72](instruct.py#L40-L72) — `format_prompt()`. Builds the Alpaca-formatted text with optional input.
- [instruct.py:75-153](instruct.py#L75-L153) — `InstructionDataset._encode()`. Builds the (input, target) pair with the right-shift and the `-100` masking on prompt positions. Read the docstring carefully — it has a worked example.
- [instruct.py:157-194](instruct.py#L157-L194) — `collate_batch()`. Custom collate function that pads variable-length sequences in a batch (different instructions have different lengths).
- [instruct.py:197-216](instruct.py#L197-L216) — `compute_instruction_loss()`. Standard cross-entropy with `ignore_index=-100`.
- [instruct.py:220-285](instruct.py#L220-L285) — `train_instruction()`. The fine-tuning loop. Generates a sample response every epoch so you can see qualitatively how the model improves.
- [instruct.py:288-331](instruct.py#L288-L331) — `generate_response()`. Builds the Alpaca prompt without the response, runs the generator, and extracts only the text after `### Response:`.
- [instruct.py:334-571](instruct.py#L334-L571) — The `__main__` pipeline plus an **interactive mode** at the end where you can type your own instructions.

**Try it yourself.**

> ⚠️ Requires `gpt2_pretrained.pth` from Step 10.

```bash
python instruct.py
```

You'll see:
- Baseline responses from the unfinetuned model (often incoherent or off-topic)
- 3 epochs of fine-tuning with sample generations after each epoch
- Improved responses on test examples
- An interactive mode at the end where you can type your own instructions

**Checkpoint:**
1. Why do we mask the prompt tokens with `-100` in the target?
2. Why do we shift the target by 1 instead of using the same indices as the input?
3. Why is the learning rate (`3e-5`) much smaller than the from-scratch training rate (`5e-4`)?

---

# Part 5 — Putting It All Together

You've now read every Step. Let's zoom out and tell the whole story in one breath:

> An LLM is a function that takes a sequence of tokens and predicts the most likely next one. Tokens are integer IDs from a 50,257-word vocabulary built by Byte-Pair Encoding. To make the model work, we look up each integer in two tables: one for the token's meaning (token embedding) and one for its position (positional embedding), then add them. The result flows through 12 transformer blocks. Each block has multi-head causal attention (which lets every token "attend to" the tokens before it, to mix information across the sequence) and a feed-forward network (which lets each token's vector be processed independently). LayerNorm and residual connections keep everything numerically stable as the data flows through deep stacks. After the last block, a single linear layer projects the final hidden state into 50,257 logits — one score per possible next token. We pick a token (greedily, or using temperature + top-k sampling) and append it to the sequence. To train, we feed the model billions of `(input, target)` pairs where `target` is `input` shifted right by one, and minimize cross-entropy loss with backpropagation. The result is a model that "knows English" — and you can take that pretrained model and fine-tune it for new tasks like spam classification or instruction following just by replacing or adapting its output layer.

That's the entire field of language modeling, in 200 words. Everything else (RLHF, mixture of experts, longer context, etc.) is engineering on top of this foundation.

## What's next (if you want to go further)

- **Train on real data.** Try a bigger text dataset (WikiText, OpenWebText, BookCorpus). You'll need a GPU.
- **Scale up.** GPT-2 medium (355M), GPT-2 large (774M), GPT-2 XL (1.5B) — same architecture, just bigger. The OpenAI weights are available for all sizes.
- **Modern improvements.** Read about **rotary positional embeddings (RoPE)**, **grouped-query attention**, **SwiGLU activations**, **RMSNorm**. These are the small upgrades modern models like Llama use.
- **RLHF / DPO.** The technique that turned GPT-3 into ChatGPT. Fine-tunes on human preferences.
- **Inference tricks.** **KV caching** is a huge speedup at inference time. Worth understanding.
- **The original papers.** "Attention Is All You Need" (2017), "Improving Language Understanding by Generative Pre-Training" (2018, the GPT-1 paper), "Language Models are Unsupervised Multitask Learners" (2019, GPT-2).

---

# Appendix A — Glossary

**Tensor.** A multi-dimensional array. A scalar is 0-D, a vector is 1-D, a matrix is 2-D, and a tensor can have any number of dimensions. PyTorch's main data type.

**Token.** A single unit in the vocabulary. Could be a word, a piece of a word, or even a single character. GPT-2's vocabulary has 50,257 tokens.

**Token ID.** An integer that uniquely identifies a token. The tokenizer maps text ↔ list of token IDs.

**Embedding.** A vector of numbers that represents a token (or a position). Stored in a lookup table: `embedding[token_id]` returns the vector.

**Logits.** The raw, unnormalized scores produced by the model's output head. There's one logit per vocabulary token. Higher = more likely. Softmax converts logits to probabilities.

**Softmax.** Converts a vector of logits into a probability distribution (numbers between 0 and 1 that sum to 1).

**Cross-entropy loss.** The standard loss for classification. Measures how surprised the model was by the correct answer. Low loss = confident correct prediction.

**Forward pass.** Running input through the model to get output (logits).

**Backward pass / backpropagation.** Computing the gradient of the loss with respect to every parameter, by applying the chain rule of calculus backwards through the network.

**Gradient.** A number for each parameter saying "if you nudge me this way, the loss will decrease". The optimizer uses gradients to update parameters.

**Optimizer.** Updates the model's parameters using their gradients. Common ones: SGD, Adam, AdamW. We use AdamW.

**Learning rate.** How big a step the optimizer takes. Too small = slow. Too big = unstable.

**Epoch.** One full pass through the training dataset.

**Batch.** A group of training samples processed together.

**Layer.** A subnetwork that takes input, applies some transformation, and produces output. PyTorch calls them `nn.Module`.

**Attention.** The mechanism that lets each token "look at" the other tokens in the sequence and update its own representation based on them.

**Causal mask.** Restricts attention so a token can only look at itself and the tokens before it. Prevents "cheating" during training.

**Multi-head attention.** Running attention multiple times in parallel with different learned projections. Each "head" can capture a different type of relationship.

**Transformer block.** A repeating unit consisting of attention + feed-forward + layer norms + residuals. GPT-2 small has 12 of them stacked.

**Residual connection.** Adding the input of a sublayer to its output. Helps gradients flow through deep networks.

**LayerNorm.** Normalizes each token's vector to mean 0, variance 1, then applies learnable scale and shift. Stabilizes training.

**GELU.** A smooth activation function used inside the feed-forward network.

**Pretraining.** Training a model from scratch on a large generic corpus. The model learns general language patterns.

**Fine-tuning.** Taking a pretrained model and continuing to train it on a smaller, task-specific dataset. Adapts it to a new task.

**Weight tying.** Sharing the same weight tensor between two layers (here: the input embedding and the output head). Saves memory and improves training.

---

# Appendix B — Troubleshooting

### `UnicodeEncodeError: 'charmap' codec can't encode character ...`

You're on Windows and the terminal uses cp1252 encoding. Set UTF-8 once per terminal:

```bash
export PYTHONIOENCODING=utf-8
```

### `ModuleNotFoundError: No module named 'torch'`

You haven't activated the virtual environment, or PyTorch isn't installed.

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

### `FileNotFoundError: gpt2_pretrained.pth`

`classify.py` and `instruct.py` need the pretrained GPT-2 weights. Run [load_gpt2_weights.py](load_gpt2_weights.py) first to download and save them.

### `FileNotFoundError: gpt_model_trained.pth`

`generate.py` needs a trained model. Run [train.py](train.py) first.

### Training is slow

That's expected on CPU. The full `train.py` takes a few minutes on a modern laptop. `load_gpt2_weights.py` is fast once the download completes. `classify.py` and `instruct.py` take a few minutes each.

### `Out of memory` errors

Lower the batch size in [train.py:27](train.py#L27), [classify.py:383-385](classify.py#L383-L385), or [instruct.py:401-405](instruct.py#L401-L405). You can also lower `max_length` (the context window) — it has a quadratic effect on attention memory.

### The training loss isn't going down

- Check that your data is loading correctly (run `python dataloader.py`).
- Check that the initial loss is approximately `-log(1/50257) ≈ 10.82`. If not, model initialization is broken.
- Make sure you're not calling `optimizer.zero_grad()` at the wrong time.
- Try reducing the learning rate by 10x.

### The model generates gibberish

That's expected for an untrained or barely-trained model. After 10 epochs on `the_verdict.txt`, the output should at least look word-like (even if it makes no sense). For coherent text, you need real pretrained weights — see Step 10.

---

You did it! You read the whole thing. You now know how an LLM actually works, end to end, with no black boxes. Go run the scripts. Edit them. Break them. Fix them. That's how the understanding becomes permanent.

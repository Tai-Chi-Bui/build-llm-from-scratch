# load_gpt2_weights.py
import os
import numpy as np
import torch
import torch.nn as nn
import urllib.request
import json
from gpt_model import GPTModel


# ── 1. Download GPT-2 weights from OpenAI ────────────────────────────────────
def download_gpt2_files(model_size, target_dir):
    """
    Download all GPT-2 checkpoint files for a given model size.

    model_size : "124M", "355M", "774M", or "1558M"
    target_dir : local directory to save files into

    OpenAI stored weights in TensorFlow checkpoint format.
    We download 7 files:
      - checkpoint         : index of checkpoint files
      - hparams.json       : architecture hyperparameters
      - encoder.json       : BPE token-to-id mapping
      - vocab.bpe          : BPE merge rules
      - model.ckpt.index   : tensor index
      - model.ckpt.meta    : graph metadata
      - model.ckpt.data-*  : actual weight tensors (the big one)
    """
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    os.makedirs(target_dir, exist_ok=True)

    for filename in filenames:
        url      = f"{base_url}/{model_size}/{filename}"
        dst_path = os.path.join(target_dir, filename)

        if os.path.exists(dst_path):
            print(f"  Already downloaded: {filename}")
            continue

        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dst_path)
            size_mb = os.path.getsize(dst_path) / (1024 ** 2)
            print(f"done ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")
            raise


# ── 2. Load weights from TensorFlow checkpoint into Python dicts ──────────────
def load_gpt2_params(checkpoint_dir, hparams):
    """
    Load GPT-2 weights from TF checkpoint into nested Python dicts.

    Returns:
      params : dict with keys "wte", "wpe", "blocks", "g", "b"
               matching GPT-2's internal naming
    """
    # TensorFlow is only needed here to read the checkpoint format
    import tensorflow as tf

    # Path to checkpoint (without file extension)
    ckpt_path = os.path.join(checkpoint_dir, "model.ckpt")

    # List all tensors in the checkpoint
    var_names = [
        name for name, _ in tf.train.list_variables(ckpt_path)
    ]

    # Load each tensor and store in a flat dict
    flat = {}
    for name in var_names:
        tensor = tf.train.load_variable(ckpt_path, name)
        flat[name] = np.squeeze(tensor)   # remove size-1 dimensions

    # ── Organise into nested structure ────────────────────────────────────────
    # GPT-2 uses names like "model/h0/attn/c_attn/w:0"
    # We parse these into params["blocks"][0]["attn"]["c_attn"]["w"]

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}

    for name, tensor in flat.items():
        # Strip "model/" prefix and ":0" suffix
        name = name.removeprefix("model/")

        # Top-level parameters
        if name in ("wte", "wpe"):          # embeddings
            params[name] = tensor
        elif name in ("g", "b"):            # final layer norm
            params[name] = tensor

        # Block parameters: "hN/..."
        elif name.startswith("h"):
            # Extract block index
            parts     = name.split("/")
            block_idx = int(parts[0][1:])     # "h0" → 0

            # Build nested dict path
            sub_dict = params["blocks"][block_idx]
            for key in parts[1:-1]:           # intermediate keys
                sub_dict = sub_dict.setdefault(key, {})
            sub_dict[parts[-1]] = tensor      # final key = tensor

    return params


# ── 3. Copy weights from params dict into our GPTModel ───────────────────────
def assign(our_param, gpt2_tensor):
    """
    Validate shapes match and return gpt2_tensor as a nn.Parameter.
    Raises a clear error if shapes don't match — catches mapping bugs early.
    """
    gpt2_tensor = np.array(gpt2_tensor)   # ensure numpy array

    if our_param.shape != gpt2_tensor.shape:
        raise ValueError(
            f"Shape mismatch!\n"
            f"  Our param : {our_param.shape}\n"
            f"  GPT-2     : {gpt2_tensor.shape}"
        )
    return nn.Parameter(
        torch.tensor(gpt2_tensor, dtype=torch.float32)
    )


def load_weights_into_gpt(model, params):
    """
    Copy all GPT-2 weights into our GPTModel.

    The main challenges:
    1. OpenAI stores Q, K, V combined → we need to split them
    2. OpenAI stores Linear weights transposed → we must transpose
    3. OpenAI uses different names → we map them manually
    """
    # ── Embedding layers ──────────────────────────────────────────────────────
    model.tok_emb.weight = assign(model.tok_emb.weight, params["wte"])
    model.pos_emb.weight = assign(model.pos_emb.weight, params["wpe"])

    # ── Transformer blocks ────────────────────────────────────────────────────
    for b_idx, block_params in enumerate(params["blocks"]):
        block = model.trf_blocks[b_idx]

        # ── Attention weights ─────────────────────────────────────────────────
        # OpenAI stores Q, K, V as one combined [768, 2304] matrix.
        # We split along last axis into three [768, 768] matrices.
        # Note: OpenAI convention is [in, out]; PyTorch Linear is [out, in]
        #       so we transpose.
        q_w, k_w, v_w = np.split(
            block_params["attn"]["c_attn"]["w"], 3, axis=-1
        )
        block.attention.W_query.weight = assign(
            block.attention.W_query.weight, q_w.T
        )
        block.attention.W_key.weight = assign(
            block.attention.W_key.weight, k_w.T
        )
        block.attention.W_value.weight = assign(
            block.attention.W_value.weight, v_w.T
        )

        # Q, K, V biases (GPT-2 uses biases in attention, we set qkv_bias=True)
        q_b, k_b, v_b = np.split(
            block_params["attn"]["c_attn"]["b"], 3, axis=-1
        )
        block.attention.W_query.bias = assign(
            block.attention.W_query.bias, q_b
        )
        block.attention.W_key.bias = assign(
            block.attention.W_key.bias, k_b
        )
        block.attention.W_value.bias = assign(
            block.attention.W_value.bias, v_b
        )

        # Output projection
        block.attention.out_proj.weight = assign(
            block.attention.out_proj.weight,
            block_params["attn"]["c_proj"]["w"].T
        )
        block.attention.out_proj.bias = assign(
            block.attention.out_proj.bias,
            block_params["attn"]["c_proj"]["b"]
        )

        # ── Feed-forward weights ──────────────────────────────────────────────
        # GPT-2 names: "mlp/c_fc"  (first linear)
        #              "mlp/c_proj" (second linear)
        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight,
            block_params["mlp"]["c_fc"]["w"].T
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias,
            block_params["mlp"]["c_fc"]["b"]
        )
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight,
            block_params["mlp"]["c_proj"]["w"].T
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias,
            block_params["mlp"]["c_proj"]["b"]
        )

        # ── Layer norm weights ────────────────────────────────────────────────
        # GPT-2 names: "ln_1" (before attention)
        #              "ln_2" (before FFN)
        # Our names:   norm1, norm2
        # "g" = gain = our "scale"
        # "b" = bias = our "shift"
        block.norm1.scale = assign(
            block.norm1.scale, block_params["ln_1"]["g"]
        )
        block.norm1.shift = assign(
            block.norm1.shift, block_params["ln_1"]["b"]
        )
        block.norm2.scale = assign(
            block.norm2.scale, block_params["ln_2"]["g"]
        )
        block.norm2.shift = assign(
            block.norm2.shift, block_params["ln_2"]["b"]
        )

    # ── Final layer norm ──────────────────────────────────────────────────────
    model.final_norm.scale = assign(model.final_norm.scale, params["g"])
    model.final_norm.shift = assign(model.final_norm.shift, params["b"])

    # ── Output head ───────────────────────────────────────────────────────────
    # Weight tying: out_head shares weights with tok_emb.
    # We already set tok_emb.weight above, so out_head is automatically correct.
    # No need to set it separately — they are the same tensor object.
    print("  Note: out_head shares weights with tok_emb (weight tying)")


# ── 4. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tiktoken
    from generate import generate, text_to_ids, ids_to_text

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    MODEL_SIZE   = "124M"
    DOWNLOAD_DIR = os.path.join("gpt2_weights", MODEL_SIZE)

    # ── GPT-2 config with qkv_bias=True ──────────────────────────────────────
    # IMPORTANT: GPT-2 uses bias vectors in attention projections.
    # Our training in Step 9 used qkv_bias=False.
    # When loading OpenAI weights we must match their convention.
    GPT_CONFIG_GPT2 = {
        "vocab_size"     : 50257,
        "context_length" : 1024,    # GPT-2 uses 1024, not 256
        "emb_dim"        : 768,
        "n_heads"        : 12,
        "n_layers"       : 12,
        "drop_rate"      : 0.0,     # disable dropout for inference
        "qkv_bias"       : True,    # GPT-2 uses biases in attention
    }

    # ── Step 1: Download ──────────────────────────────────────────────────────
    print("=" * 55)
    print("STEP 1: Downloading GPT-2 weights")
    print("=" * 55)
    print(f"\nModel size : {MODEL_SIZE}")
    print(f"Target dir : {DOWNLOAD_DIR}\n")

    download_gpt2_files(MODEL_SIZE, DOWNLOAD_DIR)
    print("\n✅ Download complete")

    # ── Step 2: Load hparams ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2: Loading architecture settings")
    print("=" * 55)

    with open(os.path.join(DOWNLOAD_DIR, "hparams.json")) as f:
        hparams = json.load(f)

    print(f"\nGPT-2 hparams: {hparams}")

    # ── Step 3: Load weight tensors ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3: Loading weight tensors from checkpoint")
    print("=" * 55)
    print("\nThis requires TensorFlow — installing if needed...")

    try:
        params = load_gpt2_params(DOWNLOAD_DIR, hparams)
        print(f"\n✅ Loaded params with keys: {list(params.keys())}")
        print(f"   Number of transformer blocks: {len(params['blocks'])}")
        print(f"   Token embedding shape       : {params['wte'].shape}")
        print(f"   Positional embedding shape  : {params['wpe'].shape}")
    except ImportError:
        print("\nTensorFlow not found. Installing...")
        os.system("pip install tensorflow --quiet")
        params = load_gpt2_params(DOWNLOAD_DIR, hparams)

    # ── Step 4: Build model and load weights ──────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4: Loading weights into GPTModel")
    print("=" * 55)

    model = GPTModel(GPT_CONFIG_GPT2)
    model.eval()

    print("\nCopying weights...")
    load_weights_into_gpt(model, params)
    model.to(device)
    print("✅ All weights loaded successfully")

    # ── Step 5: Verify with text generation ───────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5: Verification — text generation")
    print("=" * 55)

    prompts = [
        "Every effort moves you",
        "The capital of France is",
        "In machine learning, the most important",
        "def factorial(n):",
    ]

    torch.manual_seed(42)
    for prompt in prompts:
        ids = text_to_ids(prompt, tokenizer, device)
        out = generate(
            model, ids,
            max_new_tokens = 25,
            context_size   = GPT_CONFIG_GPT2["context_length"],
            temperature    = 0.7,
            top_k          = 40,
        )
        result = ids_to_text(out, tokenizer)
        print(f"\nPrompt : '{prompt}'")
        print(f"Output : '{result}'")

    # ── Step 6: Save in our format for fine-tuning ────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 6: Saving in our checkpoint format")
    print("=" * 55)

    torch.save({
        "model_state_dict" : model.state_dict(),
        "config"           : GPT_CONFIG_GPT2,
    }, "gpt2_pretrained.pth")

    size_mb = os.path.getsize("gpt2_pretrained.pth") / (1024 ** 2)
    print(f"\nSaved to gpt2_pretrained.pth ({size_mb:.1f} MB)")
    print("This file is used as the starting point for Steps 12 and 13")

    # ── Step 7: Parameter verification ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 7: Parameter count verification")
    print("=" * 55)

    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"\nUnique parameters: {unique_params:,}")
    print(f"Expected         : 124,412,160")
    assert unique_params == 124412160, "Parameter count mismatch!"
    print("✅ Matches GPT-2 small exactly")

    print("\n✅ Step 11 complete — GPT-2 weights loaded successfully!")
#!/usr/bin/env python
import os
import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from transformers import AutoConfig

# === Loader for LLaVA checkpoint using LLaVA's builder method ===
def load_llava_checkpoint(llava_checkpoint_path):
    print("🚀 Loading LLaVA-Video-7B-Qwen2 from:", llava_checkpoint_path)
    disable_torch_init()  # Disable redundant torch initializations
    model_name = get_model_name_from_path(llava_checkpoint_path)
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(llava_checkpoint_path)
    print("Loading model checkpoint...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=llava_checkpoint_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        overwrite_config={"mm_spatial_pool_stride": 4}
    )
    print("✅ LLaVA model loaded successfully!")
    return model.state_dict()

# === Loader for SPANN3R checkpoint ===
def load_spann3r_checkpoint(spann3r_checkpoint_path):
    print("🚀 Loading SPANN3R checkpoint from:", spann3r_checkpoint_path)
    checkpoint = torch.load(spann3r_checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
    print("✅ SPANN3R checkpoint loaded.")
    return checkpoint

# === Mapping function for encoder keys ===
def map_llava_to_spann3r(llava_key):
    """
    Maps a LLaVA encoder key to the corresponding SPANN3R encoder key.
    
    Assumptions:
      - LLaVA encoder keys start with:
            "model.vision_tower.vision_tower.vision_model.encoder.layers.{i}."
      - We subtract 12 from the LLaVA layer index so that layer 12 maps to SPANN3R block 0.
      - For self-attention:
            * The three projection keys ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
              are merged into a single key "attn.qkv" (to be concatenated later).
            * "self_attn.out_proj" is mapped to "attn.proj".
      - "layer_norm1" is renamed to "norm1" and "layer_norm2" to "norm2".
      - MLP keys remain unchanged.
      
    If the key does not match these rules, returns None.
    """
    base_prefix = "model.vision_tower.vision_tower.vision_model.encoder.layers."
    if not llava_key.startswith(base_prefix):
        return None

    remainder = llava_key[len(base_prefix):]  # e.g., "25.mlp.fc2.bias"
    parts = remainder.split(".")
    if len(parts) < 2:
        return None

    try:
        layer_index = int(parts[0])
    except ValueError:
        return None

    # Discard keys from layers below 12 (or any negative result)
    if layer_index < 12:
        return None

    # Compute target SPANN3R block index (subtract 12)
    target_index = layer_index - 12

    module = parts[1]
    rest = parts[2:]  # remaining components

    if module == "self_attn":
        if len(rest) < 1:
            return None
        proj = rest[0]
        if proj in ["q_proj", "k_proj", "v_proj"]:
            if rest[-1] == "weight":
                target = "attn.qkv.weight"
            elif rest[-1] == "bias":
                target = "attn.qkv.bias"
            else:
                return None
        elif proj == "out_proj":
            if rest[-1] == "weight":
                target = "attn.proj.weight"
            elif rest[-1] == "bias":
                target = "attn.proj.bias"
            else:
                return None
        else:
            return None
        return f"dust3r.enc_blocks.{target_index}.{target}"
    elif module.startswith("layer_norm"):
        if module == "layer_norm1":
            new_module = "norm1"
        elif module == "layer_norm2":
            new_module = "norm2"
        else:
            new_module = module
        return f"dust3r.enc_blocks.{target_index}.{new_module}." + ".".join(rest)
    elif module == "mlp":
        return f"dust3r.enc_blocks.{target_index}.mlp." + ".".join(rest)
    else:
        return f"dust3r.enc_blocks.{target_index}.{module}." + ".".join(rest)

# === Weight conversion function (truncation/padding) ===
def convert_weight(llava_tensor, target_shape):
    """
    Converts the LLaVA tensor to match the target shape by copying as many elements
    as possible. If the source tensor is larger along a dimension, it is truncated;
    if it is smaller, zeros are padded.
    """
    src_shape = llava_tensor.shape
    new_tensor = torch.zeros(target_shape, dtype=llava_tensor.dtype)
    # Create a slice for each dimension: copy min(src, target) elements.
    slices = tuple(slice(0, min(s, t)) for s, t in zip(src_shape, target_shape))
    new_tensor[slices] = llava_tensor[slices]
    return new_tensor

# === Update SPANN3R weights with LLaVA weights ===
def update_all_weights(llava_state, spann3r_state):
    """
    Iterates over all keys in LLaVA's encoder (those starting with the encoder prefix).
    For each key that maps to a key in the SPANN3R checkpoint (using our mapping function),
    updates the SPANN3R parameter with the LLaVA tensor. If shapes differ, converts the LLaVA
    tensor to match the target shape (using truncation/padding). Keys that have no valid mapping
    (or map to negative indices or to keys that do not exist in SPANN3R) are discarded.
    
    Returns the updated SPANN3R state dictionary and prints the total number of updated keys.
    """
    llava_prefix = "model.vision_tower.vision_tower.vision_model.encoder.layers."
    # Filter only encoder keys from LLaVA (weights and biases).
    llava_keys = [k for k in llava_state.keys() if k.startswith(llava_prefix) and ("weight" in k or "bias" in k)]
    
    updated_keys = 0
    for llava_key in sorted(llava_keys):
        mapped_key = map_llava_to_spann3r(llava_key)
        if mapped_key is None:
            # This includes LLaVA layers below 12, which we want to discard.
            # print(f"Discarding LLaVA key {llava_key} (no valid mapping).")
            continue
        if mapped_key not in spann3r_state:
            print(f"Mapped key {mapped_key} (from {llava_key}) not found in SPANN3R checkpoint. Discarding this key.")
            continue
        
        llava_tensor = llava_state[llava_key]
        target_tensor = spann3r_state[mapped_key]
        print(f"\nLLaVA key: {llava_key}")
        print(f"  → Mapped SPANN3R key: {mapped_key}")
        print(f"     LLaVA shape: {llava_tensor.shape}  |  SPANN3R shape: {target_tensor.shape}")
        
        if llava_tensor.shape == target_tensor.shape:
            print("     Shapes match. Using LLaVA tensor as-is.")
            spann3r_state[mapped_key] = llava_tensor
        else:
            print("     Shapes do not match. Converting LLaVA tensor to target shape (using truncation/padding).")
            converted = convert_weight(llava_tensor, target_tensor.shape)
            print(f"     Converted shape: {converted.shape}")
            spann3r_state[mapped_key] = converted
        updated_keys += 1

    print(f"\nTotal encoder parameters updated: {updated_keys}")
    return spann3r_state

# === Main function ===
def main():
    # Set your checkpoint paths.
    # For LLaVA, if the provided path is a directory, ensure it contains the checkpoint file.
    LLAVA_CHECKPOINT_PATH = "/data_new/spatial/huggingface/LLaVA-Video-7B-Qwen2"
    SPANN3R_CHECKPOINT_PATH = "/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/spann3r.pth"
    
    print("\n--- Loading LLaVA checkpoint ---")
    llava_state = load_llava_checkpoint(LLAVA_CHECKPOINT_PATH)
    
    print("\n--- Loading SPANN3R checkpoint ---")
    spann3r_state = load_spann3r_checkpoint(SPANN3R_CHECKPOINT_PATH)
    
    print("\n=== Updating SPANN3R encoder weights with LLaVA encoder weights ===")
    updated_spann3r_state = update_all_weights(llava_state, spann3r_state)
    
    # Save the updated SPANN3R checkpoint to a new file.
    updated_checkpoint_path = os.path.abspath("spann3r_updated_encoder.pth")
    try:
        torch.save({"model": updated_spann3r_state}, updated_checkpoint_path)
        print(f"\n✅ Updated SPANN3R checkpoint saved to: {updated_checkpoint_path}")
    except Exception as e:
        print(f"\n❌ Failed to save updated checkpoint: {e}")

if __name__ == "__main__":
    main()

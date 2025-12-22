"""
Helper functions for loading pretrained checkpoints from train_gpt_*.py

This module provides utilities to correctly load checkpoints saved by
the training scripts, handling architecture and vocabulary size mismatches.
"""

import torch
from model import GPTFlexAttention, GPTBatched


def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint and print its architecture parameters

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        dict with architecture info
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model' key")

    state_dict = checkpoint['model']

    # Remove _orig_mod prefix if present (from torch.compile)
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Detect architecture from state dict
    info = {}

    # Get vocab size from embed.weight
    if 'embed.weight' in state_dict:
        info['vocab_size'] = state_dict['embed.weight'].shape[0]
        info['model_dim'] = state_dict['embed.weight'].shape[1]
    else:
        raise ValueError("Could not find embed.weight in checkpoint")

    # Detect checkpoint format (small vs medium model)
    # Small model: qkv_w + c_proj.weight, c_fc.weight + c_proj.weight
    # Medium model: qkvo_w, fc_w + proj_w
    info['checkpoint_format'] = 'unknown'

    # Count layers and detect format
    num_layers = 0
    for i in range(100):  # Check up to 100 layers
        if f'blocks.{i}.mlp.fc_w' in state_dict:
            num_layers = i + 1
            if info['checkpoint_format'] == 'unknown':
                info['checkpoint_format'] = 'medium'
        elif f'blocks.{i}.mlp.c_fc.weight' in state_dict:
            num_layers = i + 1
            if info['checkpoint_format'] == 'unknown':
                info['checkpoint_format'] = 'small'
        else:
            break

    info['num_layers'] = num_layers

    # Get num_heads from first attention layer
    for i in range(num_layers):
        # Try medium format first
        if f'blocks.{i}.attn.qkvo_w' in state_dict:
            qkvo_w_shape = state_dict[f'blocks.{i}.attn.qkvo_w'].shape
            hdim = qkvo_w_shape[1]  # [4, hdim, dim]
            head_dim = 128  # Fixed in training script
            info['num_heads'] = hdim // head_dim
            break
        # Try small format
        elif f'blocks.{i}.attn.qkv_w' in state_dict:
            qkv_w_shape = state_dict[f'blocks.{i}.attn.qkv_w'].shape
            hdim = qkv_w_shape[1]  # [3, hdim, dim]
            head_dim = 128  # Fixed in training script
            info['num_heads'] = hdim // head_dim
            break

    # Get max_seq_len (we'll use default 1024 if not found)
    info['max_seq_len'] = 1024

    # Get scalars size
    if 'scalars' in state_dict:
        info['scalars_size'] = state_dict['scalars'].shape[0]

    print("\nCheckpoint Architecture:")
    print(f"  Format: {info['checkpoint_format']} (from train_gpt_{info['checkpoint_format']}.py)")
    print(f"  Vocab size: {info['vocab_size']}")
    print(f"  Model dim: {info['model_dim']}")
    print(f"  Num layers: {info['num_layers']}")
    print(f"  Num heads: {info['num_heads']}")
    print(f"  Max seq len: {info['max_seq_len']}")
    if 'scalars_size' in info:
        print(f"  Scalars size: {info['scalars_size']}")
    print()

    return info, checkpoint


def load_pretrained_checkpoint(checkpoint_path, device='cuda'):
    """Load a pretrained checkpoint and convert to batched model

    This function:
    1. Inspects the checkpoint to detect architecture and format
    2. Loads checkpoint weights appropriately based on format
    3. Converts to GPTBatched for fine-tuning

    Supports both checkpoint formats:
    - Small model (train_gpt_small.py): qkv_w + c_proj, c_fc + c_proj
    - Medium model (train_gpt_medium.py): qkvo_w, fc_w + proj_w

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        tuple: (batched_model, checkpoint_info, full_checkpoint)
    """
    # Inspect checkpoint and get architecture info
    info, checkpoint = inspect_checkpoint(checkpoint_path)

    state_dict = checkpoint['model']

    # Remove _orig_mod prefix if present
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    if info['checkpoint_format'] == 'small':
        # Small model format: need to load into GPTFlexAttention first, then convert
        print("Loading small model checkpoint (qkv_w format)...")
        pretrained_flex = GPTFlexAttention(
            vocab_size=info['vocab_size'],
            num_layers=info['num_layers'],
            num_heads=info['num_heads'],
            model_dim=info['model_dim'],
            max_seq_len=info['max_seq_len']
        )

        # Load weights (strict=False to handle scalars size mismatch)
        pretrained_flex.load_state_dict(state_dict, strict=False)
        print("✓ Loaded checkpoint weights successfully")

        # Convert to batched model
        print("Converting to batched model (enables batch_size > 1)...")
        batched_model = GPTBatched.from_pretrained_gpt(pretrained_flex)

    elif info['checkpoint_format'] == 'medium':
        # Medium model format: already uses qkvo_w, can load directly into GPTBatched
        print("Loading medium model checkpoint (qkvo_w format)...")
        batched_model = GPTBatched(
            vocab_size=info['vocab_size'],
            num_layers=info['num_layers'],
            num_heads=info['num_heads'],
            model_dim=info['model_dim'],
            max_seq_len=info['max_seq_len']
        )

        # Load weights (strict=False to handle scalars size mismatch)
        batched_model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded checkpoint weights directly (already in batched format)")

    else:
        raise ValueError(f"Unknown checkpoint format: {info['checkpoint_format']}")

    batched_model.to(device)

    print(f"✓ Successfully loaded checkpoint")
    print(f"✓ Model now supports batch_size > 1")

    return batched_model, info, checkpoint

#!/usr/bin/env python
"""Check pretrained model architecture from checkpoint"""

import torch
import sys

def check_model_architecture(checkpoint_path):
    """Load checkpoint and print architecture information"""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*60)
    print("CHECKPOINT CONTENTS")
    print("="*60)
    print(f"Keys: {list(ckpt.keys())}")
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    
    # Get embedding shape
    embed_shape = ckpt['model']['_orig_mod.embed.weight'].shape
    vocab_size, model_dim = embed_shape
    print(f"Embedding shape: {embed_shape}")
    print(f"  → Vocab size: {vocab_size}")
    print(f"  → Model dimension: {model_dim}")
    
    # Get LM head shape
    lm_head_shape = ckpt['model']['_orig_mod.lm_head_w'].shape
    print(f"\nLM head shape: {lm_head_shape}")
    
    # Count number of blocks (layers)
    num_blocks = sum(1 for k in ckpt['model'].keys() if 'blocks.' in k and 'mlp.fc_w' in k)
    print(f"\nNumber of layers: {num_blocks}")
    
    # Get attention head info
    qkvo_shape = ckpt['model']['_orig_mod.blocks.0.attn.qkvo_w'].shape
    head_dim = 128  # Standard head dimension
    num_heads = qkvo_shape[1] // head_dim // 4
    print(f"\nQKVO weight shape: {qkvo_shape}")
    print(f"  → Number of heads: {num_heads}")
    print(f"  → Head dimension: {head_dim}")
    
    print("\n" + "="*60)
    print("FINETUNE.PY ARGUMENTS")
    print("="*60)
    print(f"--vocab_size {vocab_size}")
    print(f"--n_embd {model_dim}")
    print(f"--n_layer {num_blocks}")
    print(f"--n_head {num_heads}")
    print(f"--block_size 1024  # (verify this matches your training)")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    check_model_architecture(checkpoint_path)
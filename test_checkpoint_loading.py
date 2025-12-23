#!/usr/bin/env python3
"""
Test script to verify checkpoint loading works correctly

Usage:
    python test_checkpoint_loading.py /path/to/checkpoint.pt
"""

import sys
import torch
from load_checkpoint_helper import load_pretrained_checkpoint, inspect_checkpoint


def test_checkpoint_loading(checkpoint_path):
    """Test loading a checkpoint"""
    print("="*80)
    print("CHECKPOINT LOADING TEST")
    print("="*80)

    # Step 1: Inspect checkpoint
    print("\n[1/3] Inspecting checkpoint...")
    try:
        info, checkpoint = inspect_checkpoint(checkpoint_path)
        print("✓ Checkpoint inspection successful")
    except Exception as e:
        print(f"✗ Failed to inspect checkpoint: {e}")
        return False

    # Step 2: Load and convert to batched model
    print("\n[2/3] Loading and converting to batched model...")
    try:
        batched_model, info, checkpoint = load_pretrained_checkpoint(checkpoint_path, device='cpu')
        print("✓ Checkpoint loaded and converted successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Verify model parameters and structure
    print("\n[3/3] Verifying model parameters...")
    try:
        # Count parameters
        total_params = sum(p.numel() for p in batched_model.parameters())
        trainable_params = sum(p.numel() for p in batched_model.parameters() if p.requires_grad)

        print(f"✓ Model structure verified:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Verify key components exist
        assert hasattr(batched_model, 'embed'), "Missing embed layer"
        assert hasattr(batched_model, 'value_embeds'), "Missing value_embeds"
        assert hasattr(batched_model, 'blocks'), "Missing blocks"
        assert hasattr(batched_model, 'scalars'), "Missing scalars"

        print(f"  Embedding dim: {batched_model.embed.embedding_dim}")
        print(f"  Vocab size: {batched_model.embed.num_embeddings}")
        print(f"  Num layers: {len(batched_model.blocks)}")
        print(f"  Scalars: {batched_model.scalars.shape}")

        print(f"\n✓ Model ready for fine-tuning!")
        print(f"  Note: Forward pass testing skipped (requires CUDA for bfloat16)")

    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nThe checkpoint can now be used for fine-tuning.")
    print(f"Use these architecture parameters in your fine-tuning script:")
    print(f"  --n_layer {info['num_layers']}")
    print(f"  --n_head {info['num_heads']}")
    print(f"  --n_embd {info['model_dim']}")
    print(f"  --block_size {info['max_seq_len']}")
    print(f"\nNote: vocab_size ({info['vocab_size']}) is auto-detected from checkpoint")

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_checkpoint_loading.py /path/to/checkpoint.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    success = test_checkpoint_loading(checkpoint_path)
    sys.exit(0 if success else 1)

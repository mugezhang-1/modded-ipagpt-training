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

    # Step 3: Verify model can run a forward pass
    print("\n[3/3] Testing forward pass...")
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 128
        dummy_input = torch.randint(0, info['vocab_size'], (batch_size, seq_len))

        # Run forward pass
        with torch.no_grad():
            features = batched_model.forward_features(dummy_input)

        expected_shape = (batch_size, seq_len, info['model_dim'])
        if features.shape == expected_shape:
            print(f"✓ Forward pass successful! Output shape: {features.shape}")
        else:
            print(f"✗ Unexpected output shape: {features.shape}, expected {expected_shape}")
            return False

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
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

#!/bin/bash
#SBATCH --job-name=test-checkpoint-loading
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=test-checkpoint-loading-%j.out
#SBATCH --error=test-checkpoint-loading-%j.err

set -euo pipefail

echo "=========================================="
echo "Checkpoint Loading GPU Test"
echo "Start: $(date)"
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "=========================================="

# Activate environment
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ipagpt_training

cd /fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Test small model checkpoint
SMALL_CKPT="/fs/scratch/PAS2836/mugezhang/clean_bbpe_50k/models/small_hin_urd_owt/000_f9c9e3a7-d7f0-48bb-8e93-ad29935ae17c/best_state_step022550_val1.2727.pt"
MEDIUM_CKPT="/fs/scratch/PAS2836/mugezhang/clean_bbpe_50k/models/medium_hin_urd_owt/000_69b1e951-6daa-4397-bce7-3b0f209ab5c4/best_state_step101500_val0.9218.pt"

echo "=========================================="
echo "Test 1: Small Model Checkpoint"
echo "=========================================="
python3 - << 'EOF_SMALL'
import torch
from load_checkpoint_helper import load_pretrained_checkpoint
import sys

checkpoint_path = "/fs/scratch/PAS2836/mugezhang/clean_bbpe_50k/models/small_hin_urd_owt/000_f9c9e3a7-d7f0-48bb-8e93-ad29935ae17c/best_state_step022550_val1.2727.pt"

print("\n[1/4] Loading small model checkpoint...")
try:
    model, info, checkpoint = load_pretrained_checkpoint(checkpoint_path, device='cuda')
    print(f"✓ Successfully loaded small model")
    print(f"  - Format: {info['checkpoint_format']}")
    print(f"  - Vocab size: {info['vocab_size']}")
    print(f"  - Model dim: {info['model_dim']}")
    print(f"  - Num layers: {info['num_layers']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/4] Testing forward pass...")
try:
    model.eval()
    batch_size = 4
    seq_len = 256
    dummy_input = torch.randint(0, info['vocab_size'], (batch_size, seq_len), device='cuda')

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            features = model.forward_features(dummy_input)

    expected_shape = (batch_size, seq_len, info['model_dim'])
    assert features.shape == expected_shape, f"Shape mismatch: {features.shape} vs {expected_shape}"
    print(f"✓ Forward pass successful!")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {features.shape}")
    print(f"  - Output dtype: {features.dtype}")
    print(f"  - Output device: {features.device}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/4] Testing batch processing (different sizes)...")
try:
    model.eval()
    test_cases = [
        (1, 128),   # Single example
        (8, 512),   # Medium batch
        (16, 256),  # Larger batch
    ]

    for bs, sl in test_cases:
        dummy_input = torch.randint(0, info['vocab_size'], (bs, sl), device='cuda')
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                features = model.forward_features(dummy_input)
        print(f"  ✓ Batch size {bs}, seq len {sl}: {features.shape}")

    print("✓ All batch sizes work correctly!")
except Exception as e:
    print(f"✗ Batch processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] Memory usage...")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print("\n" + "="*60)
print("SMALL MODEL TEST PASSED ✓")
print("="*60)
EOF_SMALL

if [ $? -ne 0 ]; then
    echo "Small model test failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test 2: Medium Model Checkpoint"
echo "=========================================="
python3 - << 'EOF_MEDIUM'
import torch
from load_checkpoint_helper import load_pretrained_checkpoint
import sys

# Clear cache from previous test
torch.cuda.empty_cache()

checkpoint_path = "/fs/scratch/PAS2836/mugezhang/clean_bbpe_50k/models/medium_hin_urd_owt/000_69b1e951-6daa-4397-bce7-3b0f209ab5c4/best_state_step101500_val0.9218.pt"

print("\n[1/4] Loading medium model checkpoint...")
try:
    model, info, checkpoint = load_pretrained_checkpoint(checkpoint_path, device='cuda')
    print(f"✓ Successfully loaded medium model")
    print(f"  - Format: {info['checkpoint_format']}")
    print(f"  - Vocab size: {info['vocab_size']}")
    print(f"  - Model dim: {info['model_dim']}")
    print(f"  - Num layers: {info['num_layers']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/4] Testing forward pass...")
try:
    model.eval()
    batch_size = 4
    seq_len = 256
    dummy_input = torch.randint(0, info['vocab_size'], (batch_size, seq_len), device='cuda')

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            features = model.forward_features(dummy_input)

    expected_shape = (batch_size, seq_len, info['model_dim'])
    assert features.shape == expected_shape, f"Shape mismatch: {features.shape} vs {expected_shape}"
    print(f"✓ Forward pass successful!")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {features.shape}")
    print(f"  - Output dtype: {features.dtype}")
    print(f"  - Output device: {features.device}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/4] Testing batch processing (different sizes)...")
try:
    model.eval()
    test_cases = [
        (1, 128),   # Single example
        (8, 512),   # Medium batch
        (16, 256),  # Larger batch
    ]

    for bs, sl in test_cases:
        dummy_input = torch.randint(0, info['vocab_size'], (bs, sl), device='cuda')
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                features = model.forward_features(dummy_input)
        print(f"  ✓ Batch size {bs}, seq len {sl}: {features.shape}")

    print("✓ All batch sizes work correctly!")
except Exception as e:
    print(f"✗ Batch processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] Memory usage...")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print("\n" + "="*60)
print("MEDIUM MODEL TEST PASSED ✓")
print("="*60)
EOF_MEDIUM

if [ $? -ne 0 ]; then
    echo "Medium model test failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test 3: Fine-tuning Integration Test"
echo "=========================================="
python3 - << 'EOF_FINETUNE'
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_checkpoint_helper import load_pretrained_checkpoint
import sys

# Clear cache
torch.cuda.empty_cache()

checkpoint_path = "/fs/scratch/PAS2836/mugezhang/clean_bbpe_50k/models/small_hin_urd_owt/000_f9c9e3a7-d7f0-48bb-8e93-ad29935ae17c/best_state_step022550_val1.2727.pt"

print("\n[1/3] Creating classification wrapper...")
try:
    # Load pretrained model
    pretrained_model, info, checkpoint = load_pretrained_checkpoint(checkpoint_path, device='cuda')

    # Create simple classification head
    num_classes = 3
    classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(info['model_dim'], info['model_dim'] // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(info['model_dim'] // 2, num_classes)
    ).to('cuda')

    print(f"✓ Created classification wrapper")
    print(f"  - Pretrained model: {sum(p.numel() for p in pretrained_model.parameters()):,} params")
    print(f"  - Classifier head: {sum(p.numel() for p in classifier.parameters()):,} params")
except Exception as e:
    print(f"✗ Failed to create wrapper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/3] Testing forward pass with classification...")
try:
    pretrained_model.eval()
    classifier.train()

    batch_size = 8
    seq_len = 256
    dummy_input = torch.randint(0, info['vocab_size'], (batch_size, seq_len), device='cuda')
    dummy_labels = torch.randint(0, num_classes, (batch_size,), device='cuda')

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Extract features
        features = pretrained_model.forward_features(dummy_input)

        # Pool features (use last token)
        pooled = features[:, -1, :]

        # Classify
        logits = classifier(pooled.float())
        loss = F.cross_entropy(logits, dummy_labels)

    print(f"✓ Forward pass with classification successful!")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Pooled shape: {pooled.shape}")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Classification forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/3] Testing backward pass (gradient computation)...")
try:
    # Create optimizer
    optimizer = torch.optim.AdamW([
        {'params': pretrained_model.parameters(), 'lr': 1e-5},
        {'params': classifier.parameters(), 'lr': 1e-4}
    ])

    pretrained_model.train()
    classifier.train()

    # Forward pass
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        features = pretrained_model.forward_features(dummy_input)
        pooled = features[:, -1, :]
        logits = classifier(pooled.float())
        loss = F.cross_entropy(logits, dummy_labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_pretrained_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in pretrained_model.parameters())
    has_classifier_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in classifier.parameters())

    print(f"✓ Backward pass successful!")
    print(f"  - Pretrained model has gradients: {has_pretrained_grads}")
    print(f"  - Classifier has gradients: {has_classifier_grads}")
    print(f"  - Loss: {loss.item():.4f}")

    if not has_pretrained_grads or not has_classifier_grads:
        print("✗ Warning: Some parameters don't have gradients!")
        sys.exit(1)

except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("FINE-TUNING INTEGRATION TEST PASSED ✓")
print("="*60)
EOF_FINETUNE

if [ $? -ne 0 ]; then
    echo "Fine-tuning integration test failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All Tests Completed Successfully!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Small model checkpoint loads and runs on GPU"
echo "  ✓ Medium model checkpoint loads and runs on GPU"
echo "  ✓ Forward pass works with bfloat16 autocast"
echo "  ✓ Batch processing works for various sizes"
echo "  ✓ Fine-tuning integration (forward + backward) works"
echo ""
echo "The checkpoint loading system is ready for production use!"

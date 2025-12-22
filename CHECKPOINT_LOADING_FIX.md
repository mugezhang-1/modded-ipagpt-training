# Checkpoint Loading Fix for Fine-tuning

## Problem Summary

The fine-tuning script was failing to load pretrained checkpoints because of architecture mismatches between the training scripts and the `GPTFlexAttention` class in `model.py`.

### Two Different Checkpoint Formats

Your training scripts use **two different architectures**:

**train_gpt_small.py format:**
- Attention: `qkv_w` [3, hdim, dim] + `c_proj.weight` [dim, hdim]
- MLP: `c_fc.weight` [hdim, dim] + `c_proj.weight` [dim, hdim]

**train_gpt_medium.py format:**
- Attention: `qkvo_w` [4, hdim, dim] (Q, K, V, O combined)
- MLP: `fc_w` [hdim, dim] + `proj_w` [dim, hdim]

### Additional Issues:
1. **Vocabulary size**: Checkpoints have vocab_size padded to multiple of 128 (e.g., 50048), but script used original size (e.g., 50000)
2. **Scalars size**: Checkpoints may have extra padding from distributed training

## Solution

I've updated the code to handle **both checkpoint formats automatically**. The key changes:

1. **Updated `GPTFlexAttention` class** (in `model.py`) to match small model checkpoint format
2. **Updated `GPTBatched.from_pretrained_gpt`** to convert small model weights to batched format
3. **Created `load_checkpoint_helper.py`** with utilities that:
   - Auto-detect checkpoint format (small vs medium)
   - Auto-detect vocabulary size and architecture
   - Load small model checkpoints with conversion
   - Load medium model checkpoints directly (no conversion needed!)
   - Handle scalars padding mismatches

## How to Fix Your Fine-tuning Script

### Option 1: Use the Helper Function (Recommended)

Replace the `load_pretrained_model` function in your fine-tuning script with this:

```python
from load_checkpoint_helper import load_pretrained_checkpoint

def load_pretrained_model(args, model):
    """Load pretrained model from training script checkpoint"""
    print(f"Loading pretrained model from {args.pretrained_ckpt_path}")

    # Use helper function to automatically detect architecture and load
    pretrained_batched, ckpt_info, checkpoint = load_pretrained_checkpoint(
        args.pretrained_ckpt_path,
        device=args.device
    )

    # Set the pretrained model
    model.pretrained_model = pretrained_batched
    model.to(args.device)

    print(f"✓ Loaded checkpoint with vocab_size={ckpt_info['vocab_size']}")
    print(f"✓ Batch size: {model.batch_size} (was 1 in FlexAttention)")

    # Create optimizer and scaler
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == 'float16'))
    optimizer = configure_optimizers(device_type, model)

    return model, optimizer, scaler
```

### Option 2: Manual Fix

If you prefer to keep the existing structure, update these lines:

**OLD CODE:**
```python
pretrained_flex = GPTFlexAttention(
    vocab_size=args.vocab_size,  # ❌ This causes size mismatch
    num_layers=args.n_layer,
    num_heads=args.n_head,
    model_dim=args.n_embd,
    max_seq_len=args.block_size
)

state_dict = checkpoint['model']
# ... unwrap _orig_mod prefix ...

pretrained_flex.load_state_dict(state_dict, strict=True)  # ❌ This will fail
```

**NEW CODE:**
```python
# First, detect vocab_size from checkpoint
state_dict = checkpoint['model']

# Remove _orig_mod prefix
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# Detect vocab_size from checkpoint
checkpoint_vocab_size = state_dict['embed.weight'].shape[0]
print(f"Detected vocab_size from checkpoint: {checkpoint_vocab_size}")

# Create model with correct vocab_size
pretrained_flex = GPTFlexAttention(
    vocab_size=checkpoint_vocab_size,  # ✓ Use detected size
    num_layers=args.n_layer,
    num_heads=args.n_head,
    model_dim=args.n_embd,
    max_seq_len=args.block_size
)

# Load with strict=False to handle scalars padding mismatch
pretrained_flex.load_state_dict(state_dict, strict=False)  # ✓ This will work
```

## Updated Command Line Arguments

Since vocab_size is now auto-detected from the checkpoint, you can remove `--vocab_size` from your command:

**OLD:**
```bash
python finetune_hindi_xnli.py \
    --dataset hindi_xnli \
    --pretrained_ckpt_path /path/to/checkpoint.pt \
    --vocab_size 50000 \  # ❌ Remove this
    --n_layer 12 \
    --n_head 6 \
    --n_embd 768
```

**NEW:**
```bash
python finetune_hindi_xnli.py \
    --dataset hindi_xnli \
    --pretrained_ckpt_path /path/to/checkpoint.pt \
    --n_layer 12 \
    --n_head 6 \
    --n_embd 768
    # vocab_size is auto-detected! ✓
```

## Testing the Fix

You can test if a checkpoint loads correctly with:

```python
from load_checkpoint_helper import inspect_checkpoint

info, checkpoint = inspect_checkpoint('/path/to/your/checkpoint.pt')
```

This will print the detected architecture parameters.

## Example: Loading a Hindi-Urdu Checkpoint

```bash
python finetune_hindi_xnli.py \
    --dataset hindi_xnli \
    --pretrained_ckpt_path /fs/scratch/PAS2836/mugezhang/clean_bbpe_50k/hin_urd_owt_run_*/best_state_*.pt \
    --tokenizer_name /fs/ess/PAS2836/mugezhang/code/modded-ipagpt-training/03train_tokenizer/tokenizers/hin_urd_owt/bpe-hin-urd-owt-tokenizer.json \
    --n_layer 12 \
    --n_head 6 \
    --n_embd 768 \
    --batch_size 16 \
    --num_epochs 3
```

## Summary of Changes Made to `model.py`

1. Added helper classes: `CastedLinear`, `TrainingScriptAttention`, `TrainingScriptMLP`, `TrainingScriptBlock`
2. Updated `GPTFlexAttention` to match small model checkpoint format
3. Updated `GPTBatched.from_pretrained_gpt` to properly convert weights from small model format
4. Added handling for scalars size mismatch (from distributed training padding)

All changes are backward compatible with existing code.

## Why Two Different Formats?

The **medium model already matches `GPTBatched` architecture**:
- Medium model uses `qkvo_w` (4D tensor combining Q, K, V, O projections)
- Medium model uses `fc_w` and `proj_w` (direct parameters, not layers)
- This is the same structure as `GPTBatched`!

The **small model uses a different format**:
- Small model uses separate `qkv_w` and `c_proj.weight`
- Small model uses `c_fc` and `c_proj` as layers (with `.weight` attributes)
- Needs conversion to work with `GPTBatched`

The loading helper automatically detects which format your checkpoint uses and handles it appropriately. You don't need to know which format you're using - it just works!

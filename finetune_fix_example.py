"""
QUICK FIX for finetune_hindi_xnli.py

Replace the load_pretrained_model function with this version:
"""

import torch
from load_checkpoint_helper import load_pretrained_checkpoint


def load_pretrained_model(args, model):
    """Load pretrained model from training script checkpoint and convert to batched version

    This version automatically detects the correct architecture from the checkpoint.
    """
    print(f"Loading pretrained model from {args.pretrained_ckpt_path}")

    # Use helper function to load checkpoint with auto-detected architecture
    pretrained_batched, ckpt_info, checkpoint = load_pretrained_checkpoint(
        str(args.pretrained_ckpt_path),
        device=args.device
    )

    # Verify architecture matches command-line args
    if ckpt_info['model_dim'] != args.n_embd:
        print(f"WARNING: Checkpoint has model_dim={ckpt_info['model_dim']} but args has n_embd={args.n_embd}")
        print(f"Using checkpoint value: {ckpt_info['model_dim']}")

    if ckpt_info['num_layers'] != args.n_layer:
        print(f"WARNING: Checkpoint has num_layers={ckpt_info['num_layers']} but args has n_layer={args.n_layer}")
        print(f"Using checkpoint value: {ckpt_info['num_layers']}")

    if ckpt_info['num_heads'] != args.n_head:
        print(f"WARNING: Checkpoint has num_heads={ckpt_info['num_heads']} but args has n_head={args.n_head}")
        print(f"Using checkpoint value: {ckpt_info['num_heads']}")

    # Set the pretrained model in the classification wrapper
    model.pretrained_model = pretrained_batched
    model.to(args.device)

    print(f"\n{'='*80}")
    print("Successfully loaded pretrained checkpoint!")
    print(f"  Vocab size: {ckpt_info['vocab_size']}")
    print(f"  Model dim: {ckpt_info['model_dim']}")
    print(f"  Num layers: {ckpt_info['num_layers']}")
    print(f"  Num heads: {ckpt_info['num_heads']}")
    print(f"  Batch size: {model.batch_size} (was 1 in FlexAttention)")
    print(f"{'='*80}\n")

    # Create optimizer and gradient scaler
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == 'float16'))

    # Import configure_optimizers from original script
    from finetune_hindi_xnli import configure_optimizers
    optimizer = configure_optimizers(device_type, model)

    return model, optimizer, scaler


# ============================================================================
# Alternative: If you want to keep the original structure, use this minimal fix
# ============================================================================

def load_pretrained_model_minimal_fix(args, model):
    """Minimal fix version - detects vocab_size from checkpoint"""
    print(f"Loading pretrained model from {args.pretrained_ckpt_path}")
    checkpoint = torch.load(args.pretrained_ckpt_path, map_location=args.device, weights_only=False)

    if 'model' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model' key")

    state_dict = checkpoint['model']

    # Remove _orig_mod prefix from torch.compile
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # ✓ FIX 1: Auto-detect vocab_size from checkpoint
    checkpoint_vocab_size = state_dict['embed.weight'].shape[0]
    print(f"Detected vocab_size from checkpoint: {checkpoint_vocab_size}")

    # ✓ FIX 2: Use detected vocab_size (not args.vocab_size)
    from model import GPTFlexAttention, GPTBatched
    pretrained_flex = GPTFlexAttention(
        vocab_size=checkpoint_vocab_size,  # ← Changed from args.vocab_size
        num_layers=args.n_layer,
        num_heads=args.n_head,
        model_dim=args.n_embd,
        max_seq_len=args.block_size
    )

    # ✓ FIX 3: Use strict=False to handle scalars size mismatch
    pretrained_flex.load_state_dict(state_dict, strict=False)  # ← Changed from strict=True

    print("Converting to batched model (enables batch_size > 1)...")
    pretrained_batched = GPTBatched.from_pretrained_gpt(pretrained_flex)
    pretrained_batched.to(args.device)

    model.pretrained_model = pretrained_batched
    model.to(args.device)

    print(f"✓ Loaded checkpoint with vocab_size={checkpoint_vocab_size}")
    print(f"✓ Batch size: {model.batch_size} (was 1 in FlexAttention)")

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == 'float16'))

    from finetune_hindi_xnli import configure_optimizers
    optimizer = configure_optimizers(device_type, model)

    return model, optimizer, scaler

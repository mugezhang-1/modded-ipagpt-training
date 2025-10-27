import argparse
import os
import pathlib
import time
from contextlib import nullcontext

from datasets import load_dataset, load_from_disk
import inspect
import numpy as np
import torch
import wandb

from model import GPTBatched, GPTClassification, GPTFlexAttention

def configure_device(args):
    """Configure device, random seed, and mixed precision context"""
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception('CUDA not available')
    
    args.out_dir = pathlib.Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return ctx 

def configure_optimizers(device_type, model): 
    """Configure optimizer with differential learning rates for pretrained vs classifier"""
    # Separate parameters into pretrained model and classification head
    pretrained_params = list(model.pretrained_model.parameters())
    pretrained_param_ids = set(id(p) for p in pretrained_params)
    
    classifier_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]
    
    pretrained_param_dict = {f"pretrained.{i}": p for i, p in enumerate(pretrained_params) if p.requires_grad}
    classifier_param_dict = {f"classifier.{i}": p for i, p in enumerate(classifier_params) if p.requires_grad}
    
    # Apply 2D weight decay rule
    pretrained_decay_params = [p for n, p in pretrained_param_dict.items() if p.dim() >= 2]
    pretrained_nodecay_params = [p for n, p in pretrained_param_dict.items() if p.dim() < 2]
    
    classifier_decay_params = [p for n, p in classifier_param_dict.items() if p.dim() >= 2]
    classifier_nodecay_params = [p for n, p in classifier_param_dict.items() if p.dim() < 2]
    
    # Create optimizer groups with different learning rates
    # Pretrained model uses 10% of base learning rate
    optim_groups = [
        {'params': pretrained_decay_params, 'weight_decay': model.weight_decay, 'lr': model.learning_rate * 0.1},
        {'params': pretrained_nodecay_params, 'weight_decay': 0.0, 'lr': model.learning_rate * 0.1},
        {'params': classifier_decay_params, 'weight_decay': model.weight_decay},
        {'params': classifier_nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Log parameter counts
    print(f"Pretrained decay params: {len(pretrained_decay_params)}, with {sum(p.numel() for p in pretrained_decay_params):,} parameters")
    print(f"Pretrained no-decay params: {len(pretrained_nodecay_params)}, with {sum(p.numel() for p in pretrained_nodecay_params):,} parameters")
    print(f"Classifier decay params: {len(classifier_decay_params)}, with {sum(p.numel() for p in classifier_decay_params):,} parameters")
    print(f"Classifier no-decay params: {len(classifier_nodecay_params)}, with {sum(p.numel() for p in classifier_nodecay_params):,} parameters")
    
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=model.learning_rate, betas=(model.beta1, model.beta2), **extra_args)
    print(f"using fused AdamW: {use_fused}")
    
    return optimizer

# ===== Data Preparation Functions =====
def prepare_datasets(args, model):
    """Load and prepare datasets for training"""
    if args.no_subset:
        if args.from_disk:
            dataset = load_from_disk(args.parent_dataset)
        else:
            dataset = load_dataset(args.parent_dataset)
    else:
        if args.from_disk:
            raise Exception('can\'t load from disk with subset.')
        dataset = load_dataset(args.parent_dataset, args.dataset, cache_dir=str(args.hf_cache))
        
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    model.prepare_if_needed(train_dataset, validation_dataset, args.force_tokenization)

def load_pretrained_model(args, model):
    """Load pretrained model from training script checkpoint and convert to batched version"""
    print(f"Loading pretrained model from {args.pretrained_ckpt_path}")
    checkpoint = torch.load(args.pretrained_ckpt_path, map_location=args.device, weights_only=False)
    
    # The training script saves checkpoints as:
    # dict(step=step, code=code, model=model.state_dict(), optimizers=[...])
    if 'model' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model' key. Make sure this is a checkpoint from the training script.")
    
    # Load into temporary FlexAttention model structure
    print("Loading FlexAttention checkpoint...")
    pretrained_flex = GPTFlexAttention(
        vocab_size=args.vocab_size,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        model_dim=args.n_embd,
        max_seq_len=args.block_size
    )
    
    # Load the state dict
    state_dict = checkpoint['model']
    
    # Clean up any '_orig_mod.' prefix (from torch.compile)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Load weights into FlexAttention model
    pretrained_flex.load_state_dict(state_dict, strict=True)
    
    # Convert to batched version (this copies all weights)
    print("Converting to batched model (enables batch_size > 1)...")
    pretrained_batched = GPTBatched.from_pretrained_gpt(pretrained_flex)
    pretrained_batched.to(args.device)
    
    # Attach pretrained model to classification wrapper
    model.pretrained_model = pretrained_batched
    model.to(args.device)
    
    print(f"converted FlexAttention model to batched version")
    print(f"Batch size: {model.batch_size} (was 1 in FlexAttention)")
    
    # Set up scaler and optimizer
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == 'float16'))
    optimizer = configure_optimizers(device_type, model)

    return model, optimizer, scaler 

# ===== Training Functions =====
def get_lr(model, it):
    """Learning rate schedule: linear warmup then linear decay"""
    if it < model.warmup_iters:
        return model.learning_rate * (it + 1) / (model.warmup_iters + 1)
    if it > model.lr_decay_iters:
        return model.min_lr
    decay_ratio = (it - model.warmup_iters) / (model.lr_decay_iters - model.warmup_iters)
    assert 0 <= decay_ratio <= 1
    return model.learning_rate - decay_ratio * (model.learning_rate - model.min_lr)

def get_max_iters(model, gradient_accumulation_steps, num_epochs):
    """Calculate number of iterations required for training"""
    tokens_per_iter = gradient_accumulation_steps * model.batch_size * model.context_window
    token_count = model.get_token_count()
    max_iters = int(np.ceil(token_count / tokens_per_iter)) * num_epochs
    model.warmup_iters = int(model.warmup_iter_ratio * max_iters)
    model.lr_decay_iters = int(model.lr_decay_iter_ratio * max_iters)
    return max_iters    

def finetune(model, max_iters, scaler, optimizer, ctx, best_val_loss, best_checkpoint_path, args):
    """Main training loop"""
    if args.wandb_log:
        config = {
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "beta1": model.beta1,
            "beta2": model.beta2,
            "grad_clip": model.grad_clip,
            "dropout": model.dropout_rate,
            "warmup_iter_ratio": model.warmup_iter_ratio,
            "lr_decay_iter_ratio": model.lr_decay_iter_ratio,
            "min_lr": model.min_lr,
            "num_epochs": args.num_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "batch_size": model.batch_size,
        }
        wandb.init(project=args.wandb_project, name=f"{args.dataset}-{time.time()}", config=config)

    get_batch = model.get_batch
    X, Y = get_batch('train') 
    iter_num = 0

    # Ensure all parameters can learn
    for param in model.pretrained_model.parameters():
        param.requires_grad = True 

    while True:
        # Learning rate scheduling
        lr = get_lr(model, iter_num) if not args.no_decay_lr else model.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation only mode
        if iter_num == 0 and args.eval_only:
            break

        # Gradient accumulation loop
        for micro_step in range(args.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        # Gradient clipping
        if model.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Evaluation
        if iter_num % args.eval_interval == 0:
            losses = model.estimate_loss(ctx, args.eval_iters)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"val f2 {losses['val_f2']:.4f}, val accuracy {losses['val_accuracy']:.4f}")
            
            # Save best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"New best validation loss: {best_val_loss:.4f}, saving checkpoint to {best_checkpoint_path}")    
        
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'val_loss': best_val_loss,
                    'val_accuracy': losses['val_accuracy'],
                    'val_f2': losses['val_f2'],
                    'args': vars(args)
                }
                torch.save(checkpoint, best_checkpoint_path)

            if args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "train/f2": losses['train_f2'],
                    "val/f2": losses['val_f2'],
                    "lr": lr,
                    "train/accuracy": losses['train_accuracy'],
                    "val/accuracy": losses['val_accuracy'],
                    "best_val_loss": best_val_loss
                })

        iter_num += 1

        # Termination
        if iter_num > max_iters:
            break

def main(): 
    parser = argparse.ArgumentParser(description="Finetune a model on a classification task with batched attention")

    # Required
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., sst2)')
    parser.add_argument('--pretrained_ckpt_path', type=pathlib.Path, required=True, 
                        help='Path to pretrained model checkpoint from training script')

    # Dataset settings
    parser.add_argument('--parent_dataset', type=str, default='nyu-mll/glue')
    parser.add_argument('--no_subset', action='store_true')
    parser.add_argument('--from_disk', action='store_true')
    parser.add_argument('--use_ipa', action='store_true')
    parser.add_argument('--force_tokenization', action='store_true')
    parser.add_argument('--text_column', type=str, nargs='+', default=['sentence'])
    parser.add_argument('--label_column', type=str, default='label',
                        help='Name of the label column in the dataset')

    # Paths
    parser.add_argument('--hf_cache', type=pathlib.Path, default=pathlib.Path('./cache'))
    parser.add_argument('--out_dir', type=pathlib.Path, default=pathlib.Path('./checkpoints'))
    parser.add_argument('--tokenizer_dir', type=pathlib.Path, default=pathlib.Path('./tokenizers'))
    parser.add_argument('--data_dir', type=pathlib.Path, default=pathlib.Path('./datasets'))
    parser.add_argument('--tokenizer_name', type=str, default='gpt2')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs (default: 1, reduced from 10 for efficiency)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16, was 1 with FlexAttention)')
    parser.add_argument('--eval_iters', type=int, default=10,
                        help='Number of iterations for evaluation (default: 10, reduced from 40)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_decay_lr', action='store_true')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for classification (default: 2 for binary)')

    # Model architecture (must match training script)
    parser.add_argument('--n_layer', type=int, default=16, help='Number of layers (must match pretrained model)')
    parser.add_argument('--n_head', type=int, default=8, help='Number of heads (must match pretrained model)')
    parser.add_argument('--n_embd', type=int, default=1024, help='Embedding dimension (must match pretrained model)')
    parser.add_argument('--block_size', type=int, default=1024, help='Context size')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size (must match pretrained model)')

    # Device settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--no_cuda', action='store_true')

    # Logging
    parser.add_argument('--wandb_project', type=str, default='ipa_finetuning')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluate every N iterations (default: 100, reduced from 5)')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    args = parser.parse_args()

    print("="*80)
    print("BATCHED ATTENTION FINE-TUNING")
    print("="*80)
    print(f"Key differences from FlexAttention version:")
    print(f"  - Batch size: {args.batch_size} (was 1)")
    print(f"  - Expected speedup: ~{args.batch_size}x")
    print(f"  - Epochs: {args.num_epochs} (recommended 1 for large datasets)")
    print(f"  - Eval interval: {args.eval_interval} (less frequent = faster)")
    print(f"  - Eval iters: {args.eval_iters} (fewer = faster)")
    print("="*80)

    # Try multiple possible tokenizer file locations and naming patterns
    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    tokenizer_name_path = pathlib.Path(args.tokenizer_name)
    
    vocab_file = None
    
    # Case 1: Absolute path provided directly
    if tokenizer_name_path.is_absolute():
        if tokenizer_name_path.exists():
            vocab_file = tokenizer_name_path
            print(f"Found tokenizer at: {vocab_file}")
        else:
            print(f"Warning: Absolute path provided but file not found: {tokenizer_name_path}")
    
    # Case 2: Search relative to tokenizer_dir
    if vocab_file is None:
        possible_files = [
            tokenizer_dir / args.tokenizer_name,  # Full filename as-is (e.g., bpe-eng-spa-tokenizer.json)
            tokenizer_dir / f"{args.tokenizer_name}.json",  # Add .json extension
            tokenizer_dir / f"{args.tokenizer_name}-tokenizer.json",  # Pattern: name-tokenizer.json
            tokenizer_dir / f"{args.tokenizer_name}-vocab.json",  # Pattern: name-vocab.json
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                vocab_file = file_path
                print(f"Found tokenizer at: {vocab_file}")
                break
    
    # Case 3: For GPT-2, download from HuggingFace
    if vocab_file is None and args.tokenizer_name == 'gpt2':
        from tokenizers import Tokenizer
        try:
            print("Loading GPT-2 tokenizer from HuggingFace...")
            tokenizer = Tokenizer.from_pretrained("gpt2")
            vocab_file = tokenizer_dir / "gpt2-tokenizer.json"
            vocab_file.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(vocab_file))
            print(f"Saved tokenizer to {vocab_file}")
        except Exception as e:
            print(f"Error: Could not load/save GPT-2 tokenizer: {e}")
            raise
    
    # Case 4: Not found anywhere
    if vocab_file is None:
        error_msg = f"Could not find tokenizer file. Tried:\n"
        for f in possible_files:
            error_msg += f"  - {f}\n"
        error_msg += f"\nMake sure the tokenizer file exists or provide the full path using --tokenizer_name"
        raise FileNotFoundError(error_msg)

    # Validate dataset
    if args.dataset not in ['sst2', 'sst-2', 'imdb', 'yelp_polarity'] and args.dataset != 'custom':
        print(f"Warning: Dataset {args.dataset} may require custom configuration. "
              f"Use --text_column and --label_column to specify correct fields.")

    # Initialize classification model with batching support
    model = GPTClassification(
        args.device, 
        vocab_file, 
        None,  # merges_file not needed for full tokenizer.json format
        args.data_dir, 
        num_classes=args.num_classes,
        num_embed=args.n_embd, 
        dropout=args.dropout, 
        context_size=args.block_size, 
        batch_size=args.batch_size,
        ipa=args.use_ipa,
        text_column=args.text_column,
        label_column=args.label_column
    )

    # Configuration
    prepare_datasets(args, model)
    ctx = configure_device(args)
    model, optimizer, scaler = load_pretrained_model(args, model)
    max_iters = get_max_iters(model, args.gradient_accumulation_steps, args.num_epochs)

    # Calculate training estimates
    tokens_per_iter = args.gradient_accumulation_steps * model.batch_size * model.context_window
    total_tokens = model.get_token_count() * args.num_epochs
    print("="*80)
    print("Training Configuration:")
    print(f"  Total training examples: {len(model.train_data):,}")
    print(f"  Total validation examples: {len(model.val_data):,}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {model.batch_size * args.gradient_accumulation_steps}")
    print(f"  Tokens per iteration: {tokens_per_iter:,}")
    print(f"  Total iterations: {max_iters:,}")
    print(f"  Evaluations: ~{max_iters // args.eval_interval:,}")
    print(f"  Estimated time (at 0.15s/iter): ~{max_iters * 0.15 / 3600:.1f} hours")
    print("="*80)

    # Check for previous best checkpoint
    best_val_loss = float('inf')
    best_checkpoint_path = pathlib.Path(args.out_dir) / f"{args.dataset}-ckpt.pt"
    if best_checkpoint_path.exists():
        print(f"Loading previous best checkpoint from {best_checkpoint_path}")
        best_checkpoint = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
        best_val_loss = best_checkpoint.get('val_loss', float('inf'))
        print(f"Previous best validation loss: {best_val_loss:.4f}")

    # Fine-tune
    print(f"Starting fine-tuning for {max_iters} iterations...")
    print(f"Warmup iterations: {model.warmup_iters}")
    print(f"LR decay iterations: {model.lr_decay_iters}")
    finetune(model, max_iters, scaler, optimizer, ctx, best_val_loss, best_checkpoint_path, args)

    print("="*80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {best_checkpoint_path}")
    print("="*80)

if __name__ == "__main__":
    main()
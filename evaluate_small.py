"""
Evaluate Fine-tuned Small GPT Model on Classification Tasks
Adapted for small model architecture (12 layers, 6 heads, 768 dimensions)
Based on evaluate_medium.py but configured for small models
"""

import argparse
import pathlib
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, f1_score, recall_score
from contextlib import nullcontext
from model_small import GPTClassificationSmall

def evaluate(model, dataset, ctx, device, batch_size=32):
    """Evaluate model on a dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    # Process in batches
    num_examples = len(dataset)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    print(f"Evaluating on {num_examples} examples in {num_batches} batches...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_examples)
        
        # Get batch data
        batch_input_ids = []
        batch_labels = []
        
        for idx in range(start_idx, end_idx):
            example = dataset[idx]
            
            # Handle single or multiple text columns
            if len(model.text_column) == 1:
                text = str(example[model.text_column[0]])
            else:
                text = ' <ENDOFTEXT> '.join([str(example[col]) for col in model.text_column])
            
            # Tokenize
            encoded = model.tokenizer.encode(text)
            input_ids = encoded.ids[:model.context_window]
            
            # Pad if necessary
            if len(input_ids) < model.context_window:
                input_ids = input_ids + [0] * (model.context_window - len(input_ids))
            
            batch_input_ids.append(input_ids)
            batch_labels.append(example[model.label_column])
        
        # Convert to tensors
        input_ids = torch.tensor(batch_input_ids, dtype=torch.int32, device=device)
        labels = torch.tensor(batch_labels, dtype=torch.int64, device=device)
        
        # Forward pass
        with torch.no_grad():
            with ctx:
                attention_mask = (input_ids != model.pad_token_id).long()
                logits = model(input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().float().numpy())  
        
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"  Processed {end_idx}/{num_examples} examples...")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_logits)

def print_results(preds, labels, dataset_name):
    """Print evaluation results for a dataset"""
    print("\n" + "="*80)
    print(f"RESULTS: {dataset_name}")
    print("="*80)
    
    accuracy = (preds == labels).mean()
    f1_score_val = f1_score(labels, preds, average='macro')
    f2_score_val = fbeta_score(labels, preds, beta=2, average='macro')
    recall_val = recall_score(labels, preds, average='macro')
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score (macro): {f1_score_val:.4f}")
    print(f"  F2 Score (macro): {f2_score_val:.4f}")
    print(f"  Recall (macro): {recall_val:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, digits=4))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for i in range(len(cm)):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Class {i}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    return accuracy, f1_score_val, f2_score_val, recall_val, cm

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SMALL model on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('--checkpoint_path', type=pathlib.Path, required=True,
                        help='Path to fine-tuned checkpoint')
    
    # Dataset specification
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of HuggingFace dataset names to evaluate on. '
                             'If not specified, defaults to XNLI English and Spanish datasets.')
    parser.add_argument('--dataset_labels', type=str, nargs='+', default=None,
                        help='Optional friendly labels for datasets (must match number of datasets). '
                             'If not specified, dataset names will be used as labels.')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='Optional dataset configuration/subset name (e.g., "mrpc" for GLUE)')
    
    # Optional
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation (default: cuda)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Data type for computation (default: bfloat16)')
    parser.add_argument('--output_dir', type=pathlib.Path, default=None,
                        help='Directory to save detailed results')
    
    args = parser.parse_args()
    
    # Setup datasets to evaluate
    if args.datasets is None:
        # Default to XNLI datasets
        datasets_to_eval = [
            ('iggy12345/xnli-en-ipa', 'English'),
            ('iggy12345/xnli-es-ipa', 'Spanish')
        ]
        print("No datasets specified. Using default XNLI datasets.")
    else:
        # Use specified datasets
        if args.dataset_labels is not None:
            if len(args.dataset_labels) != len(args.datasets):
                parser.error(f"Number of dataset labels ({len(args.dataset_labels)}) must match number of datasets ({len(args.datasets)})")
            datasets_to_eval = list(zip(args.datasets, args.dataset_labels))
        else:
            # Use dataset names as labels
            datasets_to_eval = [(ds, ds.split('/')[-1]) for ds in args.datasets]
    
    print("="*80)
    print("SMALL MODEL EVALUATION (Classification)")
    print("="*80)
    print(f"Model architecture: 12 layers, 6 heads, 768 dimensions (~70M params)")
    print(f"\nDatasets to evaluate: {len(datasets_to_eval)}")
    for ds_name, ds_label in datasets_to_eval:
        print(f"  - {ds_label}: {ds_name}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    
    # Get configuration from checkpoint
    ckpt_args = checkpoint['args']
    
    # Verify this is a small model
    if ckpt_args['n_layer'] != 12 or ckpt_args['n_head'] != 6 or ckpt_args['n_embd'] != 768:
        print(f"\nWARNING: Checkpoint architecture doesn't match small model specs!")
        print(f"  Expected: 12 layers, 6 heads, 768 dim")
        print(f"  Found: {ckpt_args['n_layer']} layers, {ckpt_args['n_head']} heads, {ckpt_args['n_embd']} dim")
        print(f"  Continuing anyway...")
    
    # Determine which columns to use based on checkpoint
    text_columns = ckpt_args['text_column']
    print(f"Checkpoint uses text columns: {text_columns}")
    print(f"Checkpoint uses label column: {ckpt_args['label_column']}")
    
    # Reconstruct model
    print("\nReconstructing model...")
    
    # Find tokenizer
    tokenizer_dir = pathlib.Path(ckpt_args['tokenizer_dir'])
    tokenizer_name = ckpt_args['tokenizer_name']
    vocab_file = tokenizer_dir / tokenizer_name
    
    if not vocab_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {vocab_file}")
    
    # Create model
    model = GPTClassificationSmall(
        device=args.device,
        vocab_file=vocab_file,
        merges_file=None,
        data_dir=pathlib.Path(ckpt_args['data_dir']),
        num_classes=ckpt_args['num_classes'],
        num_embed=ckpt_args['n_embd'],
        dropout=ckpt_args['dropout'],
        context_size=ckpt_args['block_size'],
        batch_size=1,  # Not used during eval
        text_column=text_columns,
        label_column=ckpt_args['label_column']
    )
    
    # Load pretrained model from checkpoint
    from model_small import GPTBatchedSmall
    pretrained = GPTBatchedSmall(
        vocab_size=ckpt_args['vocab_size'],
        num_layers=ckpt_args['n_layer'],
        num_heads=ckpt_args['n_head'],
        model_dim=ckpt_args['n_embd'],
        max_seq_len=ckpt_args['block_size']
    )
    model.pretrained_model = pretrained
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    
    print(f"  Model loaded successfully")
    print(f"  Val loss at save: {checkpoint['val_loss']:.4f}")
    print(f"  Val accuracy at save: {checkpoint['val_accuracy']:.4f}")
    print(f"  Val F2 at save: {checkpoint['val_f2']:.4f}")
    
    # Setup context
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Store all results
    all_results = {}
    
    # Evaluate on all specified datasets
    for dataset_name, lang_label in datasets_to_eval:
        print("\n" + "="*80)
        print(f"EVALUATING ON: {dataset_name} ({lang_label})")
        print("="*80)
        
        # Load test dataset
        try:
            print(f"Loading dataset: {dataset_name} (split: {args.split})")
            if args.dataset_config:
                test_dataset = load_dataset(dataset_name, args.dataset_config, split=args.split)
                print(f"  Using config: {args.dataset_config}")
            else:
                test_dataset = load_dataset(dataset_name, split=args.split)
            
            print(f"  Loaded {len(test_dataset)} examples")
            print(f"  Dataset features: {list(test_dataset.features.keys())}")
            print(f"  Using text columns: {text_columns}")
            print(f"  Using label column: {ckpt_args['label_column']}")
            
            # Verify required columns exist
            missing_cols = []
            for col in text_columns:
                if col not in test_dataset.features:
                    missing_cols.append(col)
            if ckpt_args['label_column'] not in test_dataset.features:
                missing_cols.append(ckpt_args['label_column'])
            
            if missing_cols:
                print(f"X Missing columns in dataset: {missing_cols}")
                print(f"  Available columns: {list(test_dataset.features.keys())}")
                print(f"  Skipping this dataset.")
                continue
                
        except Exception as e:
            print(f"X Error loading dataset: {e}")
            print(f"  Skipping this dataset.")
            continue
        
        # Evaluate
        print("\nRunning evaluation...")
        try:
            preds, labels, logits = evaluate(model, test_dataset, ctx, args.device, args.batch_size)
            
            # Print results
            accuracy, f1_score_val, f2_score_val, recall_val, cm = print_results(
                preds, labels, f"{lang_label} ({dataset_name})"
            )
            
            # Store results
            all_results[lang_label] = {
                'dataset': dataset_name,
                'config': args.dataset_config,
                'split': args.split,
                'num_examples': len(test_dataset),
                'predictions': preds.tolist(),
                'labels': labels.tolist(),
                'logits': logits.tolist(),
                'accuracy': float(accuracy),
                'f1_score': float(f1_score_val),
                'f2_score': float(f2_score_val),
                'recall': float(recall_val),
                'confusion_matrix': cm.tolist()
            }
        except Exception as e:
            print(f"X Error during evaluation: {e}")
            print(f"  Skipping this dataset.")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint_path.name}")
    print(f"Model: SMALL (12 layers, 6 heads, 768 dim)")
    print(f"Text columns used: {text_columns}")
    print(f"Label column used: {ckpt_args['label_column']}")
    print(f"\nEvaluated on {len(all_results)} dataset(s):")
    
    for lang_label, results in all_results.items():
        print(f"\n  {lang_label} ({results['dataset']}):")
        print(f"    Examples:  {results['num_examples']}")
        print(f"    Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"    F1 Score:  {results['f1_score']:.4f}")
        print(f"    F2 Score:  {results['f2_score']:.4f}")
        print(f"    Recall:    {results['recall']:.4f}")
    
    # Save results if requested
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        if len(all_results) == 1:
            # Single dataset - use its label
            output_file = args.output_dir / f"results_small_{list(all_results.keys())[0].lower().replace(' ', '_')}.json"
        else:
            # Multiple datasets
            checkpoint_type = args.checkpoint_path.stem
            output_file = args.output_dir / f"results_small_{checkpoint_type}.json"
        
        results_full = {
            'model_type': 'small',
            'model_architecture': {
                'n_layer': 12,
                'n_head': 6,
                'n_embd': 768,
                'params_approx': '70M'
            },
            'checkpoint_path': str(args.checkpoint_path),
            'checkpoint_info': {
                'val_loss': float(checkpoint['val_loss']),
                'val_accuracy': float(checkpoint['val_accuracy']),
                'val_f2': float(checkpoint['val_f2']),
            },
            'text_columns': text_columns,
            'label_column': ckpt_args['label_column'],
            'split': args.split,
            'evaluation_date': str(torch.cuda.get_device_properties(0).name) if torch.cuda.is_available() else 'CPU',
            'results': all_results
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results_full, f, indent=2)
        print(f"\nResults saved to {output_file}")

    print("="*80)
    
    # Return success status
    return len(all_results) > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Count total tokens for specified epochs in your dataset for training (SMALL MODEL)
"""

import argparse
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Count total tokens in headered .bin training shards (SMALL MODEL)"
    )
    ap.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="Directory containing data_train_*.bin files",
    )
    ap.add_argument(
        "--tokens_per_iter",
        type=int,
        default=393_216,  # Changed from 524_288 (small model: 8 GPUs × 48K)
        help="Tokens per iteration (default: 393,216 for 8 GPUs, small model)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to calculate iterations for (default: 1)",
    )
    args = ap.parse_args()

    train_files = sorted(args.data_dir.glob("data_train_*.bin"))
    if not train_files:
        print(f"No training files found in {args.data_dir}")
        return

    total_tokens = 0
    for file in train_files:
        with open(file, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            num_tokens = int(header[2])
            total_tokens += num_tokens
            print(f"{file.name}: {num_tokens:,} tokens")

    total_tokens_all_epochs = total_tokens * args.epochs
    iterations = total_tokens_all_epochs / args.tokens_per_iter

    print(f"\n{'=' * 60}")
    print(f"Total training tokens (1 epoch): {total_tokens:,}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Total tokens ({args.epochs} epoch{'s' if args.epochs != 1 else ''}): {total_tokens_all_epochs:,}")
    print(f"Tokens per iteration (8 GPUs, small model): {args.tokens_per_iter:,}")  # Updated text
    print(f"Iterations for {args.epochs} epoch{'s' if args.epochs != 1 else ''}: {iterations:.0f}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
count_tokens.py – Count total tokens in your dataset.
"""

import argparse
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Count total tokens in headered .bin training shards"
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
        default=524_288,
        help="Tokens per iteration (default: 524,288 for 8 GPUs)",
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

    print(f"\n{'=' * 60}")
    print(f"Total training tokens: {total_tokens:,}")
    print(f"Tokens per iteration (8 GPUs): {args.tokens_per_iter:,}")
    print(f"Iterations for 1 epoch: {total_tokens / args.tokens_per_iter:.0f}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import DatasetDict
from tokenizers import Tokenizer

# Allow running from repository root or as `python utils/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.xlsum_sft_common import (
    LANGUAGES,
    dataset_from_disk_or_repo,
    default_tokenizer_path,
    get_eos_token_id,
    get_rep_fields,
    maybe_subsample_by_language,
    prompt_template,
    save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tokenized XLSum SFT packs.")
    p.add_argument("--dataset_repo", type=str, default="mugezhang/xlsum_6lang_multirepr")
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--representation", type=str, required=True, choices=["text", "ipa_stripped", "romanized"])
    p.add_argument("--mix_mode", type=str, default="balanced", choices=["balanced", "natural"])
    p.add_argument("--context_len", type=int, default=2048)
    p.add_argument("--target_max_tokens", type=int, default=256)
    p.add_argument("--exp_base", type=str, default="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k")
    p.add_argument("--tokenizer_json", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_sample_frac", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--source_data_dir", type=str, default=None, help="Optional pre-built DatasetDict path.")
    return p.parse_args()


def encode_example(ex, tokenizer: Tokenizer, source_field: str, target_field: str, context_len: int, target_max_tokens: int, eos_id: int):
    lang = ex["language"]
    source = str(ex[source_field])
    target = str(ex[target_field])

    prompt = prompt_template(lang=lang, source=source)
    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(target).ids[:target_max_tokens]

    if eos_id is not None:
        target_ids = target_ids + [int(eos_id)]

    if not target_ids:
        return {"skip": True}

    max_prompt_tokens = context_len - len(target_ids)
    if max_prompt_tokens <= 0:
        return {"skip": True}

    prompt_ids = prompt_ids[:max_prompt_tokens]

    input_ids = prompt_ids + target_ids
    labels = ([-100] * len(prompt_ids)) + target_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "language": lang,
        "prompt_len": len(prompt_ids),
        "target_len": len(target_ids),
        "skip": False,
    }


def main() -> None:
    args = parse_args()
    source_field, target_field = get_rep_fields(args.representation)

    tokenizer_json = Path(args.tokenizer_json) if args.tokenizer_json else default_tokenizer_path(args.exp_base, args.representation)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos_id = get_eos_token_id(tokenizer)

    ds = dataset_from_disk_or_repo(
        data_path=args.source_data_dir,
        dataset_repo=args.dataset_repo,
        mix_mode=args.mix_mode,
        cache_dir=args.dataset_cache_dir,
        seed=args.seed,
    )

    train = ds["train"]
    validation = ds["validation"]
    test = ds["test"]

    if args.train_sample_frac < 1.0:
        train = maybe_subsample_by_language(train, frac=args.train_sample_frac, seed=args.seed)

    def _mapper(ex):
        return encode_example(
            ex,
            tokenizer=tokenizer,
            source_field=source_field,
            target_field=target_field,
            context_len=args.context_len,
            target_max_tokens=args.target_max_tokens,
            eos_id=eos_id,
        )

    remove_cols = train.column_names
    train_tok = train.map(_mapper, remove_columns=remove_cols, desc="Tokenizing train")
    val_tok = validation.map(_mapper, remove_columns=validation.column_names, desc="Tokenizing validation")
    test_tok = test.map(_mapper, remove_columns=test.column_names, desc="Tokenizing test")

    train_tok = train_tok.filter(lambda ex: not ex["skip"], desc="Filter train")
    val_tok = val_tok.filter(lambda ex: not ex["skip"], desc="Filter validation")
    test_tok = test_tok.filter(lambda ex: not ex["skip"], desc="Filter test")

    for split_name, split_ds in (("train", train_tok), ("validation", val_tok), ("test", test_tok)):
        if len(split_ds) == 0:
            raise RuntimeError(f"No rows left after tokenization for split={split_name}")

    out = DatasetDict({"train": train_tok, "validation": val_tok, "test": test_tok})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_dir))

    meta = {
        "dataset_repo": args.dataset_repo,
        "representation": args.representation,
        "mix_mode": args.mix_mode,
        "context_len": args.context_len,
        "target_max_tokens": args.target_max_tokens,
        "train_sample_frac": args.train_sample_frac,
        "tokenizer_json": str(tokenizer_json),
        "eos_id": eos_id,
        "languages": list(LANGUAGES),
        "splits": {k: len(v) for k, v in out.items()},
    }
    save_json(out_dir / "metadata.json", meta)

    print("Saved tokenized SFT packs")
    print(f"Output: {out_dir}")
    print(f"Rows: {meta['splits']}")


if __name__ == "__main__":
    main()

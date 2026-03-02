#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

# Avoid importing the repository's local `evaluate.py`.
REPO_ROOT = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == REPO_ROOT:
    sys.path.pop(0)
try:
    import evaluate as hf_evaluate  # type: ignore
except ModuleNotFoundError:
    hf_evaluate = None  # type: ignore[assignment]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sacrebleu
except ModuleNotFoundError:
    sacrebleu = None  # type: ignore[assignment]

from utils.xlsum_sft_common import (
    LANGUAGES,
    default_tokenizer_path,
    get_eos_token_id,
    get_rep_fields,
    prompt_template,
    save_json,
)
from utils.xlsum_sft_modeling import load_model_from_training_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multilingual XLSum generation (ROUGE/BLEU/token-loss).")
    p.add_argument("--model_ckpt", type=str, required=True, help="Path to SFT checkpoint (best.pt or last.pt)")
    p.add_argument("--representation", type=str, required=True, choices=["text", "ipa_stripped", "romanized"])
    p.add_argument("--size", type=str, required=True, choices=["small", "medium", "large"])
    p.add_argument("--dataset_repo", type=str, default="mugezhang/xlsum_6lang_multirepr")
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["validation", "test"])

    p.add_argument("--exp_base", type=str, default="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k")
    p.add_argument("--tokenizer_json", type=str, default=None)

    p.add_argument("--context_len", type=int, default=2048)
    p.add_argument("--target_max_tokens", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_eval_samples_per_lang", type=int, default=0)

    p.add_argument("--batch_size_token_loss", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_csv", type=str, default=None)
    return p.parse_args()


def pad_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)

    return {"input_ids": input_ids, "labels": labels}


def encode_teacher_forced(tokenizer: Tokenizer, lang: str, source: str, target: str, context_len: int, target_max_tokens: int, eos_id: int):
    prompt_ids = tokenizer.encode(prompt_template(lang, source)).ids
    target_ids = tokenizer.encode(target).ids[:target_max_tokens]
    if eos_id is not None:
        target_ids = target_ids + [int(eos_id)]
    if not target_ids:
        return None

    max_prompt = context_len - len(target_ids)
    if max_prompt <= 0:
        return None
    prompt_ids = prompt_ids[:max_prompt]

    input_ids = prompt_ids + target_ids
    labels = ([-100] * len(prompt_ids)) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def beam_search_generate(
    model,
    input_ids: list[int],
    max_new_tokens: int,
    num_beams: int,
    eos_id: int,
    device: torch.device,
    valid_vocab_size: int,
) -> list[int]:
    beams: list[tuple[float, list[int]]] = [(0.0, list(input_ids))]

    for _ in range(max_new_tokens):
        candidates: list[tuple[float, list[int]]] = []
        ended = 0

        for score, seq in beams:
            if eos_id is not None and len(seq) > len(input_ids) and seq[-1] == eos_id:
                candidates.append((score, seq))
                ended += 1
                continue

            model_input = torch.tensor(seq[-model.max_seq_len :], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(model_input)["logits"][0, -1]
            if logits.size(0) > valid_vocab_size:
                logits = logits.clone()
                logits[valid_vocab_size:] = -float("inf")

            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, k=num_beams, dim=-1)
            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                candidates.append((score + float(lp), seq + [int(tid)]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:num_beams]

        if ended == len(beams):
            break

    best = beams[0][1]
    return best[len(input_ids) :]


def compute_token_loss(model, encoded_examples: list[dict], batch_size: int, pad_token_id: int, device: torch.device) -> float:
    if not encoded_examples:
        return float("nan")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(encoded_examples), batch_size):
        batch = pad_batch(encoded_examples[i : i + batch_size], pad_token_id=pad_token_id)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            out = model(input_ids, labels=labels)
        total_loss += float(out["sum_loss"].item())
        total_tokens += int(out["num_tokens"].item())

    return total_loss / max(total_tokens, 1)


def main() -> None:
    args = parse_args()
    if hf_evaluate is None:
        raise ModuleNotFoundError(
            "Missing dependency `evaluate`. Install it in your evaluation environment: `pip install evaluate rouge_score`."
        )
    if sacrebleu is None:
        raise ModuleNotFoundError(
            "Missing dependency `sacrebleu`. Install it in your evaluation environment: `pip install sacrebleu`."
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bundle = load_model_from_training_checkpoint(args.model_ckpt, map_location="cpu")
    model = bundle.model.to(device)
    model.eval()

    tokenizer_json = Path(args.tokenizer_json) if args.tokenizer_json else Path(bundle.config.get("tokenizer_json", ""))
    if not tokenizer_json:
        tokenizer_json = default_tokenizer_path(args.exp_base, args.representation)
    if not tokenizer_json.exists():
        tokenizer_json = default_tokenizer_path(args.exp_base, args.representation)

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos_id = get_eos_token_id(tokenizer)

    source_field, target_field = get_rep_fields(args.representation)

    rouge_metric = hf_evaluate.load("rouge")
    lang_rows = []

    for lang in LANGUAGES:
        ds = load_dataset(args.dataset_repo, lang, split=args.split, cache_dir=args.dataset_cache_dir)
        if args.max_eval_samples_per_lang > 0:
            ds = ds.select(range(min(len(ds), args.max_eval_samples_per_lang)))

        preds = []
        refs = []
        encoded_for_loss = []

        for ex in ds:
            source = str(ex[source_field])
            target = str(ex[target_field])

            prompt_ids = tokenizer.encode(prompt_template(lang, source)).ids
            max_prompt = max(1, args.context_len - args.max_new_tokens)
            prompt_ids = prompt_ids[:max_prompt]

            gen_ids = beam_search_generate(
                model=model,
                input_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                eos_id=eos_id,
                device=device,
                valid_vocab_size=bundle.config["embed_vocab_size"],
            )

            if eos_id is not None and eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id)]

            pred_text = tokenizer.decode(gen_ids)
            preds.append(pred_text)
            refs.append(target)

            encoded = encode_teacher_forced(
                tokenizer=tokenizer,
                lang=lang,
                source=source,
                target=target,
                context_len=args.context_len,
                target_max_tokens=args.target_max_tokens,
                eos_id=eos_id,
            )
            if encoded is not None:
                encoded_for_loss.append(encoded)

        rouge_scores = rouge_metric.compute(predictions=preds, references=refs, use_stemmer=True)
        bleu = sacrebleu.corpus_bleu(preds, [refs]).score
        token_loss = compute_token_loss(
            model=model,
            encoded_examples=encoded_for_loss,
            batch_size=args.batch_size_token_loss,
            pad_token_id=0,
            device=device,
        )

        lang_row = {
            "language": lang,
            "split": args.split,
            "num_examples": len(refs),
            "rouge1": float(rouge_scores["rouge1"]),
            "rouge2": float(rouge_scores["rouge2"]),
            "rougeL": float(rouge_scores["rougeL"]),
            "bleu": float(bleu),
            "token_loss": float(token_loss),
        }
        lang_rows.append(lang_row)
        print(
            f"[{lang}] rougeL={lang_row['rougeL']:.4f} bleu={lang_row['bleu']:.2f} token_loss={lang_row['token_loss']:.4f}"
        )

    macro = {
        "rouge1": sum(r["rouge1"] for r in lang_rows) / len(lang_rows),
        "rouge2": sum(r["rouge2"] for r in lang_rows) / len(lang_rows),
        "rougeL": sum(r["rougeL"] for r in lang_rows) / len(lang_rows),
        "bleu": sum(r["bleu"] for r in lang_rows) / len(lang_rows),
        "token_loss": sum(r["token_loss"] for r in lang_rows) / len(lang_rows),
    }

    total_n = sum(r["num_examples"] for r in lang_rows)
    weighted = {}
    for k in ("rouge1", "rouge2", "rougeL", "bleu", "token_loss"):
        weighted[k] = sum(r[k] * r["num_examples"] for r in lang_rows) / max(total_n, 1)

    output = {
        "model_ckpt": args.model_ckpt,
        "representation": args.representation,
        "size": args.size,
        "split": args.split,
        "tokenizer_json": str(tokenizer_json),
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "max_eval_samples_per_lang": args.max_eval_samples_per_lang,
        "per_language": lang_rows,
        "macro": macro,
        "weighted": weighted,
    }

    save_json(args.output_json, output)
    print("Saved JSON:", args.output_json)

    csv_path = args.output_csv if args.output_csv else str(Path(args.output_json).with_suffix(".csv"))
    fieldnames = ["language", "split", "num_examples", "rouge1", "rouge2", "rougeL", "bleu", "token_loss"]
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in lang_rows:
            writer.writerow(row)
        writer.writerow({"language": "macro", "split": args.split, "num_examples": total_n, **macro})
        writer.writerow({"language": "weighted", "split": args.split, "num_examples": total_n, **weighted})

    print("Saved CSV:", csv_path)
    print(json.dumps({"macro": macro, "weighted": weighted}, indent=2))


if __name__ == "__main__":
    main()

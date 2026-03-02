#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from accelerate import Accelerator
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    Accelerator = None  # type: ignore[assignment]

# Ensure repository root is importable regardless of invocation path.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.xlsum_sft_common import (
    SIZE_CONFIGS,
    default_tokenizer_path,
    find_best_checkpoint,
    load_checkpoint_state,
)
from utils.xlsum_sft_modeling import load_model_from_pretrained_state


@dataclass
class RunConfig:
    representation: str
    size: str
    mix_mode: str
    context_len: int
    target_max_tokens: int
    num_train_epochs: float
    train_sample_frac: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    mixed_precision: str
    seed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF-accelerate multilingual XLSum generative SFT.")

    p.add_argument("--dataset_repo", type=str, default="mugezhang/xlsum_6lang_multirepr")
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--packed_data_dir", type=str, default=None)
    p.add_argument("--build_packs_if_missing", action="store_true")

    p.add_argument("--exp_base", type=str, default="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k")
    p.add_argument("--representation", type=str, required=True, choices=["text", "ipa_stripped", "romanized"])
    p.add_argument("--size", type=str, required=True, choices=["small", "medium", "large"])
    p.add_argument("--mix_mode", type=str, default="balanced", choices=["balanced", "natural"])

    p.add_argument("--tokenizer_json", type=str, default=None)
    p.add_argument("--pretrained_ckpt", type=str, default=None, help="Optional explicit path. If unset, auto-discover best checkpoint.")

    p.add_argument("--context_len", type=int, default=2048)
    p.add_argument("--target_max_tokens", type=int, default=256)

    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--train_sample_frac", type=float, default=1.0)

    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--eval_every_steps", type=int, default=200)
    p.add_argument("--save_every_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dataloader_pin_memory", action="store_true")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_collate(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    langs = []
    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
        langs.append(ex["language"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "language": langs,
    }


def build_default_pack_dir(args: argparse.Namespace) -> Path:
    return (
        Path(args.exp_base)
        / "xlsum_sft"
        / "packs"
        / f"{args.representation}_{args.mix_mode}_ctx{args.context_len}_tgt{args.target_max_tokens}"
    )


def ensure_tokenized_packs(args: argparse.Namespace) -> Path:
    out_dir = Path(args.packed_data_dir) if args.packed_data_dir else build_default_pack_dir(args)
    if out_dir.exists() and (out_dir / "metadata.json").exists():
        return out_dir

    if not args.build_packs_if_missing:
        raise FileNotFoundError(
            f"Tokenized pack dir not found: {out_dir}. Pass --build_packs_if_missing to auto-build."
        )

    cmd = [
        "python",
        "utils/build_xlsum_sft_packs.py",
        "--dataset_repo",
        args.dataset_repo,
        "--representation",
        args.representation,
        "--mix_mode",
        args.mix_mode,
        "--context_len",
        str(args.context_len),
        "--target_max_tokens",
        str(args.target_max_tokens),
        "--exp_base",
        args.exp_base,
        "--seed",
        str(args.seed),
        "--train_sample_frac",
        str(args.train_sample_frac),
        "--output_dir",
        str(out_dir),
    ]
    if args.dataset_cache_dir:
        cmd.extend(["--dataset_cache_dir", args.dataset_cache_dir])
    if args.tokenizer_json:
        cmd.extend(["--tokenizer_json", args.tokenizer_json])

    subprocess.run(cmd, check=True)
    return out_dir


def resolve_pretrained_checkpoint(args: argparse.Namespace) -> str:
    if args.pretrained_ckpt:
        return args.pretrained_ckpt
    best = find_best_checkpoint(args.exp_base, args.representation, args.size)
    return str(best.path)


def get_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    decay = []
    no_decay = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=(0.9, 0.95))


def get_linear_warmup_decay_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    if current_step >= total_steps:
        return 0.0
    remain = total_steps - current_step
    decay = total_steps - warmup_steps
    return float(remain) / float(max(1, decay))


@torch.no_grad()
def evaluate_token_loss(accelerator: Accelerator, model: torch.nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    total_loss = torch.zeros((), device=accelerator.device)
    total_tokens = torch.zeros((), device=accelerator.device)

    for batch in dataloader:
        out = model(batch["input_ids"], labels=batch["labels"])
        total_loss += out["sum_loss"]
        total_tokens += out["num_tokens"]

    total_loss = accelerator.gather_for_metrics(total_loss).sum()
    total_tokens = accelerator.gather_for_metrics(total_tokens).sum().clamp_min(1)

    model.train()
    return (total_loss / total_tokens).item()


def make_output_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root) if args.output_root else Path(args.exp_base) / "xlsum_sft" / "models"
    run_name = args.run_name
    if not run_name:
        run_name = f"{args.size}_{args.representation}_{args.mix_mode}_lr{args.learning_rate:g}_wd{args.weight_decay:g}_wr{args.warmup_ratio:g}_s{args.seed}"
    out_dir = output_root / args.representation / args.size / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_checkpoint(accelerator: Accelerator, model: torch.nn.Module, ckpt_path: Path, payload: dict) -> None:
    unwrapped = accelerator.unwrap_model(model)
    payload = dict(payload)
    payload["model_state"] = accelerator.get_state_dict(unwrapped)
    accelerator.save(payload, str(ckpt_path))


def main() -> None:
    args = parse_args()
    if Accelerator is None:
        raise ModuleNotFoundError(
            "Missing dependency `accelerate`. Install it in your training environment: `pip install accelerate`."
        )
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if accelerator.is_main_process:
        print("=" * 80)
        print("XLSUM MULTILINGUAL GENERATIVE SFT (ACCELERATE)")
        print("=" * 80)

    tokenizer_json = Path(args.tokenizer_json) if args.tokenizer_json else default_tokenizer_path(args.exp_base, args.representation)
    pack_dir = ensure_tokenized_packs(args)

    if accelerator.is_main_process:
        print(f"Tokenizer: {tokenizer_json}")
        print(f"Pack dir: {pack_dir}")

    ds = load_from_disk(str(pack_dir))
    pad_token_id = 0

    train_ds = ds["train"]
    val_ds = ds["validation"]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_collate(b, pad_token_id),
        num_workers=args.num_workers,
        pin_memory=args.dataloader_pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate(b, pad_token_id),
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.dataloader_pin_memory,
        drop_last=False,
    )

    pretrained_ckpt = resolve_pretrained_checkpoint(args)
    model_state = load_checkpoint_state(pretrained_ckpt)
    loaded = load_model_from_pretrained_state(state_dict=model_state, max_seq_len=args.context_len, strict=True)
    model = loaded.model

    # Safety check that checkpoint matches requested size profile.
    expected = SIZE_CONFIGS[args.size]
    got = (loaded.config["num_layers"], loaded.config["num_heads"], loaded.config["model_dim"])
    if expected != got:
        raise ValueError(f"Checkpoint architecture mismatch for size={args.size}. expected={expected}, got={got}, ckpt={pretrained_ckpt}")

    optimizer = get_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = int(math.ceil(steps_per_epoch * args.num_train_epochs))
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_linear_warmup_decay_lambda(step, warmup_steps, total_steps),
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    out_dir = make_output_dir(args)
    if accelerator.is_main_process:
        run_cfg = RunConfig(
            representation=args.representation,
            size=args.size,
            mix_mode=args.mix_mode,
            context_len=args.context_len,
            target_max_tokens=args.target_max_tokens,
            num_train_epochs=args.num_train_epochs,
            train_sample_frac=args.train_sample_frac,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
        )
        with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(run_cfg), f, indent=2)

    global_step = 0
    best_val = float("inf")
    train_loss_window = []

    t0 = time.time()
    while global_step < total_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(batch["input_ids"], labels=batch["labels"])
                loss = out["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_window.append(loss.detach().float())

            if accelerator.sync_gradients:
                global_step += 1

                if args.eval_every_steps > 0 and (global_step % args.eval_every_steps == 0 or global_step == 1):
                    val_loss = evaluate_token_loss(accelerator, model, val_loader)
                    mean_train = torch.stack(train_loss_window).mean().item() if train_loss_window else float("nan")
                    train_loss_window = []

                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        step_avg = elapsed / max(1, global_step)
                        lr = scheduler.get_last_lr()[0]
                        print(
                            f"step={global_step}/{total_steps} train_loss={mean_train:.4f} "
                            f"val_token_loss={val_loss:.4f} lr={lr:.3e} step_avg={step_avg:.3f}s"
                        )

                    if val_loss < best_val:
                        best_val = val_loss
                        if accelerator.is_main_process:
                            save_checkpoint(
                                accelerator,
                                model,
                                out_dir / "best.pt",
                                {
                                    "config": {
                                        **loaded.config,
                                        "max_seq_len": args.context_len,
                                        "representation": args.representation,
                                        "size": args.size,
                                        "tokenizer_json": str(tokenizer_json),
                                    },
                                    "training_args": vars(args),
                                    "pretrained_checkpoint": pretrained_ckpt,
                                    "global_step": global_step,
                                    "val_token_loss": best_val,
                                },
                            )

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0 and accelerator.is_main_process:
                    save_checkpoint(
                        accelerator,
                        model,
                        out_dir / f"step_{global_step:07d}.pt",
                        {
                            "config": {
                                **loaded.config,
                                "max_seq_len": args.context_len,
                                "representation": args.representation,
                                "size": args.size,
                                "tokenizer_json": str(tokenizer_json),
                            },
                            "training_args": vars(args),
                            "pretrained_checkpoint": pretrained_ckpt,
                            "global_step": global_step,
                        },
                    )

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    final_val = evaluate_token_loss(accelerator, model, val_loader)

    if accelerator.is_main_process:
        save_checkpoint(
            accelerator,
            model,
            out_dir / "last.pt",
            {
                "config": {
                    **loaded.config,
                    "max_seq_len": args.context_len,
                    "representation": args.representation,
                    "size": args.size,
                    "tokenizer_json": str(tokenizer_json),
                },
                "training_args": vars(args),
                "pretrained_checkpoint": pretrained_ckpt,
                "global_step": global_step,
                "final_val_token_loss": final_val,
                "best_val_token_loss": best_val,
            },
        )

        summary = {
            "global_step": global_step,
            "total_steps": total_steps,
            "best_val_token_loss": best_val,
            "final_val_token_loss": final_val,
            "output_dir": str(out_dir),
            "pretrained_ckpt": pretrained_ckpt,
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("Training complete")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

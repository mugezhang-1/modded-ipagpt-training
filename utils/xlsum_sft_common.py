from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from tokenizers import Tokenizer

LANGUAGES: tuple[str, ...] = (
    "english",
    "spanish",
    "hindi",
    "russian",
    "tamil",
    "urdu",
)

REP_FIELDS: dict[str, tuple[str, str]] = {
    "text": ("text", "summary"),
    "ipa_stripped": ("text_ipa_stripped", "summary_ipa_stripped"),
    "romanized": ("text_romanized", "summary_romanized"),
}

SIZE_CONFIGS: dict[str, tuple[int, int, int]] = {
    "small": (12, 6, 768),
    "medium": (16, 8, 1024),
    "large": (24, 18, 1152),
}


def get_rep_fields(representation: str) -> tuple[str, str]:
    if representation not in REP_FIELDS:
        raise ValueError(f"Unsupported representation: {representation}")
    return REP_FIELDS[representation]


def prompt_template(lang: str, source: str) -> str:
    return (
        f"[LANG={lang}] Summarize the following article.\n"
        f"Article:\n{source}\n"
        "Summary:\n"
    )


def default_tokenizer_path(exp_base: str | Path, representation: str) -> Path:
    exp_base = Path(exp_base)
    tok_dir_map = {
        "text": "8lang_text",
        "ipa_stripped": "8lang_ipa_stripped",
        "romanized": "8lang_romanized",
    }
    prefix_map = {
        "text": "bpe-8lang-text-100k-tokenizer.json",
        "ipa_stripped": "bpe-8lang-ipa-stripped-100k-tokenizer.json",
        "romanized": "bpe-8lang-romanized-100k-tokenizer.json",
    }
    out = exp_base / "tokenizers" / tok_dir_map[representation] / prefix_map[representation]
    if not out.exists():
        raise FileNotFoundError(f"Tokenizer not found: {out}")
    return out


def load_tokenizer(path: str | Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    return tok


def get_eos_token_id(tokenizer: Tokenizer, default_id: int = 50256) -> int:
    for token in ("<|endoftext|>", "<|eot_id|>", "<eos>", "</s>"):
        tid = tokenizer.token_to_id(token)
        if tid is not None:
            return int(tid)
    return int(default_id)


def next_multiple_of_n(v: int, n: int) -> int:
    return int(math.ceil(v / n) * n)


def clean_state_dict_prefix(state_dict: dict[str, object]) -> dict[str, object]:
    out = {}
    prefix = "_orig_mod."
    for k, v in state_dict.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


_BEST_STATE_PATTERN = re.compile(r"best_state_step(\d+)_val([0-9.]+)\.pt$")


@dataclass
class BestCheckpoint:
    path: Path
    step: int
    val_loss: float


def find_best_checkpoint(exp_base: str | Path, representation: str, size: str) -> BestCheckpoint:
    root = Path(exp_base) / "models" / representation / size
    if not root.exists():
        raise FileNotFoundError(f"Model directory not found: {root}")

    candidates: list[BestCheckpoint] = []
    for p in root.glob("**/best_state_step*_val*.pt"):
        m = _BEST_STATE_PATTERN.search(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        val = float(m.group(2))
        candidates.append(BestCheckpoint(path=p, step=step, val_loss=val))

    if not candidates:
        raise FileNotFoundError(f"No best_state checkpoints found under {root}")

    # Prefer lower val_loss; break ties with later step.
    candidates.sort(key=lambda c: (c.val_loss, -c.step))
    return candidates[0]


def load_checkpoint_state(path: str | Path) -> dict:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
    elif isinstance(ckpt, dict):
        model_state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return clean_state_dict_prefix(model_state)


def load_multilingual_split(
    dataset_repo: str,
    split: str,
    cache_dir: str | None = None,
) -> dict[str, Dataset]:
    out: dict[str, Dataset] = {}
    for lang in LANGUAGES:
        out[lang] = load_dataset(dataset_repo, lang, split=split, cache_dir=cache_dir)
    return out


def _upsample_dataset(ds: Dataset, target_size: int, seed: int) -> Dataset:
    if len(ds) == target_size:
        return ds
    rng = random.Random(seed)
    idx = [rng.randrange(len(ds)) for _ in range(target_size)]
    return ds.select(idx)


def mix_train_datasets(
    lang_to_ds: dict[str, Dataset],
    mix_mode: str,
    seed: int,
) -> Dataset:
    if mix_mode not in {"balanced", "natural"}:
        raise ValueError(f"Unsupported mix_mode: {mix_mode}")

    if mix_mode == "natural":
        merged = concatenate_datasets([lang_to_ds[l] for l in LANGUAGES])
        return merged.shuffle(seed=seed)

    max_n = max(len(ds) for ds in lang_to_ds.values())
    upsampled = []
    for i, lang in enumerate(LANGUAGES):
        upsampled.append(_upsample_dataset(lang_to_ds[lang], max_n, seed + i))
    merged = concatenate_datasets(upsampled)
    return merged.shuffle(seed=seed)


def attach_language_column(lang_to_ds: dict[str, Dataset]) -> dict[str, Dataset]:
    out: dict[str, Dataset] = {}
    for lang, ds in lang_to_ds.items():
        out[lang] = ds.add_column("language", [lang] * len(ds))
    return out


def build_or_load_dataset_dict(
    dataset_repo: str,
    mix_mode: str,
    cache_dir: str | None = None,
    seed: int = 42,
) -> DatasetDict:
    train_by_lang = attach_language_column(load_multilingual_split(dataset_repo, "train", cache_dir=cache_dir))
    val_by_lang = attach_language_column(load_multilingual_split(dataset_repo, "validation", cache_dir=cache_dir))
    test_by_lang = attach_language_column(load_multilingual_split(dataset_repo, "test", cache_dir=cache_dir))

    train = mix_train_datasets(train_by_lang, mix_mode=mix_mode, seed=seed)
    validation = concatenate_datasets([val_by_lang[l] for l in LANGUAGES])
    test = concatenate_datasets([test_by_lang[l] for l in LANGUAGES])

    return DatasetDict({"train": train, "validation": validation, "test": test})


def maybe_subsample_by_language(ds: Dataset, frac: float, seed: int) -> Dataset:
    if frac >= 1.0:
        return ds
    if frac <= 0:
        raise ValueError("frac must be > 0")

    # Keep per-language sampling stable to avoid language skew shifts.
    parts = []
    for i, lang in enumerate(LANGUAGES):
        lang_ds = ds.filter(lambda ex: ex["language"] == lang)
        k = max(1, int(len(lang_ds) * frac))
        parts.append(lang_ds.shuffle(seed=seed + i).select(range(k)))
    return concatenate_datasets(parts).shuffle(seed=seed)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_float_list(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def parse_int_list(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def dataset_from_disk_or_repo(data_path: str | None, dataset_repo: str, mix_mode: str, cache_dir: str | None, seed: int) -> DatasetDict:
    if data_path:
        return load_from_disk(data_path)
    return build_or_load_dataset_dict(dataset_repo=dataset_repo, mix_mode=mix_mode, cache_dir=cache_dir, seed=seed)

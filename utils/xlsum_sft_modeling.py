from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.xlsum_sft_common import clean_state_dict_prefix


def rms_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Module):
    """Small linear wrapper compatible with training checkpoints."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1.0 / 1024.0) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x_bthd: Tensor) -> Tensor:
        cos = self.cos[None, : x_bthd.size(-3), None, :]
        sin = self.sin[None, : x_bthd.size(-3), None, :]
        x1, x2 = x_bthd.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x_bthd)


def _sdpa_with_custom_scale(q: Tensor, k: Tensor, v: Tensor, attn_scale: float) -> Tensor:
    # Keep scaling consistent with pretraining code.
    head_dim = q.size(-1)
    q = q * (attn_scale * math.sqrt(head_dim))
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


class QKVOSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, sa_lambdas: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        qkv = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x)).view(
            bsz, seqlen, 3 * self.num_heads, self.head_dim
        )
        q, k, v = qkv.chunk(3, dim=-2)

        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rotary(q), self.rotary(k)
        v = rms_norm(v)

        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = _sdpa_with_custom_scale(q, k, v, attn_scale=self.attn_scale)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_heads * self.head_dim)
        return F.linear(y, self.qkvo_w[3].type_as(y))


class QKVOBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, head_dim: int):
        super().__init__()
        self.attn = QKVOSelfAttention(dim, num_heads, max_seq_len, head_dim=head_dim) if layer_idx != 7 else None
        self.mlp = QKVOMLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor) -> Tensor:
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, sa_lambdas)
        x = x + self.mlp(rms_norm(x))
        return x


class QKVOMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(torch.empty(hdim, dim))
        self.proj_w = nn.Parameter(torch.empty(dim, hdim))

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.fc_w.type_as(x))
        x = F.relu(x).square()
        return F.linear(x, self.proj_w.type_as(x))


class QKVSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim))
        self.c_proj = CastedLinear(hdim, dim)
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, sa_lambdas: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        qkv = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(
            bsz, seqlen, 3 * self.num_heads, self.head_dim
        )
        q, k, v = qkv.chunk(3, dim=-2)

        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rotary(q), self.rotary(k)

        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = _sdpa_with_custom_scale(q, k, v, attn_scale=self.attn_scale)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_heads * self.head_dim)
        return self.c_proj(y)


class QKVMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class QKVBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = QKVSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = QKVMLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor) -> Tensor:
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(rms_norm(x), ve, sa_lambdas)
        x = x + self.mlp(rms_norm(x))
        return x


class GPTQKVOGenerative(nn.Module):
    """Batched generative model for medium/large-lite checkpoints."""

    def __init__(
        self,
        embed_vocab_size: int,
        lm_head_vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        head_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.family = "qkvo"
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.embed_vocab_size = embed_vocab_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(embed_vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(embed_vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([QKVOBlock(model_dim, num_heads, max_seq_len, i, head_dim=head_dim) for i in range(num_layers)])
        self.lm_head_w = nn.Parameter(torch.empty(lm_head_vocab_size, model_dim))
        self.scalars = nn.Parameter(torch.zeros(5 * num_layers))

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        ve = [value_embed(input_ids) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]

        x = x0 = rms_norm(self.embed(input_ids))
        skip_connections: list[Tensor] = []

        skip_map = {9: 6, 10: 4, 11: 2}
        n = len(self.blocks)
        skip_weights = self.scalars[:n]
        lambdas = self.scalars[n : 3 * n].view(-1, 2)
        sa_lambdas = self.scalars[3 * n : 5 * n].view(-1, 2)

        for i, block in enumerate(self.blocks):
            if i in skip_map and skip_map[i] < len(skip_connections):
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = block(x, ve[i], x0, lambdas[i], sa_lambdas[i])
            skip_connections.append(x)
        return rms_norm(x)

    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        raw = F.linear(hidden, self.lm_head_w.type_as(hidden)).float()
        return 15.0 * raw * torch.rsqrt(raw.square() + 225.0)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        hidden = self.forward_hidden(input_ids)
        logits = self.logits_from_hidden(hidden)

        out = {"logits": logits}
        if labels is None:
            return out

        vocab = logits.size(-1)
        sum_loss = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        num_tokens = (labels != -100).sum()
        loss = sum_loss / num_tokens.clamp_min(1)
        out.update({"loss": loss, "sum_loss": sum_loss.detach(), "num_tokens": num_tokens.detach()})
        return out


class GPTSmallGenerative(nn.Module):
    """Batched generative model for small checkpoints."""

    def __init__(
        self,
        embed_vocab_size: int,
        lm_head_vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        max_seq_len: int,
        scalars_size: int,
    ):
        super().__init__()
        self.family = "qkv"
        self.model_dim = model_dim
        self.embed_vocab_size = embed_vocab_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(embed_vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(embed_vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([QKVBlock(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, lm_head_vocab_size)
        self.scalars = nn.Parameter(torch.zeros(scalars_size))

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        ve = [value_embed(input_ids) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]

        x = x0 = rms_norm(self.embed(input_ids))
        n = len(self.blocks)
        half = n // 2
        skip_connections: list[Tensor] = []

        skip_weights = self.scalars[:half]
        lambdas = self.scalars[n : 3 * n].view(-1, 2)
        sa_lambdas = self.scalars[3 * n : 5 * n].view(-1, 2)

        for i, block in enumerate(self.blocks):
            if i >= half:
                x = x + skip_weights[i - half] * skip_connections.pop()
            x = block(x, ve[i], x0, lambdas[i], sa_lambdas[i])
            if i < half:
                skip_connections.append(x)
        return rms_norm(x)

    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        raw = self.lm_head(hidden).float()
        # Keep small-model soft-capping identical to pretraining.
        return 30.0 * torch.sigmoid(raw / (7.5 * (self.model_dim ** 0.5)))

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        hidden = self.forward_hidden(input_ids)
        logits = self.logits_from_hidden(hidden)

        out = {"logits": logits}
        if labels is None:
            return out

        vocab = logits.size(-1)
        sum_loss = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        num_tokens = (labels != -100).sum()
        loss = sum_loss / num_tokens.clamp_min(1)
        out.update({"loss": loss, "sum_loss": sum_loss.detach(), "num_tokens": num_tokens.detach()})
        return out


@dataclass
class LoadedModelBundle:
    model: nn.Module
    config: dict[str, Any]


def infer_arch_from_state(state: dict[str, Tensor]) -> dict[str, Any]:
    keys = set(state.keys())
    is_qkvo = "lm_head_w" in keys
    family = "qkvo" if is_qkvo else "qkv"

    embed_vocab_size, model_dim = state["embed.weight"].shape
    if family == "qkvo":
        lm_head_vocab_size = state["lm_head_w"].shape[0]
        q_key = next(k for k in keys if k.endswith(".attn.qkvo_w"))
        qkvo_hdim = int(state[q_key].shape[1])
    else:
        lm_head_vocab_size = state["lm_head.weight"].shape[0]
        q_key = next(k for k in keys if k.endswith(".attn.qkv_w"))
        qkvo_hdim = int(state[q_key].shape[1])

    layer_ids = [int(k.split(".")[1]) for k in keys if k.startswith("blocks.") and k.split(".")[1].isdigit()]
    num_layers = max(layer_ids) + 1

    known_heads = {
        (12, 768): 6,
        (16, 1024): 8,
        (24, 1152): 18,
        (36, 1280): 20,
    }

    if family == "qkvo":
        num_heads = known_heads.get((num_layers, model_dim), max(1, qkvo_hdim // 128))
        if qkvo_hdim % num_heads != 0:
            raise ValueError(
                f"Cannot infer qkvo head_dim from checkpoint: hdim={qkvo_hdim}, num_heads={num_heads}, "
                f"layers={num_layers}, model_dim={model_dim}"
            )
        head_dim = qkvo_hdim // num_heads
    else:
        num_heads = known_heads.get((num_layers, model_dim), max(1, qkvo_hdim // 128))
        if qkvo_hdim % num_heads != 0:
            raise ValueError(
                f"Cannot infer qkv head_dim from checkpoint: hdim={qkvo_hdim}, num_heads={num_heads}, "
                f"layers={num_layers}, model_dim={model_dim}"
            )
        head_dim = qkvo_hdim // num_heads

    config = {
        "family": family,
        "embed_vocab_size": int(embed_vocab_size),
        "lm_head_vocab_size": int(lm_head_vocab_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "model_dim": int(model_dim),
        "scalars_size": int(state["scalars"].numel()),
    }
    return config


def build_model_from_config(config: dict[str, Any], max_seq_len: int) -> nn.Module:
    family = config["family"]
    if family == "qkvo":
        return GPTQKVOGenerative(
            embed_vocab_size=config["embed_vocab_size"],
            lm_head_vocab_size=config["lm_head_vocab_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            model_dim=config["model_dim"],
            head_dim=config["head_dim"],
            max_seq_len=max_seq_len,
        )

    if family == "qkv":
        return GPTSmallGenerative(
            embed_vocab_size=config["embed_vocab_size"],
            lm_head_vocab_size=config["lm_head_vocab_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            model_dim=config["model_dim"],
            max_seq_len=max_seq_len,
            scalars_size=config["scalars_size"],
        )

    raise ValueError(f"Unsupported family: {family}")


def load_model_from_pretrained_state(
    state_dict: dict[str, Tensor],
    max_seq_len: int,
    strict: bool = True,
) -> LoadedModelBundle:
    state_dict = clean_state_dict_prefix(state_dict)
    cfg = infer_arch_from_state(state_dict)
    model = build_model_from_config(cfg, max_seq_len=max_seq_len)
    model.load_state_dict(state_dict, strict=strict)
    return LoadedModelBundle(model=model, config=cfg)


def load_model_from_training_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> LoadedModelBundle:
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if "model_state" in ckpt and "config" in ckpt:
        cfg = dict(ckpt["config"])
        model = build_model_from_config(cfg, max_seq_len=cfg["max_seq_len"])
        model.load_state_dict(ckpt["model_state"], strict=True)
        return LoadedModelBundle(model=model, config=cfg)

    if "model" in ckpt:
        state = clean_state_dict_prefix(ckpt["model"])
    elif isinstance(ckpt, dict):
        state = clean_state_dict_prefix(ckpt)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    return load_model_from_pretrained_state(state_dict=state, max_seq_len=2048, strict=True)

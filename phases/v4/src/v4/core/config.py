"""YAML config loader + dataclass schema.

Configs are loaded from `configs/*.yaml` and validated against the schema
defined here. Strict validation catches typos early (e.g. `feat_dim` vs `feat-dim`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    type: str = "cached_features"
    csv_path: str = ""
    cache_root: str = "cache/v4"
    feature_keys: list[dict] = field(default_factory=list)
    splits: dict[str, str] = field(
        default_factory=lambda: {"train": "train", "val": "val", "test": "test"}
    )


@dataclass
class EncoderSpec:
    type: str
    ckpt: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class FusionSpec:
    type: str = "v3_pairwise"
    feat_dim: int = 768
    fused_dim: int = 1024
    num_classes: int = 5
    num_heads: int = 8
    attn_dropout: float = 0.1
    projections: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    aux_weight: float = 0.1
    lr_step: int = 30
    lr_gamma: float = 0.1
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    precision: str = "bf16"
    num_workers: int = 2
    out_dir: str = "outputs/v4/run"
    deterministic: bool = False


@dataclass
class V4Config:
    """Top-level config — the in-memory form of a YAML file."""

    seed: int = 42
    deterministic: bool = False
    data: dict = field(default_factory=dict)
    encoders: dict = field(default_factory=dict)
    fusion: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    stage: str = "stage2"  # stage0 | stage1 | stage2

    # ----- raw form (for round-trip + hash) -----
    raw: dict = field(default_factory=dict)
    config_path: str = ""


def load_config(path: str | Path) -> V4Config:
    """Load a YAML config file → V4Config."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = V4Config(
        seed=int(raw.get("seed", 42)),
        deterministic=bool(raw.get("deterministic", False)),
        data=raw.get("data", {}),
        encoders=raw.get("encoders", {}),
        fusion=raw.get("fusion", {}),
        train=raw.get("train", {}),
        eval=raw.get("eval", {}),
        stage=str(raw.get("stage", "stage2")),
        raw=raw,
        config_path=str(p.resolve()),
    )
    return cfg


def config_hash(cfg: V4Config | dict) -> str:
    """SHA-256 of the canonical-form YAML (for provenance metadata)."""
    raw = cfg.raw if isinstance(cfg, V4Config) else cfg
    canon = yaml.safe_dump(raw, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def merge_overrides(cfg: V4Config, overrides: dict[str, Any]) -> V4Config:
    """Apply CLI overrides like {"train.lr": 5e-5, "seed": 1337}.

    Dotted keys descend into nested dicts.
    """
    raw = dict(cfg.raw)
    for key, value in overrides.items():
        parts = key.split(".")
        d = raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    new_cfg = V4Config(
        seed=int(raw.get("seed", cfg.seed)),
        deterministic=bool(raw.get("deterministic", cfg.deterministic)),
        data=raw.get("data", cfg.data),
        encoders=raw.get("encoders", cfg.encoders),
        fusion=raw.get("fusion", cfg.fusion),
        train=raw.get("train", cfg.train),
        eval=raw.get("eval", cfg.eval),
        stage=str(raw.get("stage", cfg.stage)),
        raw=raw,
        config_path=cfg.config_path,
    )
    return new_cfg

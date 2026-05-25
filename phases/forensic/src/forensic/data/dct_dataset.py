"""DCT cache dataset for Approach 2 training + evaluation.

Loads precomputed `[1, 224, 224]` float32 DCT shards from disk and applies the
global z-score normalization from `dct_stats.json`.

Doctor's spec F.17 (MASTER_CHECKLIST):
    "Z-score: compute global DCT_mean + DCT_std from `genimage_train/train/` only,
     save to `dct_stats.json`, apply `(t-mean)/(std+1e-8)` at every stage"

Expected disk layout (built by scripts/build_splits.py + scripts/precompute_dct.py):

    <cache_root>/
    ├── train/
    │   ├── 0_real/<sid>.pt   (each .pt = torch.Tensor [1, 224, 224] float32)
    │   └── 1_fake/<sid>.pt
    ├── val/
    │   ├── 0_real/<sid>.pt
    │   └── 1_fake/<sid>.pt
    └── test/
        └── <generator>/
            ├── 0_real/<sid>.pt
            └── 1_fake/<sid>.pt

Labels from folder name: 0_real -> 0, 1_fake -> 1 (binary).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


EPS: float = 1e-8


def load_dct_stats(path: str | Path) -> tuple[float, float]:
    """Load (mean, std) from a `dct_stats.json` file."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    mean = float(d["mean"])
    std = float(d["std"])
    if std <= 0:
        raise ValueError(f"dct_stats.json has non-positive std: {std}")
    return mean, std


class DctCacheDataset(Dataset):
    """Reads `.pt` shards + applies z-score normalization.

    Args:
        root: directory containing `0_real/` and `1_fake/` subdirs of .pt files.
        dct_stats_path: path to dct_stats.json (global mean/std from training set).
        limit: optional cap on samples (for smoke tests).
    """

    def __init__(
        self,
        root: str | Path,
        dct_stats_path: str | Path,
        limit: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"DCT cache root not found: {self.root}")

        self.mean, self.std = load_dct_stats(dct_stats_path)

        # Index: list of (path, label)
        self.samples: list[tuple[Path, int]] = []
        for label_dir, label in [("0_real", 0), ("1_fake", 1)]:
            sub = self.root / label_dir
            if not sub.exists():
                continue
            for shard in sorted(sub.glob("*.pt")):
                self.samples.append((shard, label))

        if not self.samples:
            raise RuntimeError(
                f"No .pt shards found under {self.root}/{{0_real,1_fake}}/. "
                f"Run scripts/precompute_dct.py first."
            )

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        t = torch.load(path, weights_only=True)  # safe: we wrote these ourselves
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        t = t.float()
        # Doctor's z-score: (t - mean) / (std + eps)
        t = (t - self.mean) / (self.std + EPS)
        return t, label


class DctTestCacheDataset(Dataset):
    """Per-generator test set: reads <root>/<generator>/{0_real,1_fake}/*.pt.

    Used by per_generator evaluator. Yields (tensor, label).
    """

    def __init__(
        self,
        root: str | Path,
        generator: str,
        dct_stats_path: str | Path,
        limit: int | None = None,
    ):
        self.root = Path(root) / generator
        if not self.root.exists():
            raise FileNotFoundError(
                f"Test cache for generator '{generator}' not found at {self.root}"
            )
        self.generator = generator
        self.mean, self.std = load_dct_stats(dct_stats_path)

        self.samples: list[tuple[Path, int]] = []
        for label_dir, label in [("0_real", 0), ("1_fake", 1)]:
            sub = self.root / label_dir
            if not sub.exists():
                continue
            for shard in sorted(sub.glob("*.pt")):
                self.samples.append((shard, label))

        if not self.samples:
            raise RuntimeError(
                f"No .pt shards under {self.root}/{{0_real,1_fake}}/"
            )

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        t = torch.load(path, weights_only=True).float()
        t = (t - self.mean) / (self.std + EPS)
        return t, label


__all__ = ["DctCacheDataset", "DctTestCacheDataset", "load_dct_stats", "EPS"]

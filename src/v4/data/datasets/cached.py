"""CachedFeatureDataset - loads precomputed feature .pt files keyed by sample_id."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from v4.core.class_map import binary_image_label
from v4.core.registry import DATASET_REGISTRY, register
from v4.data.datasets.base import MultimodalManifestDataset


@register(DATASET_REGISTRY, "cached_features")
class CachedFeatureDataset(MultimodalManifestDataset):
    """Loads (label, source, sample_id) + arbitrary precomputed feature tensors."""

    def __init__(
        self,
        csv_path: str | Path,
        cache_root: str | Path,
        feature_keys: list[dict],
        split: str | None = None,
        drop_label_values: list[int] | None = None,
        limit: int | None = None,
        **_kwargs,
    ):
        super().__init__(csv_path=csv_path, split=split,
                         drop_label_values=drop_label_values, limit=limit)
        self.cache_root = Path(cache_root)
        self.feature_keys = feature_keys
        for spec in feature_keys:
            if "name" not in spec or "subdir" not in spec:
                raise ValueError(f"feature_keys entry needs name + subdir: {spec}")

    def load_sample(self, row: pd.Series) -> dict:
        sid = str(row["sample_id"])
        out: dict = {
            "sample_id": sid,
            "source": str(row["source"]),
            "aux_label": torch.tensor(binary_image_label(int(row["label"])), dtype=torch.long),
        }
        for spec in self.feature_keys:
            name = spec["name"]
            subdir = spec["subdir"]
            shard = self.cache_root / subdir / f"{sid}.pt"
            if shard.exists():
                t = torch.load(shard, weights_only=False)
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
                out[name] = t.float()
            else:
                dim = spec.get("dim")
                if dim is None:
                    raise FileNotFoundError(f"cache shard missing: {shard}")
                out[name] = torch.zeros(int(dim), dtype=torch.float32)
        return out


def collate_cached(batch: list[dict]) -> dict:
    """Batch a list of CachedFeatureDataset samples into batched tensors."""
    out: dict = {
        "sample_id": [b["sample_id"] for b in batch],
        "source": [b["source"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "aux_label": torch.stack([b["aux_label"] for b in batch]),
    }
    tensor_keys = {
        k for k, v in batch[0].items()
        if isinstance(v, torch.Tensor) and k not in {"label", "aux_label"}
    }
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out

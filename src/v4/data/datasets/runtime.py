"""RuntimeImageDataset - for server-time inference + precompute scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from v4.core.registry import DATASET_REGISTRY, register
from v4.data.datasets.base import MultimodalManifestDataset
from v4.data.preprocessing.patch_dct import compute_patch_dct
from v4.data.preprocessing.tokenizers import prepare_fnd_inputs


@register(DATASET_REGISTRY, "runtime_images")
class RuntimeImageDataset(MultimodalManifestDataset):
    """Reads images from disk; returns patch-DCT + FND-CLIP inputs on the fly."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str | None = None,
        limit: int | None = None,
        with_patch_dct: bool = True,
        with_fnd_inputs: bool = True,
        with_raw_text: bool = True,
        **_kwargs,
    ):
        super().__init__(csv_path=csv_path, split=split, limit=limit)
        self.with_patch_dct = with_patch_dct
        self.with_fnd_inputs = with_fnd_inputs
        self.with_raw_text = with_raw_text

    def load_sample(self, row: pd.Series) -> dict:
        sid = str(row["sample_id"])
        path = Path(row["image_path"])
        text = str(row["text"]) or ""

        out: dict = {
            "sample_id": sid,
            "source": str(row["source"]),
        }
        if self.with_raw_text:
            out["text"] = text

        try:
            pil = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            pil = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.with_patch_dct:
            out["v_imgfor_dct"] = compute_patch_dct(path) if path.exists() else \
                torch.zeros(1, 224, 224, dtype=torch.float32)

        if self.with_fnd_inputs:
            inputs = prepare_fnd_inputs(text, pil)
            for k, v in inputs.items():
                out[k] = v.squeeze(0)
        return out

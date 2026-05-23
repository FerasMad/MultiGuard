"""MultimodalManifestDataset - base class for all V4 datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from v4.data.manifest import read_manifest


class MultimodalManifestDataset(Dataset):
    """Reads canonical manifest CSV; subclasses override load_sample()."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str | None = None,
        drop_label_values: list[int] | None = None,
        limit: int | None = None,
    ):
        self.csv_path = str(csv_path)
        self.df = read_manifest(csv_path)
        if split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if drop_label_values:
            self.df = self.df[~self.df["label"].isin(drop_label_values)].reset_index(drop=True)
        if limit:
            self.df = self.df.head(limit).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def load_sample(self, row: pd.Series) -> dict:
        return {
            "sample_id": str(row["sample_id"]),
            "text": str(row["text"]),
            "image_path": str(row["image_path"]),
            "source": str(row["source"]),
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        out = self.load_sample(row)
        out["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return out

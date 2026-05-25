"""Canonical manifest CSV schema for V4 - see plan section 3.

Every dataset row has the same 6 columns regardless of source:
    sample_id   : str   - unique identifier
    text        : str   - article caption (may be empty for image-only datasets)
    image_path  : str   - absolute path to the image file
    label       : int   - 0-4 per v4.core.class_map.LABELS
    source      : str   - provenance tag, e.g. DGM4_face_attribute
    split       : str   - train | val | test
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "text", "image_path", "label", "source", "split")
SPLITS = ("train", "val", "test")


@dataclass
class ManifestRow:
    sample_id: str
    text: str
    image_path: str
    label: int
    source: str
    split: str

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "text": self.text or "",
            "image_path": str(self.image_path),
            "label": int(self.label),
            "source": self.source,
            "split": self.split,
        }


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if df.empty:
        raise ValueError("manifest is empty")
    bad_split = set(df["split"].unique()) - set(SPLITS)
    if bad_split:
        raise ValueError(f"unknown split values: {bad_split}")
    if df["sample_id"].duplicated().any():
        dups = df[df["sample_id"].duplicated()]["sample_id"].head(5).tolist()
        raise ValueError(f"duplicate sample_id values: {dups}")
    if not df["label"].between(0, 4).all():
        raise ValueError("label values must be in [0, 4]")


def write_manifest(rows: list[ManifestRow], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.as_dict() for r in rows])
    validate(df)
    df.to_csv(p, index=False, encoding="utf-8")
    return p.resolve()


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str, "source": str, "split": str})
    df["text"] = df["text"].fillna("").astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["label"] = df["label"].astype(int)
    validate(df)
    return df


def manifest_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

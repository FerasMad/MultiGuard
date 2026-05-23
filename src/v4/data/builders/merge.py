"""Merge per-source manifests into forensic_5class_v4.csv (per V3.1 section 2)."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from v4.core.logging import get_logger
from v4.core.paths import DATA_PROCESSED, ensure_dir
from v4.data.manifest import REQUIRED_COLUMNS, validate

log = get_logger(__name__)

DEFAULT_OUT = DATA_PROCESSED / "forensic_5class_v4.csv"
TARGET_PER_CLASS = 3000
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42


def _undersample(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, sub in df.groupby("label"):
        if len(sub) <= n_per_class:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=n_per_class, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _stratified_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()
    df["split"] = "train"
    for (_label, _source), sub in df.groupby(["label", "source"]):
        idxs = list(sub.index)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        for j, idx in enumerate(idxs):
            if j < n_train:
                df.at[idx, "split"] = "train"
            elif j < n_train + n_val:
                df.at[idx, "split"] = "val"
            else:
                df.at[idx, "split"] = "test"
    return df


def merge_all(out_path: str | None = None) -> str:
    sources = [
        DATA_PROCESSED / "newsclippings_rows.csv",
        DATA_PROCESSED / "dgm4_rows.csv",
        DATA_PROCESSED / "mmfakebench_rows.csv",
    ]
    dfs = []
    for p in sources:
        if not p.exists():
            log.warning("source manifest missing: %s (skipping)", p)
            continue
        df = pd.read_csv(p, dtype={"sample_id": str, "source": str, "split": str})
        log.info("loaded %s: %d rows", p.name, len(df))
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("no per-source manifests found. Run build-manifest first.")

    combined = pd.concat(dfs, ignore_index=True)
    log.info("combined: %d rows", len(combined))

    undersampled = _undersample(combined, TARGET_PER_CLASS, SEED)
    log.info("undersampled to %d/class: %d rows", TARGET_PER_CLASS, len(undersampled))

    split_df = _stratified_split(undersampled, SEED)
    log.info("splits: %s", split_df.groupby(["label", "split"]).size().to_dict())

    validate(split_df)
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    ensure_dir(out_path.parent)
    split_df[list(REQUIRED_COLUMNS)].to_csv(out_path, index=False, encoding="utf-8")
    return str(out_path)

"""Leak-free Phase 2 dataset.

Key principle: respect Phase 1's train/val/test split for OOC samples so
that v1_ooc has never seen our Phase 2 test set during its training.

  Phase 2 OOC TEST  = Phase 1 OOC TEST  (900 — v1_ooc never trained on these)
  Phase 2 OOC VAL   = Phase 1 OOC VAL   (900)
  Phase 2 OOC TRAIN = Phase 1 OOC TRAIN (4,200 — leaked into v1_ooc's training,
                                         but we don't evaluate on training)

Real (label 0): random sample from NewsCLIPpings bank, EXCLUDING any sample
v1_ooc saw in its Real (scenario 4) training.

Manipulated (label 1): DGM4 face_* — never used by v1_ooc, no constraint.
"""

import json
import os
import random
from pathlib import Path

import pandas as pd

SEED = 42
LABEL_REAL, LABEL_MANIP, LABEL_OOC = 0, 1, 2
LABEL_NAMES = {0: "Real", 1: "Manipulated", 2: "OOC"}


def load_dgm4_manipulated(dgm4_root):
    dgm4_root = Path(dgm4_root)
    rows, seen = [], set()
    for split in ("train", "val", "test"):
        ann = dgm4_root / "metadata" / f"{split}.json"
        if not ann.exists():
            continue
        with open(ann) as f:
            data = json.load(f)
        for item in data:
            fake_cls = item.get("fake_cls", "orig")
            if "face" not in fake_cls:
                continue
            rel = item.get("image", "")
            if rel.startswith("DGM4/"):
                rel = rel[len("DGM4/"):]
            image_path = str(dgm4_root / rel)
            if not os.path.exists(image_path):
                continue
            sid = f"dgm4_{item.get('id')}"
            if sid in seen:
                continue
            seen.add(sid)
            rows.append({
                "sample_id": sid,
                "text": item.get("text", ""),
                "image_path": image_path,
                "label": LABEL_MANIP,
                "source": f"DGM4_{fake_cls}",
            })
    return pd.DataFrame(rows)


def load_phase1_ooc_with_splits(phase1_csv):
    """Load Phase 1 OOC samples preserving their original split labels.
    These splits define what v1_ooc trained / validated / tested on."""
    df = pd.read_csv(phase1_csv)
    ooc = df[df["scenario"] == 1][[
        "sample_id", "text", "image_path", "split"
    ]].copy()
    ooc = ooc[ooc["image_path"].apply(os.path.exists)]
    ooc["label"] = LABEL_OOC
    ooc["source"] = "NewsCLIPpings_OOC"
    # Rename so we don't confuse it with Phase 2 split (we'll reuse it as-is).
    ooc = ooc.rename(columns={"split": "split_p1"})
    return ooc


def load_v1ooc_real_image_paths(phase1_csv):
    """Image paths v1_ooc saw as Real (scenario 4) during ANY of its splits.
    We exclude these from Phase 2 Real candidates to avoid memorized features
    leaking the Real label."""
    df = pd.read_csv(phase1_csv)
    real = df[df["scenario"] == 4]
    return set(real["image_path"].apply(os.path.abspath))


def load_newsclippings_real_random(bank_path, exclude_paths):
    """Random NewsCLIPpings genuine pairings, skipping any path that was in
    v1_ooc's Real training pool."""
    bank_root = Path(bank_path).parent
    with open(bank_path) as f:
        bank = json.load(f)
    rows = []
    for art_id, info in bank.items():
        rel = info.get("image_path")
        if not rel:
            continue
        image_path = str(bank_root / rel)
        if not os.path.exists(image_path):
            continue
        if os.path.abspath(image_path) in exclude_paths:
            continue
        rows.append({
            "sample_id": f"real_nc_{art_id}",
            "text": info.get("caption", ""),
            "image_path": image_path,
            "label": LABEL_REAL,
            "source": "NewsCLIPpings_genuine",
        })
    return pd.DataFrame(rows)


def main():
    out_path = "data/processed/forensic_3class_clean.csv"
    n_per_class = 6000

    print("Loading DGM4 manipulated (no v1_ooc overlap, just split 70/15/15)...")
    dgm4 = load_dgm4_manipulated("data/raw/DGM4")
    print(f"  available: {len(dgm4)}")

    print("Loading Phase 1 OOC with original splits (these preserve v1_ooc partition)...")
    ooc = load_phase1_ooc_with_splits("data/processed/balanced_dataset.csv")
    print(f"  available: {len(ooc)}  splits: {ooc['split_p1'].value_counts().to_dict()}")

    print("Loading v1_ooc's Real image pool (to exclude from Phase 2 Real)...")
    v1_real = load_v1ooc_real_image_paths("data/processed/balanced_dataset.csv")
    print(f"  v1_ooc Real images to exclude: {len(v1_real)}")

    print("Loading Phase 2 Real (NewsCLIPpings random, excluding v1_ooc-seen)...")
    real = load_newsclippings_real_random(
        "data/raw/NewsCLIPpings/test_dataset/visual_news_test.json",
        exclude_paths=v1_real,
    )
    print(f"  Real candidates available: {len(real)}")

    rng = random.Random(SEED)

    # --- Manipulated: balance to n_per_class, random 70/15/15 split ---
    dgm4_b = dgm4.sample(n=n_per_class, random_state=SEED).reset_index(drop=True)
    idx = list(range(len(dgm4_b)))
    rng.shuffle(idx)
    n_tr = int(round(0.70 * n_per_class))
    n_va = int(round(0.15 * n_per_class))
    train_ids = set(idx[:n_tr])
    val_ids = set(idx[n_tr:n_tr + n_va])
    dgm4_b["split"] = dgm4_b.index.map(
        lambda i: "train" if i in train_ids else "val" if i in val_ids else "test"
    )
    dgm4_b = dgm4_b.drop(columns=[c for c in dgm4_b.columns if c == "split_p1"], errors="ignore")

    # --- OOC: respect Phase 1 splits ---
    ooc_b = ooc.copy()
    ooc_b["split"] = ooc_b["split_p1"]
    ooc_b = ooc_b.drop(columns=["split_p1"])
    print("\nOOC split (using Phase 1 splits → leak-free for v1_ooc test):")
    print(ooc_b["split"].value_counts())

    # --- Real: balance to n_per_class, random 70/15/15 split ---
    real_b = real.sample(n=n_per_class, random_state=SEED).reset_index(drop=True)
    idx = list(range(len(real_b)))
    rng.shuffle(idx)
    train_ids = set(idx[:n_tr])
    val_ids = set(idx[n_tr:n_tr + n_va])
    real_b["split"] = real_b.index.map(
        lambda i: "train" if i in train_ids else "val" if i in val_ids else "test"
    )

    out = pd.concat([dgm4_b, real_b, ooc_b], ignore_index=True)
    out = out[["sample_id", "text", "image_path", "label", "source", "split"]]
    print("\nFinal split layout:")
    print(out.groupby(["label", "split"]).size().unstack(fill_value=0))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} samples to {out_path}")


if __name__ == "__main__":
    main()

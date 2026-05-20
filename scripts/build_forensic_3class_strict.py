"""Strict-PDF V2 dataset construction (no methodology fixes).

Per PDF §2:
  Real          = NewsCLIPpings genuine pairings (random sample from bank)
  Manipulated   = DGM4 face_swap / face_attribute
  OOC           = NewsCLIPpings out-of-context pairings (existing pool)

NO image alignment between Real and OOC — Real is sampled at random from
the NewsCLIPpings bank. (We already audited that this lets the forensic
encoder use a JPEG-quality shortcut. Kept here per user instruction to
follow the PDF literally.)
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


def load_newsclippings_real_random(bank_path):
    """Per PDF §2: Real samples from NewsCLIPpings. Randomly sampled from
    the genuine-pair bank — NOT aligned with OOC images."""
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
        rows.append({
            "sample_id": f"real_nc_{art_id}",
            "text": info.get("caption", ""),
            "image_path": image_path,
            "label": LABEL_REAL,
            "source": "NewsCLIPpings_genuine",
        })
    return pd.DataFrame(rows)


def load_newsclippings_ooc(existing_csv):
    df = pd.read_csv(existing_csv)
    ooc = df[df["scenario"] == 1][["sample_id", "text", "image_path"]].copy()
    ooc = ooc[ooc["image_path"].apply(os.path.exists)]
    ooc["label"] = LABEL_OOC
    ooc["source"] = "NewsCLIPpings_OOC"
    return ooc


def per_class_split(df, train_frac=0.70, val_frac=0.15, seed=SEED):
    rng = random.Random(seed)
    out = []
    for label, sub in df.groupby("label"):
        idx = sub.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        train_ids = set(idx[:n_tr])
        val_ids = set(idx[n_tr:n_tr + n_va])
        sub2 = sub.copy()
        sub2["split"] = sub2.index.map(
            lambda i: "train" if i in train_ids
                      else "val" if i in val_ids
                      else "test"
        )
        out.append(sub2)
    return pd.concat(out, ignore_index=True)


def main():
    out_path = "data/processed/forensic_3class_strict.csv"
    n_per_class = 6000

    print("Loading DGM4 manipulated...")
    dgm4 = load_dgm4_manipulated("data/raw/DGM4")
    print(f"  available: {len(dgm4)}")

    print("Loading NewsCLIPpings genuine (Real) — random sample...")
    real = load_newsclippings_real_random(
        "data/raw/NewsCLIPpings/test_dataset/visual_news_test.json")
    print(f"  available: {len(real)}")

    print("Loading NewsCLIPpings OOC...")
    ooc = load_newsclippings_ooc("data/processed/balanced_dataset.csv")
    print(f"  available: {len(ooc)}")

    full = pd.concat([dgm4, real, ooc], ignore_index=True)
    counts = full["label"].value_counts().to_dict()
    print("\nClass counts before balancing:")
    for k in sorted(counts):
        print(f"  {LABEL_NAMES[k]:12s} = {counts[k]}")

    min_count = min(min(counts.values()), n_per_class)
    print(f"\nBalancing to {min_count} per class")

    balanced = []
    for k in sorted(counts):
        sub = full[full["label"] == k].sample(n=min_count, random_state=SEED)
        balanced.append(sub)
    balanced = pd.concat(balanced, ignore_index=True)

    balanced = per_class_split(balanced)
    print("\nFinal split layout:")
    print(balanced.groupby(["label", "split"]).size().unstack(fill_value=0))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out_path, index=False)
    print(f"\nWrote {len(balanced)} samples to {out_path}")


if __name__ == "__main__":
    main()

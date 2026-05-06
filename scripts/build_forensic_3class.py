"""Build 3-class forensic dataset (Real / Manipulated / OOC).

Per-class definitions (matches Implementation Guidelines V2 §2):
  0 = Real          DGM4 origin (real news, real images)
                    + NewsCLIPpings genuine when DGM4 origin runs short
  1 = Manipulated   DGM4 face_swap / face_attribute (image-side manipulation)
  2 = OOC           NewsCLIPpings out-of-context pairs

Real and Manipulated come from the SAME source distribution (DGM4) so the
model can't shortcut by recognising dataset artifacts. We sample to the
smallest class size, then split 70/15/15 per class so every split is balanced.

Output: data/processed/forensic_3class.csv with columns
sample_id, text, image_path, label, source, split.
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

SEED = 42
LABEL_REAL, LABEL_MANIP, LABEL_OOC = 0, 1, 2
LABEL_NAMES = {0: "Real", 1: "Manipulated", 2: "OOC"}


def load_dgm4(dgm4_root):
    """Load all DGM4 samples across train/val/test, tagging each as Real or
    Manipulated based on fake_cls. Skips missing files on disk so the
    downstream pipeline doesn't choke."""
    dgm4_root = Path(dgm4_root)
    rows = []
    seen = set()
    for split in ("train", "val", "test"):
        ann = dgm4_root / "metadata" / f"{split}.json"
        if not ann.exists():
            continue
        with open(ann) as f:
            data = json.load(f)
        for item in data:
            fake_cls = item.get("fake_cls", "orig")
            image_rel = item.get("image", "")
            if image_rel.startswith("DGM4/"):
                image_rel = image_rel[len("DGM4/"):]
            image_path = str(dgm4_root / image_rel)
            if not os.path.exists(image_path):
                continue
            sid = f"dgm4_{item.get('id')}"
            if sid in seen:
                continue
            seen.add(sid)
            if fake_cls == "orig":
                label = LABEL_REAL
                src = "DGM4_origin"
            elif "face" in fake_cls:
                label = LABEL_MANIP
                src = f"DGM4_{fake_cls}"
            else:
                # text-only manipulations don't fit the forensic baseline
                continue
            rows.append({
                "sample_id": sid,
                "text": item.get("text", ""),
                "image_path": image_path,
                "label": label,
                "source": src,
            })
    return pd.DataFrame(rows)


def load_newsclippings_ooc_from_existing(csv_path):
    """The existing balanced_dataset.csv already contains NewsCLIPpings OOC
    pairs (scenario 1). Reuse those — they were validated when the original
    pipeline was built."""
    df = pd.read_csv(csv_path)
    ooc = df[df["scenario"] == 1][["sample_id", "text", "image_path"]].copy()
    ooc = ooc[ooc["image_path"].apply(os.path.exists)]
    ooc["label"] = LABEL_OOC
    ooc["source"] = "NewsCLIPpings_OOC"
    return ooc


def per_class_split(df, train_frac=0.70, val_frac=0.15, seed=SEED):
    """70/15/15 stratified split applied independently per class."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--dgm4", default="data/raw/DGM4")
    p.add_argument("--existing-csv", default="data/processed/balanced_dataset.csv")
    p.add_argument("--output", default="data/processed/forensic_3class.csv")
    args = p.parse_args()

    print("Loading DGM4 (origin + face manipulations, only existing files)...")
    dgm4 = load_dgm4(args.dgm4)
    by_label = dgm4.groupby("label").size().to_dict()
    print(f"  Real        (DGM4 origin):       {by_label.get(LABEL_REAL, 0)}")
    print(f"  Manipulated (DGM4 face_*):       {by_label.get(LABEL_MANIP, 0)}")

    print("Loading NewsCLIPpings OOC...")
    ooc = load_newsclippings_ooc_from_existing(args.existing_csv)
    print(f"  OOC         (NewsCLIPpings):     {len(ooc)}")

    full = pd.concat([dgm4, ooc], ignore_index=True)
    counts = full["label"].value_counts().to_dict()
    print("\nClass counts before balancing:")
    for k in sorted(counts):
        print(f"  {LABEL_NAMES[k]:12s} = {counts[k]}")

    min_count = min(counts.values())
    print(f"\nBalancing all classes to {min_count} samples")

    balanced = []
    for k in sorted(counts):
        sub = full[full["label"] == k].sample(n=min_count, random_state=SEED)
        balanced.append(sub)
    balanced = pd.concat(balanced, ignore_index=True)

    print("\nApplying per-class 70/15/15 split...")
    balanced = per_class_split(balanced)
    print(balanced.groupby(["label", "split"]).size().unstack(fill_value=0))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(args.output, index=False)
    print(f"\nWrote {len(balanced)} samples to {args.output}")


if __name__ == "__main__":
    main()

"""V2 dataset rebuild — strictly follows V2 PDF §2 sourcing:

  Real          = NewsCLIPpings genuine pairings (visual_news_test.json)
  Manipulated   = DGM4 face_swap / face_attribute (image-side manipulation)
  OOC           = NewsCLIPpings out-of-context pairings (existing balanced CSV)

This makes Real and OOC share the same image-pipeline distribution, so the
forensic encoder cannot shortcut between them on JPEG signature. Manipulated
remains DGM4-only because it's our only manipulation source.

Balanced 6,000 per class, 70/15/15 per-class split, seed 42.
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
    """DGM4 image-side manipulations only. Skip text-only and missing files."""
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
            if "face" not in fake_cls:
                continue
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
            rows.append({
                "sample_id": sid,
                "text": item.get("text", ""),
                "image_path": image_path,
                "label": LABEL_MANIP,
                "source": f"DGM4_{fake_cls}",
            })
    return pd.DataFrame(rows)


def load_newsclippings_real(bank_path, ooc_image_paths, n_target):
    """Real class — for each OOC sample's image, pair it with its TRUE caption
    from the bank. Guarantees Real and OOC use the *same image distribution*
    so the forensic encoder can't shortcut on file-size / JPEG signature.

    If we run out of OOC images, top up with random genuine pairings from the
    bank.
    """
    bank_root = Path(bank_path).parent
    with open(bank_path) as f:
        bank = json.load(f)

    # Build path → (article_id, caption) lookup.
    path_to_info = {}
    for art_id, info in bank.items():
        rel = info.get("image_path")
        if not rel:
            continue
        abs_p = str(bank_root / rel)
        path_to_info[os.path.abspath(abs_p)] = (art_id, info.get("caption", ""))

    rows = []
    used_paths = set()
    for ooc_path in ooc_image_paths:
        ap = os.path.abspath(ooc_path)
        if ap in used_paths or ap not in path_to_info:
            continue
        art_id, caption = path_to_info[ap]
        rows.append({
            "sample_id": f"real_nc_{art_id}",
            "text": caption,
            "image_path": ooc_path,
            "label": LABEL_REAL,
            "source": "NewsCLIPpings_genuine",
        })
        used_paths.add(ap)
    # We deliberately DO NOT top up with non-OOC bank entries — keeping Real
    # exactly aligned with OOC's image distribution so the forensic encoder
    # cannot use file-size / JPEG signature as a shortcut between the two.
    return pd.DataFrame(rows)


def load_newsclippings_ooc(existing_csv):
    """Reuse the OOC pairings already in the existing balanced CSV. They were
    validated against NewsCLIPpings annotations during the original build."""
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
    out_path = "data/processed/forensic_3class_v2.csv"
    n_per_class = 6000

    print("Loading DGM4 manipulated...")
    dgm4 = load_dgm4_manipulated("data/raw/DGM4")
    print(f"  DGM4 manipulated available: {len(dgm4)}")

    print("Loading NewsCLIPpings OOC (defines image distribution)...")
    ooc = load_newsclippings_ooc("data/processed/balanced_dataset.csv")
    print(f"  NewsCLIPpings OOC available: {len(ooc)}")

    print("Loading NewsCLIPpings genuine (Real) — using OOC images first...")
    real = load_newsclippings_real(
        "data/raw/NewsCLIPpings/test_dataset/visual_news_test.json",
        ooc_image_paths=ooc["image_path"].tolist(),
        n_target=n_per_class,
    )
    print(f"  NewsCLIPpings Real available: {len(real)}")

    full = pd.concat([dgm4, real, ooc], ignore_index=True)
    counts = full["label"].value_counts().to_dict()
    print("\nClass counts before balancing:")
    for k in sorted(counts):
        print(f"  {LABEL_NAMES[k]:12s} = {counts[k]}")

    min_count = min(min(counts.values()), n_per_class)
    print(f"\nBalancing all classes to {min_count} samples")

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

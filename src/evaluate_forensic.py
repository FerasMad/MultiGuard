"""Evaluate the Step 1 forensic baseline on the test split AND on MMFakeBench
(zero-shot transfer, no fine-tuning) per Implementation Guidelines V2 §12.

Reports: accuracy, per-class precision/recall/f1, macro-F1, AUC-ROC (one-vs-rest),
confusion matrix. Saves a YAML metrics file plus a ROC curve PNG per evaluation.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.forensic_baseline import ForensicBaseline
from train_forensic import DCTDataset, collate, path_hash, pick_device

LABEL_NAMES = {0: "Real", 1: "Manipulated", 2: "OOC"}


def map_mmfakebench_to_3class(fake_cls: str, gt_answers: str) -> int:
    """Map MMFakeBench labels into our 3-class label space.

    Actual MMFakeBench v2 release uses these `fake_cls` values:
        original                       (real text + real image, aligned)
        textual_veracity_distortion    (fake text, real image)
        visual_veracity_distortion     (real text, fake image)
        mismatch                       (real text + real image, mismatched
                                        = "out of context")

    Older docs / intermediate releases used 'cross_modal_inconsistency'
    instead of 'mismatch' — both accepted here.

    Mapping:
        original                          -> Real (0)
        contains 'visual'                 -> Manipulated (1)
        contains 'mismatch' / 'cross_modal' / 'ooc'
                                          -> OOC (2)
        contains 'textual' (but no visual / mismatch / cross_modal)
                                          -> drop (-1)
        unknown / empty                   -> drop (-1)
    """
    if not isinstance(fake_cls, str):
        # Per existing behavior: missing fake_cls is treated as 'original'.
        return 0
    fc = fake_cls.strip().lower()
    if fc == "original":
        return 0
    if fc == "":
        return -1
    if "visual" in fc:
        return 1
    if "mismatch" in fc or "cross_modal" in fc or "ooc" in fc:
        return 2
    if "textual" in fc:
        return -1
    return -1


def load_mmfakebench(root, split="val"):
    """Build a (image_path, label) DataFrame for MMFakeBench transfer eval.

    Supports two on-disk layouts:

    1. **Raw HF download** (current default):
           <root>/MMFakeBench_<split>.json
           <root>/MMFakeBench_<split>/<...image files...>
                                       (extracted from MMFakeBench_<split>.zip)

    2. **Legacy HF datasets cache**:
           <root>/liuxuannan___mm_fake_bench/MMFakeBench_<split>/
                  <version>/<hash>/dataset.arrow

    The raw JSON layout is preferred; the .arrow path is kept as a
    fallback so older local caches still work without re-downloading.

    Args:
        root: filesystem path to the MMFakeBench root
            (typically `data/raw/MMFakeBench`).
        split: one of `{"val", "test"}`.

    Returns:
        DataFrame with columns
        `sample_id, image_path, label, fake_cls, text`.
        Rows whose `fake_cls` doesn't fit the V2 3-class taxonomy
        (e.g. textual-only fakes) are dropped.

    Raises:
        FileNotFoundError: if neither layout is present under `root`.
    """
    root = Path(root)
    json_path = root / f"MMFakeBench_{split}.json"
    if json_path.exists():
        return _load_mmfakebench_json(root, split, json_path)
    return _load_mmfakebench_arrow(root, split)


def _load_mmfakebench_json(root: Path, split: str, json_path: Path) -> pd.DataFrame:
    """Read the raw HF download (JSON next to extracted images folder)."""
    with open(json_path, encoding="utf-8") as f:
        items = json.load(f)
    images_root = root / f"MMFakeBench_{split}"

    rows = []
    for idx, item in enumerate(items):
        rel = item["image_path"].lstrip("/")
        img = images_root / rel
        label = map_mmfakebench_to_3class(
            item.get("fake_cls"), item.get("gt_answers"),
        )
        if label < 0:
            continue
        rows.append({
            "sample_id": f"mmfb_{split}_{idx}",
            "image_path": str(img),
            "label": label,
            "fake_cls": item.get("fake_cls", "original"),
            "text": item.get("text", ""),
        })
    return pd.DataFrame(rows)


def _load_mmfakebench_arrow(root: Path, split: str) -> pd.DataFrame:
    """Legacy: read from the HuggingFace `datasets` on-disk cache."""
    import pyarrow as pa

    hf_path = root / "liuxuannan___mm_fake_bench" / f"MMFakeBench_{split}"
    if not hf_path.exists():
        raise FileNotFoundError(
            f"No MMFakeBench data found under {root}. Expected either:\n"
            f"  {root}/MMFakeBench_{split}.json + extracted "
            f"MMFakeBench_{split}/  (raw HF download — preferred), or\n"
            f"  {root}/liuxuannan___mm_fake_bench/MMFakeBench_{split}/"
            f"<v>/<hash>/*.arrow  (legacy datasets cache)."
        )
    arrow_file = None
    for v in hf_path.iterdir():
        if v.is_dir():
            for h in v.iterdir():
                if h.is_dir():
                    for f in h.iterdir():
                        if f.suffix == ".arrow":
                            arrow_file = f
                            break
                if arrow_file:
                    break
        if arrow_file:
            break
    if arrow_file is None:
        raise FileNotFoundError(f"No .arrow file under {hf_path}")
    table = pa.ipc.open_stream(open(arrow_file, "rb")).read_all()
    items = table.to_pylist()
    images_root = root / "images" / f"MMFakeBench_{split}"
    rows = []
    for idx, item in enumerate(items):
        rel = item["image_path"].lstrip("/")
        img = images_root / rel
        label = map_mmfakebench_to_3class(
            item.get("fake_cls"), item.get("gt_answers"),
        )
        if label < 0:
            continue
        rows.append({
            "sample_id": f"mmfb_{split}_{idx}",
            "image_path": str(img),
            "label": label,
            "fake_cls": item.get("fake_cls", "original"),
            "text": item.get("text", ""),
        })
    return pd.DataFrame(rows)


def compute_dct_runtime(image_path: str):
    """Fallback DCT computation when no cache is available (used for MMFakeBench
    where the precompute pipeline hasn't run)."""
    import cv2
    from scipy.fftpack import dct as _dct
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return torch.zeros(1, 224, 224)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)
    coeffs = _dct(_dct(y, axis=0, norm="ortho"), axis=1, norm="ortho")
    log_mag = np.log(np.abs(coeffs) + 1e-8)
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-12:
        return torch.zeros(1, 224, 224)
    normed = (log_mag - lo) / (hi - lo)
    return torch.from_numpy(normed).unsqueeze(0).float()


class DCTRuntimeDataset(torch.utils.data.Dataset):
    """For MMFakeBench: compute DCT on the fly because we may not have cached it."""

    def __init__(self, df, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.cache = Path(cache_dir) if cache_dir else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        dct = None
        if self.cache:
            pt = self.cache / f"{path_hash(path)}.pt"
            if pt.exists():
                dct = torch.load(pt, weights_only=False).float()
                if dct.dim() == 2:
                    dct = dct.unsqueeze(0)
        if dct is None:
            dct = compute_dct_runtime(path)
        return {
            "dct": dct,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def metrics_block(y_true, y_pred, y_proba, labels=(0, 1, 2)):
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro",
                                     labels=list(labels), zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0)
    for i, lab in enumerate(labels):
        out[f"{LABEL_NAMES[lab].lower()}_precision"] = float(p[i])
        out[f"{LABEL_NAMES[lab].lower()}_recall"] = float(r[i])
        out[f"{LABEL_NAMES[lab].lower()}_f1"] = float(f[i])
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    out["confusion_matrix"] = cm.tolist()
    # AUC-ROC (one-vs-rest macro)
    try:
        present = sorted(set(int(x) for x in y_true))
        ohe = np.zeros((len(y_true), len(labels)))
        for i, t in enumerate(y_true):
            ohe[i, int(t)] = 1.0
        if len(present) > 1:
            out["auc_roc_macro"] = float(roc_auc_score(
                ohe, y_proba, average="macro", multi_class="ovr"))
        else:
            out["auc_roc_macro"] = float("nan")
    except Exception as e:
        out["auc_roc_macro"] = float("nan")
        out["auc_roc_error"] = str(e)
    return out


def save_roc_curves(y_true, y_proba, out_path, labels=(0, 1, 2),
                    title="ROC (one-vs-rest)"):
    """Save a one-vs-rest ROC curve PNG.

    `title` is the plot heading. Callers should pass something descriptive
    of the model and split being evaluated (e.g. "Step 2 / Test split"
    or "Step 1 / MMFakeBench transfer").
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(6, 5))
    for i, lab in enumerate(labels):
        y_bin = (np.array(y_true) == lab).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        try:
            auc = roc_auc_score(y_bin, y_proba[:, i])
        except Exception:
            auc = float("nan")
        plt.plot(fpr, tpr, label=f"{LABEL_NAMES[lab]} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="eval"):
        dct = batch["dct"].to(device)
        labels = batch["label"]
        logits = model(dct)["logits"].cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    proba = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    return labels, preds, proba


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", default="data/processed/forensic_3class.csv")
    p.add_argument("--cache-dir", default="data/processed/dct_cache")
    p.add_argument("--mmfakebench", default="data/raw/MMFakeBench")
    p.add_argument("--mmfb-split", default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="outputs/forensic_baseline")
    p.add_argument("--skip-mmfb", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = ForensicBaseline(
        num_classes=3,
        pretrained=False,  # weights come from checkpoint
        feat_dim=cfg.get("model", {}).get("feat_dim", 768),
        dropout=cfg.get("model", {}).get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ---- Test split ----
    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"]
    test_set = DCTDataset(test_df, args.cache_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate)
    print(f"\nTest set: {len(test_set)} samples")
    y, pr, pb = run_eval(model, test_loader, device)
    test_m = metrics_block(y, pr, pb)
    print("\n=== Test split ===")
    for k, v in test_m.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    save_roc_curves(y, pb, out_dir / "test_roc.png",
                    title="Step 1 / Test split (one-vs-rest)")
    with open(out_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump(test_m, f)

    # ---- MMFakeBench transfer ----
    if not args.skip_mmfb:
        print(f"\nLoading MMFakeBench {args.mmfb_split}...")
        try:
            mmfb = load_mmfakebench(args.mmfakebench, split=args.mmfb_split)
        except Exception as e:
            print(f"  Failed to load MMFakeBench: {e}")
            mmfb = None
        if mmfb is not None and len(mmfb) > 0:
            mmfb = mmfb[mmfb["image_path"].apply(os.path.exists)]
            print(f"  Usable samples: {len(mmfb)} "
                  f"(class dist: {mmfb['label'].value_counts().to_dict()})")
            mmfb_set = DCTRuntimeDataset(mmfb, cache_dir=args.cache_dir)
            mmfb_loader = DataLoader(mmfb_set, batch_size=args.batch_size, shuffle=False,
                                     num_workers=2, collate_fn=collate)
            y, pr, pb = run_eval(model, mmfb_loader, device)
            mmfb_m = metrics_block(y, pr, pb)
            print("\n=== MMFakeBench transfer (no fine-tuning) ===")
            for k, v in mmfb_m.items():
                if isinstance(v, list):
                    print(f"  {k}: {v}")
                else:
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            save_roc_curves(y, pb, out_dir / "mmfb_roc.png",
                            title=f"Step 1 / MMFakeBench {args.mmfb_split} "
                                  "(one-vs-rest, zero-shot)")
            with open(out_dir / "mmfb_metrics.yaml", "w") as f:
                yaml.safe_dump(mmfb_m, f)

    print(f"\nAll metrics + ROC curves written to {out_dir}/")


if __name__ == "__main__":
    main()

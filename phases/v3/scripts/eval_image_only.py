"""Image-only evaluation for V3.

Uses the V3 aux head (Linear(768->2)) which was trained with BCE on the binary
target "image is AI/tampered" (class in {2 Manipulated, 4 Fully-Fab}).
That's the cleanest pure-image metric we have without retraining.

Pipeline per image:
    raw image  ->  patch-DCT [1,1,224,224]
              ->  UnivFD encoder           v_imgfor [1,768]
              ->  V3 aux_classifier        logits   [1,2]
              ->  sigmoid                   P(fake_image)

Reports binary metrics (real-image vs ai/tampered-image) on the 5-class
test split. Also prints per-source-class breakdown so you can see *which*
fake-image flavours the image side catches.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# IMPORTANT: import torch BEFORE cv2/scipy to avoid Windows DLL conflict
# that silently kills the process (observed: exit code 5, no traceback).
import numpy as np
import torch
import cv2
# scipy.fftpack.dct segfaults on Windows when scipy's pocketfft DLL conflicts
# with torch CUDA libraries (observed: access violation in _r2r). Use cv2.dct
# instead — different DLL, same orthonormal type-II DCT.
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.univfd_encoder import UnivFDEncoder  # noqa: E402
from models.v3_pipeline import V3FusionModule    # noqa: E402

UNIVFD_CKPT = ROOT / "outputs" / "univfd_genimage" / "best.pt"
V3_CKPT = ROOT / "v3" / "outputs" / "v3_pipeline_qwen" / "best.pt"
TEST_CSV = ROOT / "data" / "processed" / "forensic_5class_full.csv"
OUT_DIR = ROOT / "outputs" / "image_only_eval"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DCT_SIZE = 224
DCT_PATCH = 8
JPEG_QUALITY = 85
BATCH = 32

CLASS_NAMES = {0: "Real", 1: "Out-of-Context", 2: "Manipulated",
               3: "AI-Text", 4: "Fully-Fabricated"}


def _patch_dct(y: np.ndarray, patch_size: int = DCT_PATCH) -> np.ndarray:
    h, w = y.shape
    out = np.empty_like(y, dtype=np.float32)
    y32 = y.astype(np.float32)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            block = y32[i:i + patch_size, j:j + patch_size]
            coeffs = cv2.dct(block)
            out[i:i + patch_size, j:j + patch_size] = np.log(
                np.abs(coeffs) + 1e-8)
    return out


def compute_patch_dct(image_path: Path) -> torch.Tensor | None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if ok:
        d = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if d is not None:
            img = d
    img = cv2.resize(img, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)
    log_mag = _patch_dct(y)
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-12:
        normed = np.zeros_like(log_mag)
    else:
        normed = (log_mag - lo) / (hi - lo)
    return torch.from_numpy(normed).unsqueeze(0).float()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Test CSV: {TEST_CSV}", flush=True)

    t0 = time.time()
    univfd = UnivFDEncoder(out_dim=768, pretrained=None, dropout=0.3)
    ck = torch.load(UNIVFD_CKPT, map_location="cpu", weights_only=False)
    st = ck.get("model_state", ck)
    enc_st = {k[len("encoder."):]: v for k, v in st.items()
              if k.startswith("encoder.")}
    univfd.load_state_dict(enc_st, strict=False)
    univfd = univfd.to(DEVICE).eval()
    for p in univfd.parameters():
        p.requires_grad = False
    print(f"  UnivFD loaded ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    v3_ck = torch.load(V3_CKPT, map_location=DEVICE, weights_only=False)
    mcfg = v3_ck.get("config", {}).get("model", {})
    v3 = V3FusionModule(
        feat_dim=int(mcfg.get("feat_dim", 768)),
        fused_dim=int(mcfg.get("fused_dim", 1024)),
        num_classes=int(mcfg.get("num_classes", 5)),
        num_heads=int(mcfg.get("num_heads", 8)),
        attn_dropout=float(mcfg.get("attn_dropout", 0.1)),
    ).to(DEVICE)
    v3.load_state_dict(v3_ck["model_state"])
    v3.eval()
    aux = v3.aux_classifier.to(DEVICE).eval()
    print(f"  V3 aux head extracted ({time.time() - t0:.1f}s)", flush=True)

    with open(TEST_CSV, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == "test"]
    print(f"  test rows: {len(rows)}", flush=True)

    all_probs: list[float] = []
    all_bin_y: list[int] = []
    all_lbl: list[int] = []
    all_src: list[str] = []
    n_done = n_fail = 0
    t0 = time.time()

    batch_imgs: list[torch.Tensor] = []
    batch_meta: list[tuple[int, str]] = []

    def flush_batch():
        nonlocal n_done
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            v = univfd(x)
            logits = aux(v)
            p_fake = torch.sigmoid(logits[:, 1]).cpu().tolist()
        for (lbl, src), p in zip(batch_meta, p_fake):
            all_probs.append(p)
            all_bin_y.append(1 if lbl in (2, 4) else 0)
            all_lbl.append(lbl)
            all_src.append(src)
            n_done += 1
        batch_imgs.clear()
        batch_meta.clear()

    for idx, r in enumerate(rows):
        src_path = Path(r["image_path"])
        if not src_path.is_absolute():
            src_path = ROOT / src_path
        d = compute_patch_dct(src_path)
        if d is None:
            n_fail += 1
            continue
        batch_imgs.append(d)
        batch_meta.append((int(r["label"]), r.get("source", "")))
        if len(batch_imgs) >= BATCH:
            flush_batch()
            if n_done % 1000 < BATCH:
                rate = n_done / max(time.time() - t0, 1)
                eta = (len(rows) - idx) / max(rate, 1)
                print(f"  {n_done}/{len(rows)}  rate={rate:.0f}/s  "
                      f"eta={eta:.0f}s  fail={n_fail}", flush=True)
    flush_batch()
    print(f"\nProcessed {n_done} (failed {n_fail}) in "
          f"{time.time() - t0:.1f}s", flush=True)

    probs = np.array(all_probs)
    y_true = np.array(all_bin_y)
    y_pred = (probs > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred,
                                   target_names=["real-image", "fake-image"],
                                   digits=4)

    print("\n" + "=" * 60)
    print("BINARY IMAGE-ONLY METRICS")
    print("=" * 60)
    print(f"Accuracy            : {acc:.4f}")
    print(f"F1 (fake-image)     : {f1:.4f}")
    print(f"F1-macro            : {f1_macro:.4f}")
    print(f"Precision (fake)    : {prec:.4f}")
    print(f"Recall (fake)       : {rec:.4f}")
    print(f"AUC-ROC             : {auc:.4f}")
    print(f"Confusion matrix    : {cm}")
    print(f"  rows = ground truth (0=real, 1=fake)")
    print(f"  cols = prediction  (0=real, 1=fake)")
    print()
    print(report)

    print("=" * 60)
    print("PER-ORIGINAL-CLASS BREAKDOWN")
    print("=" * 60)
    lbl_arr = np.array(all_lbl)
    breakdown = {}
    for c in sorted(set(all_lbl)):
        mask = lbl_arr == c
        n = int(mask.sum())
        true_bin = 1 if c in (2, 4) else 0
        mean_p = float(probs[mask].mean())
        pred = (probs[mask] > 0.5).astype(int)
        pct_pred_fake = float((pred == 1).mean())
        acc_slice = float((pred == true_bin).mean())
        breakdown[c] = {
            "name": CLASS_NAMES[c],
            "n": n,
            "image_is_fake_ground_truth": bool(true_bin),
            "mean_p_fake": mean_p,
            "pct_predicted_fake": pct_pred_fake,
            "accuracy_on_slice": acc_slice,
        }
        print(f"  {c} {CLASS_NAMES[c]:<17}  n={n:>5}  "
              f"true_fake={true_bin}  mean_P(fake)={mean_p:.3f}  "
              f"pred_fake={pct_pred_fake*100:5.1f}%  "
              f"acc={acc_slice:.3f}")

    out_json = {
        "device": DEVICE,
        "n_samples": int(n_done),
        "n_failed": int(n_fail),
        "binary_metrics": {
            "accuracy": float(acc),
            "f1_fake": float(f1),
            "f1_macro": float(f1_macro),
            "precision_fake": float(prec),
            "recall_fake": float(rec),
            "auc_roc": float(auc),
            "confusion_matrix": cm,
        },
        "per_original_class": breakdown,
    }
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(out_json, indent=2), encoding="utf-8")
    (OUT_DIR / "classification_report.txt").write_text(report,
                                                       encoding="utf-8")
    with (OUT_DIR / "predictions.csv").open("w", encoding="utf-8",
                                            newline="") as f:
        w = csv.writer(f)
        w.writerow(["original_label", "binary_label", "P_fake_image",
                    "predicted_binary", "source"])
        for lbl, bin_y, p, src in zip(all_lbl, all_bin_y, all_probs,
                                       all_src):
            w.writerow([lbl, bin_y, f"{p:.4f}", int(p > 0.5), src])

    print(f"\nWrote: {OUT_DIR/'metrics.json'}")
    print(f"       {OUT_DIR/'classification_report.txt'}")
    print(f"       {OUT_DIR/'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

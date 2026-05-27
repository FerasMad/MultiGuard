"""Stage A4 audit -- image-only transfer probe on existing DCT ckpt (P14.A4).

Required by `forensic_image_branch_fix_plan.md` Step 8 "Image-only transfer
before full-pipeline integration".

Loads the existing forensic_dct_model.pth and runs it on the test split of
forensic_5class_unified_blip2.csv, mapped to binary labels:
  - classes 0, 1, 3 (real image) -> 0
  - classes 2, 4 (fake image)    -> 1

Reports binary AP, AUC, accuracy@0.5, fake-recall@0.5, fake-precision@0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, accuracy_score,
                             roc_auc_score, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phases/forensic/src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("data/processed/forensic_5class_unified_blip2.csv"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("phases/forensic/outputs/dct/forensic_dct_model.pth"))
    p.add_argument("--stats", type=Path,
                   default=Path("phases/forensic/data/dct_stats.json"))
    p.add_argument("--out", type=Path, default=Path("docs/IMAGE_TRANSFER_PROBE.md"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from forensic.preprocessing.dual_dct import compute_dual_dct
    from forensic.models.dct_resnet50 import build_dct_resnet50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[A4] device={device}")

    df = pd.read_csv(args.manifest)
    test_df = df[df["split"] == "test"].copy()
    test_df["binary_label"] = test_df["label"].map({0: 0, 1: 0, 3: 0, 2: 1, 4: 1})
    print(f"[A4] test rows: {len(test_df)}")
    print(f"  binary label balance: {test_df['binary_label'].value_counts().to_dict()}")

    if args.limit is not None:
        test_df = test_df.head(args.limit)

    print(f"[A4] loading DCT model from {args.ckpt} ...")
    model = build_dct_resnet50()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    dct_mean = float(stats["mean"])
    dct_std = float(stats["std"])
    print(f"[A4] dct_mean={dct_mean:.4f}, dct_std={dct_std:.4f}")

    from PIL import Image
    all_probs = []
    all_labels = []
    all_sources = []
    failed = 0
    t0 = time.time()
    for i, row in enumerate(test_df.itertuples(index=False)):
        try:
            img = Image.open(row.image_path).convert("RGB")
            t = compute_dual_dct(img)
            t = (t - dct_mean) / (dct_std + 1e-8)
            t = t.unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(t).squeeze(-1)
                prob = torch.sigmoid(logit).item()
            all_probs.append(prob)
            all_labels.append(int(row.binary_label))
            all_sources.append(row.source)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  FAIL row={i} sid={row.sample_id}: {type(e).__name__}: {e}")
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  progress: {i+1}/{len(test_df)} ({elapsed:.0f}s, {(i+1)/elapsed:.1f}/s)")

    total_elapsed = time.time() - t0
    print(f"\n[A4] eval done in {total_elapsed:.1f}s ({len(all_probs)} scored, {failed} failed)")

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)

    ap = float(average_precision_score(labels, probs))
    auc = float(roc_auc_score(labels, probs))
    acc = float(accuracy_score(labels, preds))
    fake_prec = float(precision_score(labels, preds, pos_label=1, zero_division=0))
    fake_recall = float(recall_score(labels, preds, pos_label=1, zero_division=0))

    src_results = {}
    for src in sorted(set(all_sources)):
        idx = [i for i, s in enumerate(all_sources) if s == src]
        if not idx:
            continue
        slabels = labels[idx]
        sprobs = probs[idx]
        if len(set(slabels.tolist())) < 2:
            src_results[src] = {"n": len(idx), "ap": None, "auc": None,
                                "all_label": int(slabels[0]),
                                "mean_prob": float(sprobs.mean())}
        else:
            src_results[src] = {
                "n": len(idx),
                "ap": float(average_precision_score(slabels, sprobs)),
                "auc": float(roc_auc_score(slabels, sprobs)),
                "mean_prob": float(sprobs.mean()),
            }

    if auc < 0.55:
        verdict = ("**Image branch DOES NOT transfer to the full pipeline's image distribution.** "
                   "AUC near 0.5 = random. Stage C data refresh is required if the image branch "
                   "is expected to contribute to the 5-class downstream task.")
    elif auc < 0.75:
        verdict = ("**Image branch transfers weakly.** Provides some signal but not enough to be "
                   "the primary detector for classes 2/4.")
    else:
        verdict = ("**Image branch transfers well.** Useful contribution to classes 2/4 expected.")

    lines = [
        "# Image-only transfer probe (Stage A4)",
        "",
        "> Generated by `phases/v4/scripts/audit_image_transfer_probe.py`",
        "",
        "## Question",
        "",
        "Does the existing `forensic_dct_model.pth` (Approach 2, val AP 0.9848 on GenImage)",
        "actually transfer to the full pipeline's image distribution (DGM4 manipulated + ",
        "MMFakeBench AI-image + NewsCLIPpings real)?",
        "",
        "If AUC near 0.5 -> image branch contributes near-zero to downstream 5-class.",
        "If AUC > 0.75  -> image branch is a meaningful signal.",
        "",
        "## Method",
        "",
        f"- Ckpt: `{args.ckpt}` (Approach 2 DCT)",
        f"- DCT stats: `{args.stats}` (mean={dct_mean:.4f}, std={dct_std:.4f})",
        f"- Test set: {len(test_df)} rows of `{args.manifest}` split=test",
        "- Binary mapping: classes 0/1/3 -> real (0); classes 2/4 -> fake (1)",
        f"- Device: {device}",
        f"- Wallclock: {total_elapsed:.1f}s, {failed} failed",
        "",
        "## Overall test metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| AP (binary fake) | **{ap:.4f}** |",
        f"| AUC | **{auc:.4f}** |",
        f"| Accuracy @ 0.5 | {acc:.4f} |",
        f"| Fake precision @ 0.5 | {fake_prec:.4f} |",
        f"| Fake recall @ 0.5 | {fake_recall:.4f} |",
        "",
        "## Per-source breakdown",
        "",
        "| Source | n | Binary label | AP | AUC | Mean P(fake) |",
        "|---|---|---|---|---|---|",
    ]
    for src, r in src_results.items():
        if "all_label" in r:
            lines.append(f"| {src} | {r['n']} | only {r['all_label']} | n/a (single class) | n/a | {r['mean_prob']:.3f} |")
        else:
            lines.append(f"| {src} | {r['n']} | mixed | {r['ap']:.3f} | {r['auc']:.3f} | {r['mean_prob']:.3f} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append("## Cross-reference with shipped GenImage per-generator eval")
    lines.append("")
    lines.append("- Approach 2 DCT GenImage in-distribution AP: **0.9863** (6/8 gens)")
    lines.append(f"- Approach 2 DCT pipeline-image transfer AP (this audit): **{ap:.4f}**")
    lines.append("")
    lines.append(f"Delta: **{0.9863 - ap:+.4f}**. ")
    lines.append("Large positive delta = severe distribution shift between GenImage train and the")
    lines.append("downstream pipeline's image mix.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Test rows scored: {len(all_probs)} of {len(test_df)} (failed: {failed})")
    import hashlib as _hashlib
    lines.append(f"- Ckpt sha256 prefix: `{_hashlib.sha256(args.ckpt.read_bytes()).hexdigest()[:16]}...`")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[A4] wrote {args.out}")

    return {"ap": ap, "auc": auc, "accuracy": acc, "fake_recall": fake_recall,
            "fake_precision": fake_prec, "verdict": verdict}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

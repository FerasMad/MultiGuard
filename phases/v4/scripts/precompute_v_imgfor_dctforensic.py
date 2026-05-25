"""Precompute v_imgfor cache for V4 using the dct_forensic_v1 encoder.

Reads:
  - data/processed/forensic_5class_c4fix.csv (V3-era V4-compatible 5-class manifest)
  - data/raw/visualnews/origin/... and other image paths in image_path column
  - phases/forensic/data/dct_stats.json (z-score)
  - phases/forensic/outputs/dct/forensic_dct_model.pth (Approach 2 trained ckpt)

Writes:
  - cache/v4/v_imgfor_dctforensic/{sample_id}.pt   (16500 shards, 768-d float32)

For each sample, applies the doctor's dual-DCT preprocessing (8x8 + 16x16
patches, scipy.fft.dctn, log, average, z-score), then forwards through the
DctForensicEncoder backbone -> 2048 -> Linear(2048, 768)+GELU -> v_imgfor.

~30 min on RTX 4070 for 16500 samples.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from forensic.data.dct_dataset import load_dct_stats
from forensic.preprocessing.dual_dct import compute_dual_dct
from v4.core.registry import ENCODER_REGISTRY, import_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/forensic_5class_c4fix.csv"),
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("phases/forensic/outputs/dct/forensic_dct_model.pth"),
    )
    p.add_argument("--stats", type=Path, default=Path("phases/forensic/data/dct_stats.json"))
    p.add_argument("--out-dir", type=Path, default=Path("cache/v4/v_imgfor_dctforensic"))
    p.add_argument("--limit", type=int, default=None, help="Smoke: cap samples")
    p.add_argument("--out-dim", type=int, default=768)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[precompute] device: {device}")
    print(f"[precompute] out_dir: {args.out_dir}")
    print(f"[precompute] ckpt: {args.ckpt}")

    # Build the V4 forensic encoder with the trained Approach 2 weights
    import_all()
    EncCls = ENCODER_REGISTRY["dct_forensic_v1"]
    enc = EncCls(out_dim=args.out_dim, ckpt=args.ckpt, freeze_backbone=False)
    enc.to(device).eval()
    print(f"[precompute] encoder built: dct_forensic_v1, out_dim={args.out_dim}")

    mean, std = load_dct_stats(args.stats)
    print(f"[precompute] dct_stats: mean={mean:.4f} std={std:.4f}")

    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"[precompute] manifest: {len(df)} rows")

    n_done = 0
    n_skipped = 0
    n_fail = 0
    t0 = time.time()
    last_log = t0

    with torch.no_grad():
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            out_path = args.out_dir / f"{sid}.pt"
            if out_path.exists():
                n_skipped += 1
                continue

            img_path = Path(row["image_path"])
            if not img_path.exists():
                n_fail += 1
                if n_fail <= 5:
                    print(f"  MISSING {img_path}")
                continue

            try:
                t = compute_dual_dct(str(img_path))  # [1, 224, 224] float32
                t = (t - mean) / (std + 1e-8)
                t = t.unsqueeze(0).to(device)  # [1, 1, 224, 224]
                v_imgfor = enc({"v_imgfor_dct": t}).squeeze(0).cpu()  # [out_dim]
                torch.save(v_imgfor, out_path)
                n_done += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  FAIL {sid}: {type(e).__name__}: {e}")

            now = time.time()
            if now - last_log > 10.0:
                elapsed = now - t0
                rate = (n_done + n_skipped) / max(elapsed, 1e-3)
                pct = 100.0 * (n_done + n_skipped + n_fail) / len(df)
                print(
                    f"  progress: {n_done + n_skipped + n_fail}/{len(df)} "
                    f"({pct:.1f}%)  rate={rate:.1f}/s  fails={n_fail}  "
                    f"skipped={n_skipped}"
                )
                last_log = now

    elapsed = time.time() - t0
    print(f"\n[precompute] DONE in {elapsed:.1f}s")
    print(f"  written : {n_done}")
    print(f"  skipped : {n_skipped} (already on disk)")
    print(f"  failed  : {n_fail}")
    print(f"  rate    : {(n_done + n_skipped) / max(elapsed, 1e-3):.1f}/s")


if __name__ == "__main__":
    main()

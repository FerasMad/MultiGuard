"""Precompute DCT maps once and cache to disk (Implementation Guidelines V2 §4).

Pipeline per image:
  1. Load with OpenCV (BGR)
  2. Resize to exactly 224 x 224
  3. BGR -> YCbCr, take Y channel
  4. 2D DCT
  5. Log scale: log(|x| + 1e-8)
  6. Per-image min-max normalize to [0, 1]
  7. Save tensor of shape [1, 224, 224] as .pt

Cached files are keyed by an md5 hash of the absolute image path so multiple
splits sharing the same image don't recompute. The training Dataset just
loads the .pt for each row, no DCT during the loop.
"""

import argparse
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.fftpack import dct
from tqdm import tqdm

DCT_SIZE = 224


def path_hash(image_path: str) -> str:
    return hashlib.md5(os.path.abspath(image_path).encode("utf-8")).hexdigest()


def compute_dct(image_path: str) -> torch.Tensor:
    """Compute DCT map for a single image. Returns [1, 224, 224] float32."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None for {image_path}")
    img = cv2.resize(img, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # OpenCV uses YCrCb
    y = ycbcr[:, :, 0].astype(np.float32)
    # 2D DCT (type-II, normalized) — apply along both axes
    coeffs = dct(dct(y, axis=0, norm="ortho"), axis=1, norm="ortho")
    log_mag = np.log(np.abs(coeffs) + 1e-8)
    # Per-image min-max normalize
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-12:
        normed = np.zeros_like(log_mag)
    else:
        normed = (log_mag - lo) / (hi - lo)
    tensor = torch.from_numpy(normed).unsqueeze(0).float()  # [1, 224, 224]
    return tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/forensic_3class.csv")
    p.add_argument("--cache-dir", default="data/processed/dct_cache")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N images (debug).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (>1 currently unused, sequential is fine).")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Deduplicate by image path so we don't recompute repeated images
    paths = sorted(set(df["image_path"].tolist()))
    print(f"Unique images to process: {len(paths)}")

    n_skip, n_done, n_fail = 0, 0, 0
    pbar = tqdm(paths, desc="DCT")
    for path in pbar:
        h = path_hash(path)
        out = cache / f"{h}.pt"
        if out.exists():
            n_skip += 1
            continue
        try:
            t = compute_dct(path)
            torch.save(t, out)
            n_done += 1
        except Exception as e:
            n_fail += 1
            if n_fail < 5:
                tqdm.write(f"FAIL {path}: {e}")
        pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail)

    print(f"\nDone. computed={n_done}  cached={n_skip}  failed={n_fail}")
    print(f"Cache dir: {cache}")


if __name__ == "__main__":
    main()

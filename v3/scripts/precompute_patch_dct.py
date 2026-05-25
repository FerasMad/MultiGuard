"""Student 1 — patch-wise DCT preprocessor.

Per-image pipeline:
  0. Load with OpenCV (BGR).
  1. Re-encode at uniform JPEG quality 85 to neutralise source-
     distribution shortcuts (same fix V2 used; disable with
     --jpeg-quality 0).
  2. Resize to exactly 224 x 224.
  3. BGR -> YCbCr, take Y channel.
  4. Tile into non-overlapping `patch_size` x `patch_size` patches.
     patch_size=8 -> 28 x 28 = 784 patches.
  5. For each patch: 2D-DCT (orthonormal). log(|DCT| + 1e-8).
  6. Reassemble.
  7. Per-image min-max normalize to [0, 1].
  8. Save as `[1, 224, 224]` float32 .pt under
     `cache/v_imgfor_dct/{sample_id}.pt`.

Input to UnivFDEncoder (`v3/src/models/univfd_encoder.py`).
Cache keyed by `sample_id` (not a path hash), so the same id
maps to the same .pt in every feature directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.fftpack import dct
from tqdm import tqdm

IMG_SIZE = 224
DEFAULT_PATCH = 8
DEFAULT_JPEG_QUALITY = 85


def normalize_jpeg_quality(img_bgr: np.ndarray, quality: int) -> np.ndarray:
    if quality <= 0:
        return img_bgr
    ok, buf = cv2.imencode(".jpg", img_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img_bgr
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img_bgr


def patch_dct(y: np.ndarray, patch_size: int = DEFAULT_PATCH) -> np.ndarray:
    h, w = y.shape
    out = np.empty_like(y, dtype=np.float32)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            block = y[i:i + patch_size, j:j + patch_size]
            coeffs = dct(dct(block, axis=0, norm="ortho"),
                         axis=1, norm="ortho")
            out[i:i + patch_size, j:j + patch_size] = np.log(
                np.abs(coeffs) + 1e-8)
    return out


def compute_patch_dct(image_path: str, patch_size: int = DEFAULT_PATCH,
                      jpeg_quality: int = DEFAULT_JPEG_QUALITY
                      ) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"cv2.imread returned None for {image_path}")
    img = normalize_jpeg_quality(img, jpeg_quality)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                     interpolation=cv2.INTER_AREA)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)
    log_mag = patch_dct(y, patch_size=patch_size)
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-12:
        normed = np.zeros_like(log_mag)
    else:
        normed = (log_mag - lo) / (hi - lo)
    return torch.from_numpy(normed).unsqueeze(0).float()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="manifest with `sample_id` and `image_path` columns")
    p.add_argument("--cache-dir", default="v3/cache/v_imgfor_dct")
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH,
                   choices=[8, 16])
    p.add_argument("--jpeg-quality", type=int,
                   default=DEFAULT_JPEG_QUALITY,
                   help="0 disables JPEG normalization (ablation)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    if "sample_id" not in df.columns or "image_path" not in df.columns:
        print("CSV must have `sample_id` and `image_path` columns.",
              file=sys.stderr)
        return 1

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    n_done, n_skip, n_fail = 0, 0, 0
    pbar = tqdm(df.itertuples(index=False), total=len(df),
                desc=f"patch-DCT ({args.patch_size}x{args.patch_size})")
    for row in pbar:
        sid = str(getattr(row, "sample_id"))
        out = cache / f"{sid}.pt"
        if out.exists():
            n_skip += 1
            continue
        try:
            t = compute_patch_dct(getattr(row, "image_path"),
                                  patch_size=args.patch_size,
                                  jpeg_quality=args.jpeg_quality)
            torch.save(t, out)
            n_done += 1
        except Exception as e:
            n_fail += 1
            if n_fail < 5:
                tqdm.write(f"FAIL {sid}: {e}")
        pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail)

    print(f"\nDone. computed={n_done} cached={n_skip} failed={n_fail}")
    print(f"Cache dir: {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

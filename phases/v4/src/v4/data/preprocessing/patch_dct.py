"""Patch-DCT preprocessing per V3.1 §3.1.

IMPORTANT: Uses cv2.dct, NOT scipy.fftpack.dct. The latter segfaults on Windows
when loaded alongside torch CUDA (observed in V3 — see plan §2 forced deviation
and docs/CONTRIBUTING.md). cv2.dct is the only canonical V4 implementation.

Per V3.1 §3.1:
  1. Patch Division: 224x224 image into 8x8 (or 16x16) non-overlapping patches
  2. Y-Channel: BGR -> YCbCr, isolate Y
  3. 2D-DCT: orthonormal type-II DCT per patch
  4. Log scaling: log(|DCT| + 1e-8)
  5. Normalization: per-image min-max -> [0, 1]
  6. Storage: save as [1, 224, 224] float32 .pt
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# Per V3.1 §3.1
DCT_SIZE = 224
DCT_PATCH = 8           # spec allows 8 or 16; V3 used 8 (proven)
JPEG_QUALITY = 85       # uniform re-encode kills source-distribution shortcut (V2 lesson)


def _normalize_jpeg(img_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
    """Re-encode + re-decode at uniform JPEG quality to normalize source artifacts."""
    if quality <= 0:
        return img_bgr
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img_bgr
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img_bgr


def _patch_dct_y(y: np.ndarray, patch_size: int = DCT_PATCH) -> np.ndarray:
    """Tile-wise 2D-DCT on the Y channel; log-magnitude per patch.

    cv2.dct performs orthonormal type-II DCT (matches scipy norm='ortho').
    """
    h, w = y.shape
    out = np.empty_like(y, dtype=np.float32)
    y32 = y.astype(np.float32)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            block = y32[i:i + patch_size, j:j + patch_size]
            coeffs = cv2.dct(block)
            out[i:i + patch_size, j:j + patch_size] = np.log(np.abs(coeffs) + 1e-8)
    return out


def compute_patch_dct(
    image_path_or_array: str | Path | np.ndarray,
    *,
    patch_size: int = DCT_PATCH,
    jpeg_quality: int = JPEG_QUALITY,
) -> torch.Tensor:
    """Compute the V3.1 §3.1 patch-DCT map for one image.

    Args:
        image_path_or_array: either a path on disk, or a BGR numpy array (already loaded)
        patch_size: 8 (default per V3) or 16 per spec
        jpeg_quality: re-encode quality (set 0 to disable)

    Returns:
        torch.Tensor float32 of shape [1, 224, 224], values in [0, 1].
    """
    # Load
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cv2.imread returned None for {image_path_or_array}")
    else:
        img = image_path_or_array
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected BGR HxWx3 array, got shape {img.shape}")

    # Per V2/V3 lesson — uniform JPEG before any DCT extraction
    img = _normalize_jpeg(img, jpeg_quality)

    # Resize to 224x224
    img = cv2.resize(img, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA)

    # BGR -> YCbCr; isolate Y
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)

    # Patch-DCT + log
    log_mag = _patch_dct_y(y, patch_size=patch_size)

    # Per-image min-max -> [0, 1]
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-12:
        normed = np.zeros_like(log_mag)
    else:
        normed = (log_mag - lo) / (hi - lo)

    return torch.from_numpy(normed).unsqueeze(0).float()  # [1, 224, 224]


def compute_patch_dct_from_pil(pil_img, **kwargs) -> torch.Tensor:
    """Convenience wrapper: PIL.Image -> patch-DCT tensor."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return compute_patch_dct(bgr, **kwargs)

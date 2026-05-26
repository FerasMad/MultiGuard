"""Doctor's dual-patch DCT preprocessing for Approach 2 (forensic image detector).

Mirrored from phases/forensic/src/forensic/preprocessing/dual_dct.py.

Pipeline per image:
    1. Resize -> 224x224 bilinear
    2. RGB -> YCbCr, extract Y channel only, float32 in [0, 255]
    3. Two patch sets (8x8 and 16x16)
    4. Per patch: 2D DCT-II (norm='ortho') via scipy.fft.dctn
    5. log(|coef| + 1e-8)
    6. Reassemble each set to [224, 224]
    7. Element-wise average -> [224, 224]
    8. Add channel dim -> [1, 224, 224]

Z-score normalization is the caller's responsibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn

IMG_SIZE: int = 224
PATCH_A: int = 8
PATCH_B: int = 16
LOG_EPS: float = 1e-8


def _to_y_channel(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    y = np.asarray(img.convert("YCbCr"), dtype=np.float32)[..., 0]
    return y


def _dct_log_set(y: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = y.shape
    n_h = h // patch_size
    n_w = w // patch_size
    patches = y.reshape(n_h, patch_size, n_w, patch_size).transpose(0, 2, 1, 3)
    coefs = dctn(patches, type=2, norm="ortho", axes=(-2, -1))
    log_coefs = np.log(np.abs(coefs) + LOG_EPS)
    out = log_coefs.transpose(0, 2, 1, 3).reshape(h, w)
    return out.astype(np.float32)


def compute_dual_dct(image_source: str | Path | np.ndarray | Image.Image) -> torch.Tensor:
    """Returns: torch.Tensor [1, 224, 224] float32. NOT z-score normalized."""
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source)
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(image_source)
    elif isinstance(image_source, Image.Image):
        img = image_source
    else:
        raise TypeError(
            f"image_source must be str/Path/ndarray/PIL.Image, got {type(image_source)}"
        )
    y = _to_y_channel(img)
    set_a = _dct_log_set(y, PATCH_A)
    set_b = _dct_log_set(y, PATCH_B)
    avg = (set_a + set_b) * 0.5
    return torch.from_numpy(avg).unsqueeze(0).contiguous()


__all__ = ["IMG_SIZE", "LOG_EPS", "PATCH_A", "PATCH_B", "compute_dual_dct"]

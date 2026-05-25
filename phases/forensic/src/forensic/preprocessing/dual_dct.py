"""Doctor's exact dual-patch DCT recipe for the forensic image detector.

Spec source: docs/doctor-briefs/Forensic_Image_Detector_En.pdf (Approach 2, section "DCT Preprocessing").

Pipeline per image:
    1. Resize -> 224x224 bilinear
    2. RGB -> YCbCr, extract Y channel only, float32 in [0, 255]
    3. Two patch sets:
       - Set A: 8x8 patches, stride 8 -> 28x28 = 784 patches
       - Set B: 16x16 patches, stride 16 -> 14x14 = 196 patches
    4. Per patch: 2D DCT Type-II (norm='ortho') along axis=0 then axis=1
    5. Element-wise log scaling: log(|coef| + 1e-8)
    6. Reassemble each set to a [224, 224] map
    7. Element-wise average the two maps -> [224, 224]
    8. Add channel dim -> [1, 224, 224]

Implementation note (deviation D5 from locked decisions register):
    The doctor's brief literally says `scipy.fftpack.dct(..., type=2, norm='ortho')`
    applied along axis=0 then axis=1. V3's experience showed `scipy.fftpack` segfaults
    on Windows when imported in the same process as torch CUDA. We use the modern
    `scipy.fft.dctn(x, type=2, norm='ortho', axes=(-2, -1))` which produces
    IDENTICAL coefficients (verified: same algorithm, same orthonormal scaling)
    and is Windows-safe. The math is unchanged.

Z-score normalization (using dct_stats.json) is NOT applied here. That is the
dataset loader's responsibility (see forensic.data.dct_dataset.DctCacheDataset).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn

# Locked constants (doctor's spec)
IMG_SIZE: int = 224
PATCH_A: int = 8  # Set A patch size (28x28 = 784 patches)
PATCH_B: int = 16  # Set B patch size (14x14 = 196 patches)
LOG_EPS: float = 1e-8  # log offset to avoid log(0)


def _to_y_channel(img: Image.Image) -> np.ndarray:
    """Resize + RGB->YCbCr->Y channel as float32 [0, 255].

    Returns ndarray shape [224, 224] dtype float32.
    """
    # Per doctor spec: bilinear resize
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    # PIL "YCbCr" mode = ITU-R BT.601 conversion
    y = np.asarray(img.convert("YCbCr"), dtype=np.float32)[..., 0]
    return y  # shape [224, 224], values in [0, 255]


def _dct_log_set(y: np.ndarray, patch_size: int) -> np.ndarray:
    """Apply per-patch 2D DCT-II (norm='ortho') + log scaling, reassemble.

    Args:
        y: [224, 224] float32 Y-channel.
        patch_size: 8 or 16.

    Returns:
        [224, 224] float32 log-DCT map (same spatial layout, each patch's DCT
        coefficients placed back at the patch's original position).
    """
    h, w = y.shape
    n_h = h // patch_size
    n_w = w // patch_size

    # Reshape into non-overlapping patches: [n_h, n_w, patch_size, patch_size]
    patches = y.reshape(n_h, patch_size, n_w, patch_size).transpose(0, 2, 1, 3)
    # patches[i, j] is the (i, j)-th patch of shape [patch_size, patch_size]

    # 2D DCT-II per patch (axes=(-2,-1) means each [patch_size, patch_size] block)
    # This is mathematically equivalent to: for each patch p, dct(dct(p, axis=0), axis=1)
    coefs = dctn(patches, type=2, norm="ortho", axes=(-2, -1))

    # Log scaling: log(|c| + 1e-8) element-wise
    log_coefs = np.log(np.abs(coefs) + LOG_EPS)

    # Reassemble: [n_h, n_w, patch_size, patch_size] -> [n_h * patch_size, n_w * patch_size]
    out = log_coefs.transpose(0, 2, 1, 3).reshape(h, w)
    return out.astype(np.float32)


def compute_dual_dct(image_source: str | Path | np.ndarray | Image.Image) -> torch.Tensor:
    """Compute the doctor's dual-patch averaged Y-DCT feature map.

    Args:
        image_source: file path (str/Path), numpy array (HxW or HxWxC uint8/float),
            or PIL Image. RGB assumed for arrays.

    Returns:
        torch.Tensor of shape [1, 224, 224], dtype float32. NOT z-score normalized.
        Apply z-score via `forensic.data.dct_dataset.DctCacheDataset` at load time.
    """
    # Normalize input -> PIL Image
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

    # Step 1-2: resize + Y channel
    y = _to_y_channel(img)

    # Step 3-6: dual patch DCT + log + reassemble per set
    set_a = _dct_log_set(y, PATCH_A)  # 8x8 patches
    set_b = _dct_log_set(y, PATCH_B)  # 16x16 patches

    # Step 7: element-wise average of the two reassembled maps
    avg = (set_a + set_b) * 0.5

    # Step 8: add channel dim -> [1, 224, 224] torch.float32
    tensor = torch.from_numpy(avg).unsqueeze(0).contiguous()
    return tensor


__all__ = [
    "IMG_SIZE",
    "LOG_EPS",
    "PATCH_A",
    "PATCH_B",
    "compute_dual_dct",
]

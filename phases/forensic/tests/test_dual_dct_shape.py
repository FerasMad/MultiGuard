"""Shape + dtype + finite-value assertions for dual_dct.compute_dual_dct.

Doctor's spec demands output [1, 224, 224] float32. Anything else fails this test.
"""
from __future__ import annotations

import numpy as np
import torch

from forensic.preprocessing.dual_dct import (
    IMG_SIZE,
    PATCH_A,
    PATCH_B,
    compute_dual_dct,
)


def _synthetic_rgb(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """Random RGB image as uint8 ndarray."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_constants_match_doctor_spec():
    """Doctor's spec lists patch sizes 8 and 16."""
    assert IMG_SIZE == 224
    assert PATCH_A == 8
    assert PATCH_B == 16


def test_output_shape_and_dtype():
    img = _synthetic_rgb()
    t = compute_dual_dct(img)
    assert isinstance(t, torch.Tensor), f"expected torch.Tensor, got {type(t)}"
    assert t.shape == (1, IMG_SIZE, IMG_SIZE), f"expected (1, 224, 224), got {tuple(t.shape)}"
    assert t.dtype == torch.float32, f"expected float32, got {t.dtype}"


def test_output_is_finite():
    """log(|x| + eps) must never produce inf/NaN."""
    img = _synthetic_rgb()
    t = compute_dual_dct(img)
    assert torch.isfinite(t).all(), "DCT output contains NaN or inf"


def test_output_changes_with_input():
    """Different images produce different DCT maps (sanity)."""
    img_a = _synthetic_rgb(seed=0)
    img_b = _synthetic_rgb(seed=42)
    t_a = compute_dual_dct(img_a)
    t_b = compute_dual_dct(img_b)
    diff = (t_a - t_b).abs().mean().item()
    assert diff > 0.01, f"DCT outputs too similar (mean abs diff = {diff}); check randomness"


def test_deterministic_for_same_input():
    """Same input -> same output, twice."""
    img = _synthetic_rgb(seed=7)
    t1 = compute_dual_dct(img)
    t2 = compute_dual_dct(img)
    assert torch.allclose(t1, t2), "compute_dual_dct is non-deterministic"


def test_grayscale_input_handled():
    """Single-channel input (HxW) should be accepted and produce same shape output."""
    rng = np.random.default_rng(0)
    img_gray = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    t = compute_dual_dct(img_gray)
    assert t.shape == (1, IMG_SIZE, IMG_SIZE)
    assert t.dtype == torch.float32

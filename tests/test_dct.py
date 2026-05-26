"""Tests for the DCT preprocessing in scripts/precompute_dct.py.

Per V2 §4 the DCT pipeline must produce a [1, 224, 224] float32 tensor
with values in [0, 1] for any 3-channel BGR input. These tests verify
the contract on synthetic images so refactoring doesn't silently break
the dataset-side invariants the trainer assumes.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "phases" / "v2" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from precompute_dct import compute_dct, path_hash  # noqa: E402


def _save_synthetic_image(tmp_path: Path, h: int, w: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), img)
    return p


def test_compute_dct_output_shape_and_dtype(tmp_path: Path):
    p = _save_synthetic_image(tmp_path, 300, 400)
    out = compute_dct(str(p))
    assert out.shape == (1, 224, 224)
    assert out.dtype == torch.float32


def test_compute_dct_values_in_unit_interval(tmp_path: Path):
    p = _save_synthetic_image(tmp_path, 400, 600)
    out = compute_dct(str(p))
    # Per-image min/max normalization guarantees [0, 1].
    assert float(out.min()) >= 0.0 - 1e-6
    assert float(out.max()) <= 1.0 + 1e-6


def test_compute_dct_handles_arbitrary_input_size(tmp_path: Path):
    """Pipeline resizes any input to 224 x 224 before DCT."""
    for shape in [(100, 100), (480, 640), (50, 1000)]:
        p = _save_synthetic_image(tmp_path / f"img_{shape[0]}_{shape[1]}",
                                   *shape)
        # Need a fresh path each time
    # Run on a few sizes and assert all give 224x224 outputs.
    for shape in [(100, 100), (480, 640)]:
        sub = tmp_path / f"sub_{shape[0]}_{shape[1]}"
        sub.mkdir(exist_ok=True)
        p = _save_synthetic_image(sub, *shape)
        out = compute_dct(str(p))
        assert out.shape == (1, 224, 224)


def test_compute_dct_constant_image_no_nan(tmp_path: Path):
    """Constant image -> degenerate DCT range; output must still be finite."""
    img = np.full((224, 224, 3), 127, dtype=np.uint8)
    p = tmp_path / "flat.png"
    cv2.imwrite(str(p), img)
    out = compute_dct(str(p))
    assert torch.isfinite(out).all()


def test_compute_dct_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        compute_dct(str(tmp_path / "does_not_exist.png"))


def test_path_hash_deterministic():
    """The cache key must be stable across calls for the same input."""
    p = "/some/abs/path/img.jpg"
    h1 = path_hash(p)
    h2 = path_hash(p)
    assert h1 == h2
    # Different paths -> different hashes.
    h3 = path_hash("/some/abs/path/other.jpg")
    assert h1 != h3

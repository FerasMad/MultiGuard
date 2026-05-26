"""Tests for metrics + MMFakeBench label mapping in src/evaluate_forensic.py.

PDF references:
  - §12  Evaluation: accuracy, per-class P/R, F1-macro, AUC-ROC, CM
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "phases" / "v2" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate_forensic import (  # noqa: E402
    LABEL_NAMES,
    map_mmfakebench_to_3class,
    metrics_block,
)


# metrics_block


def _make_proba_from_preds(y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Synthesize a probabilistic matrix that argmaxes to y_pred."""
    p = np.full((len(y_pred), n_classes), 0.05, dtype=np.float64)
    p[np.arange(len(y_pred)), y_pred] = 0.9
    p /= p.sum(axis=1, keepdims=True)
    return p


def test_metrics_block_returns_pdf_required_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])  # one wrong
    proba = _make_proba_from_preds(y_pred)
    m = metrics_block(y_true, y_pred, proba)

    # PDF §12 metrics:
    assert "accuracy" in m
    assert "f1_macro" in m
    for cls in LABEL_NAMES.values():
        cl = cls.lower()
        assert f"{cl}_precision" in m
        assert f"{cl}_recall" in m
        assert f"{cl}_f1" in m
    assert "confusion_matrix" in m
    assert "auc_roc_macro" in m


def test_metrics_block_accuracy_correct():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    proba = _make_proba_from_preds(y_pred)
    m = metrics_block(y_true, y_pred, proba)
    assert m["accuracy"] == pytest.approx(5 / 6)


def test_metrics_block_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    proba = _make_proba_from_preds(y_pred)
    m = metrics_block(y_true, y_pred, proba)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)


def test_metrics_block_confusion_matrix_3x3_sum_to_n():
    y_true = np.array([0, 1, 2, 0, 1, 2, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    proba = _make_proba_from_preds(y_pred)
    m = metrics_block(y_true, y_pred, proba)
    cm = m["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)
    assert sum(sum(row) for row in cm) == len(y_true)


def test_metrics_block_auc_in_unit_interval():
    np.random.seed(0)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 0, 2])
    proba = _make_proba_from_preds(y_pred)
    m = metrics_block(y_true, y_pred, proba)
    auc = m["auc_roc_macro"]
    assert 0.0 <= auc <= 1.0


def test_metrics_block_single_class_present_returns_nan_auc():
    """One-vs-rest AUC requires at least 2 classes present in y_true."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    proba = _make_proba_from_preds(y_pred)
    m = metrics_block(y_true, y_pred, proba)
    assert np.isnan(m["auc_roc_macro"])


# MMFakeBench label mapping


def test_mmfb_original_is_real():
    assert map_mmfakebench_to_3class("original", "True") == 0


def test_mmfb_visual_distortion_is_manipulated():
    assert map_mmfakebench_to_3class("visual_veracity_distortion", "Fake") == 1


def test_mmfb_textual_plus_visual_is_manipulated():
    """Combinations containing 'visual' map to manipulated, regardless of textual."""
    assert map_mmfakebench_to_3class("textual+visual", "Fake") == 1


def test_mmfb_cross_modal_is_ooc():
    assert map_mmfakebench_to_3class("cross_modal_inconsistency", "Fake") == 2


def test_mmfb_mismatch_is_ooc():
    """The actual MMFakeBench v2 release uses 'mismatch' for OOC samples;
    older docs called it 'cross_modal_inconsistency'. Both must map to OOC."""
    assert map_mmfakebench_to_3class("mismatch", "Fake") == 2
    assert map_mmfakebench_to_3class("Mismatch", "Fake") == 2  # case-insensitive


def test_mmfb_textual_only_is_dropped():
    """Text-only fakes don't fit our 3 classes -> drop signal (-1)."""
    assert map_mmfakebench_to_3class("textual_veracity_distortion", "Fake") == -1


def test_mmfb_unknown_is_dropped():
    assert map_mmfakebench_to_3class("", "") == -1
    # Non-string fake_cls falls back to "original" defensively
    assert map_mmfakebench_to_3class(None, "True") == 0

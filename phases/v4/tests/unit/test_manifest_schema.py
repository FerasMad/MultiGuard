"""Manifest schema invariants."""

from __future__ import annotations

import pandas as pd
import pytest

from v4.data.manifest import REQUIRED_COLUMNS, SPLITS, ManifestRow, validate


def test_required_columns():
    assert REQUIRED_COLUMNS == (
        "sample_id",
        "text",
        "image_path",
        "label",
        "source",
        "split",
    )


def test_validate_rejects_bad_split():
    df = pd.DataFrame(
        [
            {
                "sample_id": "x1",
                "text": "t",
                "image_path": "/a/b.jpg",
                "label": 0,
                "source": "test",
                "split": "INVALID",
            }
        ]
    )
    with pytest.raises(ValueError, match="unknown split"):
        validate(df)


def test_validate_rejects_duplicate_ids():
    rows = [
        ManifestRow("x1", "t", "/a/b.jpg", 0, "test", "train").as_dict(),
        ManifestRow("x1", "t", "/a/b.jpg", 0, "test", "train").as_dict(),
    ]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="duplicate"):
        validate(df)


def test_validate_rejects_bad_label():
    df = pd.DataFrame(
        [
            {
                "sample_id": "x1",
                "text": "t",
                "image_path": "/a/b.jpg",
                "label": 99,
                "source": "test",
                "split": "train",
            }
        ]
    )
    with pytest.raises(ValueError, match="label"):
        validate(df)


def test_splits_constant():
    assert SPLITS == ("train", "val", "test")

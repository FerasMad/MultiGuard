"""Class map invariants: 5 classes, correct binary mapping for aux head."""
from __future__ import annotations

from v4.core.class_map import (
    IMAGE_FAKE_CLASSES, LABELS, NUM_CLASSES,
    binary_image_label, label_to_name, name_to_label,
)


def test_five_classes():
    assert NUM_CLASSES == 5
    assert set(LABELS) == {0, 1, 2, 3, 4}


def test_binary_image_label():
    # V3.1 5.5: 1 iff image side is AI/tampered (Manipulated or Fully-Fabricated)
    assert binary_image_label(0) == 0   # Real
    assert binary_image_label(1) == 0   # OOC (image is real)
    assert binary_image_label(2) == 1   # Manipulated
    assert binary_image_label(3) == 0   # AI-Text (image is real)
    assert binary_image_label(4) == 1   # Fully-Fabricated


def test_image_fake_classes():
    assert IMAGE_FAKE_CLASSES == frozenset({2, 4})


def test_label_round_trip():
    for i in range(5):
        name = label_to_name(i)
        assert name_to_label(name) == i

"""5-class label map for the V3.1 task taxonomy.

Centralizes the label_index ↔ name mapping so future students who want a 6th
class or a 3-class collapse only edit this file. Also defines which class
indices count as "fake image" for the binary aux head per V3.1 §5.5.
"""

from __future__ import annotations

# Per V3.1 §2 table:
LABELS: dict[int, str] = {
    0: "Real",
    1: "Out-of-Context",
    2: "Manipulated",
    3: "AI-Text",
    4: "Fully-Fabricated",  # also called "Double-Fake" in the spec
}

NAMES: dict[str, int] = {v: k for k, v in LABELS.items()}

NUM_CLASSES = len(LABELS)

# Arabic translations for the bilingual UI (carried over from V3 server)
LABELS_AR: dict[int, str] = {
    0: "حقيقي",
    1: "خارج السياق",
    2: "معدَّل",
    3: "نص مولَّد",
    4: "ملفَّق بالكامل",
}

# V3.1 §5.5 binary forensic consistency: "1 iff image-side is AI/tampered"
# Class 2 = Manipulated (image tampered) and Class 4 = Double-Fake (AI image)
IMAGE_FAKE_CLASSES: frozenset[int] = frozenset({2, 4})


def binary_image_label(label: int) -> int:
    """Map 5-class label → binary "image is AI/tampered" target for aux head."""
    return 1 if int(label) in IMAGE_FAKE_CLASSES else 0


def label_to_name(label: int) -> str:
    if label not in LABELS:
        raise KeyError(f"Unknown label index {label}; valid: {list(LABELS)}")
    return LABELS[label]


def name_to_label(name: str) -> int:
    if name not in NAMES:
        raise KeyError(f"Unknown label name {name!r}; valid: {list(NAMES)}")
    return NAMES[name]

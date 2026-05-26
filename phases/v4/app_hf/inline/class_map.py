"""5-class label map for the V3.1 task taxonomy.

Mirrors phases/v4/src/v4/core/class_map.py. The display strings here are
the doctor-diagram-friendly variants ("Real news" instead of "Real",
"Real text + Fake image" instead of "Manipulated", etc.) to match the
labels under the MLP classifier in his architecture diagram.
"""

from __future__ import annotations

NUM_CLASSES: int = 5

# Internal label IDs (used by the trained checkpoint)
LABELS: dict[int, str] = {
    0: "Real",
    1: "Out-of-Context",
    2: "Manipulated",
    3: "AI-Text",
    4: "Fully-Fabricated",
}

# Doctor-diagram-friendly display names (what the demo shows the user)
DISPLAY_LABELS: dict[int, str] = {
    0: "Real news",
    1: "Out-of-context",
    2: "Real text + Fake image",
    3: "Fake text + Real image",
    4: "Fully fake",
}

# Helpful one-line explanations for the UI
LABEL_EXPLANATIONS: dict[int, str] = {
    0: "Both the text and the image are genuine.",
    1: "Both are real, but the image doesn't actually depict what the text describes.",
    2: "The text is real reporting, but the image has been tampered with or AI-generated.",
    3: "The image is real, but the text is AI-generated or fabricated.",
    4: "Both the text and the image are fabricated.",
}

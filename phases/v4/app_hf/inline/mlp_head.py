"""MLPClassifier per V3.1 section 6 (Student 3).

Hardcoded per interview lock D3/D6: strict V3.1 section 6 architecture, no
task-flexibility hooks. Future students wanting a different taxonomy fork
this file.

Architecture (V3.1 section 6):
    Layer 1: Linear(1024, 512) + BatchNorm1d(512) + GELU + Dropout(0.5)
    Layer 2: Linear(512, 256) + GELU
    Layer 3: Linear(256, 5)   (raw logits, no softmax)

The Dropout(0.5) is essential per the spec: "to prevent the model from
over-relying on a single modality, forcing it to look at the combined
relational evidence."
"""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """V3.1 section 6 classifier — exact spec, no flexibility hooks."""

    def __init__(self, in_dim: int = 1024, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """fused: [B, in_dim] -> logits: [B, num_classes] (raw, no softmax)."""
        return self.net(fused)

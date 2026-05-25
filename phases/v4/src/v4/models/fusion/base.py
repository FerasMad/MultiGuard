"""FusionBase — the contract every fusion module must satisfy."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn


class FusionBase(nn.Module):
    """Abstract base for V4 fusion modules.

    Class attributes:
        name:             registry key
        expected_inputs:  tuple of feature names the fusion consumes
                          (e.g. ("v_semantic", "v_imgfor", "v_textfor"))

    Instance attribute:
        num_classes:      int — output dimension of main classifier head

    Forward signature:
        forward(features: dict[str, Tensor]) -> dict[str, Tensor]
            Must contain at minimum: 'main_logits' [B, num_classes]
            May also contain:        'aux_logits' [B, 2], 'fused' [B, fused_dim]
    """

    name: ClassVar[str] = "_base"
    expected_inputs: ClassVar[tuple[str, ...]] = ()

    def __init__(self):
        super().__init__()
        self.num_classes: int = 0

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("subclasses must implement forward(features)")

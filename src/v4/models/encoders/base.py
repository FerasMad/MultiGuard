"""EncoderBase — the contract every encoder must satisfy.

See docs/ADD_ENCODER.md for a copy-paste template.
"""
from __future__ import annotations

from typing import ClassVar, Literal

import torch
import torch.nn as nn


class EncoderBase(nn.Module):
    """Abstract base for all V4 encoders.

    Class attributes (set on each subclass):
        name:             registry key (auto-set by @register)
        required_inputs:  tuple of dict keys this encoder consumes from the batch
        modality:         "text" | "image" | "multi"

    Instance attribute:
        output_dim:       int — width of the encoder's output vector

    Forward signature:
        forward(batch: dict[str, Tensor]) -> Tensor  of shape [B, output_dim]

    Subclasses pick the keys they need from `batch` rather than relying on
    positional args. This way a single trainer can drive encoders with
    different input shapes (FND-CLIP wants 6 tensors, UnivFD wants 1).
    """
    name: ClassVar[str] = "_base"
    required_inputs: ClassVar[tuple[str, ...]] = ()
    modality: ClassVar[Literal["text", "image", "multi"]] = "multi"

    def __init__(self):
        super().__init__()
        self.output_dim: int = 0

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("subclasses must implement forward(batch)")

    def freeze(self) -> None:
        """Disable grad on all parameters; set to eval."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def load_legacy_checkpoint(self, path: str) -> None:
        """Optional: subclasses may override to ingest V1/V2/V3 checkpoints."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load_legacy_checkpoint"
        )

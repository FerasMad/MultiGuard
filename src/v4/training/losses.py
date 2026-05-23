"""CompositeLoss - parametrized loss spec for V4 trainer.

Supports Stage 1 (single BCE) and Stage 2 (CE + 0.1 * BCE aux with detach)
via the same loss config schema.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossSpec:
    """One term of a composite loss."""
    type: str            # "ce" | "bce" | "bce_logits"
    target: str          # key in model output dict (e.g. "main_logits", "aux_logits")
    weight: float = 1.0
    detach_input: str | None = None    # for V3.1 section 5.5 aux gradient isolation


class CompositeLoss(nn.Module):
    """Sum of multiple weighted loss terms - the V4 training-loss formulation."""

    def __init__(self, terms: list[dict]):
        super().__init__()
        self.specs = [LossSpec(**t) for t in terms]
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        out: dict[str, torch.Tensor],
        labels: torch.Tensor,
        aux_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss + per-term breakdown for logging.

        Args:
            out: model output dict (main_logits, aux_logits, fused, ...)
            labels: 5-class ground truth [B]
            aux_labels: binary ground truth [B] for aux head (image-is-fake)
        """
        total = torch.zeros(1, device=labels.device, dtype=torch.float32)
        breakdown: dict[str, float] = {}

        for spec in self.specs:
            logits = out[spec.target]
            if spec.type == "ce":
                term = self.ce(logits, labels)
            elif spec.type in {"bce", "bce_logits"}:
                if aux_labels is None:
                    raise ValueError(f"loss term {spec.type} needs aux_labels")
                # Expect logits [B, 2]; convert aux_labels [B] to one-hot 2 floats
                target_oh = F.one_hot(aux_labels, num_classes=logits.shape[-1]).float()
                term = self.bce(logits, target_oh)
            else:
                raise ValueError(f"unknown loss type: {spec.type}")

            total = total + spec.weight * term
            breakdown[f"loss_{spec.target}"] = float(term.detach().item())

        breakdown["loss_total"] = float(total.detach().item())
        return total.squeeze(0), breakdown


def build_loss(terms: list[dict]) -> CompositeLoss:
    """Factory used by trainer."""
    return CompositeLoss(terms)

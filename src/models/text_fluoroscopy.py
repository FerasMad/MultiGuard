"""
Text Fluoroscopy Module — src/models/text_fluoroscopy.py
========================================================
Components for Qwen2-7B Layer-30 text forensic feature extraction.

Two classes:
  - masked_mean_pool(): Used by precompute_text_forensics.py to pool
    hidden states before caching the raw 4096-dim vectors.
  - TextForensicProjection: Linear(4096→768) + GELU. NOT used during
    precomputation — included here for Student 3/4 to import as a
    TRAINABLE layer inside the fusion pipeline (Stage 2).

Why cache at 4096 instead of 768:
  The projection layer starts with random weights and has no training
  signal in isolation. It only becomes meaningful when trained end-to-end
  with the fusion classifier. Caching at 4096 lets the projection learn
  what forensic patterns matter for the 5-class task.
"""

import torch
import torch.nn as nn


class TextForensicProjection(nn.Module):
    """
    Projects Qwen2-7B Layer-30 pooled hidden states from 4096-dim
    to the shared forensic embedding space (768-dim).

    Architecture (Implementation Guidelines V3, §4 step 5):
        Linear(4096, 768) → GELU

    This module is TRAINABLE during Stage 2 fusion training.
    Student 3 includes it in v3_pipeline.py; Student 4 trains it.

    Forward:
        input:  [Batch, 4096]
        output: [Batch, 768]
    """

    def __init__(self, input_dim: int = 4096, output_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool across the sequence dimension, excluding padding tokens.

    The Implementation Guidelines V3 (§4 step 4) explicitly require
    applying attention_mask BEFORE averaging so padding tokens do NOT
    dilute the forensic signal.

    Args:
        hidden_states: [Batch, Sequence, Hidden] from a transformer layer.
        attention_mask: [Batch, Sequence] binary (1 = real, 0 = pad).

    Returns:
        [Batch, Hidden] pooled tensor.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, S, 1]
    summed = (hidden_states * mask).sum(dim=1)                    # [B, H]
    counts = mask.sum(dim=1).clamp(min=1.0)                       # [B, 1]
    return summed / counts

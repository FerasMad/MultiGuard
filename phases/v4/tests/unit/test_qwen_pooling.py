"""Unit tests for the Qwen text-branch masked-mean pooling.

Non-image fix-plan item #3 demanded "tests for masked pooling and output shape":
  - attention_mask is applied BEFORE the mean (padding tokens contribute 0),
  - padding token VALUES never affect the embedding,
  - output is [B, H] (H = 3584 for Qwen2-7B in production),
  - works with left-padding (Qwen uses padding_side="left").

Tests the canonical `v4.data.preprocessing.tokenizers.masked_mean_pool`, which
is the exact function `Qwen2TextEncoder.encode_text` calls at both cache-build
and inference time -- so cache/runtime parity is covered by construction.
"""

from __future__ import annotations

import torch

from v4.data.preprocessing.tokenizers import masked_mean_pool


def test_output_shape_collapses_sequence_dim():
    """[B, S, H] -> [B, H]; feature dim preserved (3584 in production)."""
    B, S, H = 4, 7, 3584
    hidden = torch.randn(B, S, H)
    mask = torch.ones(B, S)
    pooled = masked_mean_pool(hidden, mask)
    assert pooled.shape == (B, H), f"expected [{B}, {H}], got {tuple(pooled.shape)}"


def test_all_real_mask_equals_plain_mean():
    """With an all-ones mask, masked-mean == ordinary mean over the seq dim."""
    hidden = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5)
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, hidden.mean(dim=1), atol=1e-6)


def test_padding_values_do_not_affect_pooling():
    """Padding-position VALUES must be irrelevant: scramble them, result unchanged."""
    B, S, H = 2, 6, 16
    hidden = torch.randn(B, S, H)
    # Mark the last 2 positions of each row as padding.
    mask = torch.ones(B, S)
    mask[:, -2:] = 0.0

    pooled_a = masked_mean_pool(hidden, mask)

    # Replace the padded positions with huge garbage; pooled output must not move.
    hidden_scrambled = hidden.clone()
    hidden_scrambled[:, -2:, :] = 1e6
    pooled_b = masked_mean_pool(hidden_scrambled, mask)

    assert torch.allclose(pooled_a, pooled_b, atol=1e-4), (
        "padding token values leaked into the pooled embedding -- mask is not "
        "applied before the sum"
    )


def test_masked_mean_matches_manual_real_token_average():
    """Pooled vector equals the mean over ONLY the real (mask=1) positions."""
    hidden = torch.randn(1, 5, 8)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])  # 3 real, 2 pad
    pooled = masked_mean_pool(hidden, mask)
    manual = hidden[0, :3, :].mean(dim=0, keepdim=True)
    assert torch.allclose(pooled, manual, atol=1e-6)


def test_left_padding_is_handled():
    """Qwen uses padding_side='left'; pad at the START must be ignored."""
    hidden = torch.randn(1, 5, 8)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])  # left-padded: 2 pad, 3 real
    pooled = masked_mean_pool(hidden, mask)
    manual = hidden[0, 2:, :].mean(dim=0, keepdim=True)
    assert torch.allclose(pooled, manual, atol=1e-6)


def test_empty_mask_does_not_divide_by_zero():
    """Defensive: an all-pad row clamps the divisor to >=1 (no NaN/inf)."""
    hidden = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4)
    pooled = masked_mean_pool(hidden, mask)
    assert torch.isfinite(pooled).all(), "all-padding row produced NaN/inf"

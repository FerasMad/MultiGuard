"""V3.1 section 5.3: directions summed (not concatenated).

Asserts the fusion module sums direction_1 + direction_2 element-wise, not concat.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.spec_compliance
def test_pairwise_sum_not_concat():
    from v4.models.fusion.v3_pairwise import PairwiseCrossAttention

    pair = PairwiseCrossAttention(dim=768, num_heads=8, dropout=0.0)
    x = torch.randn(2, 768)
    y = torch.randn(2, 768)
    out = pair(x, y)
    # Output must be [B, dim] (sum), NOT [B, 2*dim] (concat)
    assert out.shape == (2, 768), f"expected [2, 768] but got {out.shape}"


@pytest.mark.spec_compliance
def test_fusion_output_shapes():
    from v4.models.fusion.v3_pairwise import V3PairwiseFusion

    fusion = V3PairwiseFusion(feat_dim=768, fused_dim=1024, num_classes=5)
    features = {
        "v_semantic": torch.randn(4, 768),
        "v_imgfor": torch.randn(4, 768),
        "v_textfor": torch.randn(4, 768),
    }
    fusion.eval()
    # Run in eval mode so BatchNorm doesn't choke on B=1
    with torch.no_grad():
        out = fusion(features)
    assert out["main_logits"].shape == (4, 5)
    assert out["aux_logits"].shape == (4, 2)
    assert out["fused"].shape == (4, 1024)

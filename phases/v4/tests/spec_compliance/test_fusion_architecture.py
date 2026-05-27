"""Spec-compliance: end-to-end fusion architecture asserts (non-image fix plan section 5).

Four explicit contracts the doctor's plan demands:
  (a) Pairwise outputs are SUMMED, not concatenated
  (b) Conv1d operates over the 3 interaction-channels axis, NOT the feature axis
  (c) `main_logits` are RAW (no softmax applied before CrossEntropyLoss)
  (d) Aux head is `Linear(feat_dim, 2)` and consumes `v_imgfor.detach()`

Plus two consolidating checks:
  (e) BaseEvaluator does NOT consult aux_logits at inference
  (f) The new Stage-B `disable_branches` kwarg is backward-compatible

These are not redundant with `test_v3_1_section_5_3_pairwise_sum.py` -- they consolidate
the architectural invariants the external fix-plan reviewers will look for.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn


@pytest.mark.spec_compliance
def test_a_pairwise_outputs_summed_not_concatenated():
    """V3.1 5.3: bidirectional MHA directions are summed element-wise, not concat."""
    from v4.models.fusion.v3_pairwise import PairwiseCrossAttention

    pair = PairwiseCrossAttention(dim=768, num_heads=8, dropout=0.0)
    x = torch.randn(3, 768)
    y = torch.randn(3, 768)
    out = pair(x, y)
    # SUM => [B, dim]; CONCAT would have been [B, 2*dim]
    assert out.shape == (3, 768), f"expected [3, 768] (sum), got {out.shape}"
    # Symmetry sanity: swapping arguments must NOT permute output dim
    out_swap = pair(y, x)
    assert out_swap.shape == out.shape


@pytest.mark.spec_compliance
def test_b_conv1d_operates_over_interaction_channels_axis():
    """V3.1 5.4: Conv1d is applied across the 3 interaction channels.

    After stacking the 3 relational vectors [r1, r2, r3] (each [B, feat_dim]):
      stacked = stack(..., dim=1)        -> [B, 3, feat_dim]
      permute(0, 2, 1)                   -> [B, feat_dim, 3]
      Conv1d(in_channels=feat_dim, out_channels=feat_dim/fused_dim, ...)

    The Conv1d input channel-axis dim must equal feat_dim (the FEATURE axis sitting
    in the middle), and the length-axis (last) must equal 3 (the interaction count).
    """
    from v4.models.fusion.v3_pairwise import ThreeWayFusion

    feat_dim, fused_dim = 768, 1024
    fusion = ThreeWayFusion(feat_dim=feat_dim, fused_dim=fused_dim, num_heads=8, attn_dropout=0.0)

    # conv1: feat_dim -> feat_dim with k=3, p=1; conv2: feat_dim -> fused_dim with k=1
    assert isinstance(fusion.conv1, nn.Conv1d)
    assert fusion.conv1.in_channels == feat_dim, (
        f"Conv1d in_channels must be feat_dim={feat_dim} (operating over channels axis), "
        f"got {fusion.conv1.in_channels}. If this is 3, the permute is wrong."
    )
    assert fusion.conv1.out_channels == feat_dim
    assert fusion.conv1.kernel_size == (3,)
    assert fusion.conv1.padding == (1,)

    assert isinstance(fusion.conv2, nn.Conv1d)
    assert fusion.conv2.in_channels == feat_dim
    assert fusion.conv2.out_channels == fused_dim
    assert fusion.conv2.kernel_size == (1,)

    # Forward sanity: output is [B, fused_dim] after AdaptiveAvgPool1d(1).squeeze(-1)
    s = torch.randn(2, feat_dim)
    i = torch.randn(2, feat_dim)
    t = torch.randn(2, feat_dim)
    out = fusion(s, i, t)
    assert out.shape == (2, fused_dim), f"expected [2, {fused_dim}] got {out.shape}"


@pytest.mark.spec_compliance
def test_c_main_logits_are_raw_no_softmax():
    """V3.1 5.5 + 6: main_logits must be RAW logits (CE applies softmax internally)."""
    from v4.models.classifier.mlp_head import MLPClassifier
    from v4.models.fusion.v3_pairwise import V3PairwiseFusion

    fusion = V3PairwiseFusion(feat_dim=768, fused_dim=1024, num_classes=5)
    fusion.eval()

    # 1. No Softmax anywhere in the classifier head
    softmax_modules = [
        m for m in fusion.classifier.modules() if isinstance(m, (nn.Softmax, nn.LogSoftmax))
    ]
    assert softmax_modules == [], (
        f"main classifier head must not contain Softmax/LogSoftmax (CE applies internally), "
        f"found: {softmax_modules}"
    )

    # 2. Empirical: logits unbounded + row-sums do not look like a probability simplex
    features = {
        "v_semantic": torch.randn(4, 768) * 5.0,
        "v_imgfor": torch.randn(4, 768) * 5.0,
        "v_textfor": torch.randn(4, 768) * 5.0,
    }
    with torch.no_grad():
        out = fusion(features)
    logits = out["main_logits"]
    sums = logits.sum(dim=-1)
    # If somebody applied softmax, every row would sum to ~1.0 within float tolerance.
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
        "main_logits row-sums look like a probability simplex; softmax was applied "
        "before CE. Logits should be RAW."
    )

    # 3. The classifier itself, in isolation, also passes the no-softmax check
    head = MLPClassifier(in_dim=1024, num_classes=5)
    head_softmaxes = [m for m in head.modules() if isinstance(m, (nn.Softmax, nn.LogSoftmax))]
    assert head_softmaxes == [], f"MLPClassifier must not contain softmax: {head_softmaxes}"


@pytest.mark.spec_compliance
def test_d_aux_head_linear_768_2_consumes_detached_imgfor():
    """V3.1 5.5: aux head is Linear(feat_dim, 2) on v_imgfor.detach()."""
    from v4.models.fusion.v3_pairwise import V3PairwiseFusion

    feat_dim = 768
    fusion = V3PairwiseFusion(feat_dim=feat_dim, fused_dim=1024, num_classes=5)
    fusion.train()

    aux = fusion.aux_classifier
    assert isinstance(aux, nn.Linear), f"aux_classifier must be nn.Linear, got {type(aux)}"
    assert aux.in_features == feat_dim, (
        f"aux_classifier.in_features must be feat_dim={feat_dim}, got {aux.in_features}"
    )
    assert aux.out_features == 2, (
        f"aux_classifier.out_features must be 2 (binary), got {aux.out_features}"
    )

    # Empirical detach check: aux_loss.backward() must NOT populate v_imgfor.grad
    v_semantic = torch.randn(2, 768, requires_grad=True)
    v_imgfor = torch.randn(2, 768, requires_grad=True)
    v_textfor = torch.randn(2, 768, requires_grad=True)
    features = {"v_semantic": v_semantic, "v_imgfor": v_imgfor, "v_textfor": v_textfor}
    out = fusion(features)
    out["aux_logits"].sum().backward()
    assert v_imgfor.grad is None, (
        "aux loss should NOT propagate into v_imgfor (it must be detached). "
        f"Found v_imgfor.grad = {v_imgfor.grad}"
    )


@pytest.mark.spec_compliance
def test_e_aux_head_disabled_at_inference():
    """V3.1 5.5 + section 7: BaseEvaluator must only consume main_logits."""
    import inspect

    from v4.evaluation.evaluator import BaseEvaluator

    src = inspect.getsource(BaseEvaluator.predict)
    assert "main_logits" in src, "BaseEvaluator.predict must read 'main_logits'"
    assert "aux_logits" not in src, (
        "BaseEvaluator.predict must NOT read 'aux_logits' (V3.1 5.5: aux disabled at inference)."
    )


@pytest.mark.spec_compliance
def test_f_disable_branches_kwarg_is_backward_compatible():
    """Stage-B `disable_branches` kwarg defaults to None and is a no-op then."""
    from v4.models.fusion.v3_pairwise import V3PairwiseFusion

    torch.manual_seed(0)
    m1 = V3PairwiseFusion(feat_dim=768, fused_dim=1024, num_classes=5)
    torch.manual_seed(0)
    m2 = V3PairwiseFusion(
        feat_dim=768,
        fused_dim=1024,
        num_classes=5,
        disable_branches=None,
    )
    m1.eval()
    m2.eval()

    features = {k: torch.randn(2, 768) for k in ("v_semantic", "v_imgfor", "v_textfor")}
    with torch.no_grad():
        a = m1(features)["main_logits"]
        b = m2(features)["main_logits"]
    assert torch.allclose(a, b), "disable_branches=None must be identical to omitting it"

    with pytest.raises(ValueError, match="unknown keys"):
        V3PairwiseFusion(disable_branches=["v_typo"])

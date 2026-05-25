"""Tests for V3PairwiseFusion's optional pre-fusion projections (proj_dims).

Added in P5.10a so V4 Stage 2 can train on V3-era caches without rebuilding
v_semantic (raw 512-d) and v_textfor (raw 3584-d).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from v4.core.registry import FUSION_REGISTRY, import_all  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _import_v4():
    import_all()


def _build(proj_dims=None, feat_dim=768, num_classes=5):
    return FUSION_REGISTRY["v3_pairwise"](
        feat_dim=feat_dim, fused_dim=1024, num_classes=num_classes, proj_dims=proj_dims
    )


def test_default_no_proj_is_identity():
    fus = _build(proj_dims=None)
    assert type(fus.sem_proj).__name__ == "Identity"
    assert type(fus.img_proj).__name__ == "Identity"
    assert type(fus.text_proj).__name__ == "Identity"


def test_proj_dims_v3_style():
    fus = _build(proj_dims={"v_semantic": 512, "v_textfor": 3584})
    assert type(fus.sem_proj).__name__ == "Sequential"
    assert type(fus.img_proj).__name__ == "Identity"
    assert type(fus.text_proj).__name__ == "Sequential"


def test_proj_dims_eq_feat_dim_is_identity():
    fus = _build(proj_dims={"v_semantic": 768, "v_imgfor": 768, "v_textfor": 768})
    assert type(fus.sem_proj).__name__ == "Identity"
    assert type(fus.img_proj).__name__ == "Identity"
    assert type(fus.text_proj).__name__ == "Identity"


def test_forward_with_v3_cache_dims():
    """Forward with raw V3 cache shapes: v_semantic 512, v_imgfor 768, v_textfor 3584."""
    fus = _build(proj_dims={"v_semantic": 512, "v_textfor": 3584})
    fus.eval()
    B = 3
    out = fus(
        {
            "v_semantic": torch.randn(B, 512),
            "v_imgfor": torch.randn(B, 768),
            "v_textfor": torch.randn(B, 3584),
        }
    )
    assert out["main_logits"].shape == (B, 5)
    assert out["aux_logits"].shape == (B, 2)
    assert out["fused"].shape == (B, 1024)


def test_forward_default_still_works():
    """Default config (no proj_dims) still works with 768-d everywhere."""
    fus = _build(proj_dims=None)
    fus.eval()
    B = 2
    out = fus(
        {
            "v_semantic": torch.randn(B, 768),
            "v_imgfor": torch.randn(B, 768),
            "v_textfor": torch.randn(B, 768),
        }
    )
    assert out["main_logits"].shape == (B, 5)


def test_proj_dims_includes_imgfor():
    """proj_dims supports v_imgfor too (e.g. raw ResNet50 pre-pool 2048-d)."""
    fus = _build(proj_dims={"v_imgfor": 2048})
    fus.eval()
    B = 2
    out = fus(
        {
            "v_semantic": torch.randn(B, 768),
            "v_imgfor": torch.randn(B, 2048),
            "v_textfor": torch.randn(B, 768),
        }
    )
    assert out["main_logits"].shape == (B, 5)


def test_grad_flows_through_projections():
    """Backward through main_logits should reach the projection layers."""
    fus = _build(proj_dims={"v_semantic": 512, "v_textfor": 3584})
    fus.train()
    B = 2
    out = fus(
        {
            "v_semantic": torch.randn(B, 512, requires_grad=False),
            "v_imgfor": torch.randn(B, 768, requires_grad=False),
            "v_textfor": torch.randn(B, 3584, requires_grad=False),
        }
    )
    loss = out["main_logits"].sum()
    loss.backward()
    assert fus.sem_proj[0].weight.grad is not None
    assert fus.sem_proj[0].weight.grad.abs().sum().item() > 0.0
    assert fus.text_proj[0].weight.grad is not None
    assert fus.text_proj[0].weight.grad.abs().sum().item() > 0.0

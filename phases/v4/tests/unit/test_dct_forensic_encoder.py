"""Tests for the V4 DCT forensic encoder wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from v4.core.registry import ENCODER_REGISTRY, import_all  # noqa: E402
from v4.models.encoders.dct_forensic import DctForensicEncoder  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _ensure_imported():
    import_all()


def test_registered_in_encoder_registry():
    assert "dct_forensic_v1" in ENCODER_REGISTRY


def test_encoder_class_name():
    cls = ENCODER_REGISTRY["dct_forensic_v1"]
    assert cls is DctForensicEncoder
    assert cls.name == "dct_forensic_v1"
    assert cls.modality == "image"
    assert cls.required_inputs == ("v_imgfor_dct",)


def test_forward_shape_default_768():
    m = DctForensicEncoder(out_dim=768)
    x = torch.randn(3, 1, 224, 224)
    y = m({"v_imgfor_dct": x})
    assert y.shape == (3, 768)
    assert y.dtype == torch.float32


def test_forward_shape_other_out_dim():
    m = DctForensicEncoder(out_dim=512)
    x = torch.randn(2, 1, 224, 224)
    y = m({"v_imgfor_dct": x})
    assert y.shape == (2, 512)


def test_output_dim_attribute_set():
    m = DctForensicEncoder(out_dim=1024)
    assert m.output_dim == 1024


def test_freeze_backbone():
    m = DctForensicEncoder(out_dim=768, freeze_backbone=True)
    backbone_trainable = sum(p.numel() for p in m.backbone.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in m.head.parameters() if p.requires_grad)
    assert backbone_trainable == 0
    assert head_trainable > 0


def test_no_freeze_default():
    m = DctForensicEncoder(out_dim=768)
    backbone_trainable = sum(p.numel() for p in m.backbone.parameters() if p.requires_grad)
    assert backbone_trainable > 0  # ResNet50 has many params


def test_ckpt_missing_falls_back_gracefully():
    # Non-existent path should warn but not crash
    m = DctForensicEncoder(out_dim=768, ckpt=Path("nonexistent_path.pth"))
    x = torch.randn(1, 1, 224, 224)
    y = m({"v_imgfor_dct": x})
    assert y.shape == (1, 768)


def test_load_real_ckpt_if_present():
    """If the trained forensic_dct_model.pth exists, ensure it loads cleanly."""
    ckpt = Path("phases/forensic/outputs/dct/forensic_dct_model.pth")
    if not ckpt.exists():
        pytest.skip(f"Real ckpt not on disk at {ckpt}")
    m = DctForensicEncoder(out_dim=768, ckpt=ckpt)
    x = torch.randn(2, 1, 224, 224)
    y = m({"v_imgfor_dct": x})
    assert y.shape == (2, 768)
    # Backbone conv1 should be 1-channel
    assert m.backbone.conv1.in_channels == 1

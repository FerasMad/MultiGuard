"""Integration test: dct_forensic_v1 encoder + v3_pairwise fusion + classifier.

Validates that the new forensic encoder produced by the Approach 2 (DCT) trainer
plugs into V4's existing fusion + classifier without shape or contract violations.
Does NOT train or load real data — uses random tensors only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from v4.core.registry import ENCODER_REGISTRY, FUSION_REGISTRY, import_all  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _import_v4():
    import_all()


def test_dct_forensic_v1_registered():
    assert "dct_forensic_v1" in ENCODER_REGISTRY
    assert "v3_pairwise" in FUSION_REGISTRY


def test_forensic_encoder_feeds_v3_pairwise():
    """End-to-end: forensic DCT encoder -> v_imgfor -> v3_pairwise -> 5-class logits."""
    enc_cls = ENCODER_REGISTRY["dct_forensic_v1"]
    fus_cls = FUSION_REGISTRY["v3_pairwise"]

    img_encoder = enc_cls(out_dim=768)
    fusion = fus_cls(feat_dim=768, fused_dim=1024, num_classes=5)

    img_encoder.eval()
    fusion.eval()

    B = 4
    # v_imgfor_dct: patch-DCT [B,1,224,224] - what the encoder consumes
    v_imgfor_dct = torch.randn(B, 1, 224, 224)
    # v_semantic + v_textfor would normally come from FND-CLIP + Qwen encoders.
    # For the integration smoke we mock them as random 768-d vectors.
    v_semantic = torch.randn(B, 768)
    v_textfor = torch.randn(B, 768)

    with torch.no_grad():
        v_imgfor = img_encoder({"v_imgfor_dct": v_imgfor_dct})
        out = fusion({"v_semantic": v_semantic, "v_imgfor": v_imgfor, "v_textfor": v_textfor})

    assert v_imgfor.shape == (B, 768), f"forensic encoder output wrong: {v_imgfor.shape}"
    assert out["main_logits"].shape == (B, 5), f"main_logits wrong: {out['main_logits'].shape}"
    assert out["aux_logits"].shape == (B, 2), f"aux_logits wrong: {out['aux_logits'].shape}"
    assert out["fused"].shape == (B, 1024), f"fused wrong: {out['fused'].shape}"


def test_forensic_ckpt_load_when_present():
    """If forensic_dct_model.pth is on disk, the encoder loads it cleanly and forward still works."""
    ckpt = Path("phases/forensic/outputs/dct/forensic_dct_model.pth")
    if not ckpt.exists():
        pytest.skip(f"forensic_dct_model.pth not at {ckpt}")
    enc_cls = ENCODER_REGISTRY["dct_forensic_v1"]
    fus_cls = FUSION_REGISTRY["v3_pairwise"]
    img_encoder = enc_cls(out_dim=768, ckpt=ckpt)
    fusion = fus_cls(num_classes=5)
    img_encoder.eval()
    fusion.eval()

    B = 2
    v_imgfor_dct = torch.randn(B, 1, 224, 224)
    v_semantic = torch.randn(B, 768)
    v_textfor = torch.randn(B, 768)
    with torch.no_grad():
        v_imgfor = img_encoder({"v_imgfor_dct": v_imgfor_dct})
        out = fusion({"v_semantic": v_semantic, "v_imgfor": v_imgfor, "v_textfor": v_textfor})
    # The forensic ckpt was trained for binary detection; we discard its fc.* head
    # in DctForensicEncoder. v_imgfor still has to be 768-d after the V4 projection.
    assert v_imgfor.shape == (B, 768)
    assert out["main_logits"].shape == (B, 5)


def test_aux_head_uses_detached_v_imgfor():
    """V3.1 section 5.5 demands aux head sees v_imgfor.detach() so gradients don't flow."""
    enc_cls = ENCODER_REGISTRY["dct_forensic_v1"]
    fus_cls = FUSION_REGISTRY["v3_pairwise"]
    enc = enc_cls(out_dim=768)
    fus = fus_cls(num_classes=5)
    enc.train()
    fus.train()
    B = 2
    v_imgfor_dct = torch.randn(B, 1, 224, 224, requires_grad=False)
    v_semantic = torch.randn(B, 768)
    v_textfor = torch.randn(B, 768)
    v_imgfor = enc({"v_imgfor_dct": v_imgfor_dct})
    # Confirm v_imgfor has grad path back to encoder params
    assert v_imgfor.requires_grad
    out = fus({"v_semantic": v_semantic, "v_imgfor": v_imgfor, "v_textfor": v_textfor})
    # aux_logits is computed from v_imgfor.detach() inside fusion;
    # we verify by computing a loss on aux only and checking no grad reaches encoder
    aux_loss = out["aux_logits"].sum()
    enc.zero_grad()
    aux_loss.backward()
    # No encoder param should have non-zero grad from the aux head
    for _name, p in enc.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, "aux head grad leaked into encoder param"

"""Shape and behavior smoke tests for V2 model components.

Runs purely on CPU with random tensors. No data, no checkpoints, no
GPU required. These tests exist to catch regressions when refactoring
the model code — they don't validate that the model is correct, just
that the dimensions and forward/backward signals flow as the V2 PDF
specifies.

PDF references:
  - §5  ResNet18 Forensic Encoder (1-channel DCT input)
  - §6  Auxiliary Classifier
  - §7  Cross-Attention Fusion (8 heads, batch_first=True)
  - §8  MLP Classifier (1024 -> 512 -> 256 -> 3)
  - §9  Full pipeline composition
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.forensic_baseline import ForensicBaseline, ResNet18Forensic  # noqa: E402
from models.full_pipeline import (  # noqa: E402
    CrossAttentionFusion,
    FullPipeline,
    MLPClassifier,
)


# ---------------------------------------------------------------------------
# Step 1: forensic encoder + baseline
# ---------------------------------------------------------------------------


def test_resnet18_forensic_accepts_1_channel_dct():
    """V2 §5: forensic encoder takes [B, 1, 224, 224] DCT-Y."""
    model = ResNet18Forensic(pretrained=False, out_dim=768, dropout=0.3)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 768)
    assert out.dtype == torch.float32


def test_resnet18_forensic_rejects_3_channel():
    model = ResNet18Forensic(pretrained=False)
    with pytest.raises(RuntimeError):
        _ = model(torch.randn(1, 3, 224, 224))


def test_resnet18_forensic_custom_out_dim():
    model = ResNet18Forensic(pretrained=False, out_dim=512)
    out = model(torch.randn(1, 1, 224, 224))
    assert out.shape == (1, 512)


def test_forensic_baseline_returns_dict():
    """ForensicBaseline returns a dict with v_forensic and logits."""
    model = ForensicBaseline(num_classes=3, pretrained=False, feat_dim=512)
    out = model(torch.randn(2, 1, 224, 224))
    assert set(out.keys()) == {"v_forensic", "logits"}
    assert out["v_forensic"].shape == (2, 512)
    assert out["logits"].shape == (2, 3)


def test_forensic_baseline_gradients_flow():
    model = ForensicBaseline(num_classes=3, pretrained=False, feat_dim=512)
    out = model(torch.randn(2, 1, 224, 224))
    loss = out["logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# Step 2 components: cross-attention + MLP head
# ---------------------------------------------------------------------------


def test_cross_attention_concat_to_1024():
    """V2 §7: two 512-dim projections -> [B, 1024] concatenated fused."""
    fusion = CrossAttentionFusion(sem_dim=512, for_dim=512, proj_dim=512,
                                  num_heads=8, dropout=0.1)
    sem = torch.randn(4, 512)
    forensic = torch.randn(4, 512)
    fused = fusion(sem, forensic)
    assert fused.shape == (4, 1024)
    assert torch.isfinite(fused).all()


def test_cross_attention_handles_unequal_input_dims():
    """The proj_sem and proj_for layers can absorb differing input widths."""
    fusion = CrossAttentionFusion(sem_dim=768, for_dim=512, proj_dim=256,
                                  num_heads=8)
    sem = torch.randn(2, 768)
    forensic = torch.randn(2, 512)
    fused = fusion(sem, forensic)
    assert fused.shape == (2, 512)  # 2 * proj_dim


def test_cross_attention_gradients_flow():
    fusion = CrossAttentionFusion(sem_dim=512, for_dim=512, proj_dim=512,
                                  num_heads=8)
    sem = torch.randn(2, 512, requires_grad=True)
    forensic = torch.randn(2, 512, requires_grad=True)
    fused = fusion(sem, forensic)
    fused.sum().backward()
    assert sem.grad is not None
    assert forensic.grad is not None


def test_mlp_classifier_layout_matches_spec():
    """V2 §8: 1024 -> 512 -> 256 -> 3 with no softmax (raw logits)."""
    head = MLPClassifier(in_dim=1024, num_classes=3)
    out = head(torch.randn(8, 1024))
    assert out.shape == (8, 3)
    # Logits, not probabilities — softmax of these should NOT sum to 1 trivially.
    sums = out.exp().sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


# ---------------------------------------------------------------------------
# Step 2: full pipeline
# ---------------------------------------------------------------------------


class _StubFND(nn.Module):
    """Minimal FND-CLIP placeholder that yields a fixed-dim v_semantic.

    The full pipeline lets v_semantic be passed in directly (the production
    case where features are precomputed and cached). When that path is used
    the FND-CLIP module is never called, so a stub suffices for tests.

    Uses default `nn.Module.train()` / `.eval()` so FullPipeline's train()
    override (which calls `self.fnd_clip.eval()`) actually toggles training
    state — the real FND-CLIP relies on this same default behavior.
    """
    def __init__(self):
        super().__init__()
        self._buf = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward_semantic(self, *a, **k):
        raise RuntimeError("v_semantic must be precomputed in tests")


def _make_pipeline(num_classes: int = 3) -> FullPipeline:
    return FullPipeline(
        fnd_clip=_StubFND(),
        fnd_clip_feat_dim=512,
        forensic_feat_dim=512,
        fusion_proj_dim=512,
        num_classes=num_classes,
        fusion_heads=8,
        fusion_dropout=0.1,
        forensic_dropout=0.3,
    )


def test_full_pipeline_with_cached_v_semantic_returns_main_and_aux():
    """Forward returns dict with main_logits + aux_logits + intermediates."""
    pipeline = _make_pipeline(num_classes=3)
    pipeline.train()
    out = pipeline(
        dct=torch.rand(2, 1, 224, 224),
        v_semantic=torch.randn(2, 512),
    )
    assert set(out.keys()) >= {"main_logits", "aux_logits", "v_semantic",
                               "v_forensic", "fused"}
    assert out["main_logits"].shape == (2, 3)
    assert out["aux_logits"].shape == (2, 2)         # binary aux per Phase 2
    assert out["v_forensic"].shape == (2, 512)
    assert out["fused"].shape == (2, 1024)


def test_full_pipeline_trainable_only_outside_fnd_clip():
    """V2 §3: FND-CLIP is frozen; everything else trains."""
    pipeline = _make_pipeline()
    for p in pipeline.fnd_clip.parameters():
        assert p.requires_grad is False
    trainable_top = {n.split(".")[0]
                     for n, p in pipeline.named_parameters()
                     if p.requires_grad}
    # Trainable modules are forensic, aux_classifier, fusion, classifier.
    assert "forensic" in trainable_top
    assert "fusion" in trainable_top
    assert "classifier" in trainable_top
    assert "aux_classifier" in trainable_top


def test_full_pipeline_train_keeps_fnd_in_eval():
    """The .train() override must keep frozen FND-CLIP in eval mode."""
    pipeline = _make_pipeline()
    pipeline.train(True)
    assert pipeline.training is True
    assert pipeline.fnd_clip.training is False


def test_full_pipeline_joint_loss_backward_no_nan():
    """V2 §10: total_loss = CE(main) + 0.1 * CE(aux). Backward is clean."""
    pipeline = _make_pipeline()
    pipeline.train()
    out = pipeline(
        dct=torch.rand(4, 1, 224, 224),
        v_semantic=torch.randn(4, 512),
    )
    main_y = torch.randint(0, 3, (4,))
    aux_y = torch.randint(0, 2, (4,))
    ce = nn.CrossEntropyLoss()
    loss = ce(out["main_logits"], main_y) + 0.1 * ce(out["aux_logits"], aux_y)
    loss.backward()
    assert torch.isfinite(loss)


def test_full_pipeline_intermediate_v_forensic_matches_aux_input():
    """v_forensic feeds the auxiliary classifier (V2 §6)."""
    pipeline = _make_pipeline()
    pipeline.eval()
    out = pipeline(
        dct=torch.rand(2, 1, 224, 224),
        v_semantic=torch.randn(2, 512),
    )
    # aux_logits must be a function of v_forensic alone.
    expected_aux = pipeline.aux_classifier(out["v_forensic"])
    assert torch.allclose(out["aux_logits"], expected_aux)

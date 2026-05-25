"""Student 1 — UnivFD image-forensic encoder.

Input:  patch-DCT map [B, 1, 224, 224] (Y-channel only, log-normalised
        per patch, see `v3/scripts/precompute_patch_dct.py`).
Output: v_imgfor [B, 768].

Build steps (per the 4-student spec):
  1. Start from a ResNet50 (UnivFD backbone). Modify conv1 to accept
     1 input channel; initialise with Kaiming Normal.
  2. Load the pretrained `blur_jpg_v0.pth` checkpoint with
     `strict=False` so the new 1-channel conv1 and the dimension
     mismatch on the classifier head are tolerated.
  3. Replace the original 1000-way classifier with a 768-dim
     projection head: `Flatten -> Dropout(0.3) -> Linear(2048, 768)`.

Used in:
  - Stage-1 binary training (`v3/src/train_stage1_univfd.py`): adds a
    1-neuron BCE head on top of the 768 vector to train real-vs-tampered
    on DGM4 + MMFakeBench tampered subset.
  - Stage-2 5-class training (frozen): the 768 vector is the
    `v_imgfor` input to the pairwise-fusion module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def _make_1ch_conv1(old_conv1: nn.Conv2d) -> nn.Conv2d:
    """Replace 3-channel conv1 with a 1-channel version, Kaiming-init."""
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=(old_conv1.bias is not None),
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out",
                            nonlinearity="relu")
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)
    return new_conv


class UnivFDEncoder(nn.Module):
    """ResNet50-based UnivFD image-forensic encoder.

    Args:
        out_dim:      width of v_imgfor (default 768 to match fusion).
        pretrained:   if a path is given, load it as a UnivFD ckpt with
                      strict=False. If None, fall back to ImageNet
                      pretrained weights (placeholder when blur_jpg_v0
                      is not on disk yet).
        dropout:      dropout in the projection head.
    """

    def __init__(self, out_dim: int = 768,
                 pretrained: Optional[str] = None,
                 dropout: float = 0.3):
        super().__init__()
        # ImageNet weights serve as a sensible init when the UnivFD ckpt
        # is missing; the strict=False load below overrides whichever
        # tensors are present in the UnivFD ckpt.
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        backbone.conv1 = _make_1ch_conv1(backbone.conv1)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, out_dim),
        )

        if pretrained:
            path = Path(pretrained)
            if not path.exists():
                raise FileNotFoundError(
                    f"UnivFD checkpoint not found: {path}. Download "
                    f"`blur_jpg_v0.pth` from the CNNDetection repo "
                    f"(Wang et al.) and place it there.")
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.backbone.load_state_dict(
                state, strict=False)
            print(f"  UnivFD ckpt loaded with strict=False: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

    def forward(self, dct_y: torch.Tensor) -> torch.Tensor:
        """`dct_y`: [B, 1, 224, 224]. Returns [B, out_dim]."""
        feats = self.backbone(dct_y)         # [B, 2048]
        return self.head(feats)              # [B, out_dim]


class UnivFDStage1Wrapper(nn.Module):
    """Stage-1 binary head on top of the encoder. Used during
    `train_stage1_univfd.py` and discarded for Stage 2."""

    def __init__(self, encoder: UnivFDEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.head[-1].out_features, 1)

    def forward(self, dct_y: torch.Tensor) -> dict:
        feat = self.encoder(dct_y)
        logits = self.head(feat).squeeze(-1)
        return {"logits": logits, "v_imgfor": feat}


# Smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    enc = UnivFDEncoder(out_dim=768, pretrained=None)
    x = torch.randn(2, 1, 224, 224)
    v = enc(x)
    assert v.shape == (2, 768), v.shape
    print(f"encoder smoke ok | v {tuple(v.shape)}")

    stage1 = UnivFDStage1Wrapper(enc)
    out = stage1(x)
    assert out["logits"].shape == (2,), out["logits"].shape
    print(f"stage1 smoke ok | logits {tuple(out['logits'].shape)}")

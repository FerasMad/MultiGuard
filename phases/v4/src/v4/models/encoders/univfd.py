"""UnivFD image-forensic encoder per V3.1 §3.2.

Student 1's deliverable. ResNet-50 with:
  - conv1 modified to accept 1 input channel (patch-DCT Y-channel)
  - Kaiming Normal init on the new conv1
  - blur_jpg_v0.pth pre-trained weights loaded with strict=False
  - Final classifier replaced with: Flatten -> Dropout(0.3) -> Linear(2048, 768)

Input:  v_imgfor_dct [B, 1, 224, 224]   (patch-DCT map from preprocessing/patch_dct.py)
Output: v_imgfor     [B, 768]           (forensic feature vector)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from v4.core.checkpoints import strip_prefix
from v4.core.logging import get_logger
from v4.core.registry import ENCODER_REGISTRY, register
from v4.models.encoders.base import EncoderBase

log = get_logger(__name__)


def _make_1ch_conv1(old_conv1: nn.Conv2d) -> nn.Conv2d:
    """Replace 3-channel conv1 with 1-channel, Kaiming-init per V3.1 §3.2."""
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=(old_conv1.bias is not None),
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)
    return new_conv


@register(ENCODER_REGISTRY, "univfd")
class UnivFDEncoder(EncoderBase):
    """ResNet-50-based UnivFD image-forensic encoder. V3.1 §3.2."""

    required_inputs = ("v_imgfor_dct",)
    modality = "image"

    def __init__(
        self,
        out_dim: int = 768,
        pretrained: str | Path | None = None,
        dropout: float = 0.3,
        ckpt: str | Path | None = None,
        **_kwargs,
    ):
        """Build the encoder.

        Args:
            out_dim: width of v_imgfor (default 768 per V3.1 §1)
            pretrained: optional path to blur_jpg_v0.pth (CNNDetection init)
            dropout: head dropout (V3.1 §3.2 implied via UnivFD paper recipe)
            ckpt: optional V4 trained ckpt (Stage 1 output). Loaded last,
                  overriding any pretrained init.
        """
        super().__init__()

        # ImageNet init is a sensible fallback if blur_jpg_v0 is missing.
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
        self.output_dim = out_dim

        if pretrained:
            self._load_pretrained_blur_jpg(pretrained)
        if ckpt:
            self.load_legacy_checkpoint(ckpt)

    def _load_pretrained_blur_jpg(self, path: str | Path) -> None:
        """Load CNNDetection blur_jpg_v0.pth with strict=False (per V3.1 §3.2)."""
        p = Path(path)
        if not p.exists():
            log.warning("blur_jpg_v0 not found at %s; using ImageNet init only", p)
            return
        state = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # blur_jpg state keys may or may not have a 'module.' prefix from old DDP runs.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        log.info(
            "UnivFD loaded blur_jpg_v0 (missing=%d, unexpected=%d)", len(missing), len(unexpected)
        )

    def load_legacy_checkpoint(self, path: str | Path) -> None:
        """Load a Stage-1-trained UnivFD checkpoint.

        Stage-1 wrappers used to save under `encoder.*` prefix; strip it.
        """
        p = Path(path)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        state = ck.get("model_state", ck)
        # If saved by Stage-1 wrapper with encoder. prefix, strip it
        enc_state = strip_prefix(state, "encoder.")
        if enc_state:
            missing, unexpected = self.load_state_dict(enc_state, strict=False)
        else:
            missing, unexpected = self.load_state_dict(state, strict=False)
        log.info(
            "UnivFD loaded from %s (missing=%d, unexpected=%d)",
            p.name,
            len(missing),
            len(unexpected),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["v_imgfor_dct"]  # [B, 1, 224, 224]
        feats = self.backbone(x)  # [B, 2048]
        return self.head(feats)  # [B, out_dim]

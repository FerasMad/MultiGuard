"""V4 DCT-forensic encoder (image stream) — inline copy for the HF Space.

Mirrors phases/v4/src/v4/models/encoders/dct_forensic.py with v4.* imports
stripped and the @register decorator dropped.

Architecture:
  - torchvision ResNet50 with 'IMAGENET1K_V1' weights
  - conv1 modified to 1-channel (Kaiming Normal init)
  - fc replaced with Linear(2048, out_dim) + GELU projection head

Input:  v_imgfor_dct [B, 1, 224, 224]
Output: v_imgfor     [B, 768]
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)


def _make_1ch_conv1(old_conv1: nn.Conv2d) -> nn.Conv2d:
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


class DctForensicEncoder(nn.Module):
    """V4 encoder backed by the trained Approach 2 (DCT) forensic detector."""

    def __init__(
        self,
        out_dim: int = 768,
        ckpt: str | Path | None = None,
        freeze_backbone: bool = False,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        backbone.conv1 = _make_1ch_conv1(backbone.conv1)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(2048, out_dim),
            nn.GELU(),
        )
        self.output_dim = out_dim

        if ckpt is not None:
            self.load_forensic_ckpt(ckpt)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def load_forensic_ckpt(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            log.warning("forensic ckpt not found at %s; keeping ImageNet init", p)
            return
        payload = torch.load(p, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        log.info(
            "DctForensicEncoder loaded %s (missing=%d, unexpected=%d)",
            p.name,
            len(missing),
            len(unexpected),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["v_imgfor_dct"]
        feats = self.backbone(x)
        return self.head(feats)

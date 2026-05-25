"""DCT forensic encoder for V4 - wraps the trained Approach 2 detector.

Bridges the standalone forensic detector at
`phases/forensic/outputs/dct/forensic_dct_model.pth` into V4's encoder
contract (`EncoderBase`). Replaces the older `UnivFDEncoder` Stage-1 init
(`blur_jpg_v0.pth`) with the doctor-spec frequency-domain forensic
checkpoint when configured.

Architecture (matches `forensic.models.dct_resnet50.build_dct_resnet50`):
  - torchvision ResNet50 with `weights='IMAGENET1K_V1'`
  - conv1 modified to 1 input channel (Kaiming, fan_out, relu)
  - fc replaced with `Linear(2048, 1)` (binary head, doctor's F.15)

For V4 use as an image-forensic encoder, we throw away the binary fc head
and add a `Linear(2048, out_dim) + GELU` projection so the encoder emits
a 768-d feature vector compatible with V4's fusion module.

Input:  v_imgfor_dct [B, 1, 224, 224]   (patch-DCT map; SAME as UnivFDEncoder
                                          so the precompute path is identical)
Output: v_imgfor     [B, 768]
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
from torch import nn
from torchvision import models

from v4.core.logging import get_logger
from v4.core.registry import ENCODER_REGISTRY, register
from v4.models.encoders.base import EncoderBase

log = get_logger(__name__)


def _make_1ch_conv1(old_conv1: nn.Conv2d) -> nn.Conv2d:
    """Replace 3-channel conv1 with 1-channel, Kaiming-init per F.13."""
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


@register(ENCODER_REGISTRY, "dct_forensic_v1")
class DctForensicEncoder(EncoderBase):
    """V4 encoder backed by the trained Approach 2 (DCT) forensic detector."""

    required_inputs: ClassVar[tuple[str, ...]] = ("v_imgfor_dct",)
    modality: ClassVar = "image"

    def __init__(
        self,
        out_dim: int = 768,
        ckpt: str | Path | None = None,
        freeze_backbone: bool = False,
        head_dropout: float = 0.3,
        **_kwargs,
    ):
        """Build the encoder.

        Args:
            out_dim: width of `v_imgfor` (default 768 per V3.1 sec 1).
            ckpt: path to `forensic_dct_model.pth` (or any payload with
                  `model_state` containing the binary-head state_dict). If
                  None, the backbone is left at ImageNet RN50 init with a
                  fresh Kaiming 1ch conv1 - useful for ablation but loses
                  the frequency-domain advantage.
            freeze_backbone: if True, freeze the RN50 backbone and only
                  train the V4 projection head (faster Stage-2, less
                  flexible). Default False (full fine-tune like UnivFD).
            head_dropout: dropout on the [2048 -> out_dim] projection.
        """
        super().__init__()

        # Build the same backbone the forensic detector trained on
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        backbone.conv1 = _make_1ch_conv1(backbone.conv1)

        # Replace fc with Identity so backbone outputs 2048-d features.
        # The trained ckpt has a Linear(2048, 1) at fc; we strip it at load time.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # V4 projection head: 2048 -> out_dim (analogous to UnivFD's head)
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
            log.info("DctForensicEncoder: backbone frozen (head-only training)")

    def load_forensic_ckpt(self, path: str | Path) -> None:
        """Load `phases/forensic/outputs/dct/forensic_dct_model.pth` (or `best.pt`).

        The forensic ckpt has these top-level keys (per
        `forensic.training.two_phase_trainer.TwoPhaseTrainer._save_checkpoint`):
          - model_state: dict with conv1, bn1, layer{1..4}, fc.* keys
          - optimizer_state, scheduler_state, ...

        We unwrap `model_state`, drop the `fc.*` keys (incompatible binary head),
        and load the remaining backbone weights with strict=False.
        """
        p = Path(path)
        if not p.exists():
            log.warning("forensic ckpt not found at %s; keeping ImageNet init", p)
            return
        payload = torch.load(p, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        # Strip the binary fc head; it's incompatible with V4's projection.
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        log.info(
            "DctForensicEncoder loaded %s (missing=%d, unexpected=%d)",
            p.name,
            len(missing),
            len(unexpected),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per V4 contract: batch -> [B, out_dim]."""
        x = batch["v_imgfor_dct"]  # [B, 1, 224, 224]
        feats = self.backbone(x)  # [B, 2048]
        return self.head(feats)  # [B, out_dim]

"""ResNet18 forensic encoder for DCT maps (Implementation Guidelines V2 §5-6).

The encoder takes a single-channel DCT map [B, 1, 224, 224] and returns a
768-dim forensic vector. During training we attach a 3-class auxiliary
classifier so the encoder can be supervised independently. At inference,
only the main classifier from the full pipeline is used.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Forensic(nn.Module):
    """ResNet18 modified for 1-channel input, projecting features to 768-dim."""

    def __init__(self, pretrained: bool = True, out_dim: int = 768,
                 dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace first conv to accept 1 channel.
        # Initialize from the mean across the original 3 RGB channels so
        # ImageNet priors carry over to grayscale-style DCT input.
        old = backbone.conv1
        new_conv = nn.Conv2d(1, old.out_channels,
                             kernel_size=old.kernel_size,
                             stride=old.stride,
                             padding=old.padding,
                             bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # Drop the original classifier (1000-way) and the avgpool/flatten
        # come from the backbone itself. We project the 512-dim feature.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, dct_map: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(dct_map)  # [B, 512]
        return self.head(feats)         # [B, 768]


class ForensicBaseline(nn.Module):
    """ResNet18 encoder + auxiliary 3-class classifier.

    For Step 1 of the pipeline (forensic-only baseline). Returns a dict so
    the caller can decide whether to use the feature, the logits, or both.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 feat_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.encoder = ResNet18Forensic(pretrained=pretrained, out_dim=feat_dim,
                                        dropout=dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, dct_map: torch.Tensor):
        feat = self.encoder(dct_map)
        logits = self.classifier(feat)
        return {"v_forensic": feat, "logits": logits}

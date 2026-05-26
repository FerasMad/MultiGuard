"""DCT-input ResNet50 forensic detector per doctor's Approach 2 spec.

Spec source: docs/doctor-briefs/Forensic_Image_Detector_En.pdf, F.12-F.15 in MASTER_CHECKLIST.

Architecture:
    - Base: torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    - conv1: in_channels 3 -> 1 (other params unchanged: out_channels=64, kernel=7,
             stride=2, padding=3, bias=False). Weights re-initialized via Kaiming Normal
             (mode='fan_out', nonlinearity='relu'). NOT loaded from ImageNet (only 3-ch).
    - fc: replaced with Linear(2048, 1), no activation. Trained head for binary classification
          (sigmoid applied at loss/eval time, not in the model).

Training phases (see forensic.training.two_phase_trainer):
    - Phase 1 (epochs 1-5): conv1 + bn1 + layer3 + layer4 + fc trainable; layer1, layer2 frozen
    - Phase 2 (epoch 6+):  all layers trainable, reinit optimizer + scheduler + grad clip

Doctor's spec explicitly says "Do NOT use any ResNet file from the FakeImageDetection
repository" for Approach 2 - we use stock torchvision.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_dct_resnet50(pretrained: bool = True) -> nn.Module:
    """Build the doctor-spec DCT-input ResNet50.

    Steps (literal doctor spec):
        1. Build torchvision resnet50 with ImageNet V1 weights.
        2. Replace conv1 with a new 1-channel Conv2d (Kaiming Normal init).
        3. Replace fc with Linear(2048, 1).

    Args:
        pretrained: If True, load IMAGENET1K_V1 weights for all layers EXCEPT
            the modified conv1 (which is freshly Kaiming-initialized). If False,
            random init throughout (useful for unit tests).

    Returns:
        nn.Module accepting [B, 1, 224, 224] DCT-input tensor, outputting [B, 1] logits.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    # conv1 modification
    # Keep all params identical to stock conv1 except in_channels (3 -> 1).
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    # Kaiming Normal init per spec
    nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = new_conv1

    # fc replacement
    # Linear(2048, 1), no activation. BCEWithLogitsLoss applies sigmoid internally.
    model.fc = nn.Linear(2048, 1)

    return model


# Phase-1 / Phase-2 freeze helpers (doctor's training spec)

PHASE1_TRAINABLE: tuple[str, ...] = ("conv1", "bn1", "layer3", "layer4", "fc")
PHASE1_FROZEN: tuple[str, ...] = ("layer1", "layer2")

PHASE2_TRAINABLE: tuple[str, ...] = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")


def freeze_phase1(model: nn.Module) -> None:
    """Set requires_grad per doctor's Phase 1 spec.

    Trainable: conv1, bn1, layer3, layer4, fc.
    Frozen:    layer1, layer2.
    """
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top in PHASE1_FROZEN:
            param.requires_grad = False
        elif top in PHASE1_TRAINABLE:
            param.requires_grad = True
        else:
            # Anything unexpected (e.g. a new layer added by a custom subclass)
            # defaults to trainable to avoid silently freezing.
            param.requires_grad = True


def unfreeze_phase2(model: nn.Module) -> None:
    """At START of epoch 6: unfreeze layer1 + layer2 (everything trainable)."""
    for param in model.parameters():
        param.requires_grad = True


def trainable_params(model: nn.Module) -> list[torch.nn.Parameter]:
    """Return only parameters with requires_grad=True (for optimizer).

    Doctor spec: 'Pass only trainable parameters (layer3, layer4, fc) to the AdamW
    optimizer. Frozen layers receive no gradient updates at any point.'
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "PHASE1_FROZEN",
    "PHASE1_TRAINABLE",
    "PHASE2_TRAINABLE",
    "build_dct_resnet50",
    "count_trainable_params",
    "freeze_phase1",
    "trainable_params",
    "unfreeze_phase2",
]

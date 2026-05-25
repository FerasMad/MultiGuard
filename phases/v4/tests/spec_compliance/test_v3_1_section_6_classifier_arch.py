"""V3.1 section 6: classifier MLP exact shape."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


@pytest.mark.spec_compliance
def test_mlp_classifier_layers():
    from v4.models.classifier.mlp_head import MLPClassifier

    cls = MLPClassifier(in_dim=1024, num_classes=5)
    cls.eval()

    # Test forward shape
    x = torch.randn(8, 1024)
    with torch.no_grad():
        y = cls(x)
    assert y.shape == (8, 5)

    # Spec section 6: Dropout(0.5) must be present
    has_dropout_05 = any(
        isinstance(m, nn.Dropout) and abs(m.p - 0.5) < 1e-6
        for m in cls.modules()
    )
    assert has_dropout_05, "V3.1 section 6 requires Dropout(0.5)"

    # Spec: BatchNorm after first Linear
    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in cls.modules())
    assert has_bn, "V3.1 section 6 requires BatchNorm after first Linear"

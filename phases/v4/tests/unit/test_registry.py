"""Registry pattern smoke tests."""

from __future__ import annotations

import pytest


def test_registry_populated():
    from v4.core.registry import (
        DATASET_REGISTRY,
        ENCODER_REGISTRY,
        FUSION_REGISTRY,
        import_all,
    )

    import_all()
    assert "univfd" in ENCODER_REGISTRY
    assert "fnd_clip" in ENCODER_REGISTRY
    assert "qwen2_7b" in ENCODER_REGISTRY
    assert "v3_pairwise" in FUSION_REGISTRY
    assert "cached_features" in DATASET_REGISTRY
    assert "runtime_images" in DATASET_REGISTRY


def test_duplicate_register_rejected():
    from v4.core.registry import register

    reg: dict = {}

    @register(reg, "x")
    class _A: ...

    with pytest.raises(ValueError, match="duplicate"):

        @register(reg, "x")
        class _B: ...

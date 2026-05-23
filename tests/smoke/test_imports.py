"""Every public V4 module imports without error."""
from __future__ import annotations


def test_core_imports():
    import v4
    import v4.core.checkpoints
    import v4.core.class_map
    import v4.core.config
    import v4.core.logging
    import v4.core.paths
    import v4.core.registry
    import v4.core.seed
    assert v4.SPEC_VERSION == "V3.1"


def test_model_imports():
    from v4.models.encoders.base import EncoderBase
    from v4.models.fusion.base import FusionBase
    from v4.models.classifier.mlp_head import MLPClassifier
    assert EncoderBase is not None
    assert FusionBase is not None
    assert MLPClassifier is not None


def test_data_imports():
    from v4.data.manifest import REQUIRED_COLUMNS, validate
    from v4.data.datasets.base import MultimodalManifestDataset
    from v4.data.preprocessing.patch_dct import compute_patch_dct
    assert REQUIRED_COLUMNS
    assert MultimodalManifestDataset is not None
    assert compute_patch_dct is not None


def test_training_eval_imports():
    from v4.training.trainer import BaseTrainer
    from v4.training.losses import CompositeLoss
    from v4.evaluation.evaluator import BaseEvaluator
    from v4.evaluation.reporting import compute_metrics
    assert BaseTrainer is not None
    assert CompositeLoss is not None
    assert BaseEvaluator is not None
    assert compute_metrics is not None


def test_cli_import():
    from v4.cli import SUBCOMMANDS, main
    assert main is not None
    assert "train" in SUBCOMMANDS
    assert "eval" in SUBCOMMANDS
    assert "server" in SUBCOMMANDS

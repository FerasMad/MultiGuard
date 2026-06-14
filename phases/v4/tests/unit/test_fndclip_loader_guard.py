"""FND-CLIP loader guard: a missing backbone weight must fail loudly.

Regression test for the train/serve parity bug the server parity sweep caught
(273/929 FND-CLIP weights silently random because of a strict=False load with
mismatched param names). `require_semantic_weights_loaded` converts that silent
failure into a hard RuntimeError at load time.
"""

from __future__ import annotations

import pytest

from v4.core.checkpoints import require_semantic_weights_loaded


def test_passes_when_only_allowed_prefixes_missing():
    # V1 ckpt legitimately lacks the classifier head + sem_proj adapter.
    missing = ["classifier.0.weight", "classifier.0.bias", "sem_proj.0.weight"]
    require_semantic_weights_loaded(
        missing, allowed_missing_prefixes=("classifier.", "sem_proj.")
    )  # must not raise


def test_passes_when_nothing_missing():
    require_semantic_weights_loaded([])  # must not raise


def test_raises_on_missing_backbone_weight():
    # A missing attention/backbone weight = silent random init = parity break.
    missing = ["attention.scorer.0.weight", "classifier.0.weight"]
    with pytest.raises(RuntimeError, match="parity"):
        require_semantic_weights_loaded(
            missing, allowed_missing_prefixes=("classifier.", "sem_proj.")
        )


def test_error_names_the_offending_keys_and_ckpt():
    missing = ["visual.project.weight"]
    with pytest.raises(RuntimeError) as exc:
        require_semantic_weights_loaded(missing, ckpt_name="best.pt")
    msg = str(exc.value)
    assert "visual.project.weight" in msg
    assert "best.pt" in msg

"""Enforce the 4-invariant Phase-1 -> Phase-2 transition (doctor spec F.20-F.21).

Invariants asserted at start of epoch 6:
    I1. id(self.optimizer) != id(previous_optimizer)            # AdamW reinit'd
    I2. id(self.scheduler) != id(previous_scheduler)            # ReduceLROnPlateau reinit'd
    I3. self.scheduler.num_bad_epochs == 0                      # plateau counter cleared
    I4. self._apply_grad_clip == True                            # grad clipping active

Bonus assertion:
    B1. self.early_stop.patience_left preserved across transition (NOT reset to 5)
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from forensic.models.dct_resnet50 import build_dct_resnet50
from forensic.training.two_phase_trainer import (
    EARLY_STOP_PATIENCE,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE2_GRAD_CLIP_MAX_NORM,
    PHASE2_LR,
    TwoPhaseTrainer,
)


def _make_loaders(n_train: int = 16, n_val: int = 8) -> tuple[DataLoader, DataLoader]:
    """Tiny synthetic dataset of [1,224,224] DCT tensors + binary labels."""
    torch.manual_seed(0)
    x_tr = torch.randn(n_train, 1, 224, 224)
    y_tr = torch.randint(0, 2, (n_train,))
    x_va = torch.randn(n_val, 1, 224, 224)
    y_va = torch.randint(0, 2, (n_val,))
    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=4),
        DataLoader(TensorDataset(x_va, y_va), batch_size=4),
    )


def _make_trainer(tmp_path, max_epochs=7) -> TwoPhaseTrainer:
    model = build_dct_resnet50(pretrained=False)  # random init for speed
    train_loader, val_loader = _make_loaders()
    return TwoPhaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=tmp_path,
        max_epochs=max_epochs,
        device=torch.device("cpu"),
    )


def test_phase1_initial_state(tmp_path):
    """At init: phase=1, lr=1e-4, grad clip off, layer1/2 frozen."""
    t = _make_trainer(tmp_path)
    assert t.current_phase == 1
    assert t.optimizer.param_groups[0]["lr"] == PHASE1_LR
    assert t._apply_grad_clip is False
    # layer1 + layer2 must be frozen
    for name, p in t.model.named_parameters():
        top = name.split(".")[0]
        if top in {"layer1", "layer2"}:
            assert p.requires_grad is False, f"{name} should be frozen in Phase 1"


def test_phase2_transition_4_invariants(tmp_path):
    """Manually trigger the Phase 1 -> Phase 2 transition and assert all 4 invariants."""
    t = _make_trainer(tmp_path)

    # Snapshot Phase 1 state
    p1_optimizer_id = id(t.optimizer)
    p1_scheduler_id = id(t.scheduler)

    # Simulate early-stop patience having decremented during Phase 1 (e.g. 3 bad epochs)
    t.early_stop.patience_left = 2  # arbitrary non-default value
    t.early_stop.best_ap = 0.42

    # Pump fake bad epochs into the Phase 1 scheduler so num_bad_epochs > 0
    t.scheduler.step(0.0)
    t.scheduler.step(0.0)
    t.scheduler.step(0.0)
    p1_num_bad_epochs = t.scheduler.num_bad_epochs
    assert p1_num_bad_epochs > 0, "test precondition: Phase 1 scheduler should have bad epochs"

    # ---- TRANSITION ----
    t._transition_to_phase2()

    # I1: optimizer reinit'd (new object)
    assert id(t.optimizer) != p1_optimizer_id, "I1 FAIL: optimizer should be reinit'd"

    # I2: scheduler reinit'd (new object)
    assert id(t.scheduler) != p1_scheduler_id, "I2 FAIL: scheduler should be reinit'd"

    # I3: scheduler plateau counter reset
    assert t.scheduler.num_bad_epochs == 0, (
        f"I3 FAIL: scheduler.num_bad_epochs should be 0 after reinit, got {t.scheduler.num_bad_epochs}"
    )

    # I4: grad clipping is now active
    assert t._apply_grad_clip is True, "I4 FAIL: grad clip should be active in Phase 2"

    # Bonus: phase is 2, lr is 1e-5, ALL params trainable
    assert t.current_phase == 2
    assert t.optimizer.param_groups[0]["lr"] == PHASE2_LR
    for name, p in t.model.named_parameters():
        assert p.requires_grad is True, f"{name} should be trainable in Phase 2"

    # B1: early-stop patience_left preserved (NOT reset to EARLY_STOP_PATIENCE)
    assert t.early_stop.patience_left == 2, (
        f"B1 FAIL: early-stop patience_left should carry across phase boundary, "
        f"got {t.early_stop.patience_left}"
    )
    assert t.early_stop.best_ap == 0.42, "B1 FAIL: early-stop best_ap should carry"


def test_grad_clip_is_called_in_phase2(tmp_path, monkeypatch):
    """Smoke check: in Phase 2, torch.nn.utils.clip_grad_norm_ must be invoked."""
    t = _make_trainer(tmp_path, max_epochs=1)

    # Force Phase 2 immediately (skip the 5 Phase-1 epochs)
    t._transition_to_phase2()
    assert t._apply_grad_clip is True

    # Spy on clip_grad_norm_
    calls = {"n": 0, "max_norm": None}
    import torch.nn.utils as nnu

    orig = nnu.clip_grad_norm_

    def _spy(params, max_norm, **kw):
        calls["n"] += 1
        calls["max_norm"] = max_norm
        return orig(params, max_norm, **kw)

    monkeypatch.setattr(nnu, "clip_grad_norm_", _spy)

    # Run one training epoch
    t._train_epoch()
    assert calls["n"] > 0, "clip_grad_norm_ should be called in Phase 2"
    assert calls["max_norm"] == PHASE2_GRAD_CLIP_MAX_NORM, (
        f"clip_grad_norm_ called with wrong max_norm: {calls['max_norm']} (expected {PHASE2_GRAD_CLIP_MAX_NORM})"
    )


def test_grad_clip_NOT_called_in_phase1(tmp_path, monkeypatch):
    """Negative: in Phase 1, clip_grad_norm_ must NOT be invoked."""
    t = _make_trainer(tmp_path, max_epochs=1)
    assert t._apply_grad_clip is False

    calls = {"n": 0}
    import torch.nn.utils as nnu

    orig = nnu.clip_grad_norm_

    def _spy(params, max_norm, **kw):
        calls["n"] += 1
        return orig(params, max_norm, **kw)

    monkeypatch.setattr(nnu, "clip_grad_norm_", _spy)

    t._train_epoch()
    assert calls["n"] == 0, "clip_grad_norm_ should NOT be called in Phase 1"


def test_early_stop_state_increment_and_reset(tmp_path):
    """EarlyStopState: increment on no improvement, reset on improvement."""
    t = _make_trainer(tmp_path)

    # 3 improvements in a row
    t.early_stop.update(1, 0.10)
    assert t.early_stop.patience_left == EARLY_STOP_PATIENCE
    t.early_stop.update(2, 0.20)
    assert t.early_stop.patience_left == EARLY_STOP_PATIENCE
    t.early_stop.update(3, 0.30)
    assert t.early_stop.patience_left == EARLY_STOP_PATIENCE

    # Now 2 non-improvements -> patience drops
    t.early_stop.update(4, 0.29)
    assert t.early_stop.patience_left == EARLY_STOP_PATIENCE - 1
    t.early_stop.update(5, 0.10)
    assert t.early_stop.patience_left == EARLY_STOP_PATIENCE - 2

    # Improvement again -> patience reset
    t.early_stop.update(6, 0.40)
    assert t.early_stop.patience_left == EARLY_STOP_PATIENCE
    assert t.early_stop.best_ap == 0.40
    assert t.early_stop.best_epoch == 6


def test_phase1_epochs_constant():
    """Doctor's spec: Phase 1 = epochs 1-5, Phase 2 starts at epoch 6."""
    assert PHASE1_EPOCHS == 5

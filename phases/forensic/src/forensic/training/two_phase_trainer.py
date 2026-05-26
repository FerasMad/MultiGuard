"""Two-phase trainer for the doctor's Approach 2 forensic detector.

Spec source: docs/doctor-briefs/Forensic_Image_Detector_En.pdf, F.19-F.22 in MASTER_CHECKLIST.

Phase 1 (epochs 1-5):
    - Layers trainable: conv1, bn1, layer3, layer4, fc
    - Layers frozen:    layer1, layer2
    - Optimizer: AdamW(lr=1e-4, wd=1e-4) on trainable params ONLY
    - Loss:      BCEWithLogitsLoss
    - Scheduler: ReduceLROnPlateau(mode='max', factor=0.5, patience=3)  # monitors val AP
    - Gradient clipping: OFF
    - Early-stop: patience=5 on val AP

Phase 2 transition (START of epoch 6, EXECUTE EXACTLY ONCE):
    1. Set requires_grad=True for layer1 + layer2 (everything trainable)
    2. **Re-initialize** AdamW with ALL params at lr=1e-5 wd=1e-4
    3. **Re-initialize** ReduceLROnPlateau scheduler with the new optimizer
    4. Enable gradient clipping: torch.nn.utils.clip_grad_norm_ max_norm=1.0 every batch
    5. **CARRY** early-stop patience counter across the boundary (DO NOT reset)

Saves:
    - best.pt          (best val-AP checkpoint, copied to forensic_dct_model.pth at end)
    - latest.pt        (every-epoch checkpoint for resume)
    - training_history.csv

CRITICAL: the unit test `tests/test_two_phase_trainer.py` enforces 4 invariants
at the Phase 1 -> Phase 2 boundary. Read it before modifying this file.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from forensic.models.dct_resnet50 import (
    freeze_phase1,
    trainable_params,
    unfreeze_phase2,
)

# Doctor-spec constants (locked)
PHASE1_EPOCHS: int = 5  # epochs 1..5 = Phase 1
PHASE1_LR: float = 1e-4
PHASE2_LR: float = 1e-5
WEIGHT_DECAY: float = 1e-4
SCHEDULER_FACTOR: float = 0.5
SCHEDULER_PATIENCE: int = 3  # ReduceLROnPlateau patience (different from early-stop patience)
EARLY_STOP_PATIENCE: int = 5  # On val AP plateau
PHASE2_GRAD_CLIP_MAX_NORM: float = 1.0


@dataclass
class EarlyStopState:
    """Carries early-stop state ACROSS the Phase 1 -> Phase 2 boundary.

    Doctor spec F.21: 'patience=5 on val AP, **continues across phase boundary**
    (do NOT reset patience counter)'.
    """

    patience: int = EARLY_STOP_PATIENCE
    patience_left: int = EARLY_STOP_PATIENCE
    best_ap: float = -float("inf")
    best_epoch: int = 0
    history: list[dict] = field(default_factory=list)

    def update(self, epoch: int, val_ap: float) -> bool:
        """Returns True if this epoch improved best_ap."""
        if val_ap > self.best_ap:
            self.best_ap = val_ap
            self.best_epoch = epoch
            self.patience_left = self.patience
            return True
        self.patience_left -= 1
        return False

    @property
    def stopped(self) -> bool:
        return self.patience_left <= 0


class TwoPhaseTrainer:
    """Doctor-spec 2-phase trainer for binary forensic detection.

    Designed for the unit test to assert 4 invariants at Phase 1 -> Phase 2 transition:
        I1. id(optimizer) changed
        I2. id(scheduler) changed
        I3. scheduler.num_bad_epochs == 0 after reinit
        I4. grad clipping is active in Phase 2 (verified via _apply_grad_clip flag)
        BONUS: early-stop patience_left NOT reset (carries via self.early_stop)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        out_dir: Path | str,
        max_epochs: int = 30,
        device: torch.device | None = None,
        on_epoch_end: Callable | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.on_epoch_end = on_epoch_end

        # State
        self.current_phase: int = 1
        self.current_epoch: int = 0
        self._apply_grad_clip: bool = False  # I4: starts False (Phase 1), True after transition
        self.early_stop = EarlyStopState()

        # Loss is constant across phases
        self.criterion = nn.BCEWithLogitsLoss()

        # Build initial Phase 1 optimizer + scheduler
        freeze_phase1(self.model)
        self._build_optimizer_scheduler(phase=1)

    # phase mgmt
    def _build_optimizer_scheduler(self, phase: int) -> None:
        """(Re)build optimizer + scheduler for the given phase.

        Called at __init__ (phase=1) and at the START of epoch 6 (phase=2).
        After this returns: id(self.optimizer) and id(self.scheduler) are NEW.
        """
        if phase == 1:
            params = trainable_params(self.model)
            lr = PHASE1_LR
            self._apply_grad_clip = False
        elif phase == 2:
            # ALL params trainable
            params = list(self.model.parameters())
            lr = PHASE2_LR
            self._apply_grad_clip = True
        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

        # Doctor spec: AdamW, weight_decay=1e-4
        self.optimizer = AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
        # Doctor spec: ReduceLROnPlateau, mode='max', factor=0.5, patience=3, monitors val AP
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )
        self.current_phase = phase

    def _transition_to_phase2(self) -> None:
        """Execute the Phase 1 -> Phase 2 transition at START of epoch 6.

        Per doctor spec F.20:
            1. Set requires_grad=True for layer1, layer2
            2. Reinitialize AdamW with ALL params at lr=1e-5 wd=1e-4
            3. Reinitialize ReduceLROnPlateau with new optimizer
            4. Enable gradient clipping max_norm=1.0

        Per doctor spec F.21:
            - Early-stop patience_left carries across (NOT reset).
            - Achieved by NOT touching self.early_stop here.
        """
        unfreeze_phase2(self.model)
        self._build_optimizer_scheduler(phase=2)
        # self.early_stop intentionally preserved across boundary.

    # train + eval
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in self.train_loader:
            x, y = (
                batch[0].to(self.device, non_blocking=True),
                batch[1].to(self.device, non_blocking=True),
            )
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x).squeeze(-1)
            loss = self.criterion(logits, y.float())
            loss.backward()
            if self._apply_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=PHASE2_GRAD_CLIP_MAX_NORM
                )
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def _eval_epoch(self) -> tuple[float, float]:
        """Returns (val_ap, val_acc_at_threshold_0.5)."""
        self.model.eval()
        probs_all: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        for batch in self.val_loader:
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1]
            logits = self.model(x).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
        probs_concat = np.concatenate(probs_all)
        labels_concat = np.concatenate(labels_all)
        val_ap = float(average_precision_score(labels_concat, probs_concat))
        val_acc = float(((probs_concat >= 0.5).astype(int) == labels_concat).mean())
        return val_ap, val_acc

    # checkpoint
    def _save_checkpoint(self, path: Path, epoch: int, val_ap: float, val_acc: float) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "phase": self.current_phase,
            "epoch": epoch,
            "val_ap": val_ap,
            "val_acc": val_acc,
            "early_stop": {
                "patience": self.early_stop.patience,
                "patience_left": self.early_stop.patience_left,
                "best_ap": self.early_stop.best_ap,
                "best_epoch": self.early_stop.best_epoch,
            },
        }
        torch.save(payload, path)

    # fit
    def fit(self) -> dict:
        """Run the full Phase-1 -> Phase-2 training schedule with early-stop.

        Returns dict with final metrics + path to best.pt.
        """
        history_path = self.out_dir / "training_history.csv"
        best_path = self.out_dir / "best.pt"
        latest_path = self.out_dir / "latest.pt"
        forensic_named_path = self.out_dir / "forensic_dct_model.pth"  # doctor's required filename

        self.model.to(self.device)

        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "phase",
                    "train_loss",
                    "val_ap",
                    "val_acc",
                    "lr",
                    "patience_left",
                    "is_best",
                ]
            )

        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch

            # Phase 2 transition at START of epoch 6
            if epoch == PHASE1_EPOCHS + 1:
                self._transition_to_phase2()

            # Train + eval
            train_loss = self._train_epoch()
            val_ap, val_acc = self._eval_epoch()

            # Scheduler step
            self.scheduler.step(val_ap)

            # Early-stop update
            is_best = self.early_stop.update(epoch, val_ap)
            self.early_stop.history.append(
                {
                    "epoch": epoch,
                    "phase": self.current_phase,
                    "train_loss": train_loss,
                    "val_ap": val_ap,
                    "val_acc": val_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "patience_left": self.early_stop.patience_left,
                }
            )

            # Persist
            self._save_checkpoint(latest_path, epoch, val_ap, val_acc)
            if is_best:
                self._save_checkpoint(best_path, epoch, val_ap, val_acc)

            with open(history_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        self.current_phase,
                        f"{train_loss:.6f}",
                        f"{val_ap:.6f}",
                        f"{val_acc:.6f}",
                        f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        self.early_stop.patience_left,
                        int(is_best),
                    ]
                )

            if self.on_epoch_end is not None:
                self.on_epoch_end(
                    epoch=epoch,
                    phase=self.current_phase,
                    metrics={
                        "train_loss": train_loss,
                        "val_ap": val_ap,
                        "val_acc": val_acc,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "patience_left": self.early_stop.patience_left,
                        "is_best": is_best,
                    },
                )

            print(
                f"epoch={epoch:02d}  phase={self.current_phase}  "
                f"train_loss={train_loss:.4f}  val_ap={val_ap:.4f}  val_acc={val_acc:.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.1e}  "
                f"patience_left={self.early_stop.patience_left}  best={'*' if is_best else ' '}"
            )

            if self.early_stop.stopped:
                print(
                    f"Early-stop at epoch {epoch} (patience={self.early_stop.patience} on val AP)"
                )
                break

        # Copy best -> doctor-spec filename
        if best_path.exists():
            shutil.copyfile(best_path, forensic_named_path)

        return {
            "best_ap": self.early_stop.best_ap,
            "best_epoch": self.early_stop.best_epoch,
            "final_epoch": self.current_epoch,
            "final_phase": self.current_phase,
            "best_ckpt": str(best_path),
            "forensic_named_ckpt": str(forensic_named_path),
            "history_csv": str(history_path),
        }


__all__ = [
    "EARLY_STOP_PATIENCE",
    "PHASE1_EPOCHS",
    "PHASE1_LR",
    "PHASE2_GRAD_CLIP_MAX_NORM",
    "PHASE2_LR",
    "SCHEDULER_FACTOR",
    "SCHEDULER_PATIENCE",
    "WEIGHT_DECAY",
    "EarlyStopState",
    "TwoPhaseTrainer",
]

"""BaseTrainer - single training loop for all V4 stages.

Per V3.1 section 5.5: AdamW lr=1e-4 wd=1e-4 batch=64, StepLR x0.1 at epoch 30,
Early stopping patience 10 on val F1-macro, gradient clip max_norm=1.0,
bf16 mixed precision (dual 4090).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from v4.core.checkpoints import build_provenance, save_checkpoint
from v4.core.logging import get_logger
from v4.core.seed import seed_all, snapshot_rng_state
from v4.training.losses import build_loss
from v4.training.schedulers import build_scheduler

log = get_logger(__name__)


class BaseTrainer:
    """Single training loop for all V4 stages."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: torch.device,
        *,
        config_hash: str,
        data_hash: str,
        loss_terms: list[dict] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        train_cfg = cfg.train
        self.epochs = int(train_cfg.get("epochs", 50))
        self.lr = float(train_cfg.get("lr", 1e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.patience = int(train_cfg.get("early_stop_patience", 10))
        self.precision = str(train_cfg.get("precision", "bf16"))
        self.out_dir = Path(train_cfg.get("out_dir", "outputs/v4/run"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if loss_terms is None:
            loss_terms = [
                {"type": "ce", "target": "main_logits", "weight": 1.0},
                {
                    "type": "bce",
                    "target": "aux_logits",
                    "weight": float(train_cfg.get("aux_weight", 0.1)),
                },
            ]
        self.loss_fn = build_loss(loss_terms)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = build_scheduler(self.optimizer, train_cfg)

        if self.precision == "bf16" and torch.cuda.is_available():
            self.amp_dtype = torch.bfloat16
            self.amp = True
        elif self.precision == "fp16" and torch.cuda.is_available():
            self.amp_dtype = torch.float16
            self.amp = True
        else:
            self.amp_dtype = torch.float32
            self.amp = False

        self.provenance = build_provenance(
            config_hash=config_hash,
            data_hash=data_hash,
            seed=int(cfg.seed),
            stage=str(cfg.stage),
        )

    def _move_batch(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_n, total_loss = 0, 0.0
        pbar = tqdm(self.train_loader, desc=f"train ep{epoch}", leave=False)
        for batch in pbar:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
                out = self.model(batch)
                if not isinstance(out, dict):
                    out = {"main_logits": out}
                loss, breakdown = self.loss_fn(out, batch["label"], batch.get("aux_label"))
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            n = batch["label"].size(0)
            total_n += n
            total_loss += float(loss.item()) * n
            pbar.set_postfix(loss=f"{breakdown['loss_total']:.3f}")
        return {"train_loss": total_loss / max(total_n, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_n, total_loss = 0, 0.0
        all_preds, all_labels = [], []
        for batch in loader:
            batch = self._move_batch(batch)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
                out = self.model(batch)
                if not isinstance(out, dict):
                    out = {"main_logits": out}
                loss, _ = self.loss_fn(out, batch["label"], batch.get("aux_label"))
            n = batch["label"].size(0)
            total_n += n
            total_loss += float(loss.item()) * n
            preds = out["main_logits"].argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(batch["label"].cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        f1m = f1_score(all_labels, all_preds, average="macro")
        return {"val_loss": total_loss / max(total_n, 1), "f1_macro": float(f1m)}

    def fit(self) -> dict:
        """Run the full training loop. Returns a summary dict with best metrics."""
        seed_all(int(self.cfg.seed), deterministic=bool(self.cfg.deterministic))

        best_f1 = -1.0
        best_epoch = -1
        patience_left = self.patience
        history = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_m = self.train_one_epoch(epoch)
            val_m = self.evaluate(self.val_loader)
            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            log.info(
                "ep %d | lr=%.2e | train_loss=%.4f | val_loss=%.4f | val_f1_macro=%.4f | %.1fs",
                epoch,
                lr,
                train_m["train_loss"],
                val_m["val_loss"],
                val_m["f1_macro"],
                elapsed,
            )
            history.append({"epoch": epoch, **train_m, **val_m, "lr": lr, "seconds": elapsed})

            save_checkpoint(
                self.out_dir / "latest.pt",
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                rng_state=snapshot_rng_state(),
                epoch=epoch,
                val_metrics=val_m,
                config=dict(self.cfg.raw),
                provenance=self.provenance,
            )

            if val_m["f1_macro"] > best_f1:
                best_f1 = val_m["f1_macro"]
                best_epoch = epoch
                patience_left = self.patience
                save_checkpoint(
                    self.out_dir / "best.pt",
                    model_state=self.model.state_dict(),
                    epoch=epoch,
                    val_metrics=val_m,
                    config=dict(self.cfg.raw),
                    provenance=self.provenance,
                )
                log.info("  -> saved best (f1_macro=%.4f)", best_f1)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log.info("early stop at ep %d (best=%.4f at ep %d)", epoch, best_f1, best_epoch)
                    break

        import pandas as pd

        pd.DataFrame(history).to_csv(self.out_dir / "training_history.csv", index=False)
        return {"best_f1_macro": best_f1, "best_epoch": best_epoch, "history": history}

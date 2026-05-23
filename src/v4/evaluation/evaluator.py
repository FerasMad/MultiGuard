"""BaseEvaluator - runs V3.1 section 7 eval on a trained model."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from v4.core.logging import get_logger
from v4.evaluation.reporting import (
    compute_metrics,
    write_classification_report,
    write_confusion_matrix_png,
    write_metrics_json,
)

log = get_logger(__name__)


class BaseEvaluator:
    """Run V3.1 section 7 eval suite over a DataLoader."""

    def __init__(self, model, device: torch.device, num_classes: int = 5):
        self.model = model.to(device).eval()
        self.device = device
        self.num_classes = num_classes

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """Run inference, returning (y_true, y_pred, y_probs, sources)."""
        all_true, all_pred, all_prob, all_src = [], [], [], []
        for batch in loader:
            batch_dev = {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = self.model(batch_dev)
            if not isinstance(out, dict):
                out = {"main_logits": out}
            logits = out["main_logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_true.append(batch["label"].numpy())
            all_pred.append(preds)
            all_prob.append(probs)
            if "source" in batch:
                all_src.extend(batch["source"])
        import numpy as np
        return (
            np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_prob),
            all_src,
        )

    def evaluate(self, loader: DataLoader, *, out_dir: str | Path) -> dict:
        """V3.1 section 7: produce metrics.json + confusion_matrix.png + classification_report.txt."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        y_true, y_pred, y_prob, sources = self.predict(loader)
        metrics = compute_metrics(y_true, y_pred, num_classes=self.num_classes)

        write_metrics_json(metrics, out_dir / "metrics.json")
        write_classification_report(y_true, y_pred, out_dir / "classification_report.txt")
        write_confusion_matrix_png(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")

        # Per-source breakdown (V3 lesson; not in spec but cheap and informative)
        if sources:
            self._per_source_breakdown(y_true, y_pred, sources, out_dir / "per_source_metrics.json")

        log.info("eval F1-macro=%.4f -> %s", metrics["f1_macro"], out_dir)
        return metrics

    def _per_source_breakdown(self, y_true, y_pred, sources, out_path: Path) -> None:
        import numpy as np
        srcs = np.asarray(sources)
        breakdown = {}
        for s in sorted(set(srcs)):
            mask = srcs == s
            if mask.sum() == 0:
                continue
            metrics = compute_metrics(y_true[mask], y_pred[mask], num_classes=self.num_classes)
            breakdown[s] = {
                "n": int(mask.sum()),
                "f1_macro": metrics["f1_macro"],
            }
        write_metrics_json(breakdown, out_path)

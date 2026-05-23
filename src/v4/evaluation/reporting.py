"""V3.1 section 7 evaluation reporting: P, R, F1-macro, confusion matrix."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from v4.core.class_map import LABELS


def compute_metrics(y_true, y_pred, *, num_classes: int = 5) -> dict:
    """V3.1 section 7 metrics: P, R, F1-macro, per-class breakdown."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0,
    )
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    return {
        "f1_macro": f1_macro,
        "per_class": {
            int(c): {
                "name": LABELS.get(int(c), str(c)),
                "precision": float(p[c]),
                "recall": float(r[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
            for c in range(num_classes)
        },
        "confusion_matrix": cm,
    }


def write_classification_report(y_true, y_pred, out_path: Path) -> None:
    """sklearn classification_report -> .txt"""
    target_names = [LABELS[i] for i in range(len(LABELS))]
    txt = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0, digits=4,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8")


def write_confusion_matrix_png(cm, out_path: Path) -> None:
    """Render confusion matrix as a PNG heatmap."""
    cm = np.asarray(cm)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[i] for i in range(n)], rotation=45, ha="right")
    ax.set_yticklabels([LABELS[i] for i in range(n)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_metrics_json(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

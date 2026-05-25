"""Per-generator + aggregate evaluation for the forensic detector.

Doctor's spec F.23-F.24 (MASTER_CHECKLIST):
    Per generator:
        - AP (sklearn.metrics.average_precision_score)
        - Accuracy (at threshold 0.5)
        - AUC (sklearn.metrics.roc_auc_score)
    Aggregates:
        - Overall avg (mean across all 8 generators)
        - GAN avg (BigGAN only)
        - Diffusion avg (other 7: midjourney, sdv1_4, sdv1_5, wukong, vqdm, adm, glide)
        - Std Dev of AP across all 8 generators
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from forensic.data.dct_dataset import DctTestCacheDataset


# Doctor's 8 generators + their type taxonomy
GENERATORS_DIFFUSION: tuple[str, ...] = (
    "midjourney", "sdv1_4", "sdv1_5", "wukong", "vqdm", "adm", "glide",
)
GENERATORS_GAN: tuple[str, ...] = ("biggan",)
ALL_GENERATORS: tuple[str, ...] = GENERATORS_DIFFUSION + GENERATORS_GAN


@torch.no_grad()
def evaluate_one_generator(
    model: nn.Module,
    test_cache_root: str | Path,
    generator: str,
    dct_stats_path: str | Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 2,
    limit: int | None = None,
) -> dict:
    """Run inference on one generator's test set, return AP/Acc/AUC.

    Doctor's spec F.26: 'Load best ckpt, eval mode, torch.no_grad, sigmoid outputs.'
    """
    ds = DctTestCacheDataset(
        root=test_cache_root,
        generator=generator,
        dct_stats_path=dct_stats_path,
        limit=limit,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True,
    )

    model.eval().to(device)
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y))

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)

    # Need both classes present for AP/AUC to be meaningful
    if len(set(labels.tolist())) < 2:
        return {
            "generator": generator,
            "n_samples": int(len(labels)),
            "n_real": int((labels == 0).sum()),
            "n_fake": int((labels == 1).sum()),
            "ap": float("nan"),
            "accuracy": float(((probs >= 0.5).astype(int) == labels).mean()),
            "auc": float("nan"),
            "warning": "only one class present; AP/AUC undefined",
        }

    return {
        "generator": generator,
        "n_samples": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "ap": float(average_precision_score(labels, probs)),
        "accuracy": float(((probs >= 0.5).astype(int) == labels).mean()),
        "auc": float(roc_auc_score(labels, probs)),
    }


def evaluate_per_generator(
    model: nn.Module,
    test_cache_root: str | Path,
    dct_stats_path: str | Path,
    device: torch.device,
    generators: tuple[str, ...] = ALL_GENERATORS,
    batch_size: int = 64,
    num_workers: int = 2,
    limit: int | None = None,
) -> dict[str, dict]:
    """Iterate over all generators (8 by doctor's spec) and collect per-gen metrics.

    Skips generators whose test folder doesn't exist (logs a warning instead).
    """
    results: dict[str, dict] = {}
    for g in generators:
        gen_root = Path(test_cache_root) / g
        if not gen_root.exists():
            results[g] = {
                "generator": g,
                "skipped": True,
                "reason": f"test cache not found at {gen_root}",
            }
            continue
        results[g] = evaluate_one_generator(
            model=model,
            test_cache_root=test_cache_root,
            generator=g,
            dct_stats_path=dct_stats_path,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            limit=limit,
        )
    return results


def aggregate_metrics(per_gen: dict[str, dict]) -> dict:
    """Compute Overall / GAN / Diffusion / StdDev aggregates per doctor's F.24."""
    metrics_present = [
        per_gen[g] for g in per_gen
        if not per_gen[g].get("skipped") and not np.isnan(per_gen[g].get("ap", float("nan")))
    ]

    def _avg(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if not np.isnan(r.get(key, float("nan")))]
        return float(mean(vals)) if vals else float("nan")

    diffusion_rows = [
        per_gen[g] for g in GENERATORS_DIFFUSION
        if g in per_gen and not per_gen[g].get("skipped")
    ]
    gan_rows = [
        per_gen[g] for g in GENERATORS_GAN
        if g in per_gen and not per_gen[g].get("skipped")
    ]

    aps = [r["ap"] for r in metrics_present]
    std_dev_ap = float(stdev(aps)) if len(aps) >= 2 else float("nan")

    return {
        "overall_avg": {
            "ap": _avg(metrics_present, "ap"),
            "accuracy": _avg(metrics_present, "accuracy"),
            "auc": _avg(metrics_present, "auc"),
        },
        "gan_avg": {
            "ap": _avg(gan_rows, "ap"),
            "accuracy": _avg(gan_rows, "accuracy"),
            "auc": _avg(gan_rows, "auc"),
        },
        "diffusion_avg": {
            "ap": _avg(diffusion_rows, "ap"),
            "accuracy": _avg(diffusion_rows, "accuracy"),
            "auc": _avg(diffusion_rows, "auc"),
        },
        "std_dev_ap": std_dev_ap,
        "n_generators_evaluated": len(metrics_present),
    }


__all__ = [
    "ALL_GENERATORS",
    "GENERATORS_DIFFUSION",
    "GENERATORS_GAN",
    "evaluate_one_generator",
    "evaluate_per_generator",
    "aggregate_metrics",
]

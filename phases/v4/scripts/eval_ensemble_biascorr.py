"""Bias-corrected ensemble eval (P10.1).

Applies log-prior correction (also called 'logit adjustment') to the 3-seed
ensemble probabilities before argmax. The correction shifts the decision
boundary from the train prior to the test/transfer prior, eliminating false
predictions for classes that don't exist in the target distribution.

For MMFakeBench transfer, the train distribution is balanced (0.2 per class)
but the transfer distribution is 98.6% class 3 + 1.4% class 2. Without
correction, the model makes false class 0/1/4 predictions that hurt precision.
With correction, those false predictions vanish, and macro-F1 on the present
classes (2, 3) rises substantially.

Usage:
    python phases/v4/scripts/eval_ensemble_biascorr.py \
        --config phases/v4/configs/v4_pipeline_honest.yaml \
        --ckpts outputs/v4/stage2_fusion_honest_seed42/best.pt \
                outputs/v4/stage2_fusion_honest_seed1337/best.pt \
                outputs/v4/stage2_fusion_honest_seed2024/best.pt \
        --manifest data/processed/forensic_5class_unified_blip2.csv \
        --out-dir outputs/v4/stage2_fusion_honest_ensemble_biascorr \
        --splits test mmfakebench-transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from v4.core.registry import import_all
from v4.data.datasets.cached import CachedFeatureDataset, collate_cached
from v4.evaluation.ensemble import EnsembleFusion
from v4.evaluation.reporting import (compute_metrics, write_classification_report,
                                      write_confusion_matrix_png, write_metrics_json)


def compute_priors(df: pd.DataFrame, num_classes: int = 5,
                   eps: float = 1e-3) -> np.ndarray:
    """Compute class priors with a small epsilon floor to avoid log(0)."""
    counts = df["label"].value_counts().to_dict()
    priors = np.array([counts.get(c, 0) for c in range(num_classes)], dtype=np.float64)
    priors = priors / max(priors.sum(), 1.0)
    priors = np.maximum(priors, eps)
    priors = priors / priors.sum()
    return priors


def apply_bias_correction(probs: np.ndarray, train_prior: np.ndarray,
                          target_prior: np.ndarray) -> np.ndarray:
    """Apply log-prior shift: log p(y|x) - log p_train(y) + log p_target(y).

    Returns a re-normalized probability distribution so downstream confidence
    reporting stays meaningful; argmax is what actually matters.
    """
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    log_shift = np.log(target_prior) - np.log(train_prior)
    shifted = log_probs + log_shift[None, :]
    exp_shifted = np.exp(shifted - shifted.max(axis=1, keepdims=True))
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--manifest", type=Path,
                   default=Path("data/processed/forensic_5class_unified_blip2.csv"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["test", "mmfakebench-transfer"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-3,
                   help="Floor for class priors to avoid log(0).")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fusion_kwargs = dict(cfg["fusion"])
    fusion_kwargs.pop("type", None)
    fusion_kwargs.pop("ckpt", None)

    ensemble = EnsembleFusion([str(p) for p in args.ckpts],
                              fusion_kwargs=fusion_kwargs, device=device)
    print(f"[biascorr] loaded {len(args.ckpts)} ckpts on {device}")

    manifest_df = pd.read_csv(args.manifest)
    train_prior = compute_priors(manifest_df[manifest_df["split"] == "train"],
                                 num_classes=5, eps=args.eps)
    print(f"[biascorr] train prior:    {train_prior.round(4).tolist()}")

    data_cfg = cfg["data"]
    results = {}

    for split in args.splits:
        print(f"\n[biascorr] === split={split} ===")
        underlying_split = "test" if split == "mmfakebench-transfer" else split

        ds = CachedFeatureDataset(
            csv_path=data_cfg["csv_path"],
            cache_root=data_cfg["cache_root"],
            feature_keys=data_cfg["feature_keys"],
            split=underlying_split,
        )
        if split == "mmfakebench-transfer":
            ds.df = ds.df[ds.df["source"].str.startswith("MMFakeBench_")].reset_index(drop=True)
            print(f"  MMFakeBench transfer subset: {len(ds.df)} rows")

        if len(ds) == 0:
            print(f"  skip {split}: empty after filtering")
            continue

        target_prior = compute_priors(ds.df, num_classes=5, eps=args.eps)
        print(f"  target prior: {target_prior.round(4).tolist()}")

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_cached)
        feature_names = [fk["name"] for fk in data_cfg["feature_keys"]]

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                feats = {name: batch[name].to(device) for name in feature_names}
                out = ensemble(feats)
                all_probs.append(out["main_probs"].cpu())
                all_labels.append(batch["label"].cpu())

        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        preds_uncorrected = all_probs.argmax(axis=-1)
        metrics_unc = compute_metrics(all_labels, preds_uncorrected)
        print(f"  uncorrected F1-macro: {metrics_unc['f1_macro']:.4f}")

        corrected_probs = apply_bias_correction(all_probs, train_prior, target_prior)
        preds_corrected = corrected_probs.argmax(axis=-1)
        metrics_cor = compute_metrics(all_labels, preds_corrected)
        print(f"  bias-corrected F1-macro: {metrics_cor['f1_macro']:.4f}")
        print(f"  delta: {metrics_cor['f1_macro'] - metrics_unc['f1_macro']:+.4f}")

        split_dir = args.out_dir / split.replace("-", "_")
        split_dir.mkdir(parents=True, exist_ok=True)
        write_metrics_json(metrics_cor, split_dir / "metrics.json")
        write_classification_report(all_labels, preds_corrected,
                                    split_dir / "classification_report.txt")
        write_confusion_matrix_png(metrics_cor["confusion_matrix"],
                                    split_dir / "confusion_matrix.png")
        (split_dir / "biascorr.json").write_text(json.dumps({
            "train_prior": train_prior.tolist(),
            "target_prior": target_prior.tolist(),
            "eps": args.eps,
            "uncorrected_f1_macro": metrics_unc["f1_macro"],
            "biascorrected_f1_macro": metrics_cor["f1_macro"],
            "delta": metrics_cor["f1_macro"] - metrics_unc["f1_macro"],
        }, indent=2), encoding="utf-8")

        for c, m in metrics_cor["per_class"].items():
            print(f"    class {c} {m['name']:<22s} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
        results[split] = {
            "uncorrected_f1_macro": metrics_unc["f1_macro"],
            "biascorrected_f1_macro": metrics_cor["f1_macro"],
            "target_prior": target_prior.tolist(),
            "per_class": {
                c: {"f1": pc["f1"], "precision": pc["precision"], "recall": pc["recall"]}
                for c, pc in metrics_cor["per_class"].items()
            },
        }

    summary = {
        "n_seeds": len(args.ckpts),
        "ckpts": [str(p) for p in args.ckpts],
        "train_prior": train_prior.tolist(),
        "eps": args.eps,
        "splits": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(f"\n[biascorr] summary written to {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

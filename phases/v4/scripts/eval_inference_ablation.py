"""P15.2 + P15.3: inference-time leave-one-out ablation on the shipped 3-seed ensemble.

Loads outputs/v4/stage2_fusion_honest_seed{42,1337,2024}/best.pt and runs the
test split under four `disable_branches` settings, with and without the
train/test image_path leak filter (P15.3).

Answers two distinct questions left open by the prior P14 ablation:
  1. "If we drop branch X from the SHIPPED pipeline at inference, what happens
      to test F1?" -- a leave-one-out on the trained model, not a fresh-train
      per variant.
  2. "Are the headline F1 numbers inflated by the 16 image_path values that
      appear in both train and test (A3 audit finding)?"

For each (variant, split) combination, writes per-class F1 + confusion matrix
and aggregates everything into outputs/v4/inference_ablation/summary.json.

Run:
    .venv/Scripts/python.exe phases/v4/scripts/eval_inference_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from v4.core.registry import import_all
from v4.data.datasets.cached import CachedFeatureDataset, collate_cached
from v4.evaluation.ensemble import EnsembleFusion
from v4.evaluation.reporting import (
    compute_metrics,
    write_confusion_matrix_png,
    write_metrics_json,
)

VARIANTS = {
    "all": [],
    "no_image": ["v_imgfor"],
    "no_text": ["v_textfor"],
    "semantic_only": ["v_imgfor", "v_textfor"],
}


def _compute_test_leak_paths(csv_path: str) -> set[str]:
    df = pd.read_csv(csv_path)
    train_paths = set(df[df["split"] == "train"]["image_path"])
    test_paths = set(df[df["split"] == "test"]["image_path"])
    return train_paths & test_paths


def _build_loader(
    csv_path: str,
    cache_root: str,
    feature_keys: list[dict],
    *,
    leak_paths: set[str] | None,
    batch_size: int,
) -> DataLoader:
    ds = CachedFeatureDataset(
        csv_path=csv_path,
        cache_root=cache_root,
        feature_keys=feature_keys,
        split="test",
    )
    if leak_paths:
        before = len(ds.df)
        ds.df = ds.df[~ds.df["image_path"].isin(leak_paths)].reset_index(drop=True)
        print(f"  filtered {before - len(ds.df)} leak rows, kept {len(ds.df)}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_cached
    )


def _run_one(
    ensemble: EnsembleFusion,
    loader: DataLoader,
    feature_names: list[str],
    device: torch.device,
) -> dict:
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            feats = {name: batch[name].to(device) for name in feature_names}
            out = ensemble(feats)
            all_probs.append(out["main_probs"].cpu())
            all_labels.append(batch["label"].cpu())
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = probs.argmax(axis=-1)
    return compute_metrics(labels, preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("phases/v4/configs/v4_pipeline_honest.yaml"),
    )
    p.add_argument(
        "--ckpts",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/v4/stage2_fusion_honest_seed42/best.pt"),
            Path("outputs/v4/stage2_fusion_honest_seed1337/best.pt"),
            Path("outputs/v4/stage2_fusion_honest_seed2024/best.pt"),
        ],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/v4/inference_ablation"),
    )
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    feature_names = [fk["name"] for fk in data_cfg["feature_keys"]]
    leak_paths = _compute_test_leak_paths(data_cfg["csv_path"])
    print(f"[ablation] manifest: {data_cfg['csv_path']}")
    print(f"[ablation] train/test image_path leaks: {len(leak_paths)}")

    base_fusion_kwargs = dict(cfg["fusion"])
    base_fusion_kwargs.pop("type", None)
    base_fusion_kwargs.pop("ckpt", None)

    splits = {
        "test": None,
        "test_clean": leak_paths if leak_paths else None,
    }

    summary: dict[str, dict] = {}

    for variant_name, disable in VARIANTS.items():
        fusion_kwargs = dict(base_fusion_kwargs)
        fusion_kwargs["disable_branches"] = disable
        ensemble = EnsembleFusion(
            [str(p) for p in args.ckpts], fusion_kwargs=fusion_kwargs, device=device
        )
        print(
            f"\n[ablation] === variant={variant_name} "
            f"disable_branches={disable} (n_seeds={len(args.ckpts)}) ==="
        )

        for split_name, leaks in splits.items():
            if split_name == "test_clean" and not leaks:
                continue  # nothing to filter
            print(f"  [{variant_name}/{split_name}]")
            loader = _build_loader(
                data_cfg["csv_path"],
                data_cfg["cache_root"],
                data_cfg["feature_keys"],
                leak_paths=leaks,
                batch_size=args.batch_size,
            )
            metrics = _run_one(ensemble, loader, feature_names, device)
            run_dir = args.out_dir / f"{variant_name}_{split_name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_metrics_json(metrics, run_dir / "metrics.json")
            write_confusion_matrix_png(
                metrics["confusion_matrix"], run_dir / "confusion_matrix.png"
            )
            per_class = metrics["per_class"]
            print(
                f"    F1-macro={metrics['f1_macro']:.4f} "
                f"C0={per_class[0]['f1']:.3f} C1={per_class[1]['f1']:.3f} "
                f"C2={per_class[2]['f1']:.3f} C3={per_class[3]['f1']:.3f} "
                f"C4={per_class[4]['f1']:.3f}"
            )
            summary.setdefault(variant_name, {})[split_name] = {
                "f1_macro": metrics["f1_macro"],
                "per_class_f1": {c: per_class[c]["f1"] for c in per_class},
                "n": sum(per_class[c]["support"] for c in per_class),
            }

    out_path = args.out_dir / "summary.json"
    out_path.write_text(
        json.dumps(
            {
                "ckpts": [str(p) for p in args.ckpts],
                "n_leaks_filtered": len(leak_paths),
                "variants": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[ablation] summary -> {out_path}")

    print("\n## Test F1-macro per variant (shipped 3-seed ensemble, inference-only)")
    print("| Variant | test | test_clean |")
    print("|---|---|---|")
    for v in VARIANTS:
        if v not in summary:
            continue
        row = summary[v]
        t = row.get("test", {}).get("f1_macro", float("nan"))
        tc = row.get("test_clean", {}).get("f1_macro", float("nan"))
        print(f"| {v} | {t:.4f} | {tc:.4f} |")


if __name__ == "__main__":
    main()

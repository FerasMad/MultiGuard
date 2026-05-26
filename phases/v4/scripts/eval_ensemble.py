"""Evaluate the V4 multi-seed ensemble on test + transfer splits (P9.7).

Usage:
    python phases/v4/scripts/eval_ensemble.py \
        --config phases/v4/configs/v4_pipeline_honest.yaml \
        --ckpts outputs/v4/stage2_fusion_honest_seed42/best.pt \
                outputs/v4/stage2_fusion_honest_seed1337/best.pt \
                outputs/v4/stage2_fusion_honest_seed2024/best.pt \
        --out-dir outputs/v4/stage2_fusion_honest_ensemble \
        --splits test mmfakebench-transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from v4.core.registry import import_all
from v4.data.datasets.cached import CachedFeatureDataset, collate_cached
from v4.evaluation.ensemble import EnsembleFusion
from v4.evaluation.reporting import (compute_metrics, write_classification_report,
                                      write_confusion_matrix_png, write_metrics_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["test", "mmfakebench-transfer"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=None)
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
    print(f"[ensemble] loaded {len(args.ckpts)} ckpts on {device}")

    data_cfg = cfg["data"]
    results = {}

    for split in args.splits:
        print(f"\n[ensemble] === split={split} ===")
        # Mirror v4.cli.eval_main: mmfakebench-transfer is a filter on top of test split
        underlying_split = "test" if split == "mmfakebench-transfer" else split
        try:
            ds = CachedFeatureDataset(
                csv_path=data_cfg["csv_path"],
                cache_root=data_cfg["cache_root"],
                feature_keys=data_cfg["feature_keys"],
                split=underlying_split,
            )
            if split == "mmfakebench-transfer":
                ds.df = ds.df[ds.df["source"].str.startswith("MMFakeBench_")].reset_index(drop=True)
                print(f"  MMFakeBench transfer subset: {len(ds.df)} rows")
        except Exception as e:
            print(f"  skip {split}: {type(e).__name__}: {e}")
            continue
        if len(ds) == 0:
            print(f"  skip {split}: empty after filtering")
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_cached)

        feature_names = [fk["name"] for fk in data_cfg["feature_keys"]]
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                feats = {name: batch[name].to(device) for name in feature_names}
                out = ensemble(feats)
                probs = out["main_probs"]
                if args.temperature is not None and args.temperature != 1.0:
                    logp = torch.log(probs.clamp(min=1e-12))
                    probs = F.softmax(logp / args.temperature, dim=-1)
                all_probs.append(probs.cpu())
                all_labels.append(batch["label"].cpu())

        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        preds = all_probs.argmax(axis=-1)

        metrics = compute_metrics(all_labels, preds)
        split_dir = args.out_dir / split.replace("-", "_")
        split_dir.mkdir(parents=True, exist_ok=True)
        write_metrics_json(metrics, split_dir / "metrics.json")
        write_classification_report(all_labels, preds, split_dir / "classification_report.txt")
        write_confusion_matrix_png(metrics["confusion_matrix"], split_dir / "confusion_matrix.png")

        print(f"  ensemble F1-macro={metrics['f1_macro']:.4f}")
        for c, m in metrics["per_class"].items():
            print(f"    class {c} {m['name']:<22s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
        results[split] = {"f1_macro": metrics["f1_macro"]}

    (args.out_dir / "summary.json").write_text(
        json.dumps({
            "n_seeds": len(args.ckpts),
            "ckpts": [str(p) for p in args.ckpts],
            "temperature": args.temperature,
            "splits": results,
        }, indent=2), encoding="utf-8")
    print(f"\n[ensemble] summary: {results}")


if __name__ == "__main__":
    main()

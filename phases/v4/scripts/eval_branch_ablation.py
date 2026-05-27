"""Stage B4 -- Evaluate the 4 ablation ckpts on test split + aggregate (P14.B4).

After `train_branch_ablation.py --variant sweep` completes, this script runs
`python -m v4 eval` for each variant's best.pt on the test split, collects
metrics into a single comparison table, and writes
`outputs/v4/branch_ablation_eval.json` plus a markdown table fragment that
slots into `docs/BRANCH_ABLATION_REPORT.md`.

Reads `outputs/v4/stage2_ablation_<variant>/best.pt` and writes per-variant
eval artifacts into `outputs/v4/eval/ablation_<variant>_test/`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from v4.core.class_map import LABELS

VARIANTS = {
    "fndclip": "phases/v4/configs/v4_pipeline_ablation_fndclip.yaml",
    "fndclip_text": "phases/v4/configs/v4_pipeline_ablation_fndclip_text.yaml",
    "fndclip_image": "phases/v4/configs/v4_pipeline_ablation_fndclip_image.yaml",
    "all": "phases/v4/configs/v4_pipeline_ablation_all.yaml",
}


def _eval_one(variant: str, config: Path, *, python: str, split: str = "test") -> dict | None:
    ckpt = Path(f"outputs/v4/stage2_ablation_{variant}/best.pt")
    if not ckpt.exists():
        print(f"SKIP {variant}: no best.pt at {ckpt}")
        return None

    out_dir = Path(f"outputs/v4/eval/ablation_{variant}_{split}")
    cmd = [
        python, "-m", "v4", "eval",
        "--config", str(config),
        "--checkpoint", str(ckpt),
        "--split", split,
        "--out-dir", str(out_dir),
    ]
    print(f"\n--- eval {variant} ({split}) ---")
    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"FAIL {variant}: eval rc={rc}")
        return None

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"FAIL {variant}: missing {metrics_path}")
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "variant": variant,
        "config": str(config),
        "ckpt": str(ckpt),
        "split": split,
        "metrics_path": str(metrics_path),
        "f1_macro": metrics["f1_macro"],
        "per_class_f1": [metrics["per_class"][str(c)]["f1"] for c in range(len(LABELS))],
        "per_class_precision": [
            metrics["per_class"][str(c)]["precision"] for c in range(len(LABELS))
        ],
        "per_class_recall": [
            metrics["per_class"][str(c)]["recall"] for c in range(len(LABELS))
        ],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--variants", nargs="*", default=list(VARIANTS),
                   help="Which variants to evaluate.")
    p.add_argument("--python", default=str(Path(".venv/Scripts/python.exe")))
    p.add_argument("--out", default="outputs/v4/branch_ablation_eval.json")
    args = p.parse_args()

    results = []
    for v in args.variants:
        cfg = Path(VARIANTS[v])
        r = _eval_one(v, cfg, python=args.python, split=args.split)
        if r is not None:
            results.append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}  ({len(results)} variants)")

    # Print a clean comparison table.
    print(f"\n## Per-variant comparison (split={args.split})")
    print("| Variant | F1-macro | C0 Real | C1 OOC | C2 Manip | C3 AI-Text | C4 FullFab |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        f1s = r["per_class_f1"]
        print(
            f"| {r['variant']:14s} | {r['f1_macro']:.4f} | "
            f"{f1s[0]:.3f} | {f1s[1]:.3f} | {f1s[2]:.3f} | {f1s[3]:.3f} | {f1s[4]:.3f} |"
        )
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())

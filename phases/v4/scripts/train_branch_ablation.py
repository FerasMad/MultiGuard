"""Stage B1+B2 -- Branch ablation training driver (P14.B1+B2+B6).

Trains the four V3PairwiseFusion variants sequentially on the existing cached
features (no re-encoding required). Used to quantify per-branch contribution
to the 5-class downstream task, per the non-image fix plan section 7.

Variants:
  - fndclip          : disable_branches=[v_imgfor, v_textfor]  (FND-CLIP alone)
  - fndclip_text     : disable_branches=[v_imgfor]             (FND-CLIP + text)
  - fndclip_image    : disable_branches=[v_textfor]            (FND-CLIP + image)
  - all              : disable_branches=[]                     (full pipeline)

Each variant uses the same cached features (`v_semantic` from V1 leakfree,
`v_imgfor` from forensic dct ckpt, `v_textfor` from BLIP-2 honest captions)
and the same seed=42 as the shipped honest-path Stage-2.

After all variants finish, run `python phases/v4/scripts/eval_branch_ablation.py`
to compute the per-class F1 deltas for `docs/BRANCH_ABLATION_REPORT.md`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VARIANTS = {
    "fndclip": "phases/v4/configs/v4_pipeline_ablation_fndclip.yaml",
    "fndclip_text": "phases/v4/configs/v4_pipeline_ablation_fndclip_text.yaml",
    "fndclip_image": "phases/v4/configs/v4_pipeline_ablation_fndclip_image.yaml",
    "all": "phases/v4/configs/v4_pipeline_ablation_all.yaml",
}


def _run_one(variant: str, config_path: Path, *, python: str, epochs: int | None,
             limit: int | None) -> dict:
    cmd = [python, "-m", "v4", "train", "--config", str(config_path)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"\n=========== {variant} ============", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    duration = time.time() - t0
    ok = rc == 0
    print(f"  -> variant={variant} rc={rc} duration={duration:.1f}s", flush=True)

    out_dir = Path(f"outputs/v4/stage2_ablation_{variant}")
    best_f1 = None
    best_epoch = None
    if ok:
        hist = out_dir / "training_history.csv"
        if hist.exists():
            import pandas as pd
            df = pd.read_csv(hist)
            if "f1_macro" in df.columns and len(df):
                best_row = df.loc[df["f1_macro"].idxmax()]
                best_f1 = float(best_row["f1_macro"])
                best_epoch = int(best_row["epoch"])

    summary = {
        "variant": variant,
        "config": str(config_path),
        "rc": rc,
        "ok": ok,
        "duration_s": round(duration, 1),
        "best_f1_macro": best_f1,
        "best_epoch": best_epoch,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS) + ["sweep"], default="sweep",
                   help="Which ablation to run; 'sweep' runs all four sequentially.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override the YAML's max-epochs (smoke tests usually pass 1).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap dataset rows per split (smoke testing only).")
    p.add_argument("--python", default=str(Path(".venv/Scripts/python.exe")),
                   help="Python interpreter to use for the trainer subprocess.")
    args = p.parse_args()

    variants = list(VARIANTS) if args.variant == "sweep" else [args.variant]

    summaries = []
    for v in variants:
        cfg_path = Path(VARIANTS[v])
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing ablation config: {cfg_path}")
        summaries.append(
            _run_one(v, cfg_path, python=args.python, epochs=args.epochs, limit=args.limit)
        )

    out_root = Path("outputs/v4")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "branch_ablation_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print("\n=========== ablation summary ============")
    print(json.dumps(summaries, indent=2))
    print("-> wrote outputs/v4/branch_ablation_summary.json")

    bad = [s for s in summaries if not s["ok"]]
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

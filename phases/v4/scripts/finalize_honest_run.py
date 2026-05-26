"""Orchestrator: Phase 7 — per-seed eval + ensemble eval + summary write (P9.7).

Usage:
    python phases/v4/scripts/finalize_honest_run.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_cli(args: list[str]) -> tuple[int, str]:
    print("$ " + " ".join(args), flush=True)
    r = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)
    if r.stdout:
        print(r.stdout)
    if r.returncode != 0 and r.stderr:
        print("STDERR:", r.stderr)
    return r.returncode, r.stdout + r.stderr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=Path("phases/v4/configs/v4_pipeline_honest.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    p.add_argument("--ckpt-dir-template",
                   default="outputs/v4/stage2_fusion_honest_seed{seed}")
    p.add_argument("--ensemble-out", type=Path,
                   default=Path("outputs/v4/stage2_fusion_honest_ensemble"))
    args = p.parse_args()

    results: dict = {"seeds": {}, "ensemble": {}, "ensemble_calibrated": {}}

    ckpts = []
    for seed in args.seeds:
        ckpt_dir = Path(args.ckpt_dir_template.format(seed=seed))
        ckpt = ckpt_dir / "best.pt"
        if not ckpt.exists():
            print(f"WARN: seed={seed} ckpt missing at {ckpt} -- skipping")
            continue
        ckpts.append(ckpt)
        for split in ["test", "mmfakebench-transfer"]:
            out_dir = ckpt_dir / "eval" / split.replace("-", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "v4", "eval",
                   "--config", str(args.config),
                   "--checkpoint", str(ckpt),
                   "--split", split,
                   "--out-dir", str(out_dir)]
            rc, _ = run_cli(cmd)
            if rc != 0:
                continue
            mf = out_dir / "metrics.json"
            if mf.exists():
                m = json.loads(mf.read_text(encoding="utf-8"))
                results["seeds"].setdefault(str(seed), {})[split] = {
                    "f1_macro": m["f1_macro"],
                    "per_class": {
                        c: {"f1": pc["f1"], "precision": pc["precision"], "recall": pc["recall"]}
                        for c, pc in m["per_class"].items()
                    },
                }

    if not ckpts:
        print("ERROR: no seed ckpts found"); sys.exit(1)

    if len(ckpts) >= 2:
        args.ensemble_out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "phases/v4/scripts/eval_ensemble.py",
               "--config", str(args.config),
               "--ckpts"] + [str(c) for c in ckpts] + [
               "--out-dir", str(args.ensemble_out),
               "--splits", "test", "mmfakebench-transfer"]
        rc, _ = run_cli(cmd)
        if rc == 0:
            for split in ["test", "mmfakebench_transfer"]:
                mf = args.ensemble_out / split / "metrics.json"
                if mf.exists():
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    results["ensemble"][split] = {
                        "f1_macro": m["f1_macro"],
                        "per_class": {
                            c: {"f1": pc["f1"], "precision": pc["precision"], "recall": pc["recall"]}
                            for c, pc in m["per_class"].items()
                        },
                    }

    primary_ckpt = Path(args.ckpt_dir_template.format(seed=args.seeds[0])) / "best.pt"
    if primary_ckpt.exists():
        temp_file = args.ensemble_out / "temperature.json"
        cmd = [sys.executable, "phases/v4/scripts/calibrate_temperature.py",
               "--config", str(args.config),
               "--checkpoint", str(primary_ckpt),
               "--out", str(temp_file)]
        rc, _ = run_cli(cmd)
        if rc == 0 and temp_file.exists():
            t = json.loads(temp_file.read_text(encoding="utf-8"))
            results["temperature"] = t
            if len(ckpts) >= 2 and "temperature" in t:
                T = t["temperature"]
                cmd = [sys.executable, "phases/v4/scripts/eval_ensemble.py",
                       "--config", str(args.config),
                       "--ckpts"] + [str(c) for c in ckpts] + [
                       "--out-dir", str(args.ensemble_out) + "_calibrated",
                       "--splits", "test", "mmfakebench-transfer",
                       "--temperature", str(T)]
                rc, _ = run_cli(cmd)
                if rc == 0:
                    for split in ["test", "mmfakebench_transfer"]:
                        mf = Path(str(args.ensemble_out) + "_calibrated") / split / "metrics.json"
                        if mf.exists():
                            m = json.loads(mf.read_text(encoding="utf-8"))
                            results["ensemble_calibrated"][split] = {
                                "f1_macro": m["f1_macro"],
                                "per_class": {
                                    c: {"f1": pc["f1"], "precision": pc["precision"], "recall": pc["recall"]}
                                    for c, pc in m["per_class"].items()
                                },
                            }

    if len(results["seeds"]) >= 2:
        import statistics
        for split in ["test", "mmfakebench-transfer"]:
            vals = [results["seeds"][s].get(split, {}).get("f1_macro")
                    for s in results["seeds"] if split in results["seeds"][s]]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                results.setdefault("seed_summary", {})[split] = {
                    "mean": float(statistics.mean(vals)),
                    "std": float(statistics.stdev(vals)),
                    "n": len(vals),
                    "values": vals,
                }

    out_summary = REPO_ROOT / "phases/v4/docs/eval/honest_run_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[finalize] wrote {out_summary}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

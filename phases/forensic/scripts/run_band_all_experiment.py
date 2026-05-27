"""Stage C-lite -- Approach 1 retrain at band='all' + per-generator delta vs shipped (P14.C-lite).

Single retrain experiment to settle whether the shipped Approach-1 ckpt's
band='low+high' Fourier masking choice was suboptimal vs upstream's default
band='all'. See docs/FOURIER_BAND_AUDIT.md for the behavioral diff.

Workflow:
  1. Backup shipped ckpt: forensic_rgb_model.pth -> forensic_rgb_model.lowhigh.pth
     (defensive; skip if already exists)
  2. Train Approach 1 with --band all on the existing VisualNews-as-nature splits,
     same hyperparams as the shipped 0.9971-AP run.
  3. Eval shipped (band=low+high) and new (band=all) on the same 6/8 generators
     (sd_v1_4 + sd_v1_5 still missing).
  4. Emit a delta JSON; the markdown doc is written by hand after inspecting it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SHIPPED_CKPT = Path("phases/forensic/outputs/rgb/forensic_rgb_model.pth")
SHIPPED_BACKUP = Path("phases/forensic/outputs/rgb/forensic_rgb_model.lowhigh.pth")
NEW_OUT_DIR = Path("phases/forensic/outputs/rgb_bandall")
NEW_CKPT = NEW_OUT_DIR / "forensic_rgb_model.pth"


def _backup_shipped() -> bool:
    if not SHIPPED_CKPT.exists():
        print(f"WARNING: shipped ckpt missing at {SHIPPED_CKPT}; nothing to backup.")
        return False
    if SHIPPED_BACKUP.exists():
        print(f"OK backup already exists at {SHIPPED_BACKUP}")
        return True
    shutil.copyfile(SHIPPED_CKPT, SHIPPED_BACKUP)
    print(f"backup: {SHIPPED_CKPT} -> {SHIPPED_BACKUP}")
    return True


def _train_band_all(python: str, epochs: int | None, limit_batches: int | None) -> bool:
    cmd = [
        python, "phases/forensic/scripts/train_rgb_fourier.py",
        "--band", "all",
        "--out-dir", str(NEW_OUT_DIR),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if limit_batches is not None:
        cmd += ["--limit-batches", str(limit_batches)]
    print("\n--- train band=all ---")
    print(" ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    duration = time.time() - t0
    print(f"  -> rc={rc} duration={duration:.1f}s")
    return rc == 0


def _eval(ckpt: Path, out_json: Path, python: str) -> dict | None:
    cmd = [
        python, "phases/forensic/scripts/eval_rgb.py",
        "--ckpt", str(ckpt),
        "--out-json", str(out_json),
        "--out-table", str(out_json.with_suffix(".md")),
    ]
    print(f"\n--- eval {ckpt} ---")
    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"FAIL: eval returned rc={rc}")
        return None
    return json.loads(out_json.read_text(encoding="utf-8"))


def _compute_deltas(shipped: dict, new: dict) -> dict:
    """Per-generator and overall AP/Acc/AUC deltas (new - shipped)."""
    def _safe(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    s_per = shipped["per_generator"]
    n_per = new["per_generator"]
    gens = sorted(set(s_per) & set(n_per))
    rows = []
    for g in gens:
        s, n = s_per[g], n_per[g]
        if s.get("skipped") or n.get("skipped"):
            rows.append({
                "generator": g,
                "shipped_skipped": bool(s.get("skipped")),
                "new_skipped": bool(n.get("skipped")),
            })
            continue
        sap, nap = _safe(s.get("ap")), _safe(n.get("ap"))
        sac, nac = _safe(s.get("accuracy")), _safe(n.get("accuracy"))
        sau, nau = _safe(s.get("auc")), _safe(n.get("auc"))
        rows.append({
            "generator": g,
            "shipped_ap": sap,
            "new_ap": nap,
            "delta_ap": (nap - sap) if (sap is not None and nap is not None) else None,
            "shipped_acc": sac,
            "new_acc": nac,
            "delta_acc": (nac - sac) if (sac is not None and nac is not None) else None,
            "shipped_auc": sau,
            "new_auc": nau,
            "delta_auc": (nau - sau) if (sau is not None and nau is not None) else None,
        })
    return {
        "shipped_ckpt": shipped.get("ckpt"),
        "new_ckpt": new.get("ckpt"),
        "shipped_aggregates": shipped.get("aggregates"),
        "new_aggregates": new.get("aggregates"),
        "per_generator_delta": rows,
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Smoke test: cap dataset size")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training (assume NEW_CKPT already exists; only re-eval)")
    p.add_argument("--python", default=str(Path(".venv/Scripts/python.exe")))
    args = p.parse_args()

    _backup_shipped()

    if not args.skip_train:
        ok = _train_band_all(args.python, args.epochs, args.limit_batches)
        if not ok:
            print("FATAL: band=all training failed; aborting before eval.")
            return 1
    elif not NEW_CKPT.exists():
        print(f"FATAL: --skip-train set but NEW_CKPT missing at {NEW_CKPT}")
        return 1

    # Eval shipped (band=low+high) and new (band=all) using the SAME eval pipeline.
    shipped_eval_path = Path("phases/forensic/outputs/eval_rgb_lowhigh.json")
    new_eval_path = Path("phases/forensic/outputs/eval_rgb_bandall.json")

    shipped_eval = _eval(SHIPPED_CKPT, shipped_eval_path, args.python)
    new_eval = _eval(NEW_CKPT, new_eval_path, args.python)

    if shipped_eval is None or new_eval is None:
        print("FATAL: at least one eval failed; cannot compute deltas.")
        return 1

    delta = _compute_deltas(shipped_eval, new_eval)
    out_path = Path("phases/forensic/outputs/band_experiment.json")
    out_path.write_text(json.dumps(delta, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    print("\n## Delta summary (new - shipped, band='all' vs band='low+high')")
    print("| Generator | shipped AP | new AP | dAP | dAcc | dAUC |")
    print("|---|---|---|---|---|---|")
    for r in delta["per_generator_delta"]:
        if r.get("shipped_skipped") or r.get("new_skipped"):
            continue
        d_ap = r.get("delta_ap")
        d_ac = r.get("delta_acc")
        d_au = r.get("delta_auc")
        print(
            f"| {r['generator']:10s} | {r['shipped_ap']:.4f} | {r['new_ap']:.4f} | "
            f"{(d_ap or 0):+.4f} | {(d_ac or 0):+.4f} | {(d_au or 0):+.4f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Precompute DCT cache for train/val/test splits.

Doctor's spec F.18 (MASTER_CHECKLIST):
    .pt cache: float32, shape [1,224,224], precomputed for train/val/test (never recomputed)

Output layout (matches forensic.data.dct_dataset.DctCacheDataset expectations):
    phases/forensic/data/dct_cache/
      train/{0_real,1_fake}/<orig_filename>.pt
      val/{0_real,1_fake}/<orig_filename>.pt
      test/<generator>/{0_real,1_fake}/<orig_filename>.pt

Each .pt is the OUTPUT of dual_dct.compute_dual_dct (UNNORMALIZED).
Z-score is applied at load time by DctCacheDataset using dct_stats.json
(generated separately by compute_dct_stats.py).

Multiprocess: uses Pool with spawn start method to keep scipy.fft isolated
from torch CUDA (per Risk R2 mitigation in plan).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch


def _process_one(args: tuple[str, str]) -> tuple[str, bool, str]:
    """Compute DCT for one image, save as .pt. Returns (out_path, ok, error_msg)."""
    img_path, out_path = args
    # Import inside worker so scipy/numpy don't get initialized in parent
    # (parent may have torch CUDA active).
    from forensic.preprocessing.dual_dct import compute_dual_dct

    try:
        out_p = Path(out_path)
        if out_p.exists():
            return (out_path, True, "skipped_existing")
        t = compute_dual_dct(img_path)  # [1, 224, 224] float32
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(t, out_p)
        return (out_path, True, "")
    except Exception as e:
        return (out_path, False, f"{type(e).__name__}: {e}")


def _collect_jobs(src_root: Path, dst_root: Path) -> list[tuple[str, str]]:
    """Walk src_root, mirror structure under dst_root with .pt suffix."""
    jobs: list[tuple[str, str]] = []
    for img in src_root.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel = img.relative_to(src_root)
        out = dst_root / rel.with_suffix(".pt")
        jobs.append((str(img), str(out)))
    return jobs


def _safe_log(msg: str) -> None:
    print(f"[precompute_dct] {msg}", flush=True)


def precompute_split(
    src_root: Path,
    dst_root: Path,
    n_workers: int = 4,
    chunksize: int = 8,
) -> dict:
    """Precompute DCT for one split (e.g. train/, val/, test/)."""
    if not src_root.exists():
        return {"src": str(src_root), "status": "src_not_found", "ok": 0, "fail": 0}

    jobs = _collect_jobs(src_root, dst_root)
    if not jobs:
        return {"src": str(src_root), "status": "no_images", "ok": 0, "fail": 0}

    _safe_log(f"  {len(jobs)} images to process from {src_root} -> {dst_root}")

    t0 = time.time()
    ok_count = 0
    fail_count = 0
    fail_examples: list[str] = []
    skipped = 0

    # spawn start method ensures clean worker processes (no inherited torch CUDA)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for i, (out_path, ok, msg) in enumerate(
            pool.imap_unordered(_process_one, jobs, chunksize=chunksize)
        ):
            if ok:
                ok_count += 1
                if msg == "skipped_existing":
                    skipped += 1
            else:
                fail_count += 1
                if len(fail_examples) < 5:
                    fail_examples.append(f"{out_path}: {msg}")
            if (i + 1) % 500 == 0 or (i + 1) == len(jobs):
                pct = 100.0 * (i + 1) / len(jobs)
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                _safe_log(
                    f"  progress: {i + 1}/{len(jobs)} ({pct:.1f}%) at {rate:.1f}/s, fails={fail_count}"
                )

    elapsed = time.time() - t0
    return {
        "src": str(src_root),
        "dst": str(dst_root),
        "status": "ok" if fail_count == 0 else "partial",
        "total": len(jobs),
        "ok": ok_count,
        "skipped_existing": skipped,
        "fail": fail_count,
        "fail_examples": fail_examples,
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-src", type=Path, default=Path("phases/forensic/data/genimage_train"))
    p.add_argument("--test-src", type=Path, default=Path("phases/forensic/data/genimage_test"))
    p.add_argument("--out-root", type=Path, default=Path("phases/forensic/data/dct_cache"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    summary: dict = {"splits": {}}

    if "train" in args.splits:
        _safe_log("=== train split ===")
        summary["splits"]["train"] = precompute_split(
            args.train_src / "train",
            args.out_root / "train",
            n_workers=args.workers,
        )
    if "val" in args.splits:
        _safe_log("=== val split ===")
        summary["splits"]["val"] = precompute_split(
            args.train_src / "val",
            args.out_root / "val",
            n_workers=args.workers,
        )
    if "test" in args.splits:
        _safe_log("=== test split (per-generator) ===")
        summary["splits"]["test"] = precompute_split(
            args.test_src,
            args.out_root / "test",
            n_workers=args.workers,
        )

    summary_path = args.out_root / "precompute_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    _safe_log(f"\nDONE. Summary at {summary_path}")

    for split, info in summary["splits"].items():
        _safe_log(
            f"  {split}: {info.get('ok', 0)} ok, {info.get('fail', 0)} fail "
            f"({info.get('elapsed_s', 0):.0f}s)"
        )

    if any(info.get("fail", 0) > 0 for info in summary["splits"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

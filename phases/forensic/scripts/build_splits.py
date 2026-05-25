"""Restructure raw GenImage downloads into doctor's required layout.

Doctor's spec F.1 (MASTER_CHECKLIST):
    Three non-overlapping trees:
      genimage_train/train/{0_real,1_fake}/    (all generators merged)
      genimage_train/val/{0_real,1_fake}/      (held-out val)
      genimage_test/<gen>/{0_real,1_fake}/     (per-generator final eval)

Input layout (built by download_genimage.py):
    data/raw/GenImage_v2/
      <gen>/ai/*.jpg
      <gen>/nature/*.jpg

Output layout (built here):
    phases/forensic/data/genimage_train/
      train/{0_real,1_fake}/  <- 80% merged
      val/{0_real,1_fake}/    <- 20% held out
    phases/forensic/data/genimage_test/
      <gen>/{0_real,1_fake}/

Per-generator counts (locked D-A2):
    Train: 1250 (split 80/20 = 1000 train + 250 val) per class
    Test:  500 per class
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

DEFAULT_TRAIN_PER_GEN = 1250
DEFAULT_TEST_PER_GEN = 500
DEFAULT_VAL_FRAC = 0.20


def _safe_log(msg: str) -> None:
    print(f"[build_splits] {msg}", flush=True)


def _collect_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw/GenImage_v2"))
    p.add_argument("--out-train", type=Path, default=Path("phases/forensic/data/genimage_train"))
    p.add_argument("--out-test", type=Path, default=Path("phases/forensic/data/genimage_test"))
    p.add_argument("--train-per-gen", type=int, default=DEFAULT_TRAIN_PER_GEN)
    p.add_argument("--test-per-gen", type=int, default=DEFAULT_TEST_PER_GEN)
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copies (saves disk; needs admin on Windows)",
    )
    args = p.parse_args()

    if not args.raw.exists():
        _safe_log(f"FATAL: raw dir not found at {args.raw}. Run download_genimage.py first.")
        sys.exit(2)

    gens = sorted(
        [
            d.name
            for d in args.raw.iterdir()
            if d.is_dir() and (d / "ai").exists() and not d.name.startswith("_")
        ]
    )
    if not gens:
        _safe_log(f"FATAL: no generator subdirs with ai/ found under {args.raw}")
        sys.exit(2)

    _safe_log(f"Found {len(gens)} generators: {gens}")

    for sub in [
        args.out_train / "train" / "0_real",
        args.out_train / "train" / "1_fake",
        args.out_train / "val" / "0_real",
        args.out_train / "val" / "1_fake",
    ]:
        sub.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "generators": gens,
        "per_generator": {},
        "totals": {"train_real": 0, "train_fake": 0, "val_real": 0, "val_fake": 0, "test": {}},
    }

    n_val = int(round(args.train_per_gen * args.val_frac))
    n_train_actual = args.train_per_gen - n_val
    _safe_log(
        f"Per gen: {n_train_actual} train + {n_val} val + {args.test_per_gen} test (per class)"
    )

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        if args.symlink:
            try:
                dst.symlink_to(src)
                return
            except OSError:
                pass
        shutil.copyfile(src, dst)

    for g in gens:
        ai_imgs = _collect_images(args.raw / g / "ai")
        nat_imgs = _collect_images(args.raw / g / "nature")
        _safe_log(f"{g}: {len(ai_imgs)} ai, {len(nat_imgs)} nature")

        need_per_class = args.train_per_gen + args.test_per_gen
        if len(ai_imgs) < need_per_class or len(nat_imgs) < need_per_class:
            _safe_log(
                f"  WARNING: {g} short on samples (have ai={len(ai_imgs)}, nature={len(nat_imgs)}, "
                f"need={need_per_class}). Continuing with what's available."
            )

        rng_g = random.Random(args.seed + hash(g) % 1000)
        rng_g.shuffle(ai_imgs)
        rng_g.shuffle(nat_imgs)

        ai_train_full = ai_imgs[: args.train_per_gen]
        nat_train_full = nat_imgs[: args.train_per_gen]
        ai_test = ai_imgs[args.train_per_gen : args.train_per_gen + args.test_per_gen]
        nat_test = nat_imgs[args.train_per_gen : args.train_per_gen + args.test_per_gen]

        ai_train = ai_train_full[:n_train_actual]
        ai_val = ai_train_full[n_train_actual:]
        nat_train = nat_train_full[:n_train_actual]
        nat_val = nat_train_full[n_train_actual:]

        for i, src in enumerate(nat_train):
            _link_or_copy(
                src,
                args.out_train / "train" / "0_real" / f"{g}_real_train_{i:05d}{src.suffix.lower()}",
            )
        for i, src in enumerate(ai_train):
            _link_or_copy(
                src,
                args.out_train / "train" / "1_fake" / f"{g}_fake_train_{i:05d}{src.suffix.lower()}",
            )
        for i, src in enumerate(nat_val):
            _link_or_copy(
                src, args.out_train / "val" / "0_real" / f"{g}_real_val_{i:05d}{src.suffix.lower()}"
            )
        for i, src in enumerate(ai_val):
            _link_or_copy(
                src, args.out_train / "val" / "1_fake" / f"{g}_fake_val_{i:05d}{src.suffix.lower()}"
            )

        test_real_out = args.out_test / g / "0_real"
        test_fake_out = args.out_test / g / "1_fake"
        test_real_out.mkdir(parents=True, exist_ok=True)
        test_fake_out.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(nat_test):
            _link_or_copy(src, test_real_out / f"{g}_real_test_{i:05d}{src.suffix.lower()}")
        for i, src in enumerate(ai_test):
            _link_or_copy(src, test_fake_out / f"{g}_fake_test_{i:05d}{src.suffix.lower()}")

        summary["per_generator"][g] = {
            "ai_available": len(ai_imgs),
            "nature_available": len(nat_imgs),
            "train_fake": len(ai_train),
            "train_real": len(nat_train),
            "val_fake": len(ai_val),
            "val_real": len(nat_val),
            "test_fake": len(ai_test),
            "test_real": len(nat_test),
        }
        summary["totals"]["train_fake"] += len(ai_train)
        summary["totals"]["train_real"] += len(nat_train)
        summary["totals"]["val_fake"] += len(ai_val)
        summary["totals"]["val_real"] += len(nat_val)
        summary["totals"]["test"][g] = {"real": len(nat_test), "fake": len(ai_test)}

    summary_path = args.out_train.parent / "splits_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    _safe_log(f"\nDONE. Summary at {summary_path}")
    _safe_log(
        f"Totals: train {summary['totals']['train_real']}/{summary['totals']['train_fake']} (real/fake)"
    )
    _safe_log(f"        val   {summary['totals']['val_real']}/{summary['totals']['val_fake']}")


if __name__ == "__main__":
    main()

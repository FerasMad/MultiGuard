"""Compute global DCT_mean + DCT_std from train .pt cache (doctor's spec F.17).

Welford's streaming algorithm: avoids loading all 16k+ tensors into memory at once.
Output: dct_stats.json with {mean, std, n, n_files, computed_at}.

Doctor's spec F.17:
    'Z-score: compute global DCT_mean + DCT_std from `genimage_train/train/` only,
     save to `dct_stats.json`, apply (t-mean)/(std+1e-8) at every stage.'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _welford_update(count: int, mean: float, m2: float, value_count: int,
                    value_sum: float, value_sum_sq: float) -> tuple[int, float, float]:
    """Update Welford running aggregates with a batch of `value_count` scalars
    having sum `value_sum` and sum-of-squares `value_sum_sq`."""
    if value_count == 0:
        return count, mean, m2
    new_count = count + value_count
    batch_mean = value_sum / value_count
    delta = batch_mean - mean
    new_mean = mean + delta * value_count / new_count
    batch_m2 = value_sum_sq - value_count * batch_mean * batch_mean
    m2_new = m2 + batch_m2 + delta * delta * count * value_count / new_count
    return new_count, new_mean, m2_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path,
                   default=Path("phases/forensic/data/dct_cache/train"),
                   help="Directory containing train .pt shards")
    p.add_argument("--out", type=Path,
                   default=Path("phases/forensic/data/dct_stats.json"))
    args = p.parse_args()

    if not args.cache.exists():
        print(f"FATAL: cache dir not found: {args.cache}", file=sys.stderr)
        sys.exit(2)

    pts = sorted(args.cache.rglob("*.pt"))
    if not pts:
        print(f"FATAL: no .pt shards found under {args.cache}", file=sys.stderr)
        sys.exit(2)

    print(f"[compute_dct_stats] Processing {len(pts)} .pt files...", flush=True)
    t0 = time.time()

    count = 0
    mean = 0.0
    m2 = 0.0

    for i, pt in enumerate(pts):
        try:
            t = torch.load(pt, weights_only=True).float()
        except Exception as e:
            print(f"  skip {pt}: {e}", file=sys.stderr)
            continue
        flat = t.flatten()
        value_count = flat.numel()
        value_sum = float(flat.sum().item())
        value_sum_sq = float((flat * flat).sum().item())
        count, mean, m2 = _welford_update(count, mean, m2, value_count, value_sum, value_sum_sq)

        if (i + 1) % 500 == 0 or (i + 1) == len(pts):
            print(f"  {i+1}/{len(pts)}  running mean={mean:.4f}  count={count}", flush=True)

    if count < 2:
        print(f"FATAL: too few scalars accumulated ({count})", file=sys.stderr)
        sys.exit(2)

    variance = m2 / (count - 1)  # sample variance
    std = variance ** 0.5

    out_data = {
        "mean": float(mean),
        "std": float(std),
        "n_scalars": int(count),
        "n_files": int(len(pts)),
        "computed_at": time.time(),
        "source": str(args.cache),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2))

    elapsed = time.time() - t0
    print(f"\n[compute_dct_stats] DONE in {elapsed:.1f}s")
    print(f"  mean = {mean:.6f}")
    print(f"  std  = {std:.6f}")
    print(f"  n    = {count} scalars from {len(pts)} files")
    print(f"  wrote: {args.out}")


if __name__ == "__main__":
    main()

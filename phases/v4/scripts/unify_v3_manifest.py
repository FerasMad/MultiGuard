"""Rewrite V3's forensic_5class_c4fix.csv into a portable V4-compatible manifest.

V3's manifest hard-coded absolute Windows paths under the OLD repo name
`C:\\Desktop\\Multimodal-fake-news-detection\\...`. After the P1 rename this
prefix is wrong, AND the absolute path makes the manifest non-portable across
machines.

This script:
  1. Reads data/processed/forensic_5class_c4fix.csv
  2. Strips the `<drive>:/Desktop/<reponame>/` prefix from image_path
  3. Normalizes backslashes to forward slashes
  4. Writes data/processed/forensic_5class_unified.csv with REPO-RELATIVE paths

After this, the manifest is portable to any machine that has the same
data/raw/ tree (DGM4, VisualNews, NewsCLIPpings, MMFakeBench, GenImage/ai/midjourney).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


_PREFIX = re.compile(r"^[A-Za-z]:[\\/]Desktop[\\/][^\\/]+[\\/]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/processed/forensic_5class_c4fix.csv"))
    p.add_argument("--out", type=Path, default=Path("data/processed/forensic_5class_unified.csv"))
    args = p.parse_args()

    if not args.src.exists():
        raise SystemExit(f"FATAL: source manifest not found at {args.src}")

    df = pd.read_csv(args.src)
    n_total = len(df)
    df["image_path"] = df["image_path"].apply(lambda s: _PREFIX.sub("", s))
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)

    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    n_ok = int(df["exists"].sum())
    print(f"Manifest: {n_total} rows, {n_ok} resolve under repo root")

    if n_ok < n_total:
        miss = df[~df["exists"]].head(5)
        print("First missing paths:")
        for _, r in miss.iterrows():
            print(f"  {r['image_path']}")
        if n_ok < n_total * 0.95:
            raise SystemExit(
                f"FATAL: only {n_ok}/{n_total} ({100 * n_ok / n_total:.1f}%) resolve. "
                "Verify data/raw/ tree is complete."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["exists"]).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({n_ok}/{n_total} valid rows)")


if __name__ == "__main__":
    main()

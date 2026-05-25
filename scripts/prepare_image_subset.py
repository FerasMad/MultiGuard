"""prepare_image_subset - list/copy only the VisualNews images referenced by manifest."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


def list_needed(csv_path: Path) -> list[Path]:
    df = pd.read_csv(csv_path)
    return sorted({Path(p) for p in df["image_path"]
                   if "visualnews" in str(p).lower()})


def copy_subset(needed: list[Path], src_root: Path, dst_root: Path) -> int:
    copied = 0
    for p in needed:
        try:
            rel = p.resolve().relative_to(src_root.resolve())
        except ValueError:
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(p, dst)
            copied += 1
    return copied


if __name__ == "__main__":
    csv = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/forensic_5class_v4.csv")
    needed = list_needed(csv)
    print(f"manifest references {len(needed)} VisualNews images")
    for p in needed[:10]:
        print(f"  {p}")

"""Check that Phase 2 inputs are in place before running the pipeline.

Verifies the three external dataset directories, the V1 dataset CSV (which
the Phase 2 builder reuses for NewsCLIPpings OOC pairs), and the Python
runtime deps. Exits 0 if everything is ready; prints a concrete TODO list
and exits 1 otherwise.

Run this before `bash scripts/run_phase2_pipeline.sh`:

    python scripts/check_phase2_data.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_DIRS = [
    ("data/raw/DGM4",
     "Download DGM4 from https://huggingface.co/datasets/rshaojimmy/DGM4"),
    ("data/raw/NewsCLIPpings",
     "Download NewsCLIPpings from https://github.com/g-luo/news_clippings"),
]

OPTIONAL_DIRS = [
    ("data/raw/MMFakeBench",
     "Optional: MMFakeBench from https://huggingface.co/datasets/liuxuannan/MMFakeBench "
     "(only needed for the zero-shot transfer probe in evaluation)"),
]

REQUIRED_FILES = [
    ("data/processed/balanced_dataset.csv",
     "Run V1's dataset builder first: python scripts/build_balanced_dataset.py "
     "--dgm4 data/raw/DGM4 --newsclippings data/raw/NewsCLIPpings "
     "--output data/processed/balanced_dataset.csv"),
]

REQUIRED_MODULES = [
    "torch", "torchvision", "cv2", "numpy", "pandas",
    "sklearn", "scipy", "transformers", "tqdm", "yaml",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    missing_required: list[str] = []
    missing_optional: list[str] = []

    print(f"Repo root: {repo_root}\n")

    print("--- Required directories ---")
    for rel, hint in REQUIRED_DIRS:
        p = repo_root / rel
        if p.is_dir():
            print(f"  OK  {rel}")
        else:
            print(f"  MISSING  {rel}")
            print(f"           hint: {hint}")
            missing_required.append(rel)

    print("\n--- Optional directories ---")
    for rel, hint in OPTIONAL_DIRS:
        p = repo_root / rel
        if p.is_dir():
            print(f"  OK   {rel}")
        else:
            print(f"  miss {rel}  (optional)")
            print(f"        hint: {hint}")
            missing_optional.append(rel)

    print("\n--- Required files ---")
    for rel, hint in REQUIRED_FILES:
        p = repo_root / rel
        if p.is_file():
            print(f"  OK  {rel}")
        else:
            print(f"  MISSING  {rel}")
            print(f"           hint: {hint}")
            missing_required.append(rel)

    print("\n--- Python runtime modules ---")
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
            print(f"  OK  {mod}")
        except ImportError as e:
            print(f"  MISSING  {mod}  ({e})")
            missing_required.append(f"python: {mod}")

    print()
    if missing_required:
        print("=" * 60)
        print(f"NOT READY. {len(missing_required)} required item(s) missing:")
        for m in missing_required:
            print(f"  - {m}")
        print("\nFix the items above, then re-run this script.")
        return 1

    print("=" * 60)
    print("READY. You can run:")
    print("    bash scripts/run_phase2_pipeline.sh")
    if missing_optional:
        print(f"\n(Note: {len(missing_optional)} optional item(s) missing — "
              "MMFakeBench transfer eval will be skipped if not present.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

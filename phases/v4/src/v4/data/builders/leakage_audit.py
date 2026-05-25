"""Leakage audit: enforce manifest invariants per plan section 3."""
from __future__ import annotations

import sys
from pathlib import Path

from v4.core.logging import get_logger
from v4.data.manifest import read_manifest

log = get_logger(__name__)


def audit(csv_path: str | Path) -> int:
    df = read_manifest(csv_path)
    log.info("audit: %s (%d rows)", csv_path, len(df))

    dups = df[df["sample_id"].duplicated()]
    if not dups.empty:
        log.error("FAIL: %d duplicate sample_id values", len(dups))
        return 1

    by_path = df.groupby("image_path")["label"].nunique()
    bad_paths = by_path[by_path > 1]
    if not bad_paths.empty:
        log.error("FAIL: %d image_paths appear under multiple labels", len(bad_paths))
        log.error("first 5: %s", bad_paths.head(5).to_dict())
        return 2

    log.info("class counts: %s", df["label"].value_counts().to_dict())
    log.info("split counts: %s", df["split"].value_counts().to_dict())
    log.info("leakage audit PASSED")
    return 0


def verify_image_paths(csv_path: str | Path) -> int:
    df = read_manifest(csv_path)
    missing = 0
    for path in df["image_path"]:
        if not Path(path).exists():
            missing += 1
            if missing <= 5:
                log.error("missing: %s", path)
    if missing:
        log.error("FAIL: %d / %d image_paths missing", missing, len(df))
        return 1
    log.info("all %d image_paths resolve OK", len(df))
    return 0


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/processed/forensic_5class_v4.csv"
    sys.exit(audit(csv))

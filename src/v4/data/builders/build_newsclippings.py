"""Builder: NewsCLIPpings -> manifest rows for classes 0 (Real) + 1 (OOC).

Reads the NewsCLIPpings JSON annotations + VisualNews image paths.
Per V3.1 section 2:
    0 Real:  NewsCLIPpings (Matched)
    1 OOC:   NewsCLIPpings (Mismatched)

Expected layout on disk (stage these before building manifests; see docs/SETUP.md step 6):
    data/raw/NewsCLIPpings/news_clippings/data/merged_balanced.json (or similar)
    data/raw/visualnews/origin/<source>/images/<bucket>/<file>.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

from v4.core.logging import get_logger
from v4.core.paths import DATA_PROCESSED, DATA_ROOT, ensure_dir
from v4.data.manifest import ManifestRow, write_manifest

log = get_logger(__name__)

DEFAULT_OUT = DATA_PROCESSED / "newsclippings_rows.csv"


def _resolve_image_path(image_id: str, visualnews_root: Path) -> Path | None:
    """NewsCLIPpings refers to images by VisualNews image_id.

    VisualNews layout: origin/<source>/images/<bucket>/<file>.jpg
    The annotations include a per-image JSON listing source + path.
    """
    # The actual resolver depends on the NewsCLIPpings annotation format;
    # this is a best-effort search across known source subdirs.
    for src_dir in ("guardian", "usa_today", "bbc", "washington_post"):
        # Image IDs are usually <bucket>_<file> like "0155_048"
        # Try a couple of common patterns
        for candidate in (
            visualnews_root / "origin" / src_dir / "images" / image_id[:4] / f"{image_id[4:]}.jpg",
            visualnews_root / "origin" / src_dir / "images" / f"{image_id}.jpg",
        ):
            if candidate.exists():
                return candidate
    return None


def build(out_path: str | None = None) -> str:
    """Build NewsCLIPpings rows for classes 0 + 1.

    Args:
        out_path: optional override for output CSV path.

    Returns:
        str path to the written CSV.
    """
    nclip_root = DATA_ROOT / "NewsCLIPpings"
    visualnews_root = DATA_ROOT / "visualnews"
    if not nclip_root.exists():
        raise FileNotFoundError(
            f"NewsCLIPpings not found at {nclip_root}. "
            f"Stage the news_clippings annotations from https://github.com/g-luo/news_clippings under {nclip_root}; "
            f"see docs/SETUP.md step 6."
        )

    # Candidate annotation files; pick the first that exists
    candidates = [
        nclip_root / "news_clippings" / "data" / "merged_balanced.json",
        nclip_root / "merged_balanced.json",
        nclip_root / "annotations.json",
    ]
    ann_path = next((c for c in candidates if c.exists()), None)
    if ann_path is None:
        raise FileNotFoundError(
            f"NewsCLIPpings annotation JSON not found under {nclip_root}. "
            f"Looked for: {[str(c) for c in candidates]}"
        )

    log.info("loading NewsCLIPpings annotations from %s", ann_path)
    with ann_path.open(encoding="utf-8") as f:
        ann = json.load(f)

    # NewsCLIPpings JSON layout (per Luo et al.): top-level "annotations" list
    # each entry has: {id, image_id, text, source, falsified (bool)}
    # We accept several known schemas.
    if isinstance(ann, dict) and "annotations" in ann:
        records = ann["annotations"]
    elif isinstance(ann, list):
        records = ann
    else:
        raise ValueError(f"unknown NewsCLIPpings JSON structure in {ann_path}")

    rows: list[ManifestRow] = []
    n_missing = 0
    for rec in records:
        text = str(rec.get("caption") or rec.get("text") or "")
        image_id = str(rec.get("image_id") or rec.get("id") or "")
        falsified = bool(
            rec.get("falsified", False)
            or rec.get("is_ooc", False)
            or rec.get("label") == 1
        )
        if not image_id:
            n_missing += 1
            continue

        img = _resolve_image_path(image_id, visualnews_root)
        if img is None:
            n_missing += 1
            continue

        label = 1 if falsified else 0
        source = "NewsCLIPpings_OOC" if falsified else "NewsCLIPpings_genuine"
        sid = f"nclip_{image_id}_{rec.get('id', image_id)}"

        rows.append(ManifestRow(
            sample_id=sid,
            text=text,
            image_path=str(img.resolve()),
            label=label,
            source=source,
            split="train",   # split assignment happens in merge.py
        ))

    log.info("NewsCLIPpings rows: %d (missing images: %d)", len(rows), n_missing)
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    ensure_dir(out_path.parent)
    write_manifest(rows, out_path)
    return str(out_path)

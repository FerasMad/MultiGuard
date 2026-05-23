"""Build the Stage 1 binary CSV: real vs fake-image for UnivFD pretrain."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from v4.core.logging import get_logger
from v4.core.paths import DATA_PROCESSED, DATA_ROOT, ensure_dir
from v4.data.manifest import ManifestRow, write_manifest

log = get_logger(__name__)

DEFAULT_OUT = DATA_PROCESSED / "univfd_stage1.csv"
SEED = 42


def _scan_visualnews(n: int, rng: random.Random) -> list[Path]:
    vn_root = DATA_ROOT / "visualnews" / "origin"
    if not vn_root.exists():
        return []
    images: list[Path] = []
    for src in ("guardian", "usa_today", "bbc", "washington_post"):
        d = vn_root / src
        if d.exists():
            images.extend(d.rglob("*.jpg"))
    rng.shuffle(images)
    return images[:n]


def _scan_dgm4_origin(n: int, rng: random.Random) -> list[Path]:
    d = DATA_ROOT / "DGM4" / "origin"
    if not d.exists():
        return []
    images = list(d.rglob("*.jpg"))
    rng.shuffle(images)
    return images[:n]


def _scan_mmfb_real(n: int, rng: random.Random) -> list[Path]:
    root = DATA_ROOT / "MMFakeBench"
    if not root.exists():
        return []
    images: list[Path] = []
    for partition in ("MMFakeBench_val", "MMFakeBench_test"):
        d = root / partition / "real"
        if d.exists():
            images.extend(d.rglob("*.png"))
            images.extend(d.rglob("*.jpg"))
    rng.shuffle(images)
    return images[:n]


def build(out_path: str | None = None) -> str:
    rng = random.Random(SEED)
    rows: list[ManifestRow] = []

    for csv_name, source_filter in (
        ("genimage_stage1_rows.csv", lambda s: s.startswith("GenImage_")),
        ("dgm4_rows.csv", lambda s: s.startswith("DGM4_") and "origin" not in s),
        ("mmfakebench_rows.csv", lambda s: any(
            t in s for t in ("coco_image_edit", "Fakeddit_photo_edit", "Newsclipings"))),
    ):
        p = DATA_PROCESSED / csv_name
        if not p.exists():
            log.warning("source CSV missing: %s", p)
            continue
        df = pd.read_csv(p)
        df = df[df["source"].apply(source_filter)]
        for _, r in df.iterrows():
            rows.append(ManifestRow(
                sample_id=str(r["sample_id"]),
                text="",
                image_path=str(r["image_path"]),
                label=1,
                source=str(r["source"]),
                split="train",
            ))

    for src_label, fn, n in (
        ("VisualNews_genuine", _scan_visualnews, 10_000),
        ("DGM4_origin", _scan_dgm4_origin, 3_000),
        ("MMFakeBench_real", _scan_mmfb_real, 2_000),
    ):
        imgs = fn(n, rng)
        log.info("%s: %d images", src_label, len(imgs))
        for i, img in enumerate(imgs):
            rows.append(ManifestRow(
                sample_id=f"{src_label.lower()}_{i:06d}",
                text="",
                image_path=str(img.resolve()),
                label=0,
                source=src_label,
                split="train",
            ))

    rng2 = random.Random(SEED + 99)
    rng2.shuffle(rows)
    n_total = len(rows)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    for i, r in enumerate(rows):
        if i < n_train:
            r.split = "train"
        elif i < n_train + n_val:
            r.split = "val"
        else:
            r.split = "test"

    log.info("Stage 1 CSV: %d total rows", n_total)
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    ensure_dir(out_path.parent)
    write_manifest(rows, out_path)
    return str(out_path)

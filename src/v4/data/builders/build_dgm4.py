"""Builder: DGM4 -> rows for class 2 (Manipulated) + class 4 augmentation."""
from __future__ import annotations

import json
from pathlib import Path

from v4.core.logging import get_logger
from v4.core.paths import DATA_PROCESSED, DATA_ROOT, ensure_dir
from v4.data.manifest import ManifestRow, write_manifest

log = get_logger(__name__)

DEFAULT_OUT = DATA_PROCESSED / "dgm4_rows.csv"

CLASS_2_TYPES = ("HFGI", "face_swap", "face_edit", "face_attribute")
CLASS_4_TYPES = ("text_swap", "text_attribute")


def _scan_metadata(dgm4_root: Path) -> list[dict]:
    meta_dir = dgm4_root / "metadata"
    if not meta_dir.exists():
        flat = dgm4_root / "metadata.json"
        if flat.exists():
            with flat.open() as f:
                return json.load(f)
        raise FileNotFoundError(
            f"DGM4 metadata not found under {meta_dir}"
        )
    out: list[dict] = []
    for j in sorted(meta_dir.glob("*.json")):
        with j.open(encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            out.extend(payload)
        elif isinstance(payload, dict) and "annotations" in payload:
            out.extend(payload["annotations"])
    return out


def build(out_path: str | None = None) -> str:
    dgm4_root = DATA_ROOT / "DGM4"
    if not dgm4_root.exists():
        raise FileNotFoundError(
            f"DGM4 not found at {dgm4_root}. See docs/DATA_DOWNLOAD.md A.5.2."
        )

    log.info("scanning DGM4 metadata under %s", dgm4_root)
    records = _scan_metadata(dgm4_root)

    rows: list[ManifestRow] = []
    n_skip = 0
    class_4_count = 0
    CLASS_4_LIMIT = 800

    for rec in records:
        image_rel = str(rec.get("image", ""))
        text = str(rec.get("text") or rec.get("caption") or "")
        manip = str(rec.get("manipulation_type") or rec.get("fake_type") or "")
        sample_id = f"dgm4_{rec.get('id', image_rel.replace('/', '_'))}"

        if not image_rel:
            n_skip += 1
            continue

        img_path = dgm4_root / image_rel if not image_rel.startswith("/") else Path(image_rel)
        if not img_path.exists():
            alt = DATA_ROOT / image_rel.lstrip("/\\")
            if alt.exists():
                img_path = alt
            else:
                n_skip += 1
                continue

        if any(t in manip for t in CLASS_2_TYPES):
            label, source = 2, f"DGM4_{manip}"
        elif any(t in manip for t in CLASS_4_TYPES) and class_4_count < CLASS_4_LIMIT:
            label, source = 4, f"DGM4_{manip}"
            class_4_count += 1
        else:
            n_skip += 1
            continue

        rows.append(ManifestRow(
            sample_id=sample_id,
            text=text,
            image_path=str(img_path.resolve()),
            label=label,
            source=source,
            split="train",
        ))

    log.info("DGM4 rows: %d (class 4 aug: %d, skipped: %d)",
             len(rows), class_4_count, n_skip)
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    ensure_dir(out_path.parent)
    write_manifest(rows, out_path)
    return str(out_path)

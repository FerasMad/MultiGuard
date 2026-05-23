"""Builder: MMFakeBench -> rows for classes 2 (tampered), 3 (AI text), 4 (double fake)."""
from __future__ import annotations

import json
from pathlib import Path

from v4.core.logging import get_logger
from v4.core.paths import DATA_PROCESSED, DATA_ROOT, ensure_dir
from v4.data.manifest import ManifestRow, write_manifest

log = get_logger(__name__)

DEFAULT_OUT = DATA_PROCESSED / "mmfakebench_rows.csv"

SUBSET_LABEL_MAP = {
    "coco_image_edit": 2,
    "Fakeddit_photo_edit": 2,
    "Newsclipings_person": 2,
    "Newsclipings_scene": 2,
    "Newsclipings_semantic": 2,
    "DGM4_text_edit_senti": 2,
    "fever_AI": 3,
    "gossipcop_match": 3,
    "politicat_match": 3,
    "rumor_match": 3,
    "llm_rewrite": 3,
    "coco_text_edit": 3,
    "chatgpt_match": 4,
    "antifact_image_generation": 4,
    "llm_gossip_md": 4,
    "llm_science_md": 4,
    "gossipcop_midjourney": 4,
}


def _resolve_subset_name(dir_name: str) -> str | None:
    for cand in (dir_name, "_".join(dir_name.split("_")[:-2]),
                 dir_name.rsplit("_", 1)[0]):
        if cand in SUBSET_LABEL_MAP:
            return cand
    return None


def _scan_partition(part_root: Path) -> list[dict]:
    records: list[dict] = []
    for kind in ("fake", "real"):
        kind_root = part_root / kind
        if not kind_root.exists():
            continue
        for subset_dir in sorted(kind_root.iterdir()):
            if not subset_dir.is_dir():
                continue
            base_subset = _resolve_subset_name(subset_dir.name)
            if base_subset is None:
                continue
            json_files = list(subset_dir.glob("*.json"))
            if json_files:
                with json_files[0].open(encoding="utf-8") as f:
                    payload = json.load(f)
                annotated = payload if isinstance(payload, list) else payload.get("annotations", [])
                for r in annotated:
                    img_name = str(r.get("image") or r.get("filename") or "")
                    text = str(r.get("text") or r.get("caption") or "")
                    if not img_name:
                        continue
                    img_path = subset_dir / img_name
                    if img_path.exists():
                        records.append({
                            "image_path": img_path,
                            "text": text,
                            "subset": base_subset,
                            "kind": kind,
                        })
            else:
                for img in list(subset_dir.glob("*.png")) + list(subset_dir.glob("*.jpg")):
                    records.append({
                        "image_path": img,
                        "text": "",
                        "subset": base_subset,
                        "kind": kind,
                    })
    return records


def build(out_path: str | None = None) -> str:
    mmfb_root = DATA_ROOT / "MMFakeBench"
    if not mmfb_root.exists():
        raise FileNotFoundError(
            f"MMFakeBench not found at {mmfb_root}. See docs/DATA_DOWNLOAD.md A.5.3."
        )

    all_records: list[dict] = []
    for partition in ("MMFakeBench_val", "MMFakeBench_test"):
        part_root = mmfb_root / partition
        if part_root.exists():
            log.info("scanning %s", partition)
            all_records.extend(_scan_partition(part_root))

    rows: list[ManifestRow] = []
    n_skip = 0
    for i, rec in enumerate(all_records):
        if rec["kind"] != "fake":
            n_skip += 1
            continue
        subset = rec["subset"]
        rows.append(ManifestRow(
            sample_id=f"mmfb_{subset}_{i:06d}",
            text=rec["text"],
            image_path=str(rec["image_path"].resolve()),
            label=SUBSET_LABEL_MAP[subset],
            source=f"MMFakeBench_{subset}",
            split="train",
        ))

    log.info("MMFakeBench rows: %d (skipped: %d)", len(rows), n_skip)
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    ensure_dir(out_path.parent)
    write_manifest(rows, out_path)
    return str(out_path)

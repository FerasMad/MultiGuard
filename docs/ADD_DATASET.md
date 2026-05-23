# Adding a new dataset / data source

This walkthrough shows how to add a new raw data source to V4. The canonical CSV schema is the universal contract - any builder that emits valid rows can be plugged in without trainer or evaluator changes.

---

## 1. The manifest schema (the only contract)

Every dataset row in V4 has six columns, defined in `src/v4/data/manifest.py`:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Unique across all rows ever produced |
| `text` | str | Caption / claim / article text (may be empty for image-only sources) |
| `image_path` | str | Absolute or repo-relative path to the image on disk |
| `label` | int | 0-4 (Real / OOC / Manipulated / AI-Text / Fully-Fabricated) for 5-class, or 0/1 for Stage 1 binary |
| `source` | str | Free-form tag like `DGM4_face_swap` or `MMFakeBench_chatgpt_match` - used for stratified splits |
| `split` | str | One of `{train, val, test}` |

`sample_id` must be unique. Path-label collisions (same image under two labels) fail the leakage audit.

---

## 2. Write a builder

`src/v4/data/builders/build_my_source.py`:

```python
"""Build CSV rows for MySource - hypothetical new dataset.

Outputs one row per image found under DATA_ROOT/MySource, labeled by
the parent-directory convention 'fake' or 'real'.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from v4.core.logging import get_logger
from v4.core.paths import DATA_PROCESSED, DATA_ROOT, ensure_dir
from v4.data.manifest import ManifestRow, write_manifest

log = get_logger(__name__)

DEFAULT_OUT = DATA_PROCESSED / "my_source_rows.csv"


def build(out_path: str | None = None) -> str:
    rows: list[ManifestRow] = []
    root = DATA_ROOT / "MySource"
    for label_dir, label in (("real", 0), ("fake", 1)):
        for img in (root / label_dir).rglob("*.jpg"):
            rows.append(ManifestRow(
                sample_id=f"my_source_{label_dir}_{img.stem}",
                text="",
                image_path=str(img.resolve()),
                label=label,
                source=f"MySource_{label_dir}",
                split="train",       # the merger reshuffles this
            ))

    log.info("MySource: %d rows", len(rows))
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    ensure_dir(out_path.parent)
    write_manifest(rows, out_path)
    return str(out_path)


if __name__ == "__main__":
    build()
```

---

## 3. Register it in the builders package

`src/v4/data/builders/__init__.py`:

```python
from v4.data.builders import (
    build_dgm4,
    build_genimage_stage1,
    build_mmfakebench,
    build_my_source,           # <-- add
    build_newsclippings,
    build_stage1_binary,
    leakage_audit,
    merge,
)
```

---

## 4. Wire it into the CLI

`src/v4/cli.py::build_manifest_main` already routes by `--source` to the right module. Add an entry:

```python
SOURCES = {
    "newsclippings": ("v4.data.builders.build_newsclippings", "build"),
    "dgm4":          ("v4.data.builders.build_dgm4",          "build"),
    "mmfakebench":   ("v4.data.builders.build_mmfakebench",   "build"),
    "stage1":        ("v4.data.builders.build_stage1_binary", "build"),
    "my_source":     ("v4.data.builders.build_my_source",     "build"),   # <-- add
}
```

Now `python -m v4.cli build-manifest --source my_source` produces `data/processed/my_source_rows.csv`.

---

## 5. Include it in the 5-class merge (optional)

If your source contributes to one of the 5 classes, edit `src/v4/data/builders/merge.py` to include it in `SOURCE_CSVS` and the per-class accounting logic. Confirm the per-class targets still balance to 3,000.

If your source is for a brand-new task (e.g. a Stage 3 multilingual head), keep it out of `merge.py` and consume it directly from `data/processed/my_source_rows.csv`.

---

## 6. Leakage-audit and verify paths

Both audits work generically against any canonical-schema CSV:

```bash
python -m v4.cli leakage-audit --csv data/processed/my_source_rows.csv
python -m v4.cli verify-image-paths --csv data/processed/my_source_rows.csv
```

They check:

1. No duplicate `sample_id`.
2. No `image_path` appears under more than one `label`.
3. Every `image_path` resolves on disk.

---

## 7. Precompute caches

Once the manifest is built, run the standard precompute pipeline:

```bash
python -m v4.cli precompute --modality v_imgfor_dct --csv data/processed/my_source_rows.csv
python -m v4.cli precompute --modality v_semantic_fnd --csv data/processed/my_source_rows.csv --config configs/v4_pipeline_qwen.yaml
python -m v4.cli precompute --modality v_textfor_qwen --csv data/processed/my_source_rows.csv --config configs/v4_pipeline_qwen.yaml
```

Each script reads the CSV, opens the image / text, and writes a `.pt` shard keyed by `sample_id`.

---

## 8. Reproducibility

The `data_hash` (sha256 of the CSV) is stored in every checkpoint. So as long as your manifest is deterministic - same rows, same order - runs are reproducible.

If your builder is non-deterministic (e.g. random sampling), seed the RNG at the top of `build`:

```python
import random
rng = random.Random(42)
```

and document the seed in the file docstring.

---

## 9. Test

Add a smoke check in `tests/unit/test_my_source_schema.py`:

```python
from v4.data.builders import build_my_source
from v4.data.manifest import read_manifest, validate

def test_my_source_emits_canonical_schema(tmp_path):
    out = build_my_source.build(out_path=str(tmp_path / "rows.csv"))
    df = read_manifest(out)
    validate(df)        # raises if any required column is missing or invalid
```

"""Builder: GenImage -> rows for Stage 1 UnivFD binary pretrain ONLY.

Per locked decision C8: 10,000 samples stratified across 8 generators
(sd14, sd15, midjourney, adm, glide, vqdm, biggan, wukong), ~1,250 each.
GenImage NEVER enters the 5-class data per A1 deviation.
"""
from __future__ import annotations

import random
from pathlib import Path

from v4.core.logging import get_logger
from v4.core.paths import DATA_PROCESSED, DATA_ROOT, ensure_dir
from v4.data.manifest import ManifestRow, write_manifest

log = get_logger(__name__)

DEFAULT_OUT = DATA_PROCESSED / "genimage_stage1_rows.csv"

GENERATORS = ("sd14", "sd15", "midjourney", "adm", "glide", "vqdm", "biggan", "wukong")
PER_GENERATOR = 1250


def _find_ai_images(gen_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for sub in ("ai", "fake", "generated", "AI"):
        d = gen_root / sub
        if d.exists():
            candidates.extend(d.rglob("*.jpg"))
            candidates.extend(d.rglob("*.png"))
            return candidates
    candidates.extend(gen_root.rglob("*.jpg"))
    candidates.extend(gen_root.rglob("*.png"))
    return candidates


def _resolve_gen_dir(gi_root: Path, gen: str) -> Path | None:
    aliases = [gen, gen.upper(), gen.capitalize()]
    if gen == "sd14":
        aliases += ["stable_diffusion_v_1_4", "SD14"]
    if gen == "sd15":
        aliases += ["stable_diffusion_v_1_5", "SD15"]
    if gen == "midjourney":
        aliases += ["Midjourney", "MidJourney", "MJ"]
    for alias in aliases:
        p = gi_root / alias
        if p.exists():
            return p
    return None


def build(out_path: str | None = None, seed: int = 42) -> str:
    gi_root = DATA_ROOT / "GenImage"
    if not gi_root.exists():
        raise FileNotFoundError(
            f"GenImage not found at {gi_root}. Stage the 8 official generator subdirs (sd14, sd15, midjourney, adm, glide, vqdm, biggan, wukong) under {gi_root}; see docs/SETUP.md step 6."
        )

    rng = random.Random(seed)
    rows: list[ManifestRow] = []

    for gen in GENERATORS:
        gen_root = _resolve_gen_dir(gi_root, gen)
        if gen_root is None:
            log.warning("GenImage generator %r not found under %s - skipping", gen, gi_root)
            continue
        images = _find_ai_images(gen_root)
        if not images:
            log.warning("no AI images found for %r at %s", gen, gen_root)
            continue
        rng.shuffle(images)
        for i, img in enumerate(images[:PER_GENERATOR]):
            rows.append(ManifestRow(
                sample_id=f"genimg_{gen}_{i:06d}",
                text="",
                image_path=str(img.resolve()),
                label=4,
                source=f"GenImage_{gen}",
                split="train",
            ))

    log.info("GenImage Stage-1 rows: %d", len(rows))
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    ensure_dir(out_path.parent)
    write_manifest(rows, out_path)
    return str(out_path)

"""Stage SD v1.4 / SD v1.5 into data/raw/GenImage_v2/ from the shimei123/Genimage zips.

This closes the two-generator gap (sdv1_4 + sdv1_5) that prepare_genimage_v2.py
documented as "bitmind 404 / Drive-only". Source found May 2026:

    shimei123/Genimage (HF dataset)
      SD_v14.zip  (3.55 GB)  ->  SD_v14/{0_real,1_fake}/*.JPEG   (official GenImage SD v1.4)
      SD_v15.zip  (4.74 GB)  ->  SD_v15/{0_real,1_fake}/*.JPEG   (official GenImage SD v1.5)

Both zips are the OFFICIAL GenImage layout: 0_real = ImageNet ILSVRC2012 "nature",
1_fake = the SD-generated images. So this ALSO fixes deviation F-A3 (VisualNews
substitute) for these two generators -- their nature is genuine ImageNet.

Output (matches prepare_genimage_v2.py / build_splits.py auto-discovery):
    data/raw/GenImage_v2/sdv1_4/ai/sdv1_4_ai_00000.jpg ...
    data/raw/GenImage_v2/sdv1_4/nature/sdv1_4_nature_00000.jpg ...
    data/raw/GenImage_v2/sdv1_5/{ai,nature}/ ...

Images are resized to 256x256 bilinear, JPEG q92 -- identical to the bitmind
extraction path so the 8 generators are preprocessed consistently.

USAGE:
    python phases/forensic/scripts/stage_sd_generators.py \\
        --zip-dir data/raw/_genimage_sd_zips \\
        --out data/raw/GenImage_v2 --target-per-gen 1750
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("FATAL: Pillow not installed", file=sys.stderr)
    sys.exit(2)

# zip file stem -> output generator key
ZIP_TO_GEN = {
    "SD_v14": "sdv1_4",
    "SD_v15": "sdv1_5",
}

# inside the zip, which subfolder is real vs fake (auto-detected, these are the candidates)
REAL_DIR_CANDIDATES = ("0_real", "nature", "real")
FAKE_DIR_CANDIDATES = ("1_fake", "ai", "fake")


def _log(msg: str) -> None:
    print(f"[stage_sd] {msg}", flush=True)


def _detect_subdir(names: list[str], top: str, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        prefix = f"{top}/{c}/"
        if any(n.startswith(prefix) and not n.endswith("/") for n in names):
            return c
    return None


def _extract_class(
    zf: zipfile.ZipFile,
    names: list[str],
    top: str,
    subdir: str,
    dst_dir: Path,
    gen: str,
    suffix_label: str,
    n_target: int,
    img_size: int,
) -> dict:
    """Extract up to n_target images from zip:top/subdir/ -> dst as resized JPGs."""
    prefix = f"{top}/{subdir}/"
    members = sorted(
        n
        for n in names
        if n.startswith(prefix)
        and not n.endswith("/")
        and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    fails: list[str] = []
    for m in members:
        if n_ok >= n_target:
            break
        out = dst_dir / f"{gen}_{suffix_label}_{n_ok:05d}.jpg"
        if out.exists():
            n_ok += 1
            continue
        try:
            blob = zf.read(m)
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            img.save(out, format="JPEG", quality=92)
            n_ok += 1
        except Exception as e:
            if len(fails) < 5:
                fails.append(f"{Path(m).name}: {type(e).__name__}: {e}")
    return {"n_extracted": n_ok, "n_available": len(members), "fail_examples": fails}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip-dir", type=Path, default=Path("data/raw/_genimage_sd_zips"))
    p.add_argument("--out", type=Path, default=Path("data/raw/GenImage_v2"))
    p.add_argument("--target-per-gen", type=int, default=1750)
    p.add_argument("--img-size", type=int, default=256)
    args = p.parse_args()

    summary: dict = {"started_at": time.time(), "generators": {}, "img_size": args.img_size}

    for zip_stem, gen in ZIP_TO_GEN.items():
        zpath = args.zip_dir / f"{zip_stem}.zip"
        if not zpath.exists():
            _log(f"FATAL: {zpath} not found -- download SD zips first")
            sys.exit(2)
        _log(f"=== {gen} (from {zpath.name}, {zpath.stat().st_size / 1e9:.2f} GB) ===")
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            tops = {n.split("/")[0] for n in names if "/" in n}
            top = zip_stem if zip_stem in tops else next(iter(tops))
            real_sub = _detect_subdir(names, top, REAL_DIR_CANDIDATES)
            fake_sub = _detect_subdir(names, top, FAKE_DIR_CANDIDATES)
            if real_sub is None or fake_sub is None:
                _log(f"FATAL: could not detect real/fake subdirs under {top}/ (tops={tops})")
                sys.exit(2)
            _log(f"  layout: {top}/{fake_sub} -> ai, {top}/{real_sub} -> nature")

            ai_res = _extract_class(
                zf,
                names,
                top,
                fake_sub,
                args.out / gen / "ai",
                gen,
                "ai",
                args.target_per_gen,
                args.img_size,
            )
            _log(f"  ai:     {ai_res['n_extracted']}/{ai_res['n_available']} available")
            nat_res = _extract_class(
                zf,
                names,
                top,
                real_sub,
                args.out / gen / "nature",
                gen,
                "nature",
                args.target_per_gen,
                args.img_size,
            )
            _log(f"  nature: {nat_res['n_extracted']}/{nat_res['n_available']} available")

        summary["generators"][gen] = {
            "zip": zpath.name,
            "ai_source": f"shimei123/Genimage::{zip_stem}/{fake_sub}",
            "nature_source": f"shimei123/Genimage::{zip_stem}/{real_sub} (official ImageNet ILSVRC2012)",
            "ai_result": ai_res,
            "nature_result": nat_res,
        }

    summary["finished_at"] = time.time()
    summary["elapsed_s"] = round(summary["finished_at"] - summary["started_at"], 1)
    out_summary = args.out / "_sd_stage_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    _log(f"\nDONE. Summary at {out_summary}  (elapsed {summary['elapsed_s']:.0f}s)")
    for gen, info in summary["generators"].items():
        _log(
            f"  {gen}: ai={info['ai_result']['n_extracted']} "
            f"nature={info['nature_result']['n_extracted']}"
        )


if __name__ == "__main__":
    main()

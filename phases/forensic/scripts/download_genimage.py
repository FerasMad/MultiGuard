"""Download GenImage data (8 generators + paired ImageNet nature) for Approach 2.

Doctor's spec F.1-F.3 (MASTER_CHECKLIST):
    Train + test from GenImage with 8 generators (midjourney, sdv1_4, sdv1_5,
    wukong, vqdm, biggan, adm, glide). Real class = ImageNet 'nature'.

Locked decisions:
    D-A1: Official GenImage Drive primary + HF fallback - reversed in practice
          because Drive requires manual browser click; HF is automatable. We try
          HuggingFace community mirrors (bitmind/GenImage_<Gen>) for fakes first;
          if any generator 404s, log + document the Drive fallback URL in
          download_manifest.json under 'manual_followup'.
    D-A2: 1,250 ai + 1,250 nature per generator for train; 500 + 500 per gen for test.
          So we sample ~1,750 ai per generator (1,250 train + 500 test).

Run:
    python phases/forensic/scripts/download_genimage.py --gens all --out data/raw/GenImage_v2
    python phases/forensic/scripts/download_genimage.py --gens midjourney --per-gen-ai 100  # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

HF_GENERATOR_SOURCES: dict[str, list[str]] = {
    "midjourney": ["bitmind/GenImage_MidJourney"],
    "sdv1_4": ["bitmind/GenImage_SDv1_4", "bitmind/GenImage_StableDiffusion_v1_4"],
    "sdv1_5": ["bitmind/GenImage_SDv1_5", "bitmind/GenImage_StableDiffusion_v1_5"],
    "wukong": ["bitmind/GenImage_Wukong"],
    "vqdm": ["bitmind/GenImage_VQDM"],
    "biggan": ["bitmind/GenImage_BigGAN"],
    "adm": ["bitmind/GenImage_ADM"],
    "glide": ["bitmind/GenImage_GLIDE"],
}

DEFAULT_PER_GEN_AI: int = 1750  # 1250 train + 500 test
DEFAULT_PER_GEN_NATURE: int = 1750

IMAGENET_SOURCES: list[str] = [
    "evanarlian/imagenet_1k_resized_256",
    "clip-benchmark/wds_imagenet1k",
    "imagenet-1k",
]

DRIVE_FALLBACK_HINT = (
    "https://github.com/GenImage-Dataset/GenImage  (see README for per-gen Drive links)"
)


def _safe_log(msg: str) -> None:
    print(f"[download_genimage] {msg}", flush=True)


def _download_one_generator_from_hf(
    generator: str,
    out_root: Path,
    per_gen_ai: int,
    cache_dir: Path,
) -> dict:
    ai_out = out_root / generator / "ai"
    ai_out.mkdir(parents=True, exist_ok=True)

    sources_tried: list[dict] = []
    for repo_id in HF_GENERATOR_SOURCES.get(generator, []):
        try:
            _safe_log(f"  trying HF: {repo_id}")
            t0 = time.time()
            local = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.parquet"],
            )
            elapsed = time.time() - t0
            sources_tried.append(
                {"repo_id": repo_id, "status": "downloaded", "elapsed_s": elapsed, "local": local}
            )

            local_p = Path(local)
            imgs = sorted(
                [p for p in local_p.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            if not imgs:
                _safe_log(f"  {repo_id} downloaded but no images found; trying next source")
                sources_tried[-1]["status"] = "no_images_found"
                continue

            _safe_log(
                f"  {repo_id}: found {len(imgs)} images, copying {min(per_gen_ai, len(imgs))} -> {ai_out}"
            )
            rng = random.Random(42)
            rng.shuffle(imgs)
            copied = 0
            for src_img in imgs[:per_gen_ai]:
                dst = ai_out / f"{generator}_{copied:05d}{src_img.suffix.lower()}"
                if not dst.exists():
                    shutil.copy2(src_img, dst)
                copied += 1
            return {
                "generator": generator,
                "status": "ok",
                "source": repo_id,
                "elapsed_s": elapsed,
                "n_copied": copied,
                "sources_tried": sources_tried,
            }
        except Exception as e:
            _safe_log(f"  {repo_id} FAILED: {type(e).__name__}: {e}")
            sources_tried.append(
                {"repo_id": repo_id, "status": "failed", "error": f"{type(e).__name__}: {e}"}
            )
            continue

    return {
        "generator": generator,
        "status": "manual_followup",
        "sources_tried": sources_tried,
        "drive_fallback": DRIVE_FALLBACK_HINT,
        "hint": (
            f"All HF mirrors failed for {generator}. Download manually from the "
            f"official GenImage Drive (see {DRIVE_FALLBACK_HINT}) and extract "
            f"AI images into {ai_out}."
        ),
    }


def _download_nature_pool(out_root: Path, total_needed: int, cache_dir: Path) -> dict:
    pool_out = out_root / "_nature_pool"
    pool_out.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        [p for p in pool_out.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if len(existing) >= total_needed:
        return {"status": "reused_existing", "n_available": len(existing), "source": "cached_pool"}

    sources_tried = []
    for repo_id in IMAGENET_SOURCES:
        try:
            _safe_log(f"  trying ImageNet nature from HF: {repo_id}")
            local = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.parquet"],
            )
            sources_tried.append({"repo_id": repo_id, "status": "downloaded", "local": local})
            local_p = Path(local)
            imgs = sorted(
                [p for p in local_p.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            if not imgs:
                sources_tried[-1]["status"] = "no_images"
                continue

            _safe_log(f"  {repo_id}: {len(imgs)} images, copying up to {total_needed}")
            rng = random.Random(7)
            rng.shuffle(imgs)
            copied = 0
            for src_img in imgs[:total_needed]:
                dst = pool_out / f"nature_{copied:06d}{src_img.suffix.lower()}"
                if not dst.exists():
                    shutil.copy2(src_img, dst)
                copied += 1
            return {
                "status": "ok",
                "source": repo_id,
                "n_copied": copied,
                "sources_tried": sources_tried,
            }
        except Exception as e:
            _safe_log(f"  {repo_id} FAILED: {type(e).__name__}: {e}")
            sources_tried.append(
                {"repo_id": repo_id, "status": "failed", "error": f"{type(e).__name__}: {e}"}
            )
            continue

    return {
        "status": "manual_followup",
        "sources_tried": sources_tried,
        "hint": (
            "All HF ImageNet mirrors failed. Manually populate "
            f"{pool_out}/ with ~{total_needed} JPEG images from any ImageNet 1K val source."
        ),
    }


def _distribute_nature_to_generators(
    out_root: Path,
    generators: list[str],
    per_gen_nature: int,
) -> dict:
    pool = out_root / "_nature_pool"
    pool_imgs = sorted([p for p in pool.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not pool_imgs:
        return {"status": "no_pool_images", "hint": "Run _download_nature_pool first."}

    rng = random.Random(123)
    rng.shuffle(pool_imgs)

    distributed = {}
    idx = 0
    for g in generators:
        nat_out = out_root / g / "nature"
        nat_out.mkdir(parents=True, exist_ok=True)
        slot = pool_imgs[idx : idx + per_gen_nature]
        idx += per_gen_nature
        for i, src in enumerate(slot):
            dst = nat_out / f"{g}_nature_{i:05d}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)
        distributed[g] = len(slot)
    return {"status": "ok", "per_generator_nature_copied": distributed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", nargs="+", default=["all"], help="Generators to download (or 'all')")
    p.add_argument("--out", type=Path, default=Path("data/raw/GenImage_v2"), help="Output root")
    p.add_argument("--per-gen-ai", type=int, default=DEFAULT_PER_GEN_AI)
    p.add_argument("--per-gen-nature", type=int, default=DEFAULT_PER_GEN_NATURE)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
        help="HF cache directory",
    )
    args = p.parse_args()

    if args.gens == ["all"]:
        gens = list(HF_GENERATOR_SOURCES.keys())
    else:
        gens = args.gens

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "started_at": time.time(),
        "generators_requested": gens,
        "per_gen_ai": args.per_gen_ai,
        "per_gen_nature": args.per_gen_nature,
        "results": {},
    }

    _safe_log(f"Phase 1: download fakes for {len(gens)} generators")
    for g in gens:
        _safe_log(f"--- generator: {g} ---")
        manifest["results"][g] = _download_one_generator_from_hf(
            g,
            args.out,
            args.per_gen_ai,
            args.cache_dir,
        )

    _safe_log("Phase 2: download ImageNet nature pool")
    total_nature_needed = args.per_gen_nature * len(gens)
    manifest["nature_pool"] = _download_nature_pool(args.out, total_nature_needed, args.cache_dir)

    if manifest["nature_pool"].get("status") == "ok":
        _safe_log("Phase 3: distribute nature pool to per-generator nature/ subdirs")
        manifest["nature_distribution"] = _distribute_nature_to_generators(
            args.out,
            gens,
            args.per_gen_nature,
        )

    manifest["finished_at"] = time.time()
    manifest["elapsed_s"] = manifest["finished_at"] - manifest["started_at"]

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    _safe_log(f"\nDONE. Manifest at {manifest_path}")

    ok_gens = [g for g, r in manifest["results"].items() if r.get("status") == "ok"]
    bad_gens = [g for g, r in manifest["results"].items() if r.get("status") != "ok"]
    _safe_log(f"AI fakes OK for {len(ok_gens)}/{len(gens)}: {ok_gens}")
    if bad_gens:
        _safe_log(f"NEEDS MANUAL FOLLOWUP: {bad_gens}")
        _safe_log(f"  See {manifest_path} 'results.<gen>.hint' for instructions per generator")

    if manifest.get("nature_pool", {}).get("status") != "ok":
        _safe_log("NATURE POOL FAILED - see manifest for details")
        sys.exit(2)

    if not ok_gens:
        sys.exit(2)
    if bad_gens:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

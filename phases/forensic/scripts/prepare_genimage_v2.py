"""Prepare data/raw/GenImage_v2/ for the 6 generators we have access to.

Outputs the doctor's required source-side layout:

    data/raw/GenImage_v2/
      midjourney/{ai,nature}/
      wukong/{ai,nature}/
      vqdm/{ai,nature}/
      biggan/{ai,nature}/
      adm/{ai,nature}/
      glide/{ai,nature}/
      _prep_summary.json

Subsequently consumed by scripts/build_splits.py.

GENERATOR DATA SOURCES (this build, May 2026):
  midjourney  -> local files at data/raw/GenImage/ai/midjourney/ (10k JPGs @ 256px)
  wukong      -> bitmind/GenImage_Wukong  (HF parquet)
  vqdm        -> bitmind/GenImage_VQDM    (HF parquet)
  biggan      -> bitmind/GenImage_BigGAN  (HF parquet)
  adm         -> bitmind/GenImage_ADM     (HF parquet)
  glide       -> bitmind/GenImage_GLIDE   (HF parquet)

SD v1.4 / SD v1.5 are NOT available on bitmind (HF 404), so they are excluded
from this build. The eval table will mark those two rows as SKIPPED with a
data-availability reason.

REAL CLASS DEVIATION (from doctor's spec F.3):
  Spec says real = ImageNet "nature". ImageNet is not on disk on this PC, so
  we substitute VisualNews real news photos (data/raw/visualnews/origin/).
  Documented in docs/forensic/DECISIONS.md.

PER-GENERATOR TARGETS:
  AI:     1750 per generator (so build_splits can take 1250 train + 500 test)
  Nature: 1750 per generator (same)

USAGE:
    python phases/forensic/scripts/prepare_genimage_v2.py \\
        --out data/raw/GenImage_v2 \\
        --target-per-gen 1750
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("FATAL: PIL/Pillow not installed. pip install pillow", file=sys.stderr)
    sys.exit(2)


# -------- generator -> source mapping --------

BITMIND_GENERATORS: dict[str, str] = {
    "wukong": "bitmind/GenImage_Wukong",
    "vqdm":   "bitmind/GenImage_VQDM",
    "biggan": "bitmind/GenImage_BigGAN",
    "adm":    "bitmind/GenImage_ADM",
    "glide":  "bitmind/GenImage_GLIDE",
}

LOCAL_GENERATORS: dict[str, Path] = {
    "midjourney": Path("data/raw/GenImage/ai/midjourney"),
}

ALL_GENERATORS = list(LOCAL_GENERATORS) + list(BITMIND_GENERATORS)


def _log(msg: str) -> None:
    print(f"[prepare_v2] {msg}", flush=True)


# -------- step 1: midjourney from local --------

def stage_local_midjourney(src_dir: Path, dst_dir: Path, n: int, seed: int) -> dict:
    """Copy n images from src_dir to dst_dir as <idx>.jpg."""
    if not src_dir.exists():
        return {"status": "skipped_no_src", "n_copied": 0}

    imgs = sorted([p for p in src_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        return {"status": "skipped_no_imgs", "n_copied": 0}

    rng = random.Random(seed)
    rng.shuffle(imgs)
    picked = imgs[:n]

    dst_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for i, src in enumerate(picked):
        dst = dst_dir / f"midjourney_ai_{i:05d}{src.suffix.lower()}"
        if dst.exists():
            n_ok += 1
            continue
        try:
            shutil.copyfile(src, dst)
            n_ok += 1
        except Exception as e:
            _log(f"  copy fail {src.name}: {e}")

    return {"status": "ok", "n_copied": n_ok, "n_available": len(imgs)}


# -------- step 2: bitmind parquet download + extract --------

def extract_bitmind_generator(
    gen: str,
    repo_id: str,
    dst_dir: Path,
    n_target: int,
    img_size: int = 256,
    n_parquets_max: int = 6,
) -> dict:
    """Download just enough parquets from HF to extract n_target images, save as resized JPGs."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as e:
        return {"status": "skipped_dep", "n_extracted": 0, "error": str(e)}

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        return {"status": "skipped_api_error", "n_extracted": 0, "error": str(e)}

    parquet_names = sorted([f for f in files if f.endswith(".parquet")])
    if not parquet_names:
        return {"status": "skipped_no_parquets", "n_extracted": 0}

    dst_dir.mkdir(parents=True, exist_ok=True)

    n_extracted = 0
    n_parquets_used = 0
    fail_examples: list[str] = []
    start = time.time()

    for pqn in parquet_names[:n_parquets_max]:
        if n_extracted >= n_target:
            break
        _log(f"  {gen}: download {pqn}")
        try:
            local_pq = hf_hub_download(
                repo_id=repo_id,
                filename=pqn,
                repo_type="dataset",
            )
        except Exception as e:
            fail_examples.append(f"download {pqn}: {e}")
            continue

        n_parquets_used += 1
        try:
            tbl = pq.read_table(local_pq)
        except Exception as e:
            fail_examples.append(f"read {pqn}: {e}")
            continue

        rows = tbl.to_pylist()
        _log(f"    {len(rows)} rows in parquet")
        for row in rows:
            if n_extracted >= n_target:
                break
            imgdict = row.get("image")
            if imgdict is None:
                continue
            blob = imgdict.get("bytes") if isinstance(imgdict, dict) else None
            if not blob:
                continue
            try:
                img = Image.open(io.BytesIO(blob)).convert("RGB")
                img = img.resize((img_size, img_size), Image.BILINEAR)
                out = dst_dir / f"{gen}_ai_{n_extracted:05d}.jpg"
                if out.exists():
                    n_extracted += 1
                    continue
                img.save(out, format="JPEG", quality=92)
                n_extracted += 1
            except Exception as e:
                if len(fail_examples) < 5:
                    fail_examples.append(f"decode {pqn}/row: {type(e).__name__}: {e}")

        # delete parquet immediately to save disk
        try:
            Path(local_pq).unlink()
        except Exception:
            pass

    elapsed = time.time() - start
    return {
        "status": "ok" if n_extracted >= n_target else ("partial" if n_extracted else "fail"),
        "n_extracted": n_extracted,
        "n_target": n_target,
        "n_parquets_used": n_parquets_used,
        "fail_examples": fail_examples,
        "elapsed_s": round(elapsed, 1),
    }


# -------- step 3: VisualNews -> per-generator nature --------

def sample_visualnews_nature(
    vn_root: Path,
    dst_per_gen: dict[str, Path],
    n_per_gen: int,
    seed: int,
) -> dict:
    """Sample n_per_gen distinct images per generator from VisualNews JPGs."""
    if not vn_root.exists():
        return {"status": "skipped_no_src", "per_gen": {}}

    all_imgs = sorted(vn_root.rglob("*.jpg"))
    _log(f"VisualNews pool: {len(all_imgs)} JPGs found")

    if len(all_imgs) < n_per_gen * len(dst_per_gen):
        _log(f"  WARNING: pool ({len(all_imgs)}) < needed ({n_per_gen * len(dst_per_gen)}); will reuse")

    rng = random.Random(seed)
    rng.shuffle(all_imgs)

    out: dict = {}
    cursor = 0
    for gen, dst_dir in dst_per_gen.items():
        dst_dir.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for i in range(n_per_gen):
            # Wrap-around if pool is exhausted (unlikely with 72K pool, 6 gens x 1750 = 10.5K)
            src = all_imgs[(cursor + i) % len(all_imgs)]
            dst = dst_dir / f"{gen}_nature_{i:05d}.jpg"
            if dst.exists():
                n_ok += 1
                continue
            try:
                shutil.copyfile(src, dst)
                n_ok += 1
            except Exception as e:
                if n_ok < 5:
                    _log(f"  copy fail {gen}/{i}: {e}")
        cursor += n_per_gen
        out[gen] = {"n_copied": n_ok}

    return {"status": "ok", "per_gen": out}


# -------- main --------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/GenImage_v2"))
    p.add_argument("--target-per-gen", type=int, default=1750,
                   help="How many AI + nature samples per generator")
    p.add_argument("--visualnews-root", type=Path,
                   default=Path("data/raw/visualnews/origin"))
    p.add_argument("--skip-bitmind", action="store_true",
                   help="Skip bitmind downloads (use only midjourney local)")
    p.add_argument("--only-gens", nargs="+", default=None,
                   help="Subset of generators to process (default: all 6)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "generators": {},
        "missing_generators": ["sdv1_4", "sdv1_5"],
        "missing_reason": "bitmind/GenImage_StableDiffusionV1.4 and bitmind/GenImage_StableDiffusionV1.5 return HF 404 (May 2026)",
        "nature_source": "VisualNews (substitute for ImageNet 'nature' per spec F.3)",
        "started_at": time.time(),
    }

    gens_to_do = args.only_gens or ALL_GENERATORS
    _log(f"Will process: {gens_to_do}")

    # Stage 1: midjourney from local
    if "midjourney" in gens_to_do:
        _log("=== midjourney (local) ===")
        res = stage_local_midjourney(
            LOCAL_GENERATORS["midjourney"],
            args.out / "midjourney" / "ai",
            n=args.target_per_gen,
            seed=args.seed,
        )
        summary["generators"]["midjourney"] = {"ai_source": "local", "ai_result": res}

    # Stage 2: bitmind generators (serial to keep disk usage low)
    if not args.skip_bitmind:
        for gen, repo_id in BITMIND_GENERATORS.items():
            if gen not in gens_to_do:
                continue
            _log(f"=== {gen} (bitmind {repo_id}) ===")
            res = extract_bitmind_generator(
                gen=gen,
                repo_id=repo_id,
                dst_dir=args.out / gen / "ai",
                n_target=args.target_per_gen,
            )
            summary["generators"].setdefault(gen, {})["ai_source"] = repo_id
            summary["generators"][gen]["ai_result"] = res

    # Stage 3: VisualNews nature for all done generators
    _log("=== nature (VisualNews substitute) ===")
    dst_per_gen = {g: args.out / g / "nature" for g in gens_to_do}
    res = sample_visualnews_nature(
        args.visualnews_root, dst_per_gen,
        n_per_gen=args.target_per_gen, seed=args.seed + 1,
    )
    for gen, info in res.get("per_gen", {}).items():
        summary["generators"].setdefault(gen, {})["nature_source"] = "VisualNews"
        summary["generators"][gen]["nature_result"] = info

    # Write summary
    summary["finished_at"] = time.time()
    summary["elapsed_s"] = round(summary["finished_at"] - summary["started_at"], 1)
    summary_path = args.out / "_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    _log(f"\nSummary at {summary_path}")
    _log(f"Elapsed: {summary['elapsed_s']:.0f}s")

    for gen in gens_to_do:
        g_info = summary["generators"].get(gen, {})
        ai_n = g_info.get("ai_result", {}).get("n_extracted") or g_info.get("ai_result", {}).get("n_copied", 0)
        nat_n = g_info.get("nature_result", {}).get("n_copied", 0)
        _log(f"  {gen:11s} ai={ai_n}  nature={nat_n}")


if __name__ == "__main__":
    main()

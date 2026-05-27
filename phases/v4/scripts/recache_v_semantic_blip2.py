"""Build cache/v3/_features/v_semantic_blip2/ -- class 3/4 from text_blip2.

P15.1a: closes the "honest path" gap. The original v_semantic cache was built
from the original LLM-syntactic-fingerprint text for cls 3/4 (Qwen text branch
got the BLIP-2 honest captions but FND-CLIP's BERT sub-encoder still saw the
shortcut). This script re-encodes only the cls 3/4 rows with text_blip2 and
copies cls 0/1/2 verbatim from the existing cache.

For classes 0, 1, 2: copies existing cache/v3/_features/v_semantic/{sid}.pt
For classes 3, 4: regenerates 512-d FND-CLIP V1 vector from (image, text_blip2).

Run:
    python phases/v4/scripts/recache_v_semantic_blip2.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v4" / "app_hf"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/forensic_5class_unified_blip2.csv"),
    )
    p.add_argument(
        "--src-cache",
        type=Path,
        default=Path("cache/v3/_features/v_semantic"),
    )
    p.add_argument(
        "--out-cache",
        type=Path,
        default=Path("cache/v3/_features/v_semantic_blip2"),
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("outputs/v1/leakfree/best.pt"),
        help="V1 leakfree FND-CLIP ckpt -- must match the one that built v_semantic.",
    )
    p.add_argument("--target-labels", type=int, nargs="+", default=[3, 4])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from inline.fnd_clip import FNDCLIPSemanticEncoder, prepare_fnd_inputs

    args.out_cache.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    target_mask = df["label"].isin(args.target_labels)
    print(
        f"[recache_v_sem] manifest={len(df)} rows, "
        f"target (re-encode) rows={target_mask.sum()}, "
        f"copy-from-src rows={(~target_mask).sum()}"
    )

    t0 = time.time()
    print(f"[recache_v_sem] step 1: copying {(~target_mask).sum()} unchanged samples ...")
    copied = 0
    missing = 0
    for _, row in df[~target_mask].iterrows():
        sid = str(row["sample_id"])
        src = args.src_cache / f"{sid}.pt"
        dst = args.out_cache / f"{sid}.pt"
        if dst.exists():
            copied += 1
            continue
        if not src.exists():
            missing += 1
            continue
        shutil.copy(src, dst)
        copied += 1
    print(f"[recache_v_sem] copied={copied} missing_in_src={missing} took={time.time() - t0:.1f}s")

    target_df = df[target_mask].copy().reset_index(drop=True)
    if args.limit is not None:
        target_df = target_df.head(args.limit)

    print(f"[recache_v_sem] step 2: regenerating {len(target_df)} class-3/4 with FND-CLIP V1 ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[recache_v_sem] device={device}")

    encoder = FNDCLIPSemanticEncoder(ckpt=args.ckpt)
    encoder.to(device).eval()

    n_done = 0
    n_skip = 0
    fails = 0
    last_log = time.time()

    rows = list(target_df.itertuples(index=False))
    with torch.no_grad():
        for _i, row in enumerate(rows):
            sid = str(row.sample_id)
            dst = args.out_cache / f"{sid}.pt"
            if dst.exists():
                n_skip += 1
                continue

            try:
                pil_img = Image.open(row.image_path).convert("RGB")
                text = str(getattr(row, "text_blip2", None) or row.text)
                inputs = prepare_fnd_inputs(text, pil_img)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                vec = encoder.forward_semantic(**inputs).squeeze(0).float().cpu()
                assert vec.shape == (512,), f"unexpected {vec.shape}"
                torch.save(vec, dst)
                n_done += 1
            except Exception as e:
                fails += 1
                if fails <= 5:
                    print(f"  FAIL sid={sid}: {type(e).__name__}: {e}")

            now = time.time()
            if now - last_log > 15.0:
                elapsed = now - t0
                processed = n_done + n_skip + fails
                rate = processed / max(elapsed, 1e-3)
                eta = (len(target_df) - processed) / max(rate, 0.1)
                print(
                    f"  progress: {processed}/{len(target_df)} "
                    f"({100 * processed / len(target_df):.1f}%)  "
                    f"rate={rate:.1f}/s  eta={eta / 60:.0f}min  fails={fails}",
                    flush=True,
                )
                last_log = now

    total = time.time() - t0
    print(f"\n[recache_v_sem] DONE in {total:.1f}s ({total / 60:.1f}min)")
    print(f"  copied (cls 0/1/2):    {copied}")
    print(f"  regenerated (cls 3/4): {n_done}")
    print(f"  skipped:               {n_skip}")
    print(f"  failed:                {fails}")
    print(f"  out_cache: {args.out_cache} ({sum(1 for _ in args.out_cache.iterdir())} shards)")


if __name__ == "__main__":
    main()

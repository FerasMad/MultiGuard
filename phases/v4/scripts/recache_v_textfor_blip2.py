"""Build cache/v3/_features/v_textfor_qwen_blip2/ — class 3/4 from text_blip2.

For classes 0, 1, 2: copies existing v_textfor_qwen/{sid}.pt verbatim.
For classes 3, 4: regenerates the 3584-d Qwen2-7B hidden state from
the BLIP-2-rewritten caption.

Run:
    python phases/v4/scripts/recache_v_textfor_blip2.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("data/processed/forensic_5class_unified_blip2.csv"))
    p.add_argument("--src-cache", type=Path,
                   default=Path("cache/v3/_features/v_textfor_qwen"))
    p.add_argument("--out-cache", type=Path,
                   default=Path("cache/v3/_features/v_textfor_qwen_blip2"))
    p.add_argument("--target-labels", type=int, nargs="+", default=[3, 4])
    p.add_argument("--model-id", default="Qwen/Qwen2-7B-Instruct")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.out_cache.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    target_mask = df["label"].isin(args.target_labels)
    print(f"[recache] manifest={len(df)} rows, "
          f"target (re-encode) rows={target_mask.sum()}, "
          f"copy-from-src rows={(~target_mask).sum()}")

    t0 = time.time()
    print(f"[recache] step 1: copying {(~target_mask).sum()} unchanged samples ...")
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
    print(f"[recache] copied={copied} missing_in_src={missing} took={time.time()-t0:.1f}s")

    target_df = df[target_mask].copy().reset_index(drop=True)
    if args.limit is not None:
        target_df = target_df.head(args.limit)
    print(f"[recache] step 2: regenerating {len(target_df)} class-3/4 with Qwen2-7B ...")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_hf"))
    from inline.qwen_text import Qwen2TextEncoder

    qwen = Qwen2TextEncoder(model_name=args.model_id, max_length=args.max_length,
                            load_backbone=True)

    n_done = 0
    n_skip = 0
    fails = 0
    last_log = time.time()
    for _, row in target_df.iterrows():
        sid = str(row["sample_id"])
        dst = args.out_cache / f"{sid}.pt"
        if dst.exists():
            n_skip += 1
            continue
        text = str(row.get("text_blip2", row["text"]))
        try:
            with torch.no_grad():
                pooled = qwen.encode_text(text).squeeze(0).cpu()
            assert pooled.shape == (3584,), f"unexpected shape {pooled.shape}"
            torch.save(pooled.float(), dst)
            n_done += 1
        except Exception as e:
            fails += 1
            if fails <= 5:
                print(f"  FAIL {sid}: {type(e).__name__}: {e}")

        now = time.time()
        if now - last_log > 15.0:
            rate = (n_done + n_skip) / (now - t0 + 1e-3)
            print(f"  progress: {n_done + n_skip + fails}/{len(target_df)} "
                  f"rate={rate:.1f}/s fails={fails} skipped={n_skip}", flush=True)
            last_log = now

    total = time.time() - t0
    print(f"\n[recache] DONE in {total:.1f}s ({total/60:.1f}min)")
    print(f"  copied (cls 0/1/2):    {copied}")
    print(f"  regenerated (cls 3/4): {n_done}")
    print(f"  skipped:               {n_skip}")
    print(f"  failed:                {fails}")
    print(f"  out_cache: {args.out_cache} ({sum(1 for _ in args.out_cache.iterdir())} shards)")


if __name__ == "__main__":
    main()

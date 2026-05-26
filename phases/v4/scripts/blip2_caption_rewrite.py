"""BLIP-2 caption rewrite for class-3 (AI-Text) and class-4 (Fully-Fab) rows.

Removes the documented caption-syntactic-fingerprint shortcut by replacing
class-3/4 captions with BLIP-2 image-grounded descriptions.

Output: data/processed/forensic_5class_unified_blip2.csv (same schema +
extra `text_blip2` column).

Run:
    python phases/v4/scripts/blip2_caption_rewrite.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("data/processed/forensic_5class_unified.csv"))
    p.add_argument("--out-manifest", type=Path,
                   default=Path("data/processed/forensic_5class_unified_blip2.csv"))
    p.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke: cap class-3/4 rows processed")
    p.add_argument("--target-labels", type=int, nargs="+", default=[3, 4])
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"[blip2] manifest: {len(df)} rows")

    target_mask = df["label"].isin(args.target_labels)
    target_df = df[target_mask].copy().reset_index(drop=True)
    if args.limit is not None:
        target_df = target_df.head(args.limit)
    print(f"[blip2] target rows (label in {args.target_labels}): {len(target_df)}")

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[blip2] device={device}, loading {args.model_id} ...")
    processor = Blip2Processor.from_pretrained(args.model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("[blip2] model loaded.")

    new_captions = []
    fails = 0
    t0 = time.time()
    last_log = t0

    for start in range(0, len(target_df), args.batch_size):
        batch = target_df.iloc[start:start + args.batch_size]
        images = []
        for img_path in batch["image_path"]:
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224), color=0))

        try:
            inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_tokens)
            captions = processor.batch_decode(generated, skip_special_tokens=True)
            captions = [c.strip() for c in captions]
        except Exception as e:
            print(f"  FAIL batch {start}: {type(e).__name__}: {e}")
            captions = ["(blip2 failed)"] * len(images)
            fails += len(images)

        new_captions.extend(captions)
        now = time.time()
        if now - last_log > 15.0:
            done = len(new_captions)
            rate = done / (now - t0 + 1e-3)
            eta = (len(target_df) - done) / max(rate, 0.1)
            print(f"  progress: {done}/{len(target_df)} ({100*done/len(target_df):.1f}%)  "
                  f"rate={rate:.1f}/s  eta={eta/60:.0f}min  fails={fails}", flush=True)
            last_log = now

    assert len(new_captions) == len(target_df)

    df["text_blip2"] = df["text"].astype(str).copy()
    target_idx = df.index[target_mask][:len(new_captions)]
    df.loc[target_idx, "text_blip2"] = new_captions

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_manifest, index=False)

    elapsed = time.time() - t0
    print(f"\n[blip2] DONE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  captioned: {len(new_captions)}")
    print(f"  failed:    {fails}")
    print(f"  rate:      {len(new_captions)/elapsed:.1f}/s")
    print(f"  written:   {args.out_manifest}")
    print()
    print("Sample captions (first 5):")
    for i in range(min(5, len(target_df))):
        print(f"  [{target_df.iloc[i]['label']}] {target_df.iloc[i]['sample_id']}")
        print(f"    orig:  {target_df.iloc[i]['text'][:100]}")
        print(f"    blip2: {new_captions[i][:100]}")


if __name__ == "__main__":
    main()

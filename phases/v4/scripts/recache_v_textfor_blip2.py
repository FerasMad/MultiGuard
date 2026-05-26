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
    print(f"[recache] step 2: regenerating {len(target_df)} class-3/4 with Qwen2-7B (4-bit, batched) ...")

    # Use 4-bit Qwen with bitsandbytes for VRAM fit on RTX 4070 (12 GB).
    # Falls back to fp16 if bitsandbytes unavailable.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                     bnb_4bit_quant_type="nf4")
        print("[recache] loading Qwen with bnb-4bit quantization ...")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_cfg, device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"[recache] bnb-4bit failed ({e}); falling back to fp16 device_map='auto'")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True,
        )
    qwen_model.eval()
    for p in qwen_model.parameters():
        p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    print(f"[recache] Qwen loaded.")

    n_done = 0
    n_skip = 0
    fails = 0
    last_log = time.time()
    batch_size = 4  # tight on 12 GB even with 4-bit; keep small
    device = next(qwen_model.parameters()).device

    rows = list(target_df.itertuples(index=False))
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        # filter out already-done rows in this batch
        pending = [(r, args.out_cache / f"{r.sample_id}.pt") for r in batch_rows]
        skip_in_batch = [(r, dst) for (r, dst) in pending if dst.exists()]
        do_in_batch = [(r, dst) for (r, dst) in pending if not dst.exists()]
        n_skip += len(skip_in_batch)
        if not do_in_batch:
            continue

        texts = []
        sids = []
        for r, dst in do_in_batch:
            sids.append(str(r.sample_id))
            texts.append(str(getattr(r, "text_blip2", None) or r.text))

        try:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = qwen_model(**enc, output_hidden_states=True)
                h = out.hidden_states[-1]  # [B, S, 3584]
                mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = pooled.float().cpu()
            assert pooled.shape == (len(do_in_batch), 3584), f"unexpected {pooled.shape}"
            for (r, dst), vec in zip(do_in_batch, pooled):
                torch.save(vec, dst)
            n_done += len(do_in_batch)
        except Exception as e:
            fails += len(do_in_batch)
            if fails <= 5:
                print(f"  FAIL batch start={start}: {type(e).__name__}: {e}")

        now = time.time()
        if now - last_log > 15.0:
            elapsed = now - t0
            rate = (n_done + n_skip) / max(elapsed, 1e-3)
            eta = (len(target_df) - n_done - n_skip - fails) / max(rate, 0.1)
            print(f"  progress: {n_done + n_skip + fails}/{len(target_df)} "
                  f"({100*(n_done+n_skip+fails)/len(target_df):.1f}%)  "
                  f"rate={rate:.1f}/s  eta={eta/60:.0f}min  fails={fails}", flush=True)
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

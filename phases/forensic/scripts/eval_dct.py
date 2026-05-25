"""Evaluate a trained forensic_dct_model.pth across all 8 generators.

Output: outputs/dct/eval_dct.json + outputs/eval_table.md (doctor's required format).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from forensic.evaluation.per_generator import (
    ALL_GENERATORS,
    aggregate_metrics,
    evaluate_per_generator,
)
from forensic.evaluation.table import write_eval_table
from forensic.models.dct_resnet50 import build_dct_resnet50


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", type=Path, required=True, help="Path to forensic_dct_model.pth (or best.pt)"
    )
    p.add_argument("--test-cache", type=Path, default=Path("phases/forensic/data/dct_cache/test"))
    p.add_argument("--dct-stats", type=Path, default=Path("phases/forensic/data/dct_stats.json"))
    p.add_argument("--out-json", type=Path, default=Path("phases/forensic/outputs/eval_dct.json"))
    p.add_argument("--out-table", type=Path, default=Path("phases/forensic/outputs/eval_table.md"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--only", nargs="+", default=None, help="Subset of generators to eval (default: all 8)"
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_dct] device: {device}")

    print(f"[eval_dct] loading model from {args.ckpt}")
    model = build_dct_resnet50(pretrained=False)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    gens = tuple(args.only) if args.only else ALL_GENERATORS
    print(f"[eval_dct] evaluating generators: {gens}")

    per_gen = evaluate_per_generator(
        model=model,
        test_cache_root=args.test_cache,
        dct_stats_path=args.dct_stats,
        device=device,
        generators=gens,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    aggregates = aggregate_metrics(per_gen)

    print("\n[eval_dct] per-generator results:")
    for g in gens:
        row = per_gen.get(g, {})
        if row.get("skipped"):
            print(f"  {g:12s} SKIPPED: {row.get('reason')}")
            continue
        print(
            f"  {g:12s} AP={row.get('ap', float('nan')):.4f}  "
            f"Acc={row.get('accuracy', float('nan')):.4f}  "
            f"AUC={row.get('auc', float('nan')):.4f}  "
            f"n={row.get('n_samples', 0)}"
        )
    print("\n[eval_dct] aggregates:")
    for k, v in aggregates.items():
        print(f"  {k}: {v}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "per_generator": per_gen,
                "aggregates": aggregates,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n[eval_dct] wrote: {args.out_json}")

    write_eval_table(
        per_gen,
        aggregates,
        out_path=args.out_table,
        title="Forensic Image Detector - Approach 2 (DCT) - Per-Generator Eval",
        notes=f"Checkpoint: `{args.ckpt}`",
    )
    print(f"[eval_dct] wrote: {args.out_table}")


if __name__ == "__main__":
    main()

"""Server-vs-eval parity sweep.

Picks N stratified rows from the test split, POSTs each (text, image) to the
running FastAPI server's /api/analyze endpoint, and compares the predicted
class against the ground-truth label. Writes a JSON report.

Usage:
    python phases/v4/scripts/server_parity_sweep.py --per-class 10
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("data/processed/forensic_5class_unified.csv"))
    p.add_argument("--server", default="http://127.0.0.1:8081")
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path("phases/v4/docs/eval/seed42/server_parity.json"))
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.manifest)
    test = df[df["split"] == "test"]
    print(f"manifest: {len(df)} rows, {len(test)} test rows")

    sampled = []
    for cls in range(5):
        cls_rows = test[test["label"] == cls]
        n = min(args.per_class, len(cls_rows))
        idxs = random.sample(range(len(cls_rows)), n)
        sampled.extend(cls_rows.iloc[idxs].to_dict("records"))
    print(f"sampled {len(sampled)} rows ({args.per_class} per class)")

    health = requests.get(f"{args.server}/api/health", timeout=10).json()
    print(f"server health: {health.get('status')}, device={health.get('device')}")

    results = []
    correct_by_cls = Counter()
    total_by_cls = Counter()
    fail_count = 0
    t0 = time.time()
    for i, row in enumerate(sampled):
        gt = int(row["label"])
        sid = row["sample_id"]
        text = str(row["text"])
        img_path = Path(row["image_path"])
        if not img_path.exists():
            print(f"  [{i+1}/{len(sampled)}] {sid} MISSING IMAGE: {img_path}")
            fail_count += 1
            continue

        try:
            t1 = time.time()
            with open(img_path, "rb") as f:
                r = requests.post(
                    f"{args.server}/api/analyze",
                    data={"text": text},
                    files={"image": (img_path.name, f, "image/jpeg")},
                    timeout=args.timeout,
                )
            elapsed = time.time() - t1
            if r.status_code != 200:
                print(f"  [{i+1}/{len(sampled)}] {sid} HTTP {r.status_code}: {r.text[:100]}")
                fail_count += 1
                continue
            j = r.json()
            if "error" in j:
                print(f"  [{i+1}/{len(sampled)}] {sid} ERR: {j['error']}")
                fail_count += 1
                continue
            pred = int(j["label_index"])
            conf = float(j["confidence"])
            ok = pred == gt
            total_by_cls[gt] += 1
            if ok:
                correct_by_cls[gt] += 1
            results.append({
                "sample_id": sid,
                "source": row["source"],
                "gt_label": gt,
                "pred_label": pred,
                "confidence": conf,
                "probabilities": j["probabilities"],
                "modules": j["modules"],
                "match": ok,
                "elapsed_s": round(elapsed, 2),
            })
            mark = "+" if ok else "-"
            print(f"  [{i+1}/{len(sampled)}] {mark} gt={gt} pred={pred} ({conf:.1f}%) "
                  f"{elapsed:.1f}s  {sid}")
        except Exception as e:
            print(f"  [{i+1}/{len(sampled)}] {sid} EXC {type(e).__name__}: {e}")
            fail_count += 1

    total_elapsed = time.time() - t0
    overall_correct = sum(correct_by_cls.values())
    overall_total = sum(total_by_cls.values())
    overall_acc = overall_correct / max(overall_total, 1)

    per_class_acc = {}
    for cls in range(5):
        if total_by_cls[cls] == 0:
            per_class_acc[cls] = None
        else:
            per_class_acc[cls] = correct_by_cls[cls] / total_by_cls[cls]

    summary = {
        "n_sampled": len(sampled),
        "n_evaluated": overall_total,
        "n_failed": fail_count,
        "overall_accuracy": round(overall_acc, 4),
        "per_class_accuracy": {
            str(cls): (round(v, 4) if v is not None else None)
            for cls, v in per_class_acc.items()
        },
        "per_class_count": {str(cls): total_by_cls[cls] for cls in range(5)},
        "per_class_correct": {str(cls): correct_by_cls[cls] for cls in range(5)},
        "wall_clock_s": round(total_elapsed, 1),
        "per_request_seconds_mean": round(
            sum(r["elapsed_s"] for r in results) / max(len(results), 1), 2
        ),
        "server": args.server,
        "manifest": str(args.manifest),
        "seed": args.seed,
        "samples": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"PARITY SWEEP COMPLETE — {overall_correct}/{overall_total} correct "
          f"({overall_acc:.1%})")
    print(f"  per-class: " + " | ".join(
        f"{cls}: {(per_class_acc[cls] or 0):.0%}" for cls in range(5)
    ))
    print(f"  failures : {fail_count}")
    print(f"  wallclock: {total_elapsed:.1f}s "
          f"(mean {summary['per_request_seconds_mean']}s/request)")
    print(f"  written  : {args.out}")


if __name__ == "__main__":
    main()

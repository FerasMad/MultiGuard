"""Build the doctor's F.25 combined eval table (Approach 1 + Approach 2).

Reads both:
  - phases/forensic/outputs/eval_dct.json  (Approach 2 results)
  - phases/forensic/outputs/eval_rgb.json  (Approach 1 results)

Writes:
  - phases/forensic/outputs/eval_table_combined.md

The combined table has rows = 8 generators + 4 summary, columns =
AP / Acc / AUC for both approaches side-by-side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forensic.evaluation.per_generator import (
    ALL_GENERATORS,
    GENERATORS_DIFFUSION,
    GENERATORS_GAN,
)


def _fmt(v) -> str:
    """Format a metric for the table; '-' for NaN or missing."""
    if v is None:
        return "-"
    try:
        if isinstance(v, float) and v != v:  # NaN
            return "-"
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "-"


def _gen_type(g: str) -> str:
    if g in GENERATORS_GAN:
        return "GAN"
    if g in GENERATORS_DIFFUSION:
        return "Diffusion"
    return "?"


def _row(per_gen: dict, g: str) -> dict:
    """Return {ap, acc, auc} or None if skipped/missing."""
    row = per_gen.get(g, {}) if isinstance(per_gen, dict) else {}
    if row.get("skipped"):
        return {"ap": None, "acc": None, "auc": None, "skipped": True}
    return {
        "ap": row.get("ap"),
        "acc": row.get("accuracy"),
        "auc": row.get("auc"),
        "skipped": False,
    }


def _load_or_empty(path: Path) -> dict:
    if not path.exists():
        return {"per_generator": {}, "aggregates": {}, "_missing": True}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dct-json", type=Path, default=Path("phases/forensic/outputs/eval_dct.json"))
    p.add_argument("--rgb-json", type=Path, default=Path("phases/forensic/outputs/eval_rgb.json"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("phases/forensic/outputs/eval_table_combined.md"),
    )
    p.add_argument(
        "--title",
        default=(
            "Forensic Image Detector - Approach 1 (RGB+Fourier) + Approach 2 (DCT) "
            "- Per-Generator Eval"
        ),
    )
    args = p.parse_args()

    dct = _load_or_empty(args.dct_json)
    rgb = _load_or_empty(args.rgb_json)

    dct_per = dct.get("per_generator", {})
    rgb_per = rgb.get("per_generator", {})
    dct_agg = dct.get("aggregates", {})
    rgb_agg = rgb.get("aggregates", {})

    lines: list[str] = []
    lines.append(f"# {args.title}\n")
    lines.append("Per-generator AP / Accuracy / AUC for both approaches (F.25 + F.23).\n")
    if dct.get("_missing"):
        lines.append(f"> Note: `{args.dct_json}` not found — Approach 2 columns will show `-`.\n")
    if rgb.get("_missing"):
        lines.append(f"> Note: `{args.rgb_json}` not found — Approach 1 columns will show `-`.\n")

    lines.append("| Generator | Type | A1 AP | A2 AP | A1 Acc | A2 Acc | A1 AUC | A2 AUC | n |")
    lines.append("|-----------|------|-------|-------|--------|--------|--------|--------|---|")

    for g in ALL_GENERATORS:
        r1 = _row(rgb_per, g)
        r2 = _row(dct_per, g)
        if r1.get("skipped") and r2.get("skipped"):
            lines.append(
                f"| {g} | {_gen_type(g)} | _skipped_ | _skipped_ | _skipped_ | _skipped_ "
                f"| _skipped_ | _skipped_ | 0 |"
            )
            continue
        n_samples = rgb_per.get(g, {}).get("n_samples") or dct_per.get(g, {}).get("n_samples") or 0
        lines.append(
            f"| {g} | {_gen_type(g)} "
            f"| {_fmt(r1['ap'])} | {_fmt(r2['ap'])} "
            f"| {_fmt(r1['acc'])} | {_fmt(r2['acc'])} "
            f"| {_fmt(r1['auc'])} | {_fmt(r2['auc'])} "
            f"| {n_samples} |"
        )

    # Aggregate rows
    def _agg_row(label: str, key: str) -> str:
        a1 = rgb_agg.get(key, {})
        a2 = dct_agg.get(key, {})
        type_label = (
            "(all)" if key == "overall_avg" else ("GAN" if key == "gan_avg" else "Diffusion")
        )
        return (
            f"| **{label}** | {type_label} "
            f"| {_fmt(a1.get('ap'))} | {_fmt(a2.get('ap'))} "
            f"| {_fmt(a1.get('accuracy'))} | {_fmt(a2.get('accuracy'))} "
            f"| {_fmt(a1.get('auc'))} | {_fmt(a2.get('auc'))} "
            f"| - |"
        )

    lines.append(_agg_row("Overall Avg", "overall_avg"))
    lines.append(_agg_row("GAN Avg", "gan_avg"))
    lines.append(_agg_row("Diffusion Avg", "diffusion_avg"))

    # Std-dev row (single value, not nested dict)
    sd1 = rgb_agg.get("std_dev_ap")
    sd2 = dct_agg.get("std_dev_ap")
    lines.append(
        f"| **Std Dev (AP)** | (across gens) | {_fmt(sd1)} | {_fmt(sd2)} | - | - | - | - | - |"
    )

    lines.append("")
    n1 = rgb_agg.get("n_generators_evaluated", 0)
    n2 = dct_agg.get("n_generators_evaluated", 0)
    lines.append(
        f"_Approach 1 (RGB+Fourier) evaluated {n1}/8 generators; "
        f"Approach 2 (DCT) evaluated {n2}/8._"
    )
    lines.append("")
    if "ckpt" in rgb:
        lines.append(f"_Approach 1 ckpt: `{rgb['ckpt']}`_")
    if "ckpt" in dct:
        lines.append(f"_Approach 2 ckpt: `{dct['ckpt']}`_")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

"""Render the doctor's per-generator eval table to Markdown.

Doctor's spec F.25 (MASTER_CHECKLIST):
    Rows: 8 generators (midjourney, sdv1_4, sdv1_5, wukong, vqdm, biggan, adm, glide)
          + 4 summary rows (Overall Avg, GAN Avg, Diffusion Avg, Std Dev).
    Cols: Generator | Type | AP | Accuracy | AUC.
"""

from __future__ import annotations

from pathlib import Path

from forensic.evaluation.per_generator import (
    ALL_GENERATORS,
    GENERATORS_DIFFUSION,
    GENERATORS_GAN,
)


def _fmt(v: float) -> str:
    """Format a metric for the table; show '-' for NaN."""
    try:
        if v != v:  # NaN
            return "-"
        return f"{v:.4f}"
    except (TypeError, ValueError):
        return "-"


def _gen_type(g: str) -> str:
    if g in GENERATORS_GAN:
        return "GAN"
    if g in GENERATORS_DIFFUSION:
        return "Diffusion"
    return "?"


def render_eval_table(
    per_gen: dict[str, dict],
    aggregates: dict,
    title: str = "Forensic Image Detector - Per-Generator Eval",
    notes: str | None = None,
) -> str:
    """Render the doctor-spec table to a Markdown string."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    if notes:
        lines.append(notes)
        lines.append("")

    lines.append("| Generator | Type | AP | Accuracy | AUC | n_samples |")
    lines.append("|-----------|------|----|----------|-----|-----------|")
    for g in ALL_GENERATORS:
        row = per_gen.get(g, {})
        if row.get("skipped"):
            lines.append(f"| {g} | {_gen_type(g)} | _skipped_ | _skipped_ | _skipped_ | 0 |")
            continue
        lines.append(
            f"| {g} | {_gen_type(g)} "
            f"| {_fmt(row.get('ap', float('nan')))} "
            f"| {_fmt(row.get('accuracy', float('nan')))} "
            f"| {_fmt(row.get('auc', float('nan')))} "
            f"| {row.get('n_samples', 0)} |"
        )

    # Aggregate rows
    oa = aggregates["overall_avg"]
    ga = aggregates["gan_avg"]
    da = aggregates["diffusion_avg"]
    sd = aggregates["std_dev_ap"]

    lines.append(
        f"| **Overall Avg** | (all) "
        f"| {_fmt(oa['ap'])} | {_fmt(oa['accuracy'])} | {_fmt(oa['auc'])} | - |"
    )
    lines.append(
        f"| **GAN Avg** | GAN | {_fmt(ga['ap'])} | {_fmt(ga['accuracy'])} | {_fmt(ga['auc'])} | - |"
    )
    lines.append(
        f"| **Diffusion Avg** | Diffusion "
        f"| {_fmt(da['ap'])} | {_fmt(da['accuracy'])} | {_fmt(da['auc'])} | - |"
    )
    lines.append(f"| **Std Dev (AP)** | (across gens) | {_fmt(sd)} | - | - | - |")
    lines.append("")
    lines.append(f"Generators evaluated: {aggregates['n_generators_evaluated']} / 8")
    return "\n".join(lines) + "\n"


def write_eval_table(
    per_gen: dict[str, dict],
    aggregates: dict,
    out_path: str | Path,
    title: str = "Forensic Image Detector - Per-Generator Eval",
    notes: str | None = None,
) -> Path:
    """Render + write to disk; return the output Path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = render_eval_table(per_gen, aggregates, title=title, notes=notes)
    out.write_text(md, encoding="utf-8")
    return out


__all__ = ["render_eval_table", "write_eval_table"]

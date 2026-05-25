"""Plot training history from training_history.csv as a 2-panel PNG.

Inputs:
  - phases/forensic/outputs/dct/training_history.csv

Outputs:
  - phases/forensic/outputs/training_curves.png

Panels:
  - Top:    train_loss + val_ap vs epoch (twin y-axis), Phase 1/2 boundary shaded.
  - Bottom: learning rate vs epoch (log scale).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("phases/forensic/outputs/dct/training_history.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("phases/forensic/outputs/training_curves.png"),
    )
    p.add_argument(
        "--title",
        default="Approach 2 (DCT) training: train_loss + val_ap vs epoch",
        help="Plot title (default fits Approach 2; override for Approach 1).",
    )
    args = p.parse_args()

    if not args.csv.exists():
        print(f"FATAL: {args.csv} not found", file=sys.stderr)
        sys.exit(2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("FATAL: matplotlib not installed. pip install matplotlib", file=sys.stderr)
        sys.exit(2)

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    val_ap = [float(r["val_ap"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]
    is_best = [int(r["is_best"]) for r in rows]

    # Find phase 1 -> phase 2 boundary (only if CSV has 'phase' column;
    # Approach 1 training has no phase, so boundary stays None).
    boundary = None
    if rows and "phase" in rows[0]:
        phases = [int(r["phase"]) for r in rows]
        for i, ph in enumerate(phases):
            if ph == 2 and (i == 0 or phases[i - 1] == 1):
                boundary = epochs[i]
                break

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # ----- top panel: loss + val_ap -----
    color_loss = "tab:red"
    color_ap = "tab:blue"
    ax_loss = ax_top
    ax_loss.set_ylabel("train_loss", color=color_loss)
    (line_loss,) = ax_loss.plot(
        epochs,
        loss,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color=color_loss,
        label="train_loss",
    )
    ax_loss.tick_params(axis="y", labelcolor=color_loss)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, which="both", alpha=0.3)

    ax_ap = ax_loss.twinx()
    ax_ap.set_ylabel("val_ap", color=color_ap)
    (line_ap,) = ax_ap.plot(
        epochs,
        val_ap,
        marker="s",
        markersize=4,
        linewidth=1.5,
        color=color_ap,
        label="val_ap",
    )
    ax_ap.tick_params(axis="y", labelcolor=color_ap)

    # Mark best epochs with a star
    for ep, ap, bs in zip(epochs, val_ap, is_best, strict=False):
        if bs:
            ax_ap.plot(
                ep,
                ap,
                marker="*",
                markersize=12,
                color="gold",
                markeredgecolor="black",
                zorder=5,
            )

    # Shade Phase 2 region
    if boundary is not None:
        ax_loss.axvspan(
            boundary - 0.5,
            max(epochs) + 0.5,
            alpha=0.08,
            color="green",
            label=f"Phase 2 (epoch {boundary}+)",
        )
        ax_loss.axvline(boundary - 0.5, color="green", linestyle="--", alpha=0.5)
        ax_loss.text(
            boundary - 0.3,
            ax_loss.get_ylim()[1] * 0.7,
            "Phase 2 begins",
            rotation=90,
            va="top",
            color="green",
            alpha=0.7,
            fontsize=8,
        )

    ax_loss.set_title(args.title)
    ax_loss.legend(handles=[line_loss, line_ap], loc="center right", fontsize=9)

    # ----- bottom panel: lr -----
    ax_bot.plot(epochs, lrs, marker="^", markersize=4, linewidth=1.5, color="tab:purple")
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("epoch")
    ax_bot.set_ylabel("learning rate")
    ax_bot.grid(True, which="both", alpha=0.3)
    if boundary is not None:
        ax_bot.axvline(boundary - 0.5, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

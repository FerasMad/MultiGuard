#!/usr/bin/env python3
"""Generate the filled 'Experimental Setup' report for MultiGuard.

Produces, under reports/:
  - Experimental_Setup_MultiGuard.md
  - Experimental_Setup_MultiGuard.docx
  - confusion_matrix_5class_ensemble.png

All numbers are taken verbatim from the repository (sources cited inline):
  docs/MODEL_REPORT.md, phases/v4/docs/eval/*, docs/IMAGE_BRANCH_ABLATION.md,
  phases/v4/configs/*.yaml, pyproject.toml, docs/MULTIGUARD_SETUP.md, STATUS.md.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["Real", "Out-of-Context", "Manipulated", "AI-Text", "Fully-Fabricated"]

# ----------------------------------------------------------------------------
# Confusion matrix (Result R2) — deployed 3-seed ensemble, from docs/MODEL_REPORT.md
# rows = true, cols = predicted
# ----------------------------------------------------------------------------
CM = np.array([
    [183, 227,  85,   0,   0],
    [196, 217,  82,   0,   0],
    [ 49,  42, 403,   1,   0],
    [  0,   0,   0, 492,   3],
    [  0,   0,   0,   7, 488],
])
CM_PNG = os.path.join(OUT, "confusion_matrix_5class_ensemble.png")


def make_confusion_png() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(CM, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("R2 — 5-Class Confusion Matrix\n(deployed 3-seed ensemble, test n=2,475)",
                 fontsize=12)
    thr = CM.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(CM[i, j]), ha="center", va="center",
                    color="white" if CM[i, j] > thr else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Data tables shared by both MD and DOCX
# ----------------------------------------------------------------------------
HARDWARE = [
    ("GPU", "NVIDIA GeForce RTX 4070 (12 GB VRAM)", "user-provided"),
    ("CPU", "Intel Core i5-13500 (14 cores / 20 threads)", "user-provided"),
    ("RAM (system)", "32 GB", "user-provided"),
    ("Storage", "~155 GB for the full datasets (VisualNews 99 GB, DGM4 21 GB, "
                "NewsCLIPpings 21 GB, MMFakeBench 14 GB)", "docs/MULTIGUARD_SETUP.md"),
    ("CUDA", "12.4+", "docs/V4_PLAN.md, docs/MULTIGUARD_SETUP.md"),
]

SOFTWARE = [
    ("Operating System", "Windows (development machine)", "docs/V4_PLAN.md"),
    ("Programming Language", "Python 3.12", "pyproject.toml"),
    ("DL Framework", "PyTorch ≥ 2.6 (CUDA 12.4 build) · torchvision ≥ 0.21", "pyproject.toml"),
    ("LLM / Transformers", "Hugging Face Transformers ≥ 4.45, < 5.0 · accelerate ≥ 0.34 · "
                           "safetensors ≥ 0.4", "pyproject.toml"),
    ("Image pre-processing", "OpenCV ≥ 4.9 · Pillow ≥ 10.2", "pyproject.toml"),
    ("Metrics / ML utils", "scikit-learn ≥ 1.4", "pyproject.toml"),
    ("Numerics / data", "NumPy ≥ 1.26 (< 2.0) · pandas ≥ 2.2", "pyproject.toml"),
]

# Model hyperparameters (one row per model/stage)
HP_COLS = ["Component (stage)", "Backbone", "Frozen?", "Embed dim",
           "LR", "Optimizer", "Batch", "Epochs", "Dropout", "Scheduler", "Precision"]
HP = [
    ["Semantic — FND-CLIP (Stage 0)", "ResNet50 + BERT-base + CLIP", "Yes (CLIP frozen)",
     "512 → 768", "1e-4", "AdamW", "32", "20", "—", "StepLR ×0.1 @10", "bf16"],
    ["Image forensic — UnivFD (Stage 1)", "ResNet50 (1-ch conv1)", "No (fine-tuned)",
     "→ 768", "1e-4", "AdamW", "64", "30", "0.3 (head)", "StepLR ×0.1 @20", "bf16"],
    ["Image forensic — DCT ResNet50 (detector)", "ResNet50 (1-ch conv1)",
     "Partial (2-phase)", "→ 768", "1e-4 → 1e-5", "AdamW", "64", "30",
     "—", "ReduceLROnPlateau", "fp32"],
    ["Image forensic — RGB+Fourier (detector)", "ResNet50", "Partial (L1–L2 frozen)",
     "→ 768", "1e-4", "AdamW", "64", "30", "—", "ReduceLROnPlateau", "fp32"],
    ["Text forensic — Qwen2-7B-Instruct", "Qwen2-7B (last layer, masked-mean)",
     "Yes (frozen)", "3584 → 768", "—", "—", "—", "—", "—", "—", "—"],
    ["Fusion — V3PairwiseFusion (Stage 2)", "3-pair cross-attn (8 heads) + Conv1d",
     "trained", "768 → 1024", "1e-4", "AdamW", "64", "50",
     "0.5 (MLP) / 0.1 (attn)", "StepLR ×0.1 @30", "bf16"],
]
HP_NOTES = (
    "Common: weight decay = 1e-4, gradient clipping max-norm = 1.0, early stopping "
    "(patience 5–10 on val AUC/F1). Loss (Stage 2): CrossEntropy (5-class) + 0.1 × "
    "BCEWithLogits auxiliary head on detached v_imgfor. "
    "Source: phases/v4/configs/*.yaml, phases/v4/src/v4/training/{trainer,losses,schedulers}.py."
)

# Result R1 — classification report (deployed 3-seed ensemble)
R1_COLS = ["Class", "Precision", "Recall", "F1", "Support"]
R1 = [
    ["0 Real", "0.428", "0.370", "0.397", "495"],
    ["1 Out-of-Context", "0.447", "0.438", "0.442", "495"],
    ["2 Manipulated", "0.707", "0.814", "0.757", "495"],
    ["3 AI-Text", "0.984", "0.994", "0.989", "495"],
    ["4 Fully-Fabricated", "0.994", "0.986", "0.990", "495"],
    ["F1-macro (overall)", "—", "—", "0.7149", "2,475"],
]

# Result R3 — training-time branch ablation (docs/IMAGE_BRANCH_ABLATION.md)
R3_COLS = ["Variant", "Test F1", "C0 Real", "C1 OOC", "C2 Manip", "C3 AI-Text", "C4 FullFab"]
R3 = [
    ["fndclip (semantic only)", "0.6681", "0.451", "0.306", "0.738", "0.886", "0.960"],
    ["fndclip + text", "0.7103", "0.411", "0.448", "0.731", "0.981", "0.982"],
    ["fndclip + image", "0.6975", "0.391", "0.464", "0.751", "0.911", "0.970"],
    ["all three (full)", "0.7157", "0.430", "0.426", "0.752", "0.985", "0.985"],
]

# Result R4 — inference-time leave-one-out ablation (docs/IMAGE_BRANCH_ABLATION.md)
R4_COLS = ["Variant", "Test F1", "ΔF1", "C2 Manip", "Note"]
R4 = [
    ["full pipeline", "0.7149", "—", "0.757", "baseline"],
    ["disable image (v_imgfor)", "0.6954", "-2.0 pp", "0.733", "image removal hits C2 most"],
    ["disable text (v_textfor)", "0.4494", "-26.6 pp", "0.710", "text dominates C3/C4"],
    ["semantic only", "0.3146", "-40.0 pp", "0.655", "—"],
]

# Result R5 — per-seed (mean ± std)
R5_COLS = ["Seed", "Test F1-macro", "MMFakeBench transfer F1 (raw)"]
R5 = [
    ["42", "0.7152", "0.3516"],
    ["1337", "0.7189", "0.3446"],
    ["2024", "0.7009", "0.4308"],
    ["mean ± std", "0.7117 ± 0.0095", "0.3757 ± 0.0479"],
    ["3-seed ensemble", "0.7149", "0.4308 (raw) / 0.7197 (bias-corrected)"],
]

# Result R6 — forensic image detectors (8/8 generators, official nature)
R6_COLS = ["Generator", "Type", "A1 RGB+Fourier AP", "A2 DCT AP"]
R6 = [
    ["midjourney", "Diffusion", "0.965", "0.828"],
    ["sdv1_4", "Diffusion", "0.988", "0.890"],
    ["sdv1_5", "Diffusion", "0.985", "0.896"],
    ["wukong", "Diffusion", "0.980", "0.902"],
    ["vqdm", "Diffusion", "0.976", "0.830"],
    ["adm", "Diffusion", "0.997", "0.971"],
    ["glide", "Diffusion", "0.987", "0.975"],
    ["biggan", "GAN", "0.995", "0.976"],
    ["Overall AP", "", "0.9841", "0.9085"],
    ["Overall Accuracy", "", "0.940", "0.834"],
    ["Std-dev AP", "", "0.011", "0.061"],
]

# Result R2 — confusion matrix as a table (rows true, cols pred)
R2_COLS = ["true \\ pred", "Real", "OOC", "Manip", "AI-Text", "Fab"]
R2 = [[CLASSES[i]] + [str(x) for x in CM[i]] for i in range(5)]


# ----------------------------------------------------------------------------
# Markdown writer
# ----------------------------------------------------------------------------
def md_table(cols, rows):
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_md():
    p = os.path.join(OUT, "Experimental_Setup_MultiGuard.md")
    s = []
    s.append("# Experimental Setup — MultiGuard\n")
    s.append("> 5-class multimodal fake-news detector (V3.1 spec) + two standalone "
             "binary AI-image forensic detectors. All values below are taken from the "
             "project repository (sources cited per table). Items marked **[fill in]** "
             "are machine-specific and not recorded in the repo.\n")

    s.append("## I. Experimental Setup\n")
    s.append("### A. Hardware Environment\n")
    s.append(md_table(["Item", "Value", "Source"], HARDWARE) + "\n")

    s.append("### B. Software Stack\n")
    s.append(md_table(["Item", "Value", "Source"], SOFTWARE) + "\n")

    s.append("### C. Model Hyperparameters\n")
    s.append(md_table(HP_COLS, HP) + "\n")
    s.append("_" + HP_NOTES + "_\n")

    s.append("## II. Evaluation Results & Ablation Studies\n")
    s.append("_Final 5-class model = deployed 3-seed honest ensemble "
             "(seeds 42 / 1337 / 2024, softmax-averaged). Test split = 2,475 held-out "
             "samples, 495 per class._\n")

    s.append("### Result R1 — 5-Class Classification Report (final model)\n")
    s.append("_Source: docs/MODEL_REPORT.md §3._\n")
    s.append(md_table(R1_COLS, R1) + "\n")

    s.append("### Result R2 — 5-Class Confusion Matrix (final model)\n")
    s.append("_Source: docs/MODEL_REPORT.md §3. Rows = true, columns = predicted._\n")
    s.append(md_table(R2_COLS, R2) + "\n")
    s.append("![Confusion matrix](confusion_matrix_5class_ensemble.png)\n")

    s.append("### Result R3 — Ablation A: Training-time Branch Contribution\n")
    s.append("_Each variant trained from scratch with only the listed branches. "
             "Source: docs/IMAGE_BRANCH_ABLATION.md._\n")
    s.append(md_table(R3_COLS, R3) + "\n")

    s.append("### Result R4 — Ablation B: Inference-time Leave-One-Out\n")
    s.append("_One branch zeroed at inference on the shipped ensemble. "
             "Source: docs/IMAGE_BRANCH_ABLATION.md._\n")
    s.append(md_table(R4_COLS, R4) + "\n")

    s.append("### Result R5 — Per-Seed Results & External Transfer\n")
    s.append("_Source: phases/v4/docs/eval/honest_run_summary.json._\n")
    s.append(md_table(R5_COLS, R5) + "\n")

    s.append("### Result R6 — Forensic Image Detectors (8/8 generators, official nature)\n")
    s.append("_Two standalone binary real-vs-AI detectors. Metric = Average Precision (AP). "
             "Source: docs/MODEL_REPORT.md §4, outputs/eval_table_combined.md._\n")
    s.append(md_table(R6_COLS, R6) + "\n")

    s.append("### Notes / Honest Limitations\n")
    s.append("- Classes 3 (AI-Text) and 4 (Fully-Fabricated) are near-perfect "
             "(F1 ≈ 0.99); the Qwen2 text branch isolates AI-written text cleanly.\n"
             "- The error mass is in Real ↔ Out-of-Context (linear-probe AUC ≈ 0.51, "
             "at-chance) — a frozen-FND-CLIP / data-scale ceiling, not a fusion bug.\n"
             "- A1 (RGB+Fourier) is the recommended forensic detector; A2 (DCT) is the "
             "weaker frequency-domain baseline (notably on MidJourney/VQDM).\n")

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    return p


# ----------------------------------------------------------------------------
# DOCX writer
# ----------------------------------------------------------------------------
def write_docx():
    from docx import Document
    from docx.shared import Inches, Pt

    doc = Document()
    doc.add_heading("Experimental Setup — MultiGuard", level=0)
    doc.add_paragraph(
        "5-class multimodal fake-news detector (V3.1 spec) plus two standalone binary "
        "AI-image forensic detectors. All values are taken from the project repository "
        "(sources cited per table). Items marked [fill in] are machine-specific and not "
        "recorded in the repo."
    )

    def add_table(cols, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = "Light Grid Accent 1"
        hdr = t.rows[0].cells
        for i, c in enumerate(cols):
            hdr[i].text = str(c)
            for para in hdr[i].paragraphs:
                for run in para.runs:
                    run.bold = True
        for r in rows:
            cells = t.add_row().cells
            for i, c in enumerate(r):
                cells[i].text = str(c)
        # shrink font a bit for wide tables
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(8 if len(cols) > 6 else 9)
        doc.add_paragraph()

    doc.add_heading("I. Experimental Setup", level=1)
    doc.add_heading("A. Hardware Environment", level=2)
    add_table(["Item", "Value", "Source"], HARDWARE)
    doc.add_heading("B. Software Stack", level=2)
    add_table(["Item", "Value", "Source"], SOFTWARE)
    doc.add_heading("C. Model Hyperparameters", level=2)
    add_table(HP_COLS, HP)
    note = doc.add_paragraph()
    run = note.add_run(HP_NOTES)
    run.italic = True
    run.font.size = Pt(9)

    doc.add_heading("II. Evaluation Results & Ablation Studies", level=1)
    p = doc.add_paragraph()
    r = p.add_run("Final 5-class model = deployed 3-seed honest ensemble "
                  "(seeds 42 / 1337 / 2024, softmax-averaged). Test split = 2,475 "
                  "held-out samples, 495 per class.")
    r.italic = True

    doc.add_heading("Result R1 — 5-Class Classification Report (final model)", level=2)
    doc.add_paragraph("Source: docs/MODEL_REPORT.md §3.")
    add_table(R1_COLS, R1)

    doc.add_heading("Result R2 — 5-Class Confusion Matrix (final model)", level=2)
    doc.add_paragraph("Source: docs/MODEL_REPORT.md §3. Rows = true, columns = predicted.")
    add_table(R2_COLS, R2)
    if os.path.exists(CM_PNG):
        doc.add_picture(CM_PNG, width=Inches(5.0))

    doc.add_heading("Result R3 — Ablation A: Training-time Branch Contribution", level=2)
    doc.add_paragraph("Each variant trained from scratch with only the listed branches. "
                      "Source: docs/IMAGE_BRANCH_ABLATION.md.")
    add_table(R3_COLS, R3)

    doc.add_heading("Result R4 — Ablation B: Inference-time Leave-One-Out", level=2)
    doc.add_paragraph("One branch zeroed at inference on the shipped ensemble. "
                      "Source: docs/IMAGE_BRANCH_ABLATION.md.")
    add_table(R4_COLS, R4)

    doc.add_heading("Result R5 — Per-Seed Results & External Transfer", level=2)
    doc.add_paragraph("Source: phases/v4/docs/eval/honest_run_summary.json.")
    add_table(R5_COLS, R5)

    doc.add_heading("Result R6 — Forensic Image Detectors (8/8 generators)", level=2)
    doc.add_paragraph("Two standalone binary real-vs-AI detectors. Metric = Average "
                      "Precision (AP). Source: docs/MODEL_REPORT.md §4, "
                      "outputs/eval_table_combined.md.")
    add_table(R6_COLS, R6)

    doc.add_heading("Notes / Honest Limitations", level=2)
    for line in [
        "Classes 3 (AI-Text) and 4 (Fully-Fabricated) are near-perfect (F1 ≈ 0.99); "
        "the Qwen2 text branch isolates AI-written text cleanly.",
        "The error mass is in Real ↔ Out-of-Context (linear-probe AUC ≈ 0.51, "
        "at-chance) — a frozen-FND-CLIP / data-scale ceiling, not a fusion bug.",
        "A1 (RGB+Fourier) is the recommended forensic detector; A2 (DCT) is the weaker "
        "frequency-domain baseline (notably on MidJourney/VQDM).",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    p = os.path.join(OUT, "Experimental_Setup_MultiGuard.docx")
    doc.save(p)
    return p


if __name__ == "__main__":
    make_confusion_png()
    md_path = write_md()
    docx_path = write_docx()
    print("WROTE:", CM_PNG)
    print("WROTE:", md_path)
    print("WROTE:", docx_path)

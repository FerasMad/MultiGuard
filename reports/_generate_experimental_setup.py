#!/usr/bin/env python3
"""Fill the 'Experimental Setup' template (exact structure) for MultiGuard.

Produces under reports/:
  - Experimental_Setup_MultiGuard.md
  - Experimental_Setup_MultiGuard.docx
  - confusion_matrix_5class_ensemble.png

Mirrors the template 1:1 (I. Setup: Hardware / Software / Hyperparameters;
II. Results: a. Ablation studies, b. Classification report, Confusion matrix).
All numbers come from the MultiGuard repo; hardware is user-provided.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["Real", "Out-of-Context", "Manipulated", "AI-Text", "Fully-Fabricated"]

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
    ax.set_title("Confusion Matrix (test n=2,475)", fontsize=12)
    thr = CM.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(CM[i, j]), ha="center", va="center",
                    color="white" if CM[i, j] > thr else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)


# --- I.A Hardware (user-provided) ---
HARDWARE = [
    ("GPU", "NVIDIA GeForce RTX 4070 (12 GB VRAM)"),
    ("CPU", "Intel Core i5-13500 (14 cores / 20 threads)"),
    ("RAM", "32 GB"),
    ("Storage", "~155 GB (full datasets)"),
]

# --- I.B Software ---
SOFTWARE = [
    ("Operating System", "Windows (CUDA 12.4+)"),
    ("Programming Language", "Python 3.12"),
    ("Core Libraries", "PyTorch + torchvision, Hugging Face Transformers, "
                       "OpenCV / Pillow, scikit-learn, NumPy, pandas"),
]

# --- I.C Hyperparameters (one block per model) ---
HYPER = [
    ("Semantic — FND-CLIP", [
        ("Backbone", "ResNet50 + BERT + CLIP (Frozen)"),
        ("Embedding Dimension", "512 -> 768"),
        ("Learning Rate", "1e-4 (AdamW)"),
        ("Batch Size", "32"),
        ("Dropout Rate", "-"),
    ]),
    ("Image Forensic — DCT ResNet50", [
        ("Backbone", "ResNet50, 1-channel conv (fine-tuned)"),
        ("Embedding Dimension", "768"),
        ("Learning Rate", "1e-4 -> 1e-5 (AdamW)"),
        ("Batch Size", "64"),
        ("Dropout Rate", "0.3"),
    ]),
    ("Text Forensic — Qwen2-7B-Instruct", [
        ("Backbone", "Qwen2-7B (Frozen) + Linear 3584->768"),
        ("Embedding Dimension", "3584 -> 768"),
        ("Learning Rate", "1e-4 (AdamW, projection)"),
        ("Batch Size", "64"),
        ("Dropout Rate", "-"),
    ]),
    ("Fusion — V3PairwiseFusion + MLP", [
        ("Backbone", "Pairwise cross-attention (8 heads) + Conv1d + MLP"),
        ("Embedding Dimension", "768 -> 1024 -> 5"),
        ("Learning Rate", "1e-4 (AdamW)"),
        ("Batch Size", "64"),
        ("Dropout Rate", "0.5 (MLP) / 0.1 (attention)"),
    ]),
]

# --- II.a Ablation studies ---
ABLATION_COLS = ["Variant", "Accuracy", "F1-macro"]
ABLATION = [
    ["Semantic only", "-", "0.6681"],
    ["Semantic + Text", "-", "0.7103"],
    ["Semantic + Image", "-", "0.6975"],
    ["All three (full)", "0.7305", "0.7149"],
]

# --- II.b Classification report (5-class ensemble) ---
REPORT_COLS = ["Class", "Precision", "Recall", "F1", "Support"]
REPORT = [
    ["Real", "0.428", "0.370", "0.397", "495"],
    ["Out-of-Context", "0.447", "0.438", "0.442", "495"],
    ["Manipulated", "0.707", "0.814", "0.757", "495"],
    ["AI-Text", "0.984", "0.994", "0.989", "495"],
    ["Fully-Fabricated", "0.994", "0.986", "0.990", "495"],
    ["accuracy", "", "", "0.7305", "2475"],
    ["macro avg", "0.726", "0.731", "0.7149", "2475"],
]

CM_COLS = ["true \\ pred", "Real", "OOC", "Manip", "AI-Text", "Fab"]
CM_ROWS = [[CLASSES[i]] + [str(x) for x in CM[i]] for i in range(5)]


def md_table(cols, rows):
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_md():
    p = os.path.join(OUT, "Experimental_Setup_MultiGuard.md")
    s = ["# Experimental Setup\n"]

    s.append("## I. Experimental Setup\n")
    s.append("### A. Hardware Environment\n")
    for k, v in HARDWARE:
        s.append(f"- **{k}:** {v}")
    s.append("")
    s.append("### B. Software Stack\n")
    for k, v in SOFTWARE:
        s.append(f"- **{k}:** {v}")
    s.append("")
    s.append("### C. Model Hyperparameters\n")
    for model, params in HYPER:
        s.append(f"**{model}**")
        for k, v in params:
            s.append(f"- {k}: {v}")
        s.append("")

    s.append("## II. Evaluation Results & Ablation Studies\n")
    s.append("### a. Ablation studies\n")
    s.append(md_table(ABLATION_COLS, ABLATION) + "\n")
    s.append("### b. Classification report\n")
    s.append(md_table(REPORT_COLS, REPORT) + "\n")
    s.append("### Confusion matrix\n")
    s.append(md_table(CM_COLS, CM_ROWS) + "\n")
    s.append("![Confusion matrix](confusion_matrix_5class_ensemble.png)\n")

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    return p


def write_docx():
    from docx import Document
    from docx.shared import Inches, Pt

    doc = Document()
    doc.add_heading("Experimental Setup", level=0)

    def add_table(cols, rows):
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = "Light Grid Accent 1"
        for i, c in enumerate(cols):
            cell = t.rows[0].cells[i]
            cell.text = str(c)
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True
        for r in rows:
            cells = t.add_row().cells
            for i, c in enumerate(r):
                cells[i].text = str(c)
        doc.add_paragraph()

    doc.add_heading("I. Experimental Setup", level=1)
    doc.add_heading("A. Hardware Environment", level=2)
    for k, v in HARDWARE:
        para = doc.add_paragraph(style="List Bullet")
        para.add_run(f"{k}: ").bold = True
        para.add_run(v)

    doc.add_heading("B. Software Stack", level=2)
    for k, v in SOFTWARE:
        para = doc.add_paragraph(style="List Bullet")
        para.add_run(f"{k}: ").bold = True
        para.add_run(v)

    doc.add_heading("C. Model Hyperparameters", level=2)
    for model, params in HYPER:
        doc.add_paragraph().add_run(model).bold = True
        for k, v in params:
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"{k}: ").bold = True
            para.add_run(v)

    doc.add_heading("II. Evaluation Results & Ablation Studies", level=1)
    doc.add_heading("a. Ablation studies", level=2)
    add_table(ABLATION_COLS, ABLATION)
    doc.add_heading("b. Classification report", level=2)
    add_table(REPORT_COLS, REPORT)
    doc.add_heading("Confusion matrix", level=2)
    add_table(CM_COLS, CM_ROWS)
    if os.path.exists(CM_PNG):
        doc.add_picture(CM_PNG, width=Inches(5.0))

    p = os.path.join(OUT, "Experimental_Setup_MultiGuard.docx")
    doc.save(p)
    return p


if __name__ == "__main__":
    make_confusion_png()
    print("WROTE:", write_md())
    print("WROTE:", write_docx())
    print("WROTE:", CM_PNG)

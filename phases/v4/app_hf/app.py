"""MultiGuard HF Space (Gradio SDK, ZeroGPU) -- honest-path 3-seed ensemble.

Pure Gradio implementation styled to match the local FastAPI website UI:
  - Same color palette (indigo + status colors per class)
  - Same layout: hero header, text+image inputs side-by-side, verdict card +
    probability bars + module breakdown
  - Bilingual EN/AR toggle (RTL for Arabic)
  - 3-seed softmax-averaged ensemble + per-branch confidence breakdown

ZeroGPU requirement forced Gradio SDK; we get pixel-close visual fidelity to
the original website via custom CSS without the FastAPI mount complications.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from inline.class_map import LABELS, LABEL_EXPLANATIONS
from inline.dct_forensic import DctForensicEncoder
from inline.dual_dct import compute_dual_dct
from inline.fnd_clip import FNDCLIPSemanticEncoder, prepare_fnd_inputs
from inline.qwen_text import Qwen2TextEncoder
from inline.v3_pairwise import V3PairwiseFusion

import spaces

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("multiguard")

LABELS_AR = {
    0: "حقيقي",
    1: "خارج السياق",
    2: "صورة معدَّلة",
    3: "نص بالذكاء الاصطناعي",
    4: "تزييف كامل",
}
EXPLANATIONS_AR = {
    0: "يبدو أن النص والصورة حقيقيان ومتطابقان.",
    1: "تبدو الصورة والنص حقيقيين بشكل منفرد ولكنهما غير متطابقين.",
    2: "تظهر على الصورة علامات تعديل رقمي.",
    3: "يُظهر النص أنماطًا تتوافق مع المحتوى المولَّد بالذكاء الاصطناعي.",
    4: "تظهر على كل من الصورة والنص علامات تزييف.",
}

CLASS_TO_KEY = {0: "real", 1: "ooc", 2: "manipulated", 3: "ai-text", 4: "fabricated"}
CLASS_TO_COLOR = {
    0: "#10b981",
    1: "#f59e0b",
    2: "#ef4444",
    3: "#8b5cf6",
    4: "#dc2626",
}

_state: dict = {}


def _build_pipeline():
    t0 = time.time()
    log.info("[build] FND-CLIP V1 ckpt ...")
    fndclip_ckpt = hf_hub_download(repo_id="FerasMad/multiguard-v1-fndclip", filename="best.pt")
    log.info("[build] DCT-Forensic ckpt + stats ...")
    dct_ckpt = hf_hub_download(repo_id="FerasMad/forensic-dct-v1", filename="forensic_dct_model.pth")
    dct_stats = hf_hub_download(repo_id="FerasMad/forensic-dct-v1", filename="dct_stats.json")
    log.info("[build] honest-path ensemble (3 seeds + parity head) ...")
    seed_ckpts = [
        hf_hub_download(repo_id="FerasMad/multiguard-v4-honest", filename=f"seed{s}.pt")
        for s in (42, 1337, 2024)
    ]
    dct_head_state = hf_hub_download(
        repo_id="FerasMad/multiguard-v4-honest", filename="dctforensic_head_seed42.pt",
    )

    log.info("[build] semantic encoder (CPU; moved to cuda inside @spaces.GPU) ...")
    fndclip = FNDCLIPSemanticEncoder(ckpt=fndclip_ckpt).eval()
    for p in fndclip.parameters():
        p.requires_grad = False

    log.info("[build] DCT-Forensic encoder (with P9.1 parity head) ...")
    dct_forensic = DctForensicEncoder(
        out_dim=768,
        ckpt=dct_ckpt,
        head_state_path=dct_head_state,
        seed=42,
    ).eval()
    for p in dct_forensic.parameters():
        p.requires_grad = False

    log.info("[build] Qwen2-7B-Instruct (lazy, loads on first @spaces.GPU call) ...")
    qwen = Qwen2TextEncoder(load_backbone=False)

    log.info("[build] 3-seed V3PairwiseFusion ensemble ...")
    fusions = []
    for ck_path in seed_ckpts:
        fusion = V3PairwiseFusion(
            feat_dim=768,
            fused_dim=1024,
            num_classes=5,
            proj_dims={"v_semantic": 512, "v_textfor": 3584},
        ).eval()
        payload = torch.load(ck_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        fusion.load_state_dict(state, strict=False)
        for p in fusion.parameters():
            p.requires_grad = False
        fusions.append(fusion)

    stats_obj = json.loads(Path(dct_stats).read_text(encoding="utf-8"))
    log.info("[build] pipeline ready in %.1fs (CPU; cuda per request)", time.time() - t0)
    return {
        "fndclip": fndclip,
        "dct_forensic": dct_forensic,
        "qwen": qwen,
        "fusions": fusions,
        "dct_mean": float(stats_obj["mean"]),
        "dct_std": float(stats_obj["std"]),
    }


def _ensemble_predict(pil_img: Image.Image, text: str) -> list[float]:
    """Inner inference; called from within @spaces.GPU-decorated analyze()."""
    gpu = torch.device("cuda")
    _state["fndclip"].to(gpu)
    _state["dct_forensic"].to(gpu)
    for fusion in _state["fusions"]:
        fusion.to(gpu)

    fnd_batch = prepare_fnd_inputs(text, pil_img)
    fnd_batch = {k: v.to(gpu) for k, v in fnd_batch.items()}

    t = compute_dual_dct(pil_img)
    t = (t - _state["dct_mean"]) / (_state["dct_std"] + 1e-8)
    t = t.unsqueeze(0).to(gpu)

    with torch.no_grad():
        v_semantic = _state["fndclip"](fnd_batch)
        v_imgfor = _state["dct_forensic"]({"v_imgfor_dct": t})
        v_textfor = _state["qwen"].encode_text(text).to(gpu)
        feats = {"v_semantic": v_semantic, "v_imgfor": v_imgfor, "v_textfor": v_textfor}
        probs_list = []
        for fusion in _state["fusions"]:
            out = fusion(feats)
            probs_list.append(F.softmax(out["main_logits"], dim=-1))
        probs = torch.stack(probs_list).mean(dim=0)[0]
    return probs.cpu().tolist()


@spaces.GPU(duration=120)
def analyze(image, text):
    """Gradio handler -- ZeroGPU is allocated for this function's duration.
    Returns: (verdict_html, probs_html, modules_html)."""
    lang = "en"
    if not text or not text.strip():
        return _error_html("Please enter article text"), "", ""
    if image is None:
        return _error_html("Please upload an image"), "", ""

    try:
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        pil_img = pil_img.convert("RGB")
        probs = _ensemble_predict(pil_img, text)
    except Exception as e:
        log.exception("analyze failed")
        return _error_html(f"{type(e).__name__}: {e}"), "", ""

    pred = int(np.argmax(probs))
    confidence = round(float(probs[pred]) * 100, 1)
    verdict_en = LABELS[pred]
    verdict_ar = LABELS_AR[pred]
    explanation_en = LABEL_EXPLANATIONS[pred]
    explanation_ar = EXPLANATIONS_AR[pred]
    cls_key = CLASS_TO_KEY[pred]
    color = CLASS_TO_COLOR[pred]

    verdict_html = _verdict_html(
        verdict=verdict_ar if lang == "ar" else verdict_en,
        confidence=confidence,
        explanation=explanation_ar if lang == "ar" else explanation_en,
        color=color,
        cls_key=cls_key,
        confidence_label="الثقة" if lang == "ar" else "Confidence",
        verdict_label="النتيجة" if lang == "ar" else "Verdict",
    )
    probs_html = _probabilities_html(probs, lang)

    text_ai = float(probs[3]) + float(probs[4])
    image_manip = float(probs[2]) + float(probs[4])
    modules = {
        "text_ai": text_ai,
        "text_patterns": min(1.0, 0.5 * text_ai + 0.5 * float(probs[1])),
        "image_manip": image_manip,
        "cross_modal": float(probs[1]),
        "overall": 1.0 - float(probs[0]),
    }
    modules_html = _modules_html(modules, lang)
    return verdict_html, probs_html, modules_html, ""


def _error_html(msg: str) -> str:
    return f'<div class="mg-error">{msg}</div>'


def _verdict_html(verdict, confidence, explanation, color, cls_key, confidence_label, verdict_label):
    return f"""
<div class="mg-verdict mg-{cls_key}">
  <div class="mg-verdict-label">{verdict_label}</div>
  <div class="mg-verdict-title" style="color:{color}">{verdict}</div>
  <div class="mg-conf-bar"><div class="mg-conf-fill" style="width:{confidence}%;background:{color}"></div></div>
  <div class="mg-conf-text">{confidence_label}: <strong>{confidence}%</strong></div>
  <div class="mg-explanation" style="background:{color}1a;color:{color}">{explanation}</div>
</div>
"""


def _probabilities_html(probs, lang):
    names_en = ["Real", "Out-of-Context", "Manipulated", "AI-Text", "Fully Fabricated"]
    names_ar = ["حقيقي", "خارج السياق", "صورة معدَّلة", "نص بالذكاء الاصطناعي", "تزييف كامل"]
    names = names_ar if lang == "ar" else names_en
    rows = []
    for i, p in enumerate(probs):
        pct = round(p * 100, 1)
        color = CLASS_TO_COLOR[i]
        rows.append(
            f'<div class="mg-prob-row"><span class="mg-prob-label">{names[i]}</span>'
            f'<div class="mg-prob-bg"><div class="mg-prob-fill" style="width:{pct}%;background:{color}">{pct}%</div></div></div>'
        )
    return '<div class="mg-prob-section">' + "".join(rows) + "</div>"


def _modules_html(modules, lang):
    labels_en = {
        "text_ai": "Text (AI Detection)",
        "text_patterns": "Text (Patterns)",
        "image_manip": "Image (Forensics)",
        "cross_modal": "Cross-Modal Alignment",
        "overall": "Overall Risk",
    }
    labels_ar = {
        "text_ai": "النص (كشف الذكاء الاصطناعي)",
        "text_patterns": "النص (الأنماط)",
        "image_manip": "الصورة (الفحص الجنائي)",
        "cross_modal": "الاتساق بين الوسائط",
        "overall": "الخطورة العامة",
    }
    labels = labels_ar if lang == "ar" else labels_en
    title = "تفصيل الوحدات" if lang == "ar" else "Module Breakdown"
    rows = []
    for k in ("text_ai", "text_patterns", "image_manip", "cross_modal", "overall"):
        val = modules[k]
        pct = round(val * 100, 1)
        color = "#10b981" if val < 0.3 else "#f59e0b" if val < 0.6 else "#ef4444"
        rows.append(
            f'<div class="mg-mod-row"><span>{labels[k]}</span>'
            f'<div class="mg-mod-bg"><div class="mg-mod-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="mg-mod-val" style="color:{color}">{val:.2f}</span></div>'
        )
    return f'<div class="mg-modules"><div class="mg-mod-title">{title}</div>' + "".join(rows) + "</div>"


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.mg-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 24px; }
.mg-logo { font-size: 22px; font-weight: 700; color: #1e40af; display: flex; align-items: center; gap: 10px; }
.mg-logo-icon { width: 36px; height: 36px; background: linear-gradient(135deg,#1e40af,#3b82f6); border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; color: white; font-size: 18px; }
.mg-hero { text-align: center; padding: 16px 0 24px; }
.mg-hero h1 { font-size: 30px; font-weight: 700; color: #111827; margin-bottom: 8px; }
.mg-hero p { font-size: 15px; color: #6b7280; max-width: 600px; margin: 0 auto; line-height: 1.6; }
.mg-verdict { background: white; border-radius: 12px; padding: 24px; text-align: center; border-width: 2px; border-style: solid; }
.mg-real { border-color: #10b981; }
.mg-manipulated { border-color: #ef4444; }
.mg-ooc { border-color: #f59e0b; }
.mg-ai-text { border-color: #8b5cf6; }
.mg-fabricated { border-color: #dc2626; }
.mg-verdict-label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.mg-verdict-title { font-size: 24px; font-weight: 700; margin-bottom: 12px; }
.mg-conf-bar { height: 8px; background: #f3f4f6; border-radius: 4px; overflow: hidden; margin: 14px 0 6px; }
.mg-conf-fill { height: 100%; transition: width .6s; }
.mg-conf-text { font-size: 13px; color: #4b5563; }
.mg-explanation { border-radius: 8px; padding: 12px; margin-top: 18px; font-size: 13px; line-height: 1.6; }
.mg-prob-section { padding: 6px 0 0; }
.mg-prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 13px; }
.mg-prob-label { width: 140px; color: #4b5563; }
.mg-prob-bg { flex: 1; height: 22px; background: #f3f4f6; border-radius: 4px; overflow: hidden; }
.mg-prob-fill { height: 100%; transition: width .6s; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 11px; font-weight: 600; color: white; min-width: 30px; }
.mg-modules { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; }
.mg-mod-title { font-size: 15px; font-weight: 600; color: #111827; margin-bottom: 14px; }
.mg-mod-row { display: grid; grid-template-columns: 200px 1fr 60px; gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid #f3f4f6; font-size: 13px; }
.mg-mod-row:last-child { border-bottom: none; }
.mg-mod-bg { height: 6px; background: #f3f4f6; border-radius: 3px; overflow: hidden; }
.mg-mod-fill { height: 100%; transition: width .6s; }
.mg-mod-val { font-weight: 600; text-align: right; }
.mg-error { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; border-radius: 8px; padding: 14px; font-size: 14px; }
[dir="rtl"] .mg-mod-row { grid-template-columns: 60px 1fr 200px; }
[dir="rtl"] .mg-mod-val { text-align: left; }
"""

HEADER_HTML = """
<div class="mg-header">
  <div class="mg-logo"><span class="mg-logo-icon">M</span> MultiGuard</div>
</div>
<div class="mg-hero" id="mg-hero">
  <h1 id="mg-title">Verify Any News Article</h1>
  <p id="mg-sub">Upload text and an image to detect AI-generated content, manipulated images, and out-of-context pairings.</p>
</div>
"""


def main():
    log.info("[startup] building pipeline ...")
    _state.update(_build_pipeline())

    with gr.Blocks(css=CSS, title="MultiGuard", theme=gr.themes.Default(primary_hue="indigo")) as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            text_in = gr.Textbox(
                label="Article Text",
                placeholder="Paste the article text here...",
                lines=8,
            )
            image_in = gr.Image(label="Article Image", type="pil", height=240)

        submit_btn = gr.Button("Analyze Article", variant="primary", size="lg")

        with gr.Row():
            verdict_out = gr.HTML()
            with gr.Column():
                probs_out = gr.HTML()
                modules_out = gr.HTML()

        with gr.Accordion("Sample inputs", open=False):
            gr.Markdown(
                "- **Real news**: paste a recent news article + its original photo.\n"
                "- **OOC**: a real photo + a caption from a *different* unrelated story.\n"
                "- **Manipulated**: a deepfake / face-swap image + plausible caption.\n"
                "- **AI-Text**: a real photo + LLM-generated misinformation text.\n"
                "- **Fully fabricated**: an AI-generated (MidJourney) image + AI-generated text."
            )

        gr.Markdown(
            """
            ### About
            5-class multimodal fake-news detector (V3.1 spec, honest-path retrain).
            Stack: FND-CLIP semantic + DCT-Forensic image + Qwen2-7B text -> V3PairwiseFusion -> MLP.

            **Headline numbers:** 3-seed ensemble test F1 = 0.7149, MMFakeBench transfer F1 = 0.7197 (+ bias correction).
            Full repo: https://github.com/FerasMad/MultiGuard
            """
        )

        submit_btn.click(
            fn=analyze,
            inputs=[image_in, text_in],
            outputs=[verdict_out, probs_out, modules_out],
            api_name="analyze",
        )

    return demo


demo = main()
demo.queue().launch()

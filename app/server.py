"""MultiGuard V4 FastAPI server - registry-driven, V3.1-spec compliant."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from v4.core.checkpoints import load_checkpoint
from v4.core.class_map import LABELS, LABELS_AR
from v4.core.logging import configure, get_logger
from v4.core.registry import build_encoder, build_fusion, import_all
from v4.data.preprocessing.patch_dct import compute_patch_dct_from_pil
from v4.data.preprocessing.tokenizers import prepare_fnd_inputs

configure("INFO")
log = get_logger(__name__)

SERVER_CONFIG_PATH = os.environ.get("V4_SERVER_CONFIG", "app/server_config.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_server() -> dict:
    log.info("loading server config: %s", SERVER_CONFIG_PATH)
    with open(SERVER_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import_all()

    t0 = time.time()
    encoders = {}
    for name, spec in cfg["encoders"].items():
        log.info("building encoder %r: %s", name, spec["type"])
        enc = build_encoder(dict(spec))
        enc.to(DEVICE).eval()
        for p in enc.parameters():
            p.requires_grad = False
        encoders[name] = enc

    fusion_spec = dict(cfg["fusion"])
    fusion_ckpt = fusion_spec.pop("ckpt", None)
    fusion = build_fusion(fusion_spec)
    if fusion_ckpt and Path(fusion_ckpt).exists():
        ck = load_checkpoint(fusion_ckpt, map_location=DEVICE)
        fusion.load_state_dict(ck["model_state"])
        log.info("loaded fusion ckpt: %s", fusion_ckpt)
    fusion.to(DEVICE).eval()

    log.info("server ready in %.1fs", time.time() - t0)
    return {"encoders": encoders, "fusion": fusion}


_state: dict = {}
app = FastAPI(title="MultiGuard V4")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    _state.update(_load_server())


EXPLANATIONS_EN = {
    0: "The text and image appear genuine and aligned.",
    1: "The image and text appear genuine but mismatched (out-of-context).",
    2: "The image shows signs of digital manipulation.",
    3: "The text shows patterns consistent with AI-generated content.",
    4: "Both image and text show signs of fabrication.",
}
EXPLANATIONS_AR = {
    0: "يبدو أن النص والصورة حقيقيان ومتطابقان.",
    1: "تبدو الصورة والنص حقيقيين بشكل منفرد ولكنهما غير متطابقين.",
    2: "تظهر على الصورة علامات تعديل رقمي.",
    3: "يُظهر النص أنماطًا تتوافق مع المحتوى المولَّد بالذكاء الاصطناعي.",
    4: "تظهر على كل من الصورة والنص علامات تزييف.",
}


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "config": SERVER_CONFIG_PATH,
        "spec_version": "V3.1",
        "encoders": sorted(_state.get("encoders", {})),
    }


@app.post("/api/analyze")
async def analyze(text: str = Form(...), image: UploadFile = File(...)) -> JSONResponse:
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        dct = compute_patch_dct_from_pil(pil_img).unsqueeze(0).to(DEVICE)

        fnd = prepare_fnd_inputs(text, pil_img)
        fnd = {k: v.to(DEVICE) for k, v in fnd.items()}

        encs = _state["encoders"]
        with torch.no_grad():
            features = {}
            features["v_semantic"] = encs["semantic"](fnd)
            features["v_imgfor"] = encs["image"]({"v_imgfor_dct": dct})
            features["v_textfor"] = encs["text"].runtime_forward(text)
            out = _state["fusion"](features)

        logits = out["main_logits"][0]
        probs = F.softmax(logits, dim=-1).cpu().tolist()
        pred = int(np.argmax(probs))
        confidence = round(float(probs[pred]) * 100, 1)

        prob_dict = {
            "Real":             round(float(probs[0]), 4),
            "Out-of-Context":   round(float(probs[1]), 4),
            "Manipulated":      round(float(probs[2]), 4),
            "AI-Text":          round(float(probs[3]), 4),
            "Fully-Fabricated": round(float(probs[4]), 4),
        }
        text_ai = float(probs[3]) + float(probs[4])
        image_manip = float(probs[2]) + float(probs[4])
        modules = {
            "text_ai":       round(text_ai, 2),
            "text_patterns": round(min(1.0, 0.5 * text_ai + 0.5 * float(probs[1])), 2),
            "image_manip":   round(image_manip, 2),
            "cross_modal":   round(float(probs[1]), 2),
            "overall":       round(1.0 - float(probs[0]), 2),
        }

        return JSONResponse({
            "verdict": LABELS[pred],
            "verdict_ar": LABELS_AR[pred],
            "confidence": confidence,
            "label_index": pred,
            "probabilities": prob_dict,
            "modules": modules,
            "explanation": EXPLANATIONS_EN[pred],
            "explanation_ar": EXPLANATIONS_AR[pred],
        })
    except Exception as e:
        log.exception("analyze failed")
        return JSONResponse({"error": str(e)}, status_code=500)


_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")

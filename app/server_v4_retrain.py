"""MultiGuard FastAPI server — uses the V4 retrain (seed=42) checkpoint.

Serves the same bilingual UI at `app/static/index.html`. /api/analyze runs the
3-branch V3.1 pipeline: FND-CLIP V1 + DCT-Forensic + Qwen2-7B-Instruct →
V3PairwiseFusion (with proj_dims for raw V3-era caches) → MLP classifier.

Variant of app/server.py adapted for the retrain checkpoint
(outputs/v4/stage2_fusion_dctforensic/best.pt). Run:

    python -m uvicorn app.server_v4_retrain:app --host 0.0.0.0 --port 8081
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "phases" / "v4" / "app_hf"))

from inline.class_map import LABEL_EXPLANATIONS, LABELS  # noqa: E402
from inline.dct_forensic import DctForensicEncoder  # noqa: E402
from inline.dual_dct import compute_dual_dct  # noqa: E402
from inline.fnd_clip import FNDCLIPSemanticEncoder, prepare_fnd_inputs  # noqa: E402
from inline.qwen_text import Qwen2TextEncoder  # noqa: E402
from inline.v3_pairwise import V3PairwiseFusion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("multiguard.server")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS_AR = {
    0: "حقيقي",
    1: "خارج السياق",
    2: "صورة معدَّلة",
    3: "نص بالذكاء الاصطناعي",
    4: "تزييف كامل",
}

EXPLANATIONS_EN = {i: LABEL_EXPLANATIONS[i] for i in range(5)}
EXPLANATIONS_AR = {
    0: "يبدو أن النص والصورة حقيقيان ومتطابقان.",
    1: "تبدو الصورة والنص حقيقيين بشكل منفرد ولكنهما غير متطابقين.",
    2: "تظهر على الصورة علامات تعديل رقمي.",
    3: "يُظهر النص أنماطًا تتوافق مع المحتوى المولَّد بالذكاء الاصطناعي.",
    4: "تظهر على كل من الصورة والنص علامات تزييف.",
}


def _resolve_ckpt(env_var: str, default_path: str) -> Path:
    p = Path(os.environ.get(env_var, default_path))
    if not p.is_absolute():
        p = ROOT / p
    return p


_state: dict = {}


def _build_pipeline():
    fndclip_ckpt = _resolve_ckpt("MULTIGUARD_FNDCLIP_CKPT", "outputs/v1/leakfree/best.pt")
    dct_ckpt = _resolve_ckpt(
        "MULTIGUARD_DCT_CKPT", "phases/forensic/outputs/dct/forensic_dct_model.pth"
    )
    dct_stats = _resolve_ckpt("MULTIGUARD_DCT_STATS", "phases/forensic/data/dct_stats.json")
    fusion_ckpt = _resolve_ckpt(
        "MULTIGUARD_FUSION_CKPT", "outputs/v4/stage2_fusion_dctforensic/best.pt"
    )
    # P9.1 server-parity fix: load the deterministic encoder.head state
    # produced by precompute_v_imgfor_dctforensic.py --seed 42
    dct_head_state = _resolve_ckpt(
        "MULTIGUARD_DCT_HEAD_STATE", "outputs/v4/dctforensic_head_seed42.pt"
    )

    for label, p in [
        ("fndclip", fndclip_ckpt),
        ("dct", dct_ckpt),
        ("dct_stats", dct_stats),
        ("fusion", fusion_ckpt),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} ckpt not found at {p}")

    t0 = time.time()

    log.info("[server] building FND-CLIP V1 (semantic, 512-d native)...")
    fndclip = FNDCLIPSemanticEncoder(ckpt=fndclip_ckpt).to(DEVICE).eval()
    for p in fndclip.parameters():
        p.requires_grad = False

    log.info("[server] building DCT-Forensic encoder (image, 768-d)...")
    # P9.1: pass head_state_path so the head Linear(2048, 768) matches the cache
    dct_forensic = (
        DctForensicEncoder(
            out_dim=768,
            ckpt=dct_ckpt,
            head_state_path=dct_head_state if dct_head_state.exists() else None,
            seed=42,  # belt-and-suspenders: also seed before init
        )
        .to(DEVICE)
        .eval()
    )
    for p in dct_forensic.parameters():
        p.requires_grad = False

    log.info("[server] building Qwen2-7B-Instruct (text, 3584-d) on %s...", DEVICE)
    qwen = Qwen2TextEncoder(load_backbone=True)

    log.info("[server] building V3PairwiseFusion + loading retrain ckpt...")
    fusion = (
        V3PairwiseFusion(
            feat_dim=768,
            fused_dim=1024,
            num_classes=5,
            proj_dims={"v_semantic": 512, "v_textfor": 3584},
        )
        .to(DEVICE)
        .eval()
    )
    payload = torch.load(fusion_ckpt, map_location=DEVICE, weights_only=False)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    missing, unexpected = fusion.load_state_dict(state, strict=False)
    log.info("[server] fusion loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))
    for p in fusion.parameters():
        p.requires_grad = False

    stats_obj = json.loads(dct_stats.read_text(encoding="utf-8"))
    dct_mean = float(stats_obj["mean"])
    dct_std = float(stats_obj["std"])

    log.info("[server] pipeline ready in %.1fs (device=%s)", time.time() - t0, DEVICE)
    return {
        "fndclip": fndclip,
        "dct_forensic": dct_forensic,
        "qwen": qwen,
        "fusion": fusion,
        "dct_mean": dct_mean,
        "dct_std": dct_std,
    }


app = FastAPI(title="MultiGuard V4 (retrain)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    _state.update(_build_pipeline())


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "spec_version": "V3.1",
        "checkpoint": "outputs/v4/stage2_fusion_dctforensic/best.pt",
        "val_f1_macro": 0.7334,
        "test_f1_macro": 0.7267,
        "transfer_f1_macro": 0.4805,
    }


@app.post("/api/analyze")
async def analyze(text: str = Form(...), image: UploadFile = File(...)) -> JSONResponse:
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        fnd_batch = prepare_fnd_inputs(text, pil_img)
        fnd_batch = {k: v.to(DEVICE) for k, v in fnd_batch.items()}

        t = compute_dual_dct(pil_img)
        t = (t - _state["dct_mean"]) / (_state["dct_std"] + 1e-8)
        t = t.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            v_semantic = _state["fndclip"](fnd_batch)  # [1, 512]
            v_imgfor = _state["dct_forensic"]({"v_imgfor_dct": t})  # [1, 768]
            v_textfor = _state["qwen"].encode_text(text).to(DEVICE)  # [1, 3584]
            out = _state["fusion"](
                {
                    "v_semantic": v_semantic,
                    "v_imgfor": v_imgfor,
                    "v_textfor": v_textfor,
                }
            )

        logits = out["main_logits"][0]
        probs = F.softmax(logits, dim=-1).cpu().tolist()
        pred = int(np.argmax(probs))
        confidence = round(float(probs[pred]) * 100, 1)

        prob_dict = {
            "Real": round(float(probs[0]), 4),
            "Out-of-Context": round(float(probs[1]), 4),
            "Manipulated": round(float(probs[2]), 4),
            "AI-Text": round(float(probs[3]), 4),
            "Fully-Fabricated": round(float(probs[4]), 4),
        }

        text_ai = float(probs[3]) + float(probs[4])
        image_manip = float(probs[2]) + float(probs[4])
        modules = {
            "text_ai": round(text_ai, 2),
            "text_patterns": round(min(1.0, 0.5 * text_ai + 0.5 * float(probs[1])), 2),
            "image_manip": round(image_manip, 2),
            "cross_modal": round(float(probs[1]), 2),
            "overall": round(1.0 - float(probs[0]), 2),
        }

        return JSONResponse(
            {
                "verdict": LABELS[pred],
                "verdict_ar": LABELS_AR[pred],
                "confidence": confidence,
                "label_index": pred,
                "probabilities": prob_dict,
                "modules": modules,
                "explanation": EXPLANATIONS_EN[pred],
                "explanation_ar": EXPLANATIONS_AR[pred],
            }
        )
    except Exception as e:
        log.exception("analyze failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8081")), log_level="info")

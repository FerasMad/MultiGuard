"""MultiGuard HF Space -- Gradio SDK with FastAPI mounted (ZeroGPU-compatible).

HF Spaces ZeroGPU only works with the Gradio SDK. To keep the same bilingual
EN/AR website UI from app/static/, we:
  1. Build a FastAPI app with POST /api/analyze and static UI at /
  2. Build a tiny Gradio Blocks (required so HF detects this as a Gradio Space)
  3. Mount Gradio at /gradio inside the FastAPI app
  4. Wrap inference with @spaces.GPU so each /api/analyze request gets ZeroGPU

The user sees the same UI as the local FastAPI server; the API contract is
identical so the existing JavaScript in index.html keeps working.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image
from starlette.concurrency import run_in_threadpool

from inline.class_map import LABELS, LABEL_EXPLANATIONS
from inline.dct_forensic import DctForensicEncoder
from inline.dual_dct import compute_dual_dct
from inline.fnd_clip import FNDCLIPSemanticEncoder, prepare_fnd_inputs
from inline.qwen_text import Qwen2TextEncoder
from inline.v3_pairwise import V3PairwiseFusion

import spaces

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("multiguard")

# ZeroGPU note: CUDA is NOT available at module import / startup time.
# It only becomes available inside @spaces.GPU functions. Therefore we build
# the pipeline on CPU and move tensors+models to "cuda" inside the
# decorated _ensemble_predict (the .to('cuda') call is idempotent so this
# is cheap on subsequent calls within the same ZeroGPU window).
GPU_DEVICE = torch.device("cuda")

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

_state: dict = {}


def _build_pipeline():
    t0 = time.time()
    log.info("[build] downloading FND-CLIP V1 ckpt ...")
    fndclip_ckpt = hf_hub_download(
        repo_id="FerasMad/multiguard-v1-fndclip", filename="best.pt",
    )
    log.info("[build] downloading DCT-Forensic ckpt + stats ...")
    dct_ckpt = hf_hub_download(
        repo_id="FerasMad/forensic-dct-v1", filename="forensic_dct_model.pth",
    )
    dct_stats = hf_hub_download(
        repo_id="FerasMad/forensic-dct-v1", filename="dct_stats.json",
    )
    log.info("[build] downloading honest-path ensemble (3 seeds + parity head) ...")
    seed_ckpts = [
        hf_hub_download(repo_id="FerasMad/multiguard-v4-honest", filename=f"seed{s}.pt")
        for s in (42, 1337, 2024)
    ]
    dct_head_state = hf_hub_download(
        repo_id="FerasMad/multiguard-v4-honest", filename="dctforensic_head_seed42.pt",
    )

    log.info("[build] FND-CLIP semantic encoder (CPU; moved to GPU inside @spaces.GPU) ...")
    fndclip = FNDCLIPSemanticEncoder(ckpt=fndclip_ckpt).eval()
    for p in fndclip.parameters():
        p.requires_grad = False

    log.info("[build] DCT-Forensic encoder with P9.1 parity head (CPU) ...")
    dct_forensic = DctForensicEncoder(
        out_dim=768,
        ckpt=dct_ckpt,
        head_state_path=dct_head_state,
        seed=42,
    ).eval()
    for p in dct_forensic.parameters():
        p.requires_grad = False

    log.info("[build] Qwen2-7B-Instruct (lazy load on first @spaces.GPU call) ...")
    qwen = Qwen2TextEncoder(load_backbone=False)

    log.info("[build] 3-seed V3PairwiseFusion ensemble (CPU) ...")
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
    log.info("[build] pipeline ready in %.1fs (CPU; will move to cuda per request)",
             time.time() - t0)
    return {
        "fndclip": fndclip,
        "dct_forensic": dct_forensic,
        "qwen": qwen,
        "fusions": fusions,
        "dct_mean": float(stats_obj["mean"]),
        "dct_std": float(stats_obj["std"]),
    }


@spaces.GPU(duration=120)
def _ensemble_predict(pil_img: Image.Image, text: str) -> list[float]:
    """3-seed softmax-average ensemble. Returns 5-vector of probabilities.

    Inside the @spaces.GPU window CUDA is available. Move models to GPU
    (idempotent after first call) and run inference there.
    """
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


fastapi_app = FastAPI(title="MultiGuard (honest-path ensemble)")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.on_event("startup")
async def _startup() -> None:
    _state.update(_build_pipeline())


@fastapi_app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": "zero-gpu (cuda allocated per /api/analyze call)",
        "spec_version": "V3.1 honest-path",
        "ensemble_seeds": [42, 1337, 2024],
        "val_f1_macro_mean": 0.7292,
        "val_f1_macro_std": 0.0055,
        "test_f1_macro_ensemble": 0.7149,
        "transfer_f1_macro_ensemble": 0.4308,
        "transfer_f1_macro_biascorr": 0.7197,
        "ckpts": "FerasMad/multiguard-v4-honest (seeds 42/1337/2024)",
    }


@fastapi_app.post("/api/analyze")
async def analyze(text: str = Form(...), image: UploadFile = File(...)) -> JSONResponse:
    try:
        img_bytes = await image.read()
        if len(img_bytes) > 50_000_000:
            return JSONResponse(
                {"error": "image too large (max 50 MB)"}, status_code=413,
            )
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        probs = await run_in_threadpool(_ensemble_predict, pil_img, text)

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
        return JSONResponse({
            "verdict": LABELS[pred],
            "verdict_ar": LABELS_AR[pred],
            "confidence": confidence,
            "label_index": pred,
            "probabilities": prob_dict,
            "modules": modules,
            "explanation": LABEL_EXPLANATIONS[pred],
            "explanation_ar": EXPLANATIONS_AR[pred],
        })
    except Exception as e:
        log.exception("analyze failed")
        return JSONResponse(
            {"error": f"{type(e).__name__}: {e}"}, status_code=500,
        )


_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    fastapi_app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(
        "## MultiGuard API\n\n"
        "The bilingual web UI is at the root of this Space.\n"
        "POST `/api/analyze` with multipart `text` + `image` returns a verdict."
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# HF Spaces Gradio SDK launches the mounted FastAPI via `app` at module level.
# Do NOT add an `if __name__ == "__main__"` uvicorn block here; it conflicts
# with HF's runner and triggers an immediate shutdown.

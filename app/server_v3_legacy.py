"""MultiGuard - FastAPI backend for multimodal fake-news detection.

V3.1 5-class pipeline (Implementation Guidelines V3):
    FND-CLIP (frozen)  -> v_semantic [512]   --sem_proj-->  [768]
    UnivFD   (frozen)  -> v_imgfor   [768]
    Qwen2-7B-Instruct  -> v_textfor  [3584]  --text_proj--> [768]
                          (last hidden, masked mean pool)

    V3FusionModule(v_semantic, v_imgfor, v_textfor) -> main_logits [5]
        0 = Real / 1 = Out-of-Context / 2 = Manipulated /
        3 = AI-Text / 4 = Fully-Fabricated

Checkpoint: v3/outputs/v3_pipeline_qwen/best.pt
Run with  : uvicorn app.server:app --host 0.0.0.0 --port 8080
Public URL: cloudflared tunnel --url http://localhost:8080
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback
from pathlib import Path

import cv2

# IMPORTANT: import torch BEFORE cv2/scipy/torchvision to avoid Windows
# MKL/OpenBLAS DLL conflicts that silently kill the process.
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# NOTE: scipy.fftpack.dct segfaults on Windows when scipy's pocketfft DLL
# conflicts with torch CUDA libs (observed: access violation in _r2r).
# Use cv2.dct instead — different DLL, same orthonormal type-II DCT.
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, CLIPProcessor

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models.fnd_clip import FNDCLIP  # noqa: E402
from models.text_fluoroscopy import TextForensicProjection, masked_mean_pool  # noqa: E402
from models.univfd_encoder import UnivFDEncoder  # noqa: E402
from models.v3_pipeline import V3FusionModule  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
V3_CKPT = ROOT / "v3" / "outputs" / "v3_pipeline_qwen" / "best.pt"
FND_CKPT = ROOT / "outputs" / "v1_leakfree" / "best.pt"
UNIVFD_CKPT = ROOT / "outputs" / "univfd_genimage" / "best.pt"
QWEN_MODEL = "Qwen/Qwen2-7B-Instruct"

# Qwen fp16 + CPU offload via accelerate. 10 GiB cap keeps a 12 GB 4070 happy.
QWEN_GPU_MEM_GIB = int(os.environ.get("QWEN_GPU_MEM_GIB", "10"))
QWEN_MAX_LEN = 512

# Patch-DCT preprocessing (matches v3/scripts/precompute_patch_dct.py).
DCT_SIZE = 224
DCT_PATCH = 8
JPEG_QUALITY = 85

LABEL_NAMES = {
    0: "Real",
    1: "Out-of-Context",
    2: "Manipulated",
    3: "AI-Text",
    4: "Fully Fabricated",
}
LABEL_NAMES_AR = {0: "حقيقي", 1: "خارج السياق", 2: "معدَّل", 3: "نص مولَّد", 4: "ملفَّق بالكامل"}

# ---------------------------------------------------------------------------
# Model loading (runs once at startup)
# ---------------------------------------------------------------------------
print(f"[startup] device={DEVICE}, num_classes={NUM_CLASSES}", flush=True)

print(f"[startup] loading FND-CLIP from {FND_CKPT.name} ...", flush=True)
t0 = time.time()
fnd_clip = FNDCLIP(feat_dim=512, num_classes=1)
fnd_state = torch.load(FND_CKPT, map_location="cpu", weights_only=False)
fnd_state = fnd_state.get("model_state", fnd_state)
compat = {
    k: v
    for k, v in fnd_state.items()
    if k in fnd_clip.state_dict() and fnd_clip.state_dict()[k].shape == v.shape
}
fnd_clip.load_state_dict(compat, strict=False)
fnd_clip = fnd_clip.to(DEVICE).eval()
for p in fnd_clip.parameters():
    p.requires_grad = False
print(
    f"  loaded {len(compat)}/{len(fnd_clip.state_dict())} tensors ({time.time() - t0:.1f}s)",
    flush=True,
)

print(f"[startup] loading UnivFD from {UNIVFD_CKPT.name} ...", flush=True)
t0 = time.time()
univfd = UnivFDEncoder(out_dim=768, pretrained=None, dropout=0.3)
univfd_ck = torch.load(UNIVFD_CKPT, map_location="cpu", weights_only=False)
univfd_state = univfd_ck.get("model_state", univfd_ck)
# Stage-1 wraps UnivFDEncoder as `self.encoder.*` - strip prefix.
univfd_enc_state = {
    k[len("encoder.") :]: v for k, v in univfd_state.items() if k.startswith("encoder.")
}
missing, unexpected = univfd.load_state_dict(univfd_enc_state, strict=False)
univfd = univfd.to(DEVICE).eval()
for p in univfd.parameters():
    p.requires_grad = False
print(
    f"  loaded {len(univfd_enc_state)}/{len(univfd.state_dict())} tensors "
    f"(missing={len(missing)}, unexpected={len(unexpected)}) "
    f"({time.time() - t0:.1f}s)",
    flush=True,
)

print(
    f"[startup] loading Qwen2-7B-Instruct (fp16, device_map=auto, "
    f"GPU cap {QWEN_GPU_MEM_GIB} GiB) ...",
    flush=True,
)
t0 = time.time()
qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
if qwen_tok.pad_token_id is None:
    qwen_tok.pad_token = qwen_tok.eos_token
    qwen_tok.pad_token_id = qwen_tok.eos_token_id
qwen_tok.padding_side = "left"
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: f"{QWEN_GPU_MEM_GIB}GiB", "cpu": "20GiB"},
    low_cpu_mem_usage=True,
)
qwen_model.eval()
QWEN_HIDDEN = qwen_model.config.hidden_size  # 3584 for Qwen2-7B-Instruct
QWEN_LAYERS = qwen_model.config.num_hidden_layers
print(
    f"  hidden_size={QWEN_HIDDEN}  num_hidden_layers={QWEN_LAYERS}  ({time.time() - t0:.1f}s)",
    flush=True,
)

print(f"[startup] loading V3 fusion ckpt {V3_CKPT.name} ...", flush=True)
t0 = time.time()
v3_ck = torch.load(V3_CKPT, map_location=DEVICE, weights_only=False)
mcfg = v3_ck.get("config", {}).get("model", {})
FEAT_DIM = int(mcfg.get("feat_dim", 768))
SEM_IN = int(mcfg.get("sem_in_dim", 512))
TXT_IN = int(mcfg.get("txt_in_dim", QWEN_HIDDEN))
if TXT_IN != QWEN_HIDDEN:
    print(f"  WARNING: ckpt txt_in_dim={TXT_IN} != Qwen hidden={QWEN_HIDDEN}", flush=True)


class _SemProj(nn.Module):
    """Mirror of v3/src/train_v3_pipeline.py::_Projector - Linear + GELU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


sem_proj = _SemProj(SEM_IN, FEAT_DIM).to(DEVICE)
text_proj = TextForensicProjection(input_dim=TXT_IN, output_dim=FEAT_DIM).to(DEVICE)
v3_fusion = V3FusionModule(
    feat_dim=FEAT_DIM,
    fused_dim=int(mcfg.get("fused_dim", 1024)),
    num_classes=int(mcfg.get("num_classes", NUM_CLASSES)),
    num_heads=int(mcfg.get("num_heads", 8)),
    attn_dropout=float(mcfg.get("attn_dropout", 0.1)),
).to(DEVICE)

v3_fusion.load_state_dict(v3_ck["model_state"])
sem_proj.load_state_dict(v3_ck["sem_proj_state"])
text_proj.load_state_dict(v3_ck["text_proj_state"])
v3_fusion.eval()
sem_proj.eval()
text_proj.eval()
for m in (v3_fusion, sem_proj, text_proj):
    for p in m.parameters():
        p.requires_grad = False
val_f1 = v3_ck.get("val_metrics", {}).get("f1_macro", float("nan"))
print(
    f"  loaded ckpt epoch {v3_ck.get('epoch', '?')} "
    f"(val F1-macro {val_f1:.4f}) ({time.time() - t0:.1f}s)",
    flush=True,
)

# Tokenizers / preprocessors for FND-CLIP front-end.
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

print("[startup] all models loaded.", flush=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_dct(y: np.ndarray, patch_size: int = DCT_PATCH) -> np.ndarray:
    """Tile-wise 2D-DCT on the Y channel, log-magnitude per patch.
    Uses cv2.dct (orthonormal type-II) to avoid the scipy/torch DLL crash
    on Windows."""
    h, w = y.shape
    out = np.empty_like(y, dtype=np.float32)
    y32 = y.astype(np.float32)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            block = y32[i : i + patch_size, j : j + patch_size]
            coeffs = cv2.dct(block)
            out[i : i + patch_size, j : j + patch_size] = np.log(np.abs(coeffs) + 1e-8)
    return out


def compute_patch_dct_from_pil(pil_img: Image.Image) -> torch.Tensor:
    """Replica of v3/scripts/precompute_patch_dct.py::compute_patch_dct,
    but takes a PIL image instead of a file path. Returns [1, 224, 224]."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Uniform JPEG re-encode (kills source-distribution shortcut).
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if ok:
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is not None:
            bgr = decoded
    bgr = cv2.resize(bgr, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA)
    ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)
    log_mag = _patch_dct(y, patch_size=DCT_PATCH)
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-12:
        normed = np.zeros_like(log_mag)
    else:
        normed = (log_mag - lo) / (hi - lo)
    return torch.from_numpy(normed).unsqueeze(0).float()  # [1, 224, 224]


def prepare_fnd_inputs(text: str, pil_img: Image.Image) -> dict:
    """Build the inputs FND-CLIP.forward_semantic expects."""
    img_tensor = image_tf(pil_img)
    bert = bert_tok(
        text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )
    clip = clip_proc(
        images=pil_img,
        text=text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    return {
        "image": img_tensor.unsqueeze(0),
        "bert_ids": bert["input_ids"],
        "bert_mask": bert["attention_mask"],
        "clip_pixels": clip["pixel_values"],
        "clip_ids": clip["input_ids"],
        "clip_mask": clip["attention_mask"],
    }


def encode_text_qwen(text: str) -> torch.Tensor:
    """Tokenize -> Qwen forward -> last hidden -> masked mean pool.
    Returns [1, QWEN_HIDDEN] float32 on DEVICE."""
    enc = qwen_tok(
        [text], return_tensors="pt", padding=True, truncation=True, max_length=QWEN_MAX_LEN
    )
    # accelerate (device_map='auto') accepts inputs on cuda:0; it forwards
    # tensors across devices internally for CPU-offloaded layers.
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = qwen_model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]  # [1, S, H], fp16
    pooled = masked_mean_pool(h.float(), enc["attention_mask"])  # [1, H], fp32
    return pooled.to(DEVICE)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MultiGuard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


EXPLANATIONS_EN = {
    0: (
        "The text and image appear genuine and well-aligned. No "
        "significant manipulation or out-of-context usage detected."
    ),
    1: (
        "The image and text appear genuine individually, but the "
        "cross-modal module detected low semantic alignment between "
        "them, suggesting the image was used out of context."
    ),
    2: (
        "The image shows signs of digital manipulation. The forensic "
        "encoder detected inconsistencies in the pixel-level artifacts, "
        "suggesting the image has been altered."
    ),
    3: (
        "The text shows patterns consistent with AI-generated content. "
        "The text forensic module detected linguistic signatures "
        "typical of large language model outputs."
    ),
    4: (
        "Both image and text show signs of fabrication. The forensic "
        "modules detected manipulated visuals paired with AI-generated "
        "text."
    ),
}
EXPLANATIONS_AR = {
    0: "يبدو أن النص والصورة حقيقيان ومتطابقان. لم يتم اكتشاف أي تعديل أو استخدام خارج السياق.",
    1: "تبدو الصورة والنص حقيقيين بشكل منفرد، إلا أن وحدة المطابقة رصدت ضعف التطابق الدلالي بينهما، مما يشير إلى أن الصورة استُخدمت خارج سياقها.",
    2: "تظهر على الصورة علامات تعديل رقمي. اكتشف المحلل الجنائي تناقضات في بنية البيكسلات، مما يشير إلى أن الصورة معدلة.",
    3: "يُظهر النص أنماطًا تتوافق مع المحتوى المولَّد بالذكاء الاصطناعي. رصدت وحدة تحليل النص بصمات لغوية نموذجية لمخرجات النماذج اللغوية الكبيرة.",
    4: "تظهر على كل من الصورة والنص علامات تزييف. رصدت الوحدات الجنائية صورًا معدَّلة مقترنة بنص مولَّد بالذكاء الاصطناعي.",
}


@app.post("/api/analyze")
async def analyze(text: str = Form(...), image: UploadFile = File(...)):
    """Run the V3 5-class pipeline on a text+image pair."""
    try:
        # ---- 1. Decode image ----
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ---- 2. Build inputs for each stream ----
        fnd_in = prepare_fnd_inputs(text, pil_img)
        dct_map = compute_patch_dct_from_pil(pil_img)  # [1, 224, 224]
        dct_batch = dct_map.unsqueeze(0).to(DEVICE)  # [1, 1, 224, 224]

        # ---- 3. Run all three encoders + fusion ----
        with torch.no_grad():
            # 3a. FND-CLIP -> v_semantic [1, 512]
            v_semantic = fnd_clip.forward_semantic(
                image=fnd_in["image"].to(DEVICE),
                bert_ids=fnd_in["bert_ids"].to(DEVICE),
                bert_mask=fnd_in["bert_mask"].to(DEVICE),
                clip_pixels=fnd_in["clip_pixels"].to(DEVICE),
                clip_ids=fnd_in["clip_ids"].to(DEVICE),
                clip_mask=fnd_in["clip_mask"].to(DEVICE),
            )
            # 3b. UnivFD on patch-DCT -> v_imgfor [1, 768]
            v_imgfor = univfd(dct_batch)
            # 3c. Qwen2-7B -> v_textfor [1, 3584]
            v_textfor = encode_text_qwen(text)

            # 3d. Project sem (512->768) and text (3584->768), then fuse.
            v_sem768 = sem_proj(v_semantic)
            v_txt768 = text_proj(v_textfor)
            out = v3_fusion(v_sem768, v_imgfor, v_txt768)

        # ---- 4. Probabilities ----
        main_logits = out["main_logits"][0]  # [5]
        aux_logits = out["aux_logits"][0]  # [2]
        probs = F.softmax(main_logits, dim=0).cpu().tolist()
        aux_probs = torch.sigmoid(aux_logits).cpu().tolist()  # [P(real), P(fake)]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Class-index mapping matches both training and UI:
        # 0=Real, 1=OOC, 2=Manipulated, 3=AI-Text, 4=Fully-Fabricated.
        prob_dict = {
            "Real": round(float(probs[0]), 4),
            "Out-of-Context": round(float(probs[1]), 4),
            "Manipulated": round(float(probs[2]), 4),
            "AI-Text": round(float(probs[3]), 4),
            "Fully-Fabricated": round(float(probs[4]), 4),
        }

        # Module scores - interpretable summaries built from class probs +
        # the aux image-forensic head (which was trained on binary
        # "image is AI/tampered" = class in {Manipulated, Fully-Fab}).
        text_ai = float(probs[3]) + float(probs[4])  # AI-Text + Fab
        image_manip = float(probs[2]) + float(probs[4])  # Manip + Fab
        cross_modal = float(probs[1])  # OOC
        text_patterns = min(1.0, 0.5 * text_ai + 0.5 * cross_modal)
        overall = 1.0 - float(probs[0])  # 1 - P(Real)

        modules = {
            "text_ai": round(text_ai, 2),
            "text_patterns": round(text_patterns, 2),
            "image_manip": round(max(image_manip, float(aux_probs[1])), 2),
            "cross_modal": round(cross_modal, 2),
            "overall": round(overall, 2),
        }

        return JSONResponse(
            {
                "verdict": LABEL_NAMES[pred_idx],
                "verdict_ar": LABEL_NAMES_AR[pred_idx],
                "confidence": round(confidence * 100, 1),
                "label_index": pred_idx,
                "probabilities": prob_dict,
                "modules": modules,
                "explanation": EXPLANATIONS_EN[pred_idx],
                "explanation_ar": EXPLANATIONS_AR[pred_idx],
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": "v3_pipeline_qwen",
        "classes": NUM_CLASSES,
        "device": DEVICE,
        "encoders": {
            "fnd_clip": str(FND_CKPT.name),
            "univfd": str(UNIVFD_CKPT.name),
            "qwen": QWEN_MODEL,
            "fusion": str(V3_CKPT.relative_to(ROOT)),
        },
    }


# Serve static files (UI).
app.mount(
    "/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static"
)


if __name__ == "__main__":
    # Run uvicorn programmatically to avoid the `python -m uvicorn` import
    # dance, which on Windows can cause the module to be imported twice in
    # the reloader/worker process and OOM during the heavy model loads.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

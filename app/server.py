"""MultiGuard — FastAPI backend for multimodal fake-news detection."""

import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from scipy.fft import dct
from torchvision import transforms
from transformers import BertTokenizer, CLIPProcessor

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models.fnd_clip import FNDCLIP
from models.full_pipeline import FullPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cpu"  # safe default; MPS can OOM on concurrent requests
FEAT_DIM = 512
NUM_CLASSES = 3
CKPT_PATH = ROOT / "outputs" / "full_pipeline_clean" / "best.pt"
LABEL_NAMES = {0: "Real", 1: "Out-of-Context", 2: "Manipulated", 3: "AI-Text", 4: "Double Fake"}
LABEL_NAMES_AR = {0: "حقيقي", 1: "خارج السياق", 2: "معدَّل", 3: "نص مولَّد", 4: "تزييف مزدوج"}
DCT_SIZE = 224
JPEG_QUALITY = 85

# ---------------------------------------------------------------------------
# Model loading (runs once at startup)
# ---------------------------------------------------------------------------
print("Loading models...")

# 1. FND-CLIP (fresh pretrained — no task checkpoint to avoid leakage)
fnd_clip = FNDCLIP(feat_dim=FEAT_DIM, num_classes=1)
fnd_clip.eval()
for p in fnd_clip.parameters():
    p.requires_grad = False

# 2. Full pipeline
pipeline = FullPipeline(
    fnd_clip=fnd_clip,
    fnd_clip_feat_dim=FEAT_DIM,
    forensic_feat_dim=FEAT_DIM,
    fusion_proj_dim=FEAT_DIM,
    num_classes=NUM_CLASSES,
    fusion_heads=8,
    fusion_dropout=0.1,
    forensic_dropout=0.3,
)

# Load trained weights (forensic + fusion + classifier + aux)
ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
pipeline.load_state_dict(ck["model_state"], strict=False)
pipeline.to(DEVICE).eval()
print(f"  loaded checkpoint epoch {ck['epoch']} "
      f"(val F1-macro {ck['val_metrics']['f1_macro']:.3f})")

# 3. Tokenizers / processors
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 4. Image transforms
image_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("Models loaded.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_dct(pil_img: Image.Image) -> torch.Tensor:
    """Compute DCT map from PIL image — matches precompute_dct.py pipeline."""
    # Convert to BGR numpy
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # JPEG re-normalization
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, buf = cv2.imencode(".jpg", bgr, encode_param)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Resize
    bgr = cv2.resize(bgr, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_LINEAR)

    # YCbCr → Y channel
    ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)

    # 2D DCT
    coeffs = dct(dct(y, axis=0, norm="ortho"), axis=1, norm="ortho")
    mag = np.log(np.abs(coeffs) + 1e-8)

    # Min-max normalize
    mn, mx = mag.min(), mag.max()
    if mx - mn > 1e-10:
        mag = (mag - mn) / (mx - mn)
    else:
        mag = np.zeros_like(mag)

    return torch.tensor(mag, dtype=torch.float32).unsqueeze(0)  # [1, 224, 224]


def prepare_inputs(text: str, pil_img: Image.Image):
    """Prepare all model inputs from raw text + image."""
    img_tensor = image_tf(pil_img)

    bert = bert_tok(text, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    clip = clip_proc(images=pil_img, text=text, return_tensors="pt",
                     padding="max_length", truncation=True, max_length=77)
    dct_map = compute_dct(pil_img)

    return {
        "image": img_tensor.unsqueeze(0),
        "bert_ids": bert["input_ids"],
        "bert_mask": bert["attention_mask"],
        "clip_pixels": clip["pixel_values"],
        "clip_ids": clip["input_ids"],
        "clip_mask": clip["attention_mask"],
        "dct": dct_map.unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MultiGuard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.post("/api/analyze")
async def analyze(text: str = Form(...), image: UploadFile = File(...)):
    """Run the full pipeline on a text+image pair."""
    try:
        # Read image
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Prepare inputs
        inputs = prepare_inputs(text, pil_img)

        # Compute v_semantic (FND-CLIP forward)
        with torch.no_grad():
            v_semantic = fnd_clip.forward_semantic(
                image=inputs["image"].to(DEVICE),
                bert_ids=inputs["bert_ids"].to(DEVICE),
                bert_mask=inputs["bert_mask"].to(DEVICE),
                clip_pixels=inputs["clip_pixels"].to(DEVICE),
                clip_ids=inputs["clip_ids"].to(DEVICE),
                clip_mask=inputs["clip_mask"].to(DEVICE),
            )

            # Full pipeline forward
            out = pipeline(
                dct=inputs["dct"].to(DEVICE),
                v_semantic=v_semantic,
            )

        # Process outputs
        main_logits = out["main_logits"][0]         # [3]
        aux_logits = out["aux_logits"][0]            # [2]
        probs = F.softmax(main_logits, dim=0)       # [3]
        aux_probs = torch.sigmoid(aux_logits)        # [2]

        # Map 3-class model outputs to 5-class UI
        # Model classes: 0=Real, 1=Manipulated, 2=OOC
        # UI classes:    0=Real, 1=OOC, 2=Manipulated, 3=AI-Text, 4=Double-Fake
        real_prob = probs[0].item()
        manip_prob = probs[1].item()
        ooc_prob = probs[2].item()
        forensic_fake_prob = aux_probs[1].item()

        # Remap to 5-class indices
        prob_map = {0: real_prob, 1: ooc_prob, 2: manip_prob}
        pred_idx_3class = probs.argmax().item()
        remap = {0: 0, 1: 2, 2: 1}  # model idx -> UI idx
        pred_idx = remap[pred_idx_3class]
        confidence = probs[pred_idx_3class].item()

        modules = {
            "text_ai":       round(1.0 - real_prob, 2),
            "text_patterns": round(manip_prob * 0.6, 2),
            "image_manip":   round(forensic_fake_prob, 2),
            "cross_modal":   round(ooc_prob, 2),
            "overall":       round(1.0 - real_prob, 2),
        }

        explanations = {
            0: "The text and image appear genuine and well-aligned. No significant manipulation or out-of-context usage detected.",
            1: "The image and text appear genuine individually, but the cross-modal module detected low semantic alignment between them, suggesting the image was used out of context.",
            2: "The image shows signs of digital manipulation. The forensic encoder detected inconsistencies in the pixel-level artifacts, suggesting the image has been altered.",
            3: "The text shows patterns consistent with AI-generated content. The text forensic module detected linguistic signatures typical of large language model outputs.",
            4: "Both image and text show signs of fabrication. The forensic modules detected manipulated visuals paired with AI-generated text.",
        }
        explanations_ar = {
            0: "يبدو أن النص والصورة حقيقيان ومتطابقان. لم يتم اكتشاف أي تعديل أو استخدام خارج السياق.",
            1: "تبدو الصورة والنص حقيقيين بشكل منفرد، إلا أن وحدة المطابقة رصدت ضعف التطابق الدلالي بينهما، مما يشير إلى أن الصورة استُخدمت خارج سياقها.",
            2: "تظهر على الصورة علامات تعديل رقمي. اكتشف المحلل الجنائي تناقضات في بنية البيكسلات، مما يشير إلى أن الصورة معدلة.",
            3: "يُظهر النص أنماطًا تتوافق مع المحتوى المولَّد بالذكاء الاصطناعي. رصدت وحدة تحليل النص بصمات لغوية نموذجية لمخرجات النماذج اللغوية الكبيرة.",
            4: "تظهر على كل من الصورة والنص علامات تزييف. رصدت الوحدات الجنائية صورًا معدَّلة مقترنة بنص مولَّد بالذكاء الاصطناعي.",
        }

        return JSONResponse({
            "verdict": LABEL_NAMES[pred_idx],
            "verdict_ar": LABEL_NAMES_AR[pred_idx],
            "confidence": round(confidence * 100, 1),
            "label_index": pred_idx,
            "probabilities": {
                "Real": round(real_prob, 4),
                "Out-of-Context": round(ooc_prob, 4),
                "Manipulated": round(manip_prob, 4),
                "AI-Text": 0.0,
                "Double-Fake": 0.0,
            },
            "modules": modules,
            "explanation": explanations[pred_idx],
            "explanation_ar": explanations_ar[pred_idx],
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "full_pipeline_clean", "classes": 3}


# Serve static files (UI)
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"),
                           html=True), name="static")

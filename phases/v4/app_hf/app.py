"""MultiGuard — HF Spaces Gradio demo.

Three tabs:
  1. Forensic A1 (RGB + Fourier mask)       -- binary fake-image detector, CPU
  2. Forensic A2 (Dual-DCT)                 -- binary fake-image detector, CPU
  3. V4 5-class multimodal fake-news        -- ZeroGPU (A100), Qwen2-7B + FND-CLIP + DCT-Forensic

The V4 5-class tab implements the doctor's 3-branch architecture (V3.1 spec).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from inline.class_map import DISPLAY_LABELS, LABEL_EXPLANATIONS  # noqa: E402
from inline.dct_forensic import DctForensicEncoder  # noqa: E402
from inline.dct_resnet50 import build_dct_resnet50  # noqa: E402
from inline.dual_dct import compute_dual_dct  # noqa: E402
from inline.fnd_clip import FNDCLIPSemanticEncoder, prepare_fnd_inputs  # noqa: E402
from inline.qwen_text import Qwen2TextEncoder  # noqa: E402
from inline.v3_pairwise import V3PairwiseFusion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("multiguard.app")

try:
    import spaces

    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False

    class _NoopGPU:
        def GPU(self, *args, **kwargs):  # noqa: N802
            def decorator(fn):
                return fn

            return decorator

    spaces = _NoopGPU()  # type: ignore[assignment]


CKPT_CACHE = HERE / ".ckpts"
CKPT_CACHE.mkdir(parents=True, exist_ok=True)

_A1_CKPT_PATH: Path | None = None
_A2_CKPT_PATH: Path | None = None
_A2_STATS_PATH: Path | None = None
_V4_FUSION_PATH: Path | None = None
_V1_FNDCLIP_PATH: Path | None = None

_a1_model = None
_a2_model = None
_a2_mean: float | None = None
_a2_std: float | None = None
_v4_pipeline = None


def _download(repo_id: str, filename: str) -> Path:
    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        local_dir=str(CKPT_CACHE / repo_id.replace("/", "__")),
    )
    return Path(p)


def _ensure_a1_ckpt() -> Path:
    global _A1_CKPT_PATH
    if _A1_CKPT_PATH is None:
        log.info("downloading A1 ckpt from HF Hub...")
        _A1_CKPT_PATH = _download("FerasMad/forensic-rgb-v1", "forensic_rgb_model.pth")
    return _A1_CKPT_PATH


def _ensure_a2_files() -> tuple[Path, Path]:
    global _A2_CKPT_PATH, _A2_STATS_PATH
    if _A2_CKPT_PATH is None:
        log.info("downloading A2 ckpt from HF Hub...")
        _A2_CKPT_PATH = _download("FerasMad/forensic-dct-v1", "forensic_dct_model.pth")
    if _A2_STATS_PATH is None:
        log.info("downloading dct_stats.json from HF Hub...")
        _A2_STATS_PATH = _download("FerasMad/forensic-dct-v1", "dct_stats.json")
    return _A2_CKPT_PATH, _A2_STATS_PATH


def _ensure_v4_ckpts() -> tuple[Path, Path, Path]:
    global _V4_FUSION_PATH, _V1_FNDCLIP_PATH
    if _V4_FUSION_PATH is None:
        log.info("downloading V4 fusion ckpt from HF Hub...")
        _V4_FUSION_PATH = _download("FerasMad/multiguard-v4-fusion", "best.pt")
    if _V1_FNDCLIP_PATH is None:
        log.info("downloading V1 FND-CLIP ckpt from HF Hub...")
        _V1_FNDCLIP_PATH = _download("FerasMad/multiguard-v1-fndclip", "best.pt")
    a2_ckpt, _ = _ensure_a2_files()
    return _V4_FUSION_PATH, _V1_FNDCLIP_PATH, a2_ckpt


# --- Approach 1 ----------------------------------------------------------

A1_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_a1_model():
    global _a1_model
    if _a1_model is None:
        from inline.a1_resnet import resnet50

        ckpt_path = _ensure_a1_ckpt()
        log.info("loading A1 model from %s", ckpt_path)
        model = resnet50(pretrained=False)
        model.change_output(1)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload.get("model_state_dict", payload.get("model_state", payload))
        if isinstance(sd, dict):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        _a1_model = model
    return _a1_model


def predict_a1(image: Image.Image | None) -> str:
    if image is None:
        return "(no image)"
    model = _get_a1_model()
    x = A1_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).squeeze().item()
        prob = float(torch.sigmoid(torch.tensor(logit)))
    verdict = "FAKE / AI-generated" if prob >= 0.5 else "REAL / authentic"
    return (
        f"**P(fake) = {prob:.4f}**\n\n"
        f"Verdict: **{verdict}** (threshold 0.5)\n\n"
        f"_Detector: Approach 1 — RGB + Fourier mask (FakeImageDetection ResNet50)._"
    )


# --- Approach 2 ----------------------------------------------------------


def _load_dct_stats(stats_path: Path) -> tuple[float, float]:
    obj = json.loads(stats_path.read_text(encoding="utf-8"))
    return float(obj["mean"]), float(obj["std"])


def _get_a2_model():
    global _a2_model, _a2_mean, _a2_std
    if _a2_model is None:
        ckpt_path, stats_path = _ensure_a2_files()
        log.info("loading A2 model from %s", ckpt_path)
        model = build_dct_resnet50(pretrained=False)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(sd, strict=True)
        model.eval()
        _a2_model = model
        _a2_mean, _a2_std = _load_dct_stats(stats_path)
        log.info("DCT stats: mean=%.4f std=%.4f", _a2_mean, _a2_std)
    return _a2_model


def predict_a2(image: Image.Image | None) -> str:
    if image is None:
        return "(no image)"
    model = _get_a2_model()
    t = compute_dual_dct(image.convert("RGB"))
    t = (t - _a2_mean) / (_a2_std + 1e-8)
    t = t.unsqueeze(0)
    with torch.no_grad():
        logit = model(t).squeeze().item()
        prob = float(torch.sigmoid(torch.tensor(logit)))
    verdict = "FAKE / AI-generated" if prob >= 0.5 else "REAL / authentic"
    return (
        f"**P(fake) = {prob:.4f}**\n\n"
        f"Verdict: **{verdict}** (threshold 0.5)\n\n"
        f"_Detector: Approach 2 — Dual-DCT (8x8 + 16x16 patches, ResNet50 1-ch conv1)._"
    )


# --- V4 5-class ----------------------------------------------------------


def _build_v4_pipeline():
    fusion_ckpt, fndclip_ckpt, dct_ckpt = _ensure_v4_ckpts()

    log.info("[V4] building FND-CLIP V1 (semantic, 512-d)...")
    fndclip = FNDCLIPSemanticEncoder(ckpt=fndclip_ckpt)
    fndclip.eval()
    for p in fndclip.parameters():
        p.requires_grad = False

    log.info("[V4] building DCT-Forensic encoder (image, 768-d)...")
    dct_forensic = DctForensicEncoder(out_dim=768, ckpt=dct_ckpt)
    dct_forensic.eval()
    for p in dct_forensic.parameters():
        p.requires_grad = False

    log.info("[V4] building Qwen2-7B-Instruct (text, 3584-d)...")
    qwen = Qwen2TextEncoder(load_backbone=False)

    log.info("[V4] building V3PairwiseFusion + loading fusion ckpt...")
    fusion = V3PairwiseFusion(
        feat_dim=768,
        fused_dim=1024,
        num_classes=5,
        proj_dims={"v_semantic": 512, "v_textfor": 3584},
    )
    payload = torch.load(fusion_ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    missing, unexpected = fusion.load_state_dict(state, strict=False)
    log.info("[V4] fusion loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))
    fusion.eval()
    for p in fusion.parameters():
        p.requires_grad = False

    if _a2_mean is None or _a2_std is None:
        _, stats_path = _ensure_a2_files()
        a2_mean, a2_std = _load_dct_stats(stats_path)
    else:
        a2_mean, a2_std = _a2_mean, _a2_std

    return {
        "fndclip": fndclip,
        "dct_forensic": dct_forensic,
        "qwen": qwen,
        "fusion": fusion,
        "dct_mean": a2_mean,
        "dct_std": a2_std,
    }


@spaces.GPU(duration=180)
def _v4_forward(image: Image.Image, text: str) -> dict[int, float]:
    global _v4_pipeline
    if _v4_pipeline is None:
        _v4_pipeline = _build_v4_pipeline()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[V4] forward on device=%s", device)

    pl = _v4_pipeline
    fndclip = pl["fndclip"].to(device)
    dct_enc = pl["dct_forensic"].to(device)
    qwen = pl["qwen"]
    fusion = pl["fusion"].to(device)

    fnd_batch = prepare_fnd_inputs(text, image.convert("RGB"))
    fnd_batch = {k: v.to(device) for k, v in fnd_batch.items()}
    with torch.no_grad():
        v_semantic = fndclip(fnd_batch)

    t = compute_dual_dct(image.convert("RGB"))
    t = (t - pl["dct_mean"]) / (pl["dct_std"] + 1e-8)
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        v_imgfor = dct_enc({"v_imgfor_dct": t})

    v_textfor = qwen.encode_text(text).to(device)

    with torch.no_grad():
        out = fusion({
            "v_semantic": v_semantic,
            "v_imgfor": v_imgfor,
            "v_textfor": v_textfor,
        })
        probs = F.softmax(out["main_logits"], dim=-1).squeeze(0).cpu().tolist()

    return {i: float(p) for i, p in enumerate(probs)}


V4_CPU_DEGRADED_MSG = """
### V4 5-class — live inference requires GPU

The V4 multimodal pipeline calls Qwen2-7B-Instruct (15 GB fp16 weights),
which doesn't fit on this Space's CPU-basic tier (16 GB RAM). To enable
live inference, upgrade the Space to **ZeroGPU** (requires HuggingFace Pro)
or **A10G Small**.

In the meantime, the trained model checkpoint is available at
[`FerasMad/multiguard-v4-fusion`](https://huggingface.co/FerasMad/multiguard-v4-fusion)
and the offline evaluation numbers are below.

#### Offline eval (FSOS seed=42 retrain)

| Split | F1-macro | Notes |
|---|---|---|
| Val (in-distribution) | **0.7334** | V3 baseline 0.7215 (+1.19 pp) |
| Test (in-distribution) | **0.7267** | First V3.1 §7 5-class test |
| MMFakeBench transfer | **0.4805** | V3 baseline 0.3832 (+9.7 pp) |

#### Per-class test F1

| Class | Display | F1 | Notes |
|---|---|---|---|
| 0 | Real news | 0.41 | Confused with Out-of-context |
| 1 | Out-of-context | 0.47 | Confused with Real news |
| 2 | Real text + Fake image | 0.76 | DGM4 + MMFakeBench |
| 3 | Fake text + Real image | **0.997** | V3 caption-shortcut persists |
| 4 | Fully fake | **0.998** | Same shortcut |

The two **forensic detector tabs** (A1, A2) on this Space DO work on CPU —
try them out on any image.
"""


def predict_v4(image: Image.Image | None, text: str | None):
    """Try the V4 pipeline; on failure (OOM, no GPU, no Pro), show static numbers."""
    if image is None:
        return None, "(please upload an image)"
    if not text or not text.strip():
        return None, "(please paste the news caption / claim)"
    try:
        probs = _v4_forward(image, text.strip())
    except Exception as e:  # noqa: BLE001
        log.exception("V4 forward failed — degrading to static numbers")
        return None, V4_CPU_DEGRADED_MSG + f"\n\n_Underlying error (for debug): `{type(e).__name__}: {e}`_"

    labels = {DISPLAY_LABELS[i]: probs[i] for i in range(5)}
    top_id = max(probs, key=probs.get)
    top_label = DISPLAY_LABELS[top_id]
    explanation = LABEL_EXPLANATIONS[top_id]
    summary = (
        f"### Top prediction: **{top_label}** ({probs[top_id]:.1%})\n\n"
        f"_{explanation}_\n\n"
        f"---\n\n"
        f"Scored by the 3-branch V3.1 pipeline: FND-CLIP (cross-modal consistency), "
        f"DCT-Forensic (image authenticity), Qwen2-7B-Instruct (text credibility) "
        f"→ attention fusion → MLP classifier."
    )
    return labels, summary


# --- Gradio UI ----------------------------------------------------------

INTRO_MD = """
# 🛡️ MultiGuard — multimodal fake-news detector

Live demo for the V3.1 implementation guidelines (3-branch architecture)
plus the two binary forensic image detectors from the Forensic Image Detector brief.

**Three tabs:**
1. **Forensic — RGB + Fourier (Approach 1)** — single-image binary detector.
2. **Forensic — Dual-DCT (Approach 2)** — single-image binary detector.
3. **5-class fake-news (V4)** — multimodal (image + caption) → 5-way classification.

Code + checkpoints: [github.com/FerasMad/MultiGuard](https://github.com/FerasMad/MultiGuard).
"""

with gr.Blocks(title="MultiGuard — fake-news detector") as demo:
    gr.Markdown(INTRO_MD)

    with gr.Tab("Forensic — RGB + Fourier (Approach 1)"):
        gr.Markdown(
            "**Binary AI-image detector.** Upload an image — the model reports "
            "P(fake) and a verdict (threshold 0.5).\n\n"
            "_Per-generator AP averaged across 6 generators: **0.9979** (StdDev 0.0023)._"
        )
        with gr.Row():
            a1_image = gr.Image(type="pil", label="News image")
            a1_output = gr.Markdown(label="Result")
        gr.Button("Detect", variant="primary").click(predict_a1, inputs=a1_image, outputs=a1_output)

    with gr.Tab("Forensic — Dual-DCT (Approach 2)"):
        gr.Markdown(
            "**Binary AI-image detector** using frequency-domain input.\n\n"
            "_Dual-patch DCT (8x8 + 16x16) → ResNet50 with 1-channel conv1. "
            "Per-generator AP averaged across 6 generators: **0.9863** (StdDev 0.0158)._"
        )
        with gr.Row():
            a2_image = gr.Image(type="pil", label="News image")
            a2_output = gr.Markdown(label="Result")
        gr.Button("Detect", variant="primary").click(predict_a2, inputs=a2_image, outputs=a2_output)

    with gr.Tab("5-class fake-news (V4)"):
        gr.Markdown(
            "**Multimodal 5-class detector.** Upload an image AND paste the news caption / claim.\n\n"
            "Categories:\n"
            "- **Real news** — both genuine\n"
            "- **Out-of-context** — both real but mismatched\n"
            "- **Real text + Fake image** — text real, image tampered / AI-generated\n"
            "- **Fake text + Real image** — image real, text AI-generated / fabricated\n"
            "- **Fully fake** — both fabricated\n\n"
            "_First call ~30 s (loads Qwen2-7B-Instruct on GPU). Subsequent ~5 s._\n\n"
            "_Reported numbers (FSOS seed=42): val F1=0.7334, test F1=0.7267, "
            "MMFakeBench transfer F1=0.4805._"
        )
        with gr.Row():
            with gr.Column():
                v4_image = gr.Image(type="pil", label="News image")
                v4_text = gr.Textbox(
                    lines=3, label="News caption / claim",
                    placeholder="e.g. 'Saudi Arabia announces a new electric vehicle plant in Riyadh.'",
                )
                v4_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                v4_probs = gr.Label(num_top_classes=5, label="5-class probabilities")
                v4_summary = gr.Markdown(label="Summary")
        v4_btn.click(predict_v4, inputs=[v4_image, v4_text], outputs=[v4_probs, v4_summary])

    gr.Markdown(
        "---\n"
        "Built by Feras Madkhali (KSU CSC 429). "
        "Models on HF Hub: "
        "[`forensic-rgb-v1`](https://huggingface.co/FerasMad/forensic-rgb-v1) · "
        "[`forensic-dct-v1`](https://huggingface.co/FerasMad/forensic-dct-v1) · "
        "[`multiguard-v4-fusion`](https://huggingface.co/FerasMad/multiguard-v4-fusion). "
        "MIT license."
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
    )

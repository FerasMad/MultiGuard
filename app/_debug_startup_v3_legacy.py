"""Debug helper - step through app/server.py startup to find the crash point."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def step(label):
    print(f"[step] {label}", flush=True)


try:
    step("1: import torch")
    import torch

    print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    step("2: import fastapi + uvicorn")

    step("3: import transformers")

    step("4: import models.fnd_clip")
    from models.fnd_clip import FNDCLIP

    step("5: import models.text_fluoroscopy")

    step("6: import models.univfd_encoder")
    from models.univfd_encoder import UnivFDEncoder

    step("7: import models.v3_pipeline")

    step("8: build FNDCLIP")
    fnd_clip = FNDCLIP(feat_dim=512, num_classes=1)

    step("9: load FND-CLIP ckpt")
    t0 = time.time()
    FND_CKPT = ROOT / "outputs" / "v1_leakfree" / "best.pt"
    ck = torch.load(FND_CKPT, map_location="cpu", weights_only=False)
    print(f"  loaded in {time.time() - t0:.1f}s  keys={list(ck.keys())}", flush=True)

    step("10: load_state_dict")
    state = ck.get("model_state", ck)
    compat = {
        k: v
        for k, v in state.items()
        if k in fnd_clip.state_dict() and fnd_clip.state_dict()[k].shape == v.shape
    }
    fnd_clip.load_state_dict(compat, strict=False)
    print(f"  loaded {len(compat)} tensors", flush=True)

    step("11: move to cuda")
    fnd_clip = fnd_clip.to("cuda").eval()
    for p in fnd_clip.parameters():
        p.requires_grad = False
    print(f"  vram={torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    step("12: build + load UnivFD")
    UNIVFD_CKPT = ROOT / "outputs" / "univfd_genimage" / "best.pt"
    univfd = UnivFDEncoder(out_dim=768, pretrained=None, dropout=0.3)
    univfd_ck = torch.load(UNIVFD_CKPT, map_location="cpu", weights_only=False)
    state = univfd_ck.get("model_state", univfd_ck)
    enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    univfd.load_state_dict(enc_state, strict=False)
    univfd = univfd.to("cuda").eval()
    print(f"  vram={torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    step("13: ALL OK up to UnivFD")

except SystemExit:
    print("SystemExit raised", flush=True)
except BaseException as e:
    print(f"!!! exception {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

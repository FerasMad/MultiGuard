"""Test-Time Augmentation for V4 5-class (P9.6).

For each input, runs inference on BOTH the original and a horizontally-flipped
copy of the image, then averages the encoder outputs. Within V3.1 spec because
§7 doesn't forbid TTA at inference.
"""

from __future__ import annotations

import torch
from PIL import Image


def horizontal_flip_dct(
    pil_image: Image.Image, compute_dual_dct_fn, dct_mean: float, dct_std: float
) -> torch.Tensor:
    flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    t = compute_dual_dct_fn(flipped)
    t = (t - dct_mean) / (dct_std + 1e-8)
    return t.unsqueeze(0)


@torch.no_grad()
def average_v_imgfor(
    dct_encoder, t_original: torch.Tensor, t_flipped: torch.Tensor
) -> torch.Tensor:
    v1 = dct_encoder({"v_imgfor_dct": t_original})
    v2 = dct_encoder({"v_imgfor_dct": t_flipped})
    return 0.5 * (v1 + v2)


@torch.no_grad()
def average_v_semantic(
    fnd_encoder,
    fnd_inputs_original: dict[str, torch.Tensor],
    pil_image: Image.Image,
    prepare_fnd_fn,
) -> torch.Tensor:
    flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    fnd_flipped = prepare_fnd_fn(text="", pil_img=flipped)
    fnd_flipped_merged = dict(fnd_inputs_original)
    fnd_flipped_merged["image"] = fnd_flipped["image"].to(fnd_inputs_original["image"].device)
    fnd_flipped_merged["clip_pixels"] = fnd_flipped["clip_pixels"].to(
        fnd_inputs_original["clip_pixels"].device
    )
    v1 = fnd_encoder(fnd_inputs_original)
    v2 = fnd_encoder(fnd_flipped_merged)
    return 0.5 * (v1 + v2)


def tta_predict(
    fnd_encoder,
    dct_encoder,
    qwen_encoder,
    fusion,
    pil_image: Image.Image,
    text: str,
    *,
    compute_dual_dct_fn,
    prepare_fnd_fn,
    dct_mean: float,
    dct_std: float,
    device: torch.device,
):
    import torch.nn.functional as F

    fnd_orig = prepare_fnd_fn(text=text, pil_img=pil_image)
    fnd_orig = {k: v.to(device) for k, v in fnd_orig.items()}
    v_semantic = average_v_semantic(fnd_encoder, fnd_orig, pil_image, prepare_fnd_fn)

    t_orig = compute_dual_dct_fn(pil_image)
    t_orig = ((t_orig - dct_mean) / (dct_std + 1e-8)).unsqueeze(0).to(device)
    t_flip = horizontal_flip_dct(pil_image, compute_dual_dct_fn, dct_mean, dct_std).to(device)
    v_imgfor = average_v_imgfor(dct_encoder, t_orig, t_flip)

    v_textfor = qwen_encoder.encode_text(text).to(device)

    out = fusion(
        {
            "v_semantic": v_semantic,
            "v_imgfor": v_imgfor,
            "v_textfor": v_textfor,
        }
    )
    return F.softmax(out["main_logits"], dim=-1).squeeze(0)

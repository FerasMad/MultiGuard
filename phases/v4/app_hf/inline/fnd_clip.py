"""FND-CLIP V1 semantic encoder — inline copy that matches V1's exact param naming.

Faithful re-implementation of `phases/v3/src/models/fnd_clip.py` (which mirrors
`phases/v1/src/models/fnd_clip.py`). The V1 leakfree ckpt was trained with
THIS exact architecture, and the cached `v_semantic` features in
`cache/v3/_features/v_semantic/*.pt` were produced by it. Earlier versions of
this inline file used V4-style naming (visual.backbone.*, visual.proj, single-
linear ModalityAttention) which caused 273/929 weights to fall back to random
init at server load time — that bug is what the parity sweep caught.

Key naming (must match V1 ckpt exactly):
  visual.features.0..X          -- ResNet50 wrapped as nn.Sequential
  visual.project.{weight,bias}  -- Linear(2048, 512)
  text.bert.*                   -- HF BertModel
  text.project.{weight,bias}    -- Linear(768, 512)
  clip.clip.*                   -- HF CLIPModel
  clip_project.{weight,bias}    -- Linear(1024, 512)
  attention.scorer.{0,2}.*      -- nn.Sequential(Linear*3 -> Tanh -> Linear*3)
  classifier.{0,3}.*            -- 4-layer head (unused at inference)

Output of forward_semantic: [B, 512] — same as V1. The fusion's sem_proj
(Linear 512->768 + GELU, lives inside V3PairwiseFusion) handles the projection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)


def _require_semantic_weights_loaded(missing, *, allowed_missing_prefixes=("classifier.",),
                                     ckpt_name: str = "") -> None:
    """Raise if a backbone/attention weight failed to load (silent parity break).

    Self-contained mirror of v4.core.checkpoints.require_semantic_weights_loaded
    (this file is copied verbatim into the HF Space, so it can't import from v4).
    classifier.* is the V1 binary head and is legitimately absent; anything else
    missing means the encoder is running at random init -- the exact bug the
    parity sweep caught -- so fail loudly here instead of mispredicting at serve time.
    """
    critical = [k for k in missing if not k.startswith(tuple(allowed_missing_prefixes))]
    if critical:
        where = f" from {ckpt_name}" if ckpt_name else ""
        raise RuntimeError(
            f"FND-CLIP: {len(critical)} semantic-path weight(s) failed to load{where} "
            f"(first 5: {critical[:5]}). The encoder would run at random init and break "
            "train/serve parity. Check that the checkpoint's parameter names match."
        )


class VisualStream(nn.Module):
    """ResNet50 wrapped as nn.Sequential (V1 layout: visual.features.0..)."""

    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        # children()[:-1] drops the fc head; len == 9 modules
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.project = nn.Linear(2048, out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image).flatten(1)
        return self.project(x)


class TextStream(nn.Module):
    """BERT-base CLS pooled -> Linear(768, out_dim). Uses `project` not `proj`."""

    def __init__(self, out_dim: int = 512, pretrained: str = "bert-base-uncased"):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(pretrained)
        self.project = nn.Linear(768, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.project(cls)


class CLIPStream(nn.Module):
    """CLIP-vit-base-patch32 paired encoders + cosine-similarity gate."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPModel

        self.clip = CLIPModel.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        img_pooled = (
            vision_out.pooler_output
            if hasattr(vision_out, "pooler_output")
            else vision_out.last_hidden_state[:, 0]
        )
        img_feat = self.clip.visual_projection(img_pooled)

        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_pooled = (
            text_out.pooler_output
            if hasattr(text_out, "pooler_output")
            else text_out.last_hidden_state[:, 0]
        )
        txt_feat = self.clip.text_projection(txt_pooled)

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        sim = (img_feat * txt_feat).sum(dim=-1)
        fused = torch.cat([img_feat, txt_feat], dim=-1)
        return fused, sim


class ModalityAttention(nn.Module):
    """V1's 3-layer modality-attention scorer.

    Concatenates the three [B, feat_dim] streams -> [B, 3*feat_dim],
    runs through Linear -> Tanh -> Linear -> softmax over 3, then takes
    a weighted sum of the original streams.

    NOTE: param path is `attention.scorer.{0,2}.{weight,bias}` to match V1 ckpt.
    The earlier inline version used a single `Linear(feat_dim, 1)` named `score`
    which is a completely different architecture — that was the parity bug.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.Tanh(),
            nn.Linear(feat_dim, 3),
        )

    def forward(self, text_feat, image_feat, clip_feat):
        stack = torch.stack([text_feat, image_feat, clip_feat], dim=1)
        concat = stack.flatten(1)
        weights = F.softmax(self.scorer(concat), dim=-1)
        weighted = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return weighted, weights


class FNDCLIPSemanticEncoder(nn.Module):
    """V1 FND-CLIP encoder. Outputs 512-d native — fusion's sem_proj handles 512->768.

    Architecture + param paths match V1 ckpt (`outputs/v1/leakfree/best.pt`)
    exactly. All 929 weights load with `strict=False` cleanly (missing=0).
    """

    def __init__(
        self,
        feat_dim: int = 512,
        bert_name: str = "bert-base-uncased",
        clip_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 5,  # V1 trained as binary OOC (1) or 5-class
        ckpt: str | Path | None = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.visual = VisualStream(out_dim=feat_dim)
        self.text = TextStream(out_dim=feat_dim, pretrained=bert_name)
        self.clip = CLIPStream(model_name=clip_name)
        self.clip_project = nn.Linear(1024, feat_dim)
        self.attention = ModalityAttention(feat_dim)

        # Classifier head — V1 layout uses indices 0,3 (Linear, Linear with
        # ReLU+Dropout in between). Inference doesn't use this; it's kept for
        # ckpt key compatibility only.
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

        self.output_dim = feat_dim
        if ckpt:
            self.load_legacy_checkpoint(ckpt)

    def load_legacy_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck

        # Shape-compat filter: skip keys whose shape doesn't match (in case the
        # classifier was trained for a different num_classes, etc.)
        target = self.state_dict()
        compat = {k: v for k, v in state.items()
                  if k in target and target[k].shape == v.shape}
        missing, unexpected = self.load_state_dict(compat, strict=False)
        log.info(
            "FND-CLIP loaded %d/%d tensors from %s (missing=%d, unexpected=%d)",
            len(compat),
            len(target),
            p.name,
            len(missing),
            len(unexpected),
        )
        _require_semantic_weights_loaded(
            missing, allowed_missing_prefixes=("classifier.",), ckpt_name=p.name
        )

    def forward_semantic(
        self,
        *,
        image: torch.Tensor,
        bert_ids: torch.Tensor,
        bert_mask: torch.Tensor,
        clip_pixels: torch.Tensor,
        clip_ids: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Identical to V1's forward_semantic. Returns [B, feat_dim] post-attention."""
        image_feat = self.visual(image)
        text_feat = self.text(bert_ids, bert_mask)
        clip_fused, clip_sim = self.clip(clip_pixels, clip_ids, clip_mask)
        clip_feat = self.clip_project(clip_fused)
        clip_feat = clip_feat * clip_sim.unsqueeze(-1)
        weighted, _ = self.attention(text_feat, image_feat, clip_feat)
        return weighted

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_semantic(
            image=batch["image"],
            bert_ids=batch["bert_ids"],
            bert_mask=batch["bert_mask"],
            clip_pixels=batch["clip_pixels"],
            clip_ids=batch["clip_ids"],
            clip_mask=batch["clip_mask"],
        )


def prepare_fnd_inputs(text: str, pil_img, *, bert_max_len: int = 128, clip_max_len: int = 77) -> dict:
    """Build the dict FNDCLIPSemanticEncoder.forward_semantic expects.

    Match the V1 / V3 preprocessing exactly — IMPORTANT for parity with the
    cached v_semantic features. Source of truth:
        phases/v3/scripts/precompute_v_semantic.py (or equivalent V1 cache build).
    """
    from torchvision import transforms
    from transformers import BertTokenizer, CLIPProcessor

    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # V1 used Resize(256) + CenterCrop(224) per the FND-CLIP paper. ImageNet
    # mean/std normalization is critical (RN50 expects it).
    image_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_tensor = image_tf(pil_img)
    bert = bert_tok(text, padding="max_length", truncation=True,
                    max_length=bert_max_len, return_tensors="pt")
    clip = clip_proc(images=pil_img, text=text, return_tensors="pt",
                     padding="max_length", truncation=True, max_length=clip_max_len)
    return {
        "image": img_tensor.unsqueeze(0),
        "bert_ids": bert["input_ids"],
        "bert_mask": bert["attention_mask"],
        "clip_pixels": clip["pixel_values"],
        "clip_ids": clip["input_ids"],
        "clip_mask": clip["attention_mask"],
    }

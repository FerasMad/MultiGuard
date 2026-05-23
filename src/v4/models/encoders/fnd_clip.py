"""FND-CLIP V1 semantic encoder per V3.1 section 1 Stage 1 (Student 4 Stage 0).

Per the Dataset+V1 PDF, FND-CLIP comprises:
  - ResNet-50 visual stream
  - BERT-base-uncased textual stream
  - Two CLIP encoders (image + text)
  - CLIP cross-modal alignment via cosine similarity
  - Modality-wise attention over (text, image, CLIP-fused)
  - Binary MLP head for V1 OOC pre-training

Forced deviation F1: published FND-CLIP outputs 512; V3.1 spec table claims
v_semantic = 768. We resolve via a registered sem_proj: Linear(512, 768) + GELU
adapter included as part of this encoder so output_dim = 768.

Input:  image + bert_ids + bert_mask + clip_pixels + clip_ids + clip_mask
Output: v_semantic [B, 768]

Faithful re-implementation of FND-CLIP from Zhou et al., IEEE ICME 2023.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from v4.core.checkpoints import shape_compat_filter
from v4.core.logging import get_logger
from v4.core.registry import ENCODER_REGISTRY, register
from v4.models.encoders.base import EncoderBase

log = get_logger(__name__)


class VisualStream(nn.Module):
    """ResNet-50 -> Linear(2048, feat_dim). Pretrained on ImageNet, fine-tuned in Stage 0."""

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(2048, feat_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(image))


class TextStream(nn.Module):
    """BERT-base-uncased CLS token -> Linear(768, feat_dim)."""

    def __init__(self, feat_dim: int = 512, bert_name: str = "bert-base-uncased"):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, feat_dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)


class CLIPStream(nn.Module):
    """Paired CLIP encoders + cosine-similarity reweighted fusion."""

    def __init__(self, clip_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained(clip_name)
        for p in self.clip.parameters():
            p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.clip.config.projection_dim * 2

    def forward(self, pixels: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor):
        img_emb = self.clip.get_image_features(pixel_values=pixels)
        txt_emb = self.clip.get_text_features(input_ids=ids, attention_mask=mask)
        img_n = F.normalize(img_emb, dim=-1)
        txt_n = F.normalize(txt_emb, dim=-1)
        sim = (img_n * txt_n).sum(dim=-1)
        fused = torch.cat([img_emb, txt_emb], dim=-1)
        return fused, sim


class ModalityAttention(nn.Module):
    """Learned 3-scalar attention over (text, image, clip) streams."""

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, t: torch.Tensor, i: torch.Tensor, c: torch.Tensor):
        st = self.score(t)
        si = self.score(i)
        sc = self.score(c)
        scores = torch.cat([st, si, sc], dim=-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        stacked = torch.stack([t, i, c], dim=1)
        weighted = (weights * stacked).sum(dim=1)
        return weighted, weights.squeeze(-1)


@register(ENCODER_REGISTRY, "fnd_clip")
class FNDCLIPSemanticEncoder(EncoderBase):
    """FND-CLIP V1 semantic encoder + sem_proj(512 -> 768) per V3.1 section 1, deviation F1."""

    required_inputs = ("image", "bert_ids", "bert_mask", "clip_pixels", "clip_ids", "clip_mask")
    modality = "multi"

    def __init__(
        self,
        feat_dim: int = 512,
        output_dim: int = 768,
        bert_name: str = "bert-base-uncased",
        clip_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 1,
        ckpt: str | Path | None = None,
        **_kwargs,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        self.visual = VisualStream(feat_dim=feat_dim)
        self.text = TextStream(feat_dim=feat_dim, bert_name=bert_name)
        self.clip = CLIPStream(clip_name=clip_name)
        self.clip_project = nn.Linear(self.clip.output_dim, feat_dim)
        self.attention = ModalityAttention(feat_dim=feat_dim)

        # Stage-0 binary head (V1 fine-tune target)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

        # F1 deviation: project 512 -> 768 for V3 fusion compatibility
        self.sem_proj = nn.Sequential(
            nn.Linear(feat_dim, output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

        if ckpt:
            self.load_legacy_checkpoint(ckpt)

    def load_legacy_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        state = ck.get("model_state", ck)
        compat = shape_compat_filter(state, self.state_dict())
        missing, unexpected = self.load_state_dict(compat, strict=False)
        log.info(
            "FND-CLIP loaded %d/%d tensors from %s (missing=%d, unexpected=%d)",
            len(compat), len(self.state_dict()), p.name, len(missing), len(unexpected),
        )

    def forward_semantic(
        self, *,
        image: torch.Tensor, bert_ids: torch.Tensor, bert_mask: torch.Tensor,
        clip_pixels: torch.Tensor, clip_ids: torch.Tensor, clip_mask: torch.Tensor,
    ) -> torch.Tensor:
        v_img = self.visual(image)
        v_txt = self.text(bert_ids, bert_mask)
        fused_clip, sim = self.clip(clip_pixels, clip_ids, clip_mask)
        v_clip = self.clip_project(fused_clip) * sim.unsqueeze(-1)
        weighted, _ = self.attention(v_txt, v_img, v_clip)
        return weighted

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        v_sem_native = self.forward_semantic(
            image=batch["image"],
            bert_ids=batch["bert_ids"], bert_mask=batch["bert_mask"],
            clip_pixels=batch["clip_pixels"],
            clip_ids=batch["clip_ids"], clip_mask=batch["clip_mask"],
        )
        return self.sem_proj(v_sem_native)

    def forward_for_stage0(self, **batch_kwargs) -> dict[str, torch.Tensor]:
        """Stage-0 fine-tune forward: returns logits for the binary OOC loss."""
        v_sem = self.forward_semantic(**batch_kwargs)
        logits = self.classifier(v_sem)
        return {"logits": logits, "v_semantic": v_sem}

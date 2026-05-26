"""FND-CLIP V1 semantic encoder — inline copy for the HF Space.

Mirrors phases/v4/src/v4/models/encoders/fnd_clip.py with v4.* imports stripped
and the @register decorator dropped (instantiated directly in app.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)


def _shape_compat_filter(source_state: dict, target_state: dict) -> dict:
    return {
        k: v
        for k, v in source_state.items()
        if k in target_state and target_state[k].shape == v.shape
    }


class VisualStream(nn.Module):
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(2048, feat_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(image))


class TextStream(nn.Module):
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
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, t, i, c):
        st = self.score(t)
        si = self.score(i)
        sc = self.score(c)
        scores = torch.cat([st, si, sc], dim=-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        stacked = torch.stack([t, i, c], dim=1)
        weighted = (weights * stacked).sum(dim=1)
        return weighted, weights.squeeze(-1)


class FNDCLIPSemanticEncoder(nn.Module):
    """FND-CLIP V1 encoder. Outputs 512-d native feature (the fusion's sem_proj projects to 768)."""

    def __init__(
        self,
        feat_dim: int = 512,
        bert_name: str = "bert-base-uncased",
        clip_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 1,
        ckpt: str | Path | None = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.visual = VisualStream(feat_dim=feat_dim)
        self.text = TextStream(feat_dim=feat_dim, bert_name=bert_name)
        self.clip = CLIPStream(clip_name=clip_name)
        self.clip_project = nn.Linear(self.clip.output_dim, feat_dim)
        self.attention = ModalityAttention(feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )
        self.output_dim = feat_dim
        if ckpt:
            self.load_legacy_checkpoint(ckpt)

    def load_legacy_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        compat = _shape_compat_filter(state, self.state_dict())
        missing, unexpected = self.load_state_dict(compat, strict=False)
        log.info(
            "FND-CLIP loaded %d/%d tensors from %s (missing=%d, unexpected=%d)",
            len(compat),
            len(self.state_dict()),
            p.name,
            len(missing),
            len(unexpected),
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
        v_img = self.visual(image)
        v_txt = self.text(bert_ids, bert_mask)
        fused_clip, sim = self.clip(clip_pixels, clip_ids, clip_mask)
        v_clip = self.clip_project(fused_clip) * sim.unsqueeze(-1)
        weighted, _ = self.attention(v_txt, v_img, v_clip)
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
    """Build the dict FNDCLIPSemanticEncoder.forward_semantic expects."""
    from torchvision import transforms
    from transformers import BertTokenizer, CLIPProcessor

    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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

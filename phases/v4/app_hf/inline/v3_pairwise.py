"""V3 Pairwise Cross-Attention Fusion — inline copy for the HF Space.

Mirrors phases/v4/src/v4/models/fusion/v3_pairwise.py.
"""

from __future__ import annotations

import torch
from torch import nn

from .mlp_head import MLPClassifier


class PairwiseCrossAttention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_x_to_y = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_y_to_x = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(1)
        y_seq = y.unsqueeze(1)
        dir1, _ = self.attn_x_to_y(query=x_seq, key=y_seq, value=y_seq)
        dir2, _ = self.attn_y_to_x(query=y_seq, key=x_seq, value=x_seq)
        return self.norm(dir1.squeeze(1) + dir2.squeeze(1))


class ThreeWayFusion(nn.Module):
    def __init__(self, feat_dim: int = 768, fused_dim: int = 1024,
                 num_heads: int = 8, attn_dropout: float = 0.1):
        super().__init__()
        self.ln_semantic = nn.LayerNorm(feat_dim)
        self.ln_imgfor = nn.LayerNorm(feat_dim)
        self.ln_textfor = nn.LayerNorm(feat_dim)
        self.pair_sem_img = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)
        self.pair_sem_text = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)
        self.pair_img_text = PairwiseCrossAttention(feat_dim, num_heads, attn_dropout)
        self.conv1 = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(feat_dim, fused_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, v_semantic, v_imgfor, v_textfor):
        s = self.ln_semantic(v_semantic)
        i = self.ln_imgfor(v_imgfor)
        t = self.ln_textfor(v_textfor)
        r1 = self.pair_sem_img(s, i)
        r2 = self.pair_sem_text(s, t)
        r3 = self.pair_img_text(i, t)
        stacked = torch.stack([r1, r2, r3], dim=1)
        x = stacked.permute(0, 2, 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.pool(x).squeeze(-1)


class V3PairwiseFusion(nn.Module):
    """V3.1 sec 5-6 fusion + classifier head + sec 5.5 aux head."""

    _DISABLE_KEYS = frozenset({"v_semantic", "v_imgfor", "v_textfor"})

    def __init__(
        self,
        feat_dim: int = 768,
        fused_dim: int = 1024,
        num_classes: int = 5,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        proj_dims: dict | None = None,
        disable_branches: list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.fused_dim = fused_dim
        self.num_classes = num_classes

        # Mirror src/v4/models/fusion/v3_pairwise.py Stage-B ablation hook.
        disable_branches = list(disable_branches or [])
        unknown = set(disable_branches) - self._DISABLE_KEYS
        if unknown:
            raise ValueError(
                f"disable_branches contains unknown keys {sorted(unknown)}; "
                f"valid keys are {sorted(self._DISABLE_KEYS)}"
            )
        self.disable_branches: frozenset[str] = frozenset(disable_branches)

        proj_dims = proj_dims or {}
        self.sem_proj = self._build_proj(proj_dims.get("v_semantic"), feat_dim)
        self.img_proj = self._build_proj(proj_dims.get("v_imgfor"), feat_dim)
        self.text_proj = self._build_proj(proj_dims.get("v_textfor"), feat_dim)

        self.fusion = ThreeWayFusion(
            feat_dim=feat_dim, fused_dim=fused_dim,
            num_heads=num_heads, attn_dropout=attn_dropout,
        )
        self.classifier = MLPClassifier(in_dim=fused_dim, num_classes=num_classes)
        self.aux_classifier = nn.Linear(feat_dim, 2)

    @staticmethod
    def _build_proj(in_dim: int | None, out_dim: int) -> nn.Module:
        if in_dim is None or int(in_dim) == int(out_dim):
            return nn.Identity()
        return nn.Sequential(nn.Linear(int(in_dim), int(out_dim)), nn.GELU())

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        v_semantic = self.sem_proj(features["v_semantic"])
        v_imgfor = self.img_proj(features["v_imgfor"])
        v_textfor = self.text_proj(features["v_textfor"])
        if self.disable_branches:
            if "v_semantic" in self.disable_branches:
                v_semantic = torch.zeros_like(v_semantic)
            if "v_imgfor" in self.disable_branches:
                v_imgfor = torch.zeros_like(v_imgfor)
            if "v_textfor" in self.disable_branches:
                v_textfor = torch.zeros_like(v_textfor)
        fused = self.fusion(v_semantic, v_imgfor, v_textfor)
        main_logits = self.classifier(fused)
        aux_logits = self.aux_classifier(v_imgfor.detach())
        return {"main_logits": main_logits, "aux_logits": aux_logits, "fused": fused}

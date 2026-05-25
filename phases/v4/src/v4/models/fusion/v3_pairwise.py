"""V3 Pairwise Cross-Attention Fusion per V3.1 section 5 (Student 3).

Per V3.1 spec 5.1-5.4:
  5.1  LayerNorm per signal (v_semantic, v_imgfor, v_textfor)
  5.2  Three pairwise channels:
         Pair 1: semantic <-> image-forensic
         Pair 2: semantic <-> text-forensic
         Pair 3: image-forensic <-> text-forensic
  5.3  Bidirectional cross-attention per pair (MHA, 8 heads), directions SUMMED (not concat)
  5.4  Stack three relational vectors -> [B, 3, 768] -> permute [B, 768, 3]
       Conv1d(768, 768, k=3, p=1) + GELU -> Conv1d(768, 1024, k=1) + GELU -> AdaptiveAvgPool1d(1)
       Final fused vector: [B, 1024]

Plus V3.1 section 6 classifier MLP (in models/classifier/mlp_head.py).
Plus V3.1 section 5.5 auxiliary binary head with v_imgfor.detach().
"""

from __future__ import annotations

import torch
from torch import nn

from v4.core.registry import FUSION_REGISTRY, register
from v4.models.classifier.mlp_head import MLPClassifier
from v4.models.fusion.base import FusionBase


class PairwiseCrossAttention(nn.Module):
    """Bidirectional cross-attention between two [B, dim] vectors per V3.1 section 5.3.

    Direction 1: x queries y  (Q=x, K=y, V=y)
    Direction 2: y queries x  (Q=y, K=x, V=x)
    Output:      dir1 + dir2  (element-wise SUM, NOT concatenation per spec)
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_x_to_y = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_y_to_x = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(1)
        y_seq = y.unsqueeze(1)
        dir1, _ = self.attn_x_to_y(query=x_seq, key=y_seq, value=y_seq)
        dir2, _ = self.attn_y_to_x(query=y_seq, key=x_seq, value=x_seq)
        dir1 = dir1.squeeze(1)
        dir2 = dir2.squeeze(1)
        return self.norm(dir1 + dir2)


class ThreeWayFusion(nn.Module):
    """Three-signal pairwise xattn + Conv1D integration per V3.1 sections 5.1-5.4."""

    def __init__(
        self,
        feat_dim: int = 768,
        fused_dim: int = 1024,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
    ):
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

    def forward(
        self,
        v_semantic: torch.Tensor,
        v_imgfor: torch.Tensor,
        v_textfor: torch.Tensor,
    ) -> torch.Tensor:
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


@register(FUSION_REGISTRY, "v3_pairwise")
class V3PairwiseFusion(FusionBase):
    """Complete V3 fusion + classifier head per V3.1 sections 5-6 with section 5.5 aux head."""

    expected_inputs = ("v_semantic", "v_imgfor", "v_textfor")

    def __init__(
        self,
        feat_dim: int = 768,
        fused_dim: int = 1024,
        num_classes: int = 5,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        **_kwargs,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.fused_dim = fused_dim
        self.num_classes = num_classes

        self.fusion = ThreeWayFusion(
            feat_dim=feat_dim,
            fused_dim=fused_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )
        self.classifier = MLPClassifier(in_dim=fused_dim, num_classes=num_classes)

        # V3.1 section 5.5 auxiliary binary head on v_imgfor.detach()
        self.aux_classifier = nn.Linear(feat_dim, 2)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        v_semantic = features["v_semantic"]
        v_imgfor = features["v_imgfor"]
        v_textfor = features["v_textfor"]

        fused = self.fusion(v_semantic, v_imgfor, v_textfor)
        main_logits = self.classifier(fused)
        # V3.1 section 5.5: detach v_imgfor to isolate aux loss gradients
        aux_logits = self.aux_classifier(v_imgfor.detach())

        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "fused": fused,
        }

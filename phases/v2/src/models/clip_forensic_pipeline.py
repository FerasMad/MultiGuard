"""V2 architecture, exactly as specified by the user:

  CLIP (frozen, off-the-shelf)  → v_semantic [1024]
  DCT  → ResNet18 (trains)      → v_forensic [768]
                ↓
       Cross-Attention Fusion   → fused [1024]
                ↓
              MLP (3-layer)     → 3 logits

Auxiliary forensic classifier is BINARY (real-image / fake-image), not
3-class — see notes from the project. The forensic encoder cannot tell
OOC from Real because both have real images, so binary is the only honest
auxiliary task for that branch.

CLIP features are precomputed and loaded from disk during training, so
the model itself only holds the trainable parts.
"""

import torch
import torch.nn as nn

from .forensic_baseline import ResNet18Forensic


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention with residual + LayerNorm both sides,
    then concat. Following V2 §7."""

    def __init__(self, sem_dim: int, for_dim: int, proj_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj_sem = nn.Linear(sem_dim, proj_dim)
        self.proj_for = nn.Linear(for_dim, proj_dim)

        self.attn_sem_to_for = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.attn_for_to_sem = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)

        self.norm_sem = nn.LayerNorm(proj_dim)
        self.norm_for = nn.LayerNorm(proj_dim)

    def forward(self, v_semantic: torch.Tensor, v_forensic: torch.Tensor):
        sem = self.proj_sem(v_semantic).unsqueeze(1)   # [B, 1, P]
        forz = self.proj_for(v_forensic).unsqueeze(1)  # [B, 1, P]

        upd_sem, _ = self.attn_sem_to_for(query=sem, key=forz, value=forz)
        upd_sem = self.norm_sem(upd_sem + sem)

        upd_for, _ = self.attn_for_to_sem(query=forz, key=sem, value=sem)
        upd_for = self.norm_for(upd_for + forz)

        return torch.cat([upd_sem.squeeze(1), upd_for.squeeze(1)], dim=-1)


class MLPClassifier(nn.Module):
    """3-layer MLP per V2 §8."""

    def __init__(self, in_dim: int = 1024, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, fused):
        return self.net(fused)


class ClipForensicPipeline(nn.Module):
    """CLIP (precomputed, frozen) + ResNet18 forensic + cross-attention + MLP."""

    def __init__(self,
                 clip_dim: int = 1024,
                 forensic_feat_dim: int = 768,
                 fusion_proj_dim: int = 512,
                 num_classes: int = 3,
                 fusion_heads: int = 8,
                 fusion_dropout: float = 0.1,
                 forensic_dropout: float = 0.3):
        super().__init__()
        self.forensic = ResNet18Forensic(
            pretrained=True, out_dim=forensic_feat_dim,
            dropout=forensic_dropout,
        )
        # Binary aux: real-image vs fake-image. Aux label rules:
        #   main label 0 (Real) or 2 (OOC) -> aux 0 (real image)
        #   main label 1 (Manipulated)     -> aux 1 (fake image)
        self.aux_classifier = nn.Linear(forensic_feat_dim, 2)

        self.fusion = CrossAttentionFusion(
            sem_dim=clip_dim,
            for_dim=forensic_feat_dim,
            proj_dim=fusion_proj_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.classifier = MLPClassifier(
            in_dim=2 * fusion_proj_dim, num_classes=num_classes)

    def forward(self, dct: torch.Tensor, v_semantic: torch.Tensor):
        v_forensic = self.forensic(dct)               # [B, 768]
        aux_logits = self.aux_classifier(v_forensic)  # [B, 2]
        fused = self.fusion(v_semantic, v_forensic)   # [B, 1024]
        main_logits = self.classifier(fused)          # [B, 3]
        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "v_semantic": v_semantic,
            "v_forensic": v_forensic,
            "fused": fused,
        }

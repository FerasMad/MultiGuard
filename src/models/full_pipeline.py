"""Full Step-2 pipeline: frozen FND-CLIP + trainable ResNet18 forensic encoder
+ Cross-Attention Fusion (FCINet 2024 style) + MLP classifier.

Implementation Guidelines V2 §7-9. Returns main + auxiliary logits during
training; only main_logits at inference.

Note: The PDF spec uses 768-dim for v_semantic and v_forensic. We use the
existing pretrained FND-CLIP (feat_dim=512) and match v_forensic to 512.
The cross-attention block still projects both inputs to a common dim
(here 512 — a no-op projection) before attending, and concatenates two
512-dim vectors to give the same 1024-dim fused output specified in §7.
"""

import torch
import torch.nn as nn

from .fnd_clip import FNDCLIP
from .forensic_baseline import ResNet18Forensic


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention with residual + LayerNorm both directions,
    then concatenation. (V2 §7).

    Direction 1: semantic asks forensic   (Q=sem, K=for, V=for)
    Direction 2: forensic asks semantic   (Q=for, K=sem, V=sem)
    """

    def __init__(self, sem_dim: int, for_dim: int, proj_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj_sem = nn.Linear(sem_dim, proj_dim)
        self.proj_for = nn.Linear(for_dim, proj_dim)

        # Two attention modules, one per direction.
        self.attn_sem_to_for = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.attn_for_to_sem = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)

        self.norm_sem = nn.LayerNorm(proj_dim)
        self.norm_for = nn.LayerNorm(proj_dim)

    def forward(self, v_semantic: torch.Tensor, v_forensic: torch.Tensor):
        """v_semantic: [B, sem_dim], v_forensic: [B, for_dim]
        returns: [B, 2*proj_dim] fused vector."""
        sem = self.proj_sem(v_semantic)        # [B, proj_dim]
        forz = self.proj_for(v_forensic)        # [B, proj_dim]

        # Add sequence dim for nn.MultiheadAttention.
        sem_seq = sem.unsqueeze(1)              # [B, 1, proj_dim]
        for_seq = forz.unsqueeze(1)             # [B, 1, proj_dim]

        # Direction 1: semantic queries forensic.
        upd_sem, _ = self.attn_sem_to_for(query=sem_seq, key=for_seq, value=for_seq)
        upd_sem = self.norm_sem(upd_sem + sem_seq)  # residual + LN

        # Direction 2: forensic queries semantic.
        upd_for, _ = self.attn_for_to_sem(query=for_seq, key=sem_seq, value=sem_seq)
        upd_for = self.norm_for(upd_for + for_seq)

        # Drop the seq dim and concat.
        upd_sem = upd_sem.squeeze(1)            # [B, proj_dim]
        upd_for = upd_for.squeeze(1)            # [B, proj_dim]
        fused = torch.cat([upd_sem, upd_for], dim=-1)  # [B, 2*proj_dim]
        return fused


class MLPClassifier(nn.Module):
    """Three-layer MLP per V2 §8.

    Linear 1024 -> 512  ReLU  Dropout 0.5
    Linear  512 -> 256  ReLU  Dropout 0.3
    Linear  256 ->   3  raw logits (no softmax — CE applies it).
    """

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

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class FullPipeline(nn.Module):
    """Frozen FND-CLIP + trainable ResNet18 forensic + cross-attention + MLP.

    Returns dict with main_logits (always) and aux_logits (training only).
    The forensic auxiliary classifier is a small Linear(for_dim, 3) per V2 §6.
    """

    def __init__(self, fnd_clip: FNDCLIP,
                 fnd_clip_feat_dim: int = 512,
                 forensic_feat_dim: int = 512,
                 fusion_proj_dim: int = 512,
                 num_classes: int = 3,
                 fusion_heads: int = 8,
                 fusion_dropout: float = 0.1,
                 forensic_dropout: float = 0.3):
        super().__init__()
        # Frozen FND-CLIP — eval mode + no_grad in forward.
        self.fnd_clip = fnd_clip
        for p in self.fnd_clip.parameters():
            p.requires_grad = False
        self.fnd_clip.eval()

        # Trainable forensic encoder.
        self.forensic = ResNet18Forensic(
            pretrained=True, out_dim=forensic_feat_dim,
            dropout=forensic_dropout,
        )
        # Auxiliary classifier on the forensic feature (training only).
        # BINARY (real image / fake image) — not 3-class. The forensic
        # encoder sees only DCT pixel features so it cannot distinguish OOC
        # from Real (both have real images). Forcing 3-class here was the
        # cause of the Step-1 baseline plateauing at F1=0.49. Binary keeps
        # the auxiliary task aligned with what the encoder can actually see;
        # the 3-class job lives at the fusion head where all signals are
        # present. (Deviation from V2 §6.)
        self.aux_classifier = nn.Linear(forensic_feat_dim, 2)

        # Cross-attention fusion + MLP head.
        self.fusion = CrossAttentionFusion(
            sem_dim=fnd_clip_feat_dim,
            for_dim=forensic_feat_dim,
            proj_dim=fusion_proj_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.classifier = MLPClassifier(
            in_dim=2 * fusion_proj_dim, num_classes=num_classes)

    def train(self, mode: bool = True):
        """Override so FND-CLIP stays in eval mode regardless of training flag."""
        super().train(mode)
        self.fnd_clip.eval()
        return self

    def forward(self, dct, v_semantic=None,
                image=None, bert_ids=None, bert_mask=None,
                clip_pixels=None, clip_ids=None, clip_mask=None):
        # Stage 1 — frozen semantic features. If v_semantic is provided
        # (precomputed and cached on disk), skip the heavy FND-CLIP forward
        # entirely — this is the main optimization that makes 50-epoch
        # training tractable.
        if v_semantic is None:
            with torch.no_grad():
                v_semantic = self.fnd_clip.forward_semantic(
                    image, bert_ids, bert_mask,
                    clip_pixels, clip_ids, clip_mask,
                )

        # Stage 2 — trainable forensic features.
        v_forensic = self.forensic(dct)

        # Auxiliary classifier (only used at training time, but always
        # computable; the trainer decides whether to backprop through it).
        aux_logits = self.aux_classifier(v_forensic)

        # Cross-attention fusion -> MLP main classifier.
        fused = self.fusion(v_semantic, v_forensic)
        main_logits = self.classifier(fused)

        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "v_semantic": v_semantic,
            "v_forensic": v_forensic,
            "fused": fused,
        }

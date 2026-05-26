"""Multi-seed softmax ensemble for V4 Stage 2 fusion (P9.6).

Loads N V3PairwiseFusion ckpts and averages their softmax outputs.
Inference-only.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from v4.models.fusion.v3_pairwise import V3PairwiseFusion


class EnsembleFusion(nn.Module):
    def __init__(
        self,
        ckpt_paths: list[str | Path],
        fusion_kwargs: dict | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if not ckpt_paths:
            raise ValueError("EnsembleFusion requires at least 1 ckpt path")
        self.device = torch.device(device)
        fusion_kwargs = dict(fusion_kwargs or {})
        fusion_kwargs.setdefault("feat_dim", 768)
        fusion_kwargs.setdefault("fused_dim", 1024)
        fusion_kwargs.setdefault("num_classes", 5)

        self.fusions = nn.ModuleList()
        self.ckpt_paths = [Path(p) for p in ckpt_paths]
        for cp in self.ckpt_paths:
            m = V3PairwiseFusion(**fusion_kwargs)
            payload = torch.load(cp, map_location="cpu", weights_only=False)
            state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
            m.load_state_dict(state, strict=False)
            m.to(self.device).eval()
            for p in m.parameters():
                p.requires_grad = False
            self.fusions.append(m)
        self.num_classes = fusion_kwargs["num_classes"]
        self.fused_dim = fusion_kwargs["fused_dim"]

    @torch.no_grad()
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        probs_list = []
        fused_list = []
        for m in self.fusions:
            out = m({k: v.to(self.device) for k, v in features.items()})
            probs_list.append(F.softmax(out["main_logits"], dim=-1))
            fused_list.append(out["fused"])
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        avg_fused = torch.stack(fused_list, dim=0).mean(dim=0)
        avg_logits = torch.log(avg_probs.clamp(min=1e-12))
        return {
            "main_logits": avg_logits,
            "main_probs": avg_probs,
            "fused": avg_fused,
            "num_seeds": len(self.fusions),
        }

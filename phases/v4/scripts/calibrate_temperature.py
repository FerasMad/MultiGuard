"""Temperature scaling on val split (P9.6).

Fits a single scalar T such that softmax(logits / T) minimizes NLL on val.
Doesn't change predictions (argmax invariant) but produces honest confidence.

Usage:
    python phases/v4/scripts/calibrate_temperature.py \
        --config phases/v4/configs/v4_pipeline_honest.yaml \
        --checkpoint outputs/v4/stage2_fusion_honest_seed42/best.pt \
        --out outputs/v4/stage2_fusion_honest_seed42/temperature.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from v4.core.checkpoints import load_checkpoint
from v4.core.registry import build_fusion, import_all
from v4.data.datasets.cached import CachedFeatureDataset, collate_cached


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    init_t: float = 1.0, lr: float = 0.01, n_iter: int = 200) -> float:
    T = torch.tensor(init_t, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.1, max=10.0), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.clamp(min=0.1, max=10.0).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_spec = dict(cfg["fusion"])
    fusion_spec.pop("ckpt", None)
    fusion = build_fusion(fusion_spec)
    ck = load_checkpoint(args.checkpoint, map_location=device)
    fusion.load_state_dict(ck["model_state"])
    fusion.to(device).eval()

    data_cfg = cfg["data"]
    val_ds = CachedFeatureDataset(
        csv_path=data_cfg["csv_path"],
        cache_root=data_cfg["cache_root"],
        feature_keys=data_cfg["feature_keys"],
        split="val",
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_cached)

    feature_names = [fk["name"] for fk in data_cfg["feature_keys"]]
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            feats = {name: batch[name].to(device) for name in feature_names}
            out = fusion(feats)
            all_logits.append(out["main_logits"].cpu())
            all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"[calib] val set: {len(labels)} samples, logits.shape={tuple(logits.shape)}")

    nll_before = float(F.cross_entropy(logits, labels).item())
    acc_before = float((logits.argmax(dim=-1) == labels).float().mean().item())
    print(f"[calib] BEFORE: NLL={nll_before:.4f}  acc={acc_before:.4f}")

    T = fit_temperature(logits, labels)
    print(f"[calib] fitted T = {T:.4f}")

    nll_after = float(F.cross_entropy(logits / T, labels).item())
    acc_after = float(((logits / T).argmax(dim=-1) == labels).float().mean().item())
    print(f"[calib] AFTER:  NLL={nll_after:.4f}  acc={acc_after:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "temperature": T,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "n_val": int(len(labels)),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
    }, indent=2), encoding="utf-8")
    print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()

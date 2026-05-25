"""Train the doctor's Approach 2 forensic detector.

Doctor's spec F.12-F.22 (MASTER_CHECKLIST):
    - torchvision ResNet50 with ImageNet V1 weights, 1ch Kaiming conv1, Linear(2048,1) head
    - Phase 1 (epochs 1-5): freeze layer1/2, AdamW lr=1e-4, no grad clip
    - Phase 2 (epoch 6+):  unfreeze + reinit optimizer at lr=1e-5 + grad clip max_norm=1.0
    - BCEWithLogitsLoss, batch 64, ReduceLROnPlateau (mode=max factor=0.5 patience=3)
    - Early-stop patience=5 on val AP (carries across phase boundary)
    - Save best as forensic_dct_model.pth

Locked decision D6: fp32 for forensic training (safer with fresh 1ch conv1).
Locked decision D7: batch 64 per doctor spec.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from forensic.data.dct_dataset import DctCacheDataset
from forensic.models.dct_resnet50 import build_dct_resnet50
from forensic.training.two_phase_trainer import TwoPhaseTrainer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=Path("phases/forensic/configs/dct_resnet50.yaml"))
    p.add_argument("--cache-train", type=Path,
                   default=Path("phases/forensic/data/dct_cache/train"))
    p.add_argument("--cache-val", type=Path,
                   default=Path("phases/forensic/data/dct_cache/val"))
    p.add_argument("--dct-stats", type=Path,
                   default=Path("phases/forensic/data/dct_stats.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("phases/forensic/outputs/dct"))
    p.add_argument("--epochs", type=int, default=None, help="Override config max_epochs")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Smoke test: cap dataset to this many samples")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_all(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    train_cfg = cfg.get("train", {})
    max_epochs = args.epochs if args.epochs is not None else train_cfg.get("max_epochs", 30)
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 4)

    print(f"[train_dct] config: {args.config}")
    print(f"[train_dct] max_epochs={max_epochs} batch={batch_size} workers={num_workers}")
    print(f"[train_dct] seed={args.seed}")
    print(f"[train_dct] device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    train_ds = DctCacheDataset(args.cache_train, args.dct_stats, limit=args.limit_batches)
    val_ds = DctCacheDataset(args.cache_val, args.dct_stats, limit=args.limit_batches)
    print(f"[train_dct] train n={len(train_ds)}  val n={len(val_ds)}")
    print(f"[train_dct] dct_stats: mean={train_ds.mean:.4f}  std={train_ds.std:.4f}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    model = build_dct_resnet50(pretrained=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer = TwoPhaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=args.out_dir,
        max_epochs=max_epochs,
    )
    result = trainer.fit()

    summary_path = args.out_dir / "train_summary.json"
    summary_path.write_text(json.dumps({
        "config_path": str(args.config),
        "seed": args.seed,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        **result,
    }, indent=2, default=str))
    print(f"\n[train_dct] DONE.  best_ap={result['best_ap']:.4f} at epoch {result['best_epoch']}")
    print(f"[train_dct] checkpoint: {result['forensic_named_ckpt']}")


if __name__ == "__main__":
    main()

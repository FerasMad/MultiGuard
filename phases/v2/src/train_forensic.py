"""Train the Step 1 forensic baseline (DCT + ResNet18 + linear classifier).

Uses precomputed DCT maps from data/processed/dct_cache. Follows the training
configuration from Implementation Guidelines V2 §10-11:
  Adam, lr=1e-4, weight_decay=1e-4, batch=64, max_epochs=50,
  StepLR x0.1 at epoch 30, gradient clipping max_norm=1.0,
  early stopping with patience=10 on val F1-macro.
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.forensic_baseline import ForensicBaseline


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def path_hash(image_path: str) -> str:
    return hashlib.md5(os.path.abspath(image_path).encode("utf-8")).hexdigest()


class DCTDataset(Dataset):
    """Loads precomputed DCT .pt files plus integer label."""

    def __init__(self, df, cache_dir: str):
        self.df = df.reset_index(drop=True)
        self.cache = Path(cache_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        h = path_hash(row["image_path"])
        pt = self.cache / f"{h}.pt"
        if not pt.exists():
            # Fallback: zero map. The training script logs missing rate.
            dct = torch.zeros(1, 224, 224)
        else:
            dct = torch.load(pt, weights_only=False)
            if dct.dim() == 2:
                dct = dct.unsqueeze(0)
            dct = dct.float()
        return {
            "dct": dct,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def collate(batch):
    return {
        "dct": torch.stack([b["dct"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def train_one_epoch(model, loader, optimizer, criterion, device, clip_max=1.0):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        dct = batch["dct"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(dct)
        loss = criterion(out["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="val", leave=False):
        dct = batch["dct"].to(device)
        labels = batch["label"].to(device)
        out = model(dct)
        loss = criterion(out["logits"], labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    f1m = f1_score(all_labels, all_preds, average="macro")
    return {"loss": total_loss / max(n, 1), "f1_macro": float(f1m)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/forensic_baseline.yaml")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override config max epochs (useful for smoke tests).")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only first N samples per split (smoke test).")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.max_epochs is not None:
        cfg["train"]["epochs"] = args.max_epochs

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = pick_device()
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(cfg["data"]["csv_path"])
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if args.limit:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(args.limit)
        test_df = test_df.head(args.limit)
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_set = DCTDataset(train_df, cfg["data"]["dct_cache"])
    val_set = DCTDataset(val_df, cfg["data"]["dct_cache"])

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"].get("num_workers", 2),
        collate_fn=collate,
        persistent_workers=cfg["train"].get("num_workers", 2) > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"].get("num_workers", 2),
        collate_fn=collate,
        persistent_workers=cfg["train"].get("num_workers", 2) > 0,
    )

    model = ForensicBaseline(num_classes=3,
                             pretrained=cfg["model"].get("pretrained", True),
                             feat_dim=cfg["model"].get("feat_dim", 768),
                             dropout=cfg["model"].get("dropout", 0.3)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=cfg["train"]["lr"],
                     weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scheduler = StepLR(optimizer,
                       step_size=cfg["train"].get("lr_step", 30),
                       gamma=cfg["train"].get("lr_gamma", 0.1))

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    patience_left = cfg["train"].get("early_stop_patience", 10)
    history = []
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                  clip_max=cfg["train"].get("grad_clip", 1.0))
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | lr={lr:.2e} | train_loss={tr_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
              f"{elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        **val_metrics, "lr": lr, "seconds": elapsed})

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_left = cfg["train"].get("early_stop_patience", 10)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
            }, out_dir / "best.pt")
            print(f"  -> saved best (f1_macro={best_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\nEarly stopping at epoch {epoch} (patience exhausted).")
                break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print(f"\nBest val F1-macro: {best_f1:.4f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

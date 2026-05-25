"""V2 training: CLIP + DCT/ResNet18 forensic + cross-attention + MLP.

CLIP features and DCT maps are precomputed and loaded from disk.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from models.clip_forensic_pipeline import ClipForensicPipeline
from precompute_clip_features import feature_hash


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def path_hash(image_path: str) -> str:
    return hashlib.md5(os.path.abspath(image_path).encode("utf-8")).hexdigest()


def main_label_to_aux(label: int) -> int:
    return 1 if int(label) == 1 else 0


class CachedDataset(Dataset):
    def __init__(self, df, dct_cache, clip_cache, clip_dim=1024):
        self.df = df.reset_index(drop=True)
        self.dct_cache = Path(dct_cache)
        self.clip_cache = Path(clip_cache)
        self.clip_dim = clip_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        text = str(row["text"])

        dpt = self.dct_cache / f"{path_hash(path)}.pt"
        if dpt.exists():
            dct = torch.load(dpt, weights_only=False).float()
            if dct.dim() == 2:
                dct = dct.unsqueeze(0)
        else:
            dct = torch.zeros(1, 224, 224)

        cpt = self.clip_cache / f"{feature_hash(text, path)}.pt"
        if cpt.exists():
            v_sem = torch.load(cpt, weights_only=False).float()
        else:
            v_sem = torch.zeros(self.clip_dim)

        main_label = int(row["label"])
        return {
            "dct": dct,
            "v_semantic": v_sem,
            "main_label": torch.tensor(main_label, dtype=torch.long),
            "aux_label": torch.tensor(main_label_to_aux(main_label),
                                      dtype=torch.long),
        }


def collate(batch):
    keys = list(batch[0].keys())
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def train_one_epoch(model, loader, optimizer, criterion, device,
                    aux_weight=0.1, clip_max=1.0):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        out = model(
            dct=batch["dct"].to(device),
            v_semantic=batch["v_semantic"].to(device),
        )
        main_label = batch["main_label"].to(device)
        aux_label = batch["aux_label"].to(device)
        main_loss = criterion(out["main_logits"], main_label)
        aux_loss = criterion(out["aux_logits"], aux_label)
        loss = main_loss + aux_weight * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip_max)
        optimizer.step()
        total_loss += loss.item() * main_label.size(0)
        n += main_label.size(0)
        pbar.set_postfix(main=f"{main_loss.item():.3f}",
                         aux=f"{aux_loss.item():.3f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="val", leave=False):
        out = model(
            dct=batch["dct"].to(device),
            v_semantic=batch["v_semantic"].to(device),
        )
        labels = batch["main_label"].to(device)
        loss = criterion(out["main_logits"], labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        preds = out["main_logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    f1m = f1_score(all_labels, all_preds, average="macro")
    return {"loss": total_loss / max(n, 1), "f1_macro": float(f1m)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/clip_forensic.yaml")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.max_epochs is not None:
        cfg["train"]["epochs"] = args.max_epochs

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = pick_device()
    print(f"Device: {device}")

    df = pd.read_csv(cfg["data"]["csv_path"])
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    if args.limit:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(args.limit)
    print(f"Splits: train={len(train_df)}, val={len(val_df)}")

    train_set = CachedDataset(train_df,
                              cfg["data"]["dct_cache"],
                              cfg["data"]["clip_cache"],
                              clip_dim=cfg["model"].get("clip_dim", 1024))
    val_set = CachedDataset(val_df,
                            cfg["data"]["dct_cache"],
                            cfg["data"]["clip_cache"],
                            clip_dim=cfg["model"].get("clip_dim", 1024))

    nw = cfg["train"].get("num_workers", 2)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=nw,
                              collate_fn=collate,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=nw,
                            collate_fn=collate,
                            persistent_workers=nw > 0)

    model = ClipForensicPipeline(
        clip_dim=cfg["model"].get("clip_dim", 1024),
        forensic_feat_dim=cfg["model"].get("forensic_feat_dim", 768),
        fusion_proj_dim=cfg["model"].get("fusion_proj_dim", 512),
        num_classes=3,
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        fusion_dropout=cfg["model"].get("fusion_dropout", 0.1),
        forensic_dropout=cfg["model"].get("forensic_dropout", 0.3),
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total={n_total:,}  trainable={n_train:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=cfg["train"]["lr"],
                     weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scheduler = StepLR(optimizer,
                       step_size=cfg["train"].get("lr_step", 30),
                       gamma=cfg["train"].get("lr_gamma", 0.1))

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    aux_weight = cfg["train"].get("aux_weight", 0.1)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    patience = cfg["train"].get("early_stop_patience", 10)
    best_f1 = -1.0
    patience_left = patience
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, aux_weight=aux_weight,
                                  clip_max=grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | lr={lr:.2e} | train_loss={tr_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_f1_macro={val_metrics['f1_macro']:.4f} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        **val_metrics, "lr": lr, "seconds": elapsed})

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_left = patience
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

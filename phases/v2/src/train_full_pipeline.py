"""Train the full Step-2 pipeline (V2 §9-11):
frozen FND-CLIP + trainable ResNet18-forensic + cross-attention fusion + MLP.

Loss = main_CE(main_logits, label) + 0.1 * aux_CE(aux_logits, aux_label)
       where aux_label = 1 if main_label == Manipulated else 0
       (binary "fake image vs real image" — see full_pipeline.py for rationale)
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
from models.fnd_clip import FNDCLIP
from models.full_pipeline import FullPipeline
from precompute_fnd_features import feature_hash


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def path_hash(image_path: str) -> str:
    return hashlib.md5(os.path.abspath(image_path).encode("utf-8")).hexdigest()


# Strict V2 §6: aux is 3-class so aux_label == main_label.
def main_label_to_aux(label: int) -> int:
    return int(label)


class CachedFeatureDataset(Dataset):
    """Loads precomputed v_semantic + DCT map + labels. No FND-CLIP, no BERT,
    no CLIP — those features are already cached on disk. This is what makes
    each training epoch fast enough to run 50 of them on MPS."""

    def __init__(self, df, dct_cache, fnd_cache, feat_dim=512):
        self.df = df.reset_index(drop=True)
        self.dct_cache = Path(dct_cache)
        self.fnd_cache = Path(fnd_cache)
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        text = str(row["text"])

        # DCT map.
        dpt = self.dct_cache / f"{path_hash(path)}.pt"
        if dpt.exists():
            dct = torch.load(dpt, weights_only=False).float()
            if dct.dim() == 2:
                dct = dct.unsqueeze(0)
        else:
            dct = torch.zeros(1, 224, 224)

        # Precomputed FND-CLIP semantic vector.
        fpt = self.fnd_cache / f"{feature_hash(text, path)}.pt"
        if fpt.exists():
            v_semantic = torch.load(fpt, weights_only=False).float()
        else:
            v_semantic = torch.zeros(self.feat_dim)

        main_label = int(row["label"])
        aux_label = main_label_to_aux(main_label)

        return {
            "dct": dct,
            "v_semantic": v_semantic,
            "main_label": torch.tensor(main_label, dtype=torch.long),
            "aux_label": torch.tensor(aux_label, dtype=torch.long),
        }


def collate(batch):
    keys = list(batch[0].keys())
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def build_model(cfg, device):
    """Build the trainable parts of the pipeline. We DON'T instantiate the
    full FND-CLIP here because v_semantic is precomputed and cached — only
    the trainable head needs to live in memory during training. We pass a
    minimal FND-CLIP-shaped object that the FullPipeline ignores when
    v_semantic is provided directly. (See FullPipeline.forward.)"""

    fnd_feat = cfg["model"].get("fnd_feat_dim", 512)

    class _NoOpFND(nn.Module):
        """Placeholder so FullPipeline can hold a reference but never run it."""
        def __init__(self):
            super().__init__()
            self._buf = nn.Parameter(torch.zeros(1), requires_grad=False)
        def forward_semantic(self, *a, **k):
            raise RuntimeError("v_semantic must be precomputed and passed in")
        def eval(self):
            return self
        def train(self, mode=True):
            return self

    pipeline = FullPipeline(
        fnd_clip=_NoOpFND(),
        fnd_clip_feat_dim=fnd_feat,
        forensic_feat_dim=cfg["model"].get("forensic_feat_dim", 512),
        fusion_proj_dim=cfg["model"].get("fusion_proj_dim", 512),
        num_classes=3,
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        fusion_dropout=cfg["model"].get("fusion_dropout", 0.1),
        forensic_dropout=cfg["model"].get("forensic_dropout", 0.3),
    ).to(device)

    n_total = sum(p.numel() for p in pipeline.parameters())
    n_train = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"Params: total={n_total:,}  trainable={n_train:,}  "
          f"(FND-CLIP not loaded — using cached v_semantic)")
    return pipeline


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
    p.add_argument("--config", default="config/full_pipeline.yaml")
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

    train_set = CachedFeatureDataset(
        train_df, cfg["data"]["dct_cache"], cfg["data"]["fnd_cache"],
        feat_dim=cfg["model"].get("fnd_feat_dim", 512))
    val_set = CachedFeatureDataset(
        val_df, cfg["data"]["dct_cache"], cfg["data"]["fnd_cache"],
        feat_dim=cfg["model"].get("fnd_feat_dim", 512))

    nw = cfg["train"].get("num_workers", 2)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=nw,
                              collate_fn=collate,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=nw,
                            collate_fn=collate,
                            persistent_workers=nw > 0)

    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
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
              f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
              f"{elapsed:.1f}s")
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

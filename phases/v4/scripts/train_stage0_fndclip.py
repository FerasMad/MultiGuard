"""Stage-0 fresh FND-CLIP V1-style fine-tune on the unified manifest.

Trains the V1 binary OOC task (matched vs mismatched) using only
classes 0 (Real, NewsCLIPpings matched) and 1 (OOC, NewsCLIPpings mismatched)
rows of forensic_5class_unified.csv. Output state_dict matches V1's
module-key layout exactly so inline/fnd_clip.py loads it cleanly.

Run:
    python phases/v4/scripts/train_stage0_fndclip.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer, CLIPProcessor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phases/v3/src"))
from models.fnd_clip import FNDCLIP  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tf(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class StageZeroDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool,
                 bert_name: str = "bert-base-uncased",
                 clip_name: str = "openai/clip-vit-base-patch32",
                 max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.image_tf = build_tf(train=train)
        self.bert_tok = BertTokenizer.from_pretrained(bert_name)
        self.clip_proc = CLIPProcessor.from_pretrained(clip_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color=0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["text"])
        pil = self._load(str(row["image_path"]))
        image = self.image_tf(pil)
        bert = self.bert_tok(text, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        clip = self.clip_proc(images=pil, text=text, return_tensors="pt",
                              padding="max_length", truncation=True, max_length=77)
        binary = 0.0 if int(row["label"]) == 0 else 1.0
        return {
            "image": image,
            "bert_ids": bert["input_ids"].squeeze(0),
            "bert_mask": bert["attention_mask"].squeeze(0),
            "clip_pixels": clip["pixel_values"].squeeze(0),
            "clip_ids": clip["input_ids"].squeeze(0),
            "clip_mask": clip["attention_mask"].squeeze(0),
            "label": torch.tensor(binary, dtype=torch.float32),
        }


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def train_epoch(model, loader, opt, crit, device, epoch):
    model.train()
    total = 0.0
    n = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        opt.zero_grad()
        out = model(
            image=batch["image"].to(device),
            bert_ids=batch["bert_ids"].to(device),
            bert_mask=batch["bert_mask"].to(device),
            clip_pixels=batch["clip_pixels"].to(device),
            clip_ids=batch["clip_ids"].to(device),
            clip_mask=batch["clip_mask"].to(device),
        )
        logits = out["logits"].squeeze(-1)
        loss = crit(logits, batch["label"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(batch["label"])
        n += len(batch["label"])
        if i % 50 == 0:
            print(f"  ep{epoch} batch {i}/{len(loader)}  loss={loss.item():.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    model.eval()
    all_probs, all_labels = [], []
    total = 0.0
    n = 0
    for batch in loader:
        out = model(
            image=batch["image"].to(device),
            bert_ids=batch["bert_ids"].to(device),
            bert_mask=batch["bert_mask"].to(device),
            clip_pixels=batch["clip_pixels"].to(device),
            clip_ids=batch["clip_ids"].to(device),
            clip_mask=batch["clip_mask"].to(device),
        )
        logits = out["logits"].squeeze(-1)
        labels = batch["label"].to(device)
        total += crit(logits, labels).item() * len(labels)
        n += len(labels)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": total / max(n, 1),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("data/processed/forensic_5class_unified.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/v4/stage0_fndclip"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", type=Path,
                   default=Path("outputs/v1/leakfree/best.pt"))
    args = p.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage0] device={device}, batch={args.batch_size}, lr={args.lr}, "
          f"epochs={args.epochs}, patience={args.patience}")

    df = pd.read_csv(args.manifest)
    bin_df = df[df["label"].isin([0, 1])].copy()
    print(f"[stage0] full manifest: {len(df)} rows; binary OOC subset: {len(bin_df)} rows")

    train_df = bin_df[bin_df["split"] == "train"].copy()
    val_df = bin_df[bin_df["split"] == "val"].copy()
    print(f"[stage0] splits: train={len(train_df)} val={len(val_df)}")
    print(f"[stage0] train label balance: {train_df['label'].value_counts().to_dict()}")
    print(f"[stage0] val label balance:   {val_df['label'].value_counts().to_dict()}")

    train_ds = StageZeroDataset(train_df, train=True)
    val_ds = StageZeroDataset(val_df, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)

    model = FNDCLIP(feat_dim=512, num_classes=1).to(device)

    if args.init_from is not None and args.init_from.exists():
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        target = model.state_dict()
        compat = {k: v for k, v in state.items()
                  if k in target and target[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(compat, strict=False)
        print(f"[stage0] warm-started from {args.init_from.name}: "
              f"{len(compat)}/{len(target)} loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    crit = nn.BCEWithLogitsLoss()
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)

    best_auc = 0.0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, crit, device, epoch)
        val_metrics = eval_epoch(model, val_loader, crit, device)
        dt = time.time() - t0
        msg = (f"[stage0] ep{epoch:>2} | train_loss={train_loss:.4f} | "
               f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
               f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f} | "
               f"dt={dt:.0f}s")
        print(msg, flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics,
                        "dt_seconds": dt})

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_metrics": val_metrics,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "history": history,
        }, args.out_dir / "latest.pt")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_left = args.patience
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "history": history,
            }, args.out_dir / "best.pt")
            print(f"  -> new best AUC={best_auc:.4f}, saved best.pt")
        else:
            patience_left -= 1
            print(f"  -> no improvement ({patience_left} patience left)")
            if patience_left <= 0:
                print(f"[stage0] early stop at ep{epoch}, best AUC={best_auc:.4f}")
                break

    import json
    (args.out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n[stage0] DONE. best AUC={best_auc:.4f}. ckpt at {args.out_dir/'best.pt'}")


if __name__ == "__main__":
    main()

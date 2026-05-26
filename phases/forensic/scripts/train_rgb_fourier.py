"""Train Approach 1 (RGB + Fourier mask) forensic detector per doctor's spec F.4-F.11.

Adapts `chandlerbing65nm/FakeImageDetection` (cloned under
`phases/forensic/external/FakeImageDetection/`) into a clean single-GPU,
Windows-friendly trainer that uses our existing
`phases/forensic/data/genimage_train/{train,val}/{0_real,1_fake}/` layout.

Spec mapping:
  F.4  - fine-tune the FakeImageDetection RN50 with Fourier masking
  F.5  - load checkpoint mask_15/rn50ft_fouriermask.pth (if available;
         falls back to ImageNet RN50 init with a documented deviation)
  F.6  - freeze conv1, bn1, layer1, layer2; train layer3, layer4, fc only
  F.7  - call model.change_output(1) (NOT manual nn.Linear creation)
  F.8  - Fourier masking 50% probability, mask_ratio=0.15, training-only
  F.9  - Resize 224x224 bilinear, ToTensor, Normalize ImageNet stats
  F.10 - BCEWithLogitsLoss, AdamW lr=1e-4 wd=1e-4, batch=64, max 30 epochs,
         ReduceLROnPlateau(mode='max', factor=0.5, patience=3) on val AP,
         early-stop patience=5
  F.11 - save best as forensic_rgb_model.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import the external FakeImageDetection components
EXTERNAL_ROOT = Path("phases/forensic/external/FakeImageDetection")
if not EXTERNAL_ROOT.exists():
    print(
        f"FATAL: {EXTERNAL_ROOT} not found. Clone first:\n"
        f"  git clone https://github.com/chandlerbing65nm/FakeImageDetection.git "
        f"{EXTERNAL_ROOT}",
        file=sys.stderr,
    )
    sys.exit(2)

sys.path.insert(0, str(EXTERNAL_ROOT.resolve()))

try:
    from mask import FrequencyMaskGenerator
    from networks.resnet import resnet50
except Exception as e:
    print(f"FATAL: could not import FakeImageDetection components: {e}", file=sys.stderr)
    sys.exit(2)

# --- doctor-spec hyperparameters (locked, F.10) -----------------------------
LR: float = 1e-4
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 30
SCHEDULER_FACTOR: float = 0.5
SCHEDULER_PATIENCE: int = 3
EARLY_STOP_PATIENCE: int = 5

# F.8
FOURIER_MASK_PROB: float = 0.5
FOURIER_MASK_RATIO: float = 0.15


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# transforms (F.8 + F.9)


class RandomFourierMask:
    """50% probability wrapper around FrequencyMaskGenerator (training-only, F.8)."""

    def __init__(self, p: float = FOURIER_MASK_PROB, ratio: float = FOURIER_MASK_RATIO):
        self.p = p
        self.ratio = ratio
        # Doctor's spec says Fourier masking. Defaults match the repo's
        # FrequencyMaskGenerator(ratio, band='low+high', transform_type='fourier',
        # channel='all') i.e. full-channel Fourier mask covering low+high bands
        # at the specified ratio.
        self.gen = FrequencyMaskGenerator(
            ratio=ratio, band="low+high", transform_type="fourier", channel="all"
        )

    def __call__(self, img):
        if random.random() < self.p:
            return self.gen.transform(img)
        return img


def make_train_transform() -> transforms.Compose:
    """F.8 + F.9: Fourier mask (50%), Resize 224 bilinear, ToTensor, Normalize."""
    return transforms.Compose(
        [
            RandomFourierMask(p=FOURIER_MASK_PROB, ratio=FOURIER_MASK_RATIO),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_eval_transform() -> transforms.Compose:
    """F.9 (eval-time; no Fourier mask)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# model loader (F.5 + F.6 + F.7)


def _resolve_checkpoint(preferred: Path | None) -> Path | None:
    """Try the preferred path; fall back to the repo's renamed equivalents.

    The doctor's spec F.5 literally names `rn50ft_fouriermask.pth`, but the
    upstream `chandlerbing65nm/FakeImageDetection` repo renamed all "fouriermask"
    files to "spectralmask" at some point. We accept the spectralmask variants
    as the spec-intended successors and document the rename as F-A8 in
    docs/DECISIONS.md.
    """
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(preferred)
        # Add same-dir spectralmask siblings
        candidates.append(preferred.parent / "rn50ft_spectralmask.pth")
        candidates.append(preferred.parent / "rn50ft_spectralmask(0.5).pth")
    for c in candidates:
        if c.exists():
            return c
    return None


def build_rgb_model(checkpoint_path: Path | None, device: torch.device) -> tuple[nn.Module, str]:
    """Build the Approach 1 model:

    - resnet50 from networks/resnet.py (this repo's variant)
    - load checkpoint_path if exists (F.5)
    - call model.change_output(1) (F.7)
    - freeze conv1, bn1, layer1, layer2 (F.6)

    Returns (model, init_method).
    """
    # F.5: try the preferred name + renamed siblings
    resolved = _resolve_checkpoint(checkpoint_path)
    pretrained_fallback = resolved is None
    model = resnet50(pretrained=pretrained_fallback)

    if not pretrained_fallback:
        print(f"[train_rgb] loading checkpoint: {resolved}")
        sd = torch.load(resolved, map_location="cpu", weights_only=False)
        # Upstream FakeImageDetection saves under "model_state_dict" with DDP "module." prefix.
        # Our DCT trainer saves under "model_state" with no prefix. Handle both.
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # Drop the upstream fc head: it's a binary head with the same Linear(2048, 1)
        # shape ours expects, but trained on a different binary task (Wang_CVPR2020 vs
        # our GenImage). Loading it would bias training; better to start fc from scratch.
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[train_rgb] loaded ckpt; missing={len(missing)} unexpected={len(unexpected)}")
        if resolved.name == "rn50ft_fouriermask.pth":
            init_method = f"FakeImageDetection mask_15 ckpt ({resolved.name}) [spec-literal]"
        else:
            init_method = (
                f"FakeImageDetection mask_15 ckpt ({resolved.name}) "
                f"[deviation F-A8: upstream renamed fouriermask->spectralmask]"
            )
    else:
        init_method = "ImageNet RN50 (deviation: F.5 ckpt unavailable)"
        print(f"[train_rgb] {init_method}")

    # F.7: change_output to single binary logit (preserves model.num_features etc.)
    model.change_output(1)

    # F.6: freeze conv1, bn1, layer1, layer2
    frozen_modules = ["conv1", "bn1", "layer1", "layer2"]
    n_frozen = 0
    n_trainable = 0
    for name, p in model.named_parameters():
        if any(name.startswith(prefix + ".") or name == prefix for prefix in frozen_modules):
            p.requires_grad = False
            n_frozen += p.numel()
        else:
            p.requires_grad = True
            n_trainable += p.numel()
    print(
        f"[train_rgb] freeze conv1/bn1/layer1/layer2: "
        f"{n_frozen:,} frozen, {n_trainable:,} trainable"
    )

    model = model.to(device)
    return model, init_method


# training loop


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    val_ap = float(average_precision_score(labels, probs))
    val_acc = float(((probs >= 0.5).astype(int) == labels).mean())
    return val_ap, val_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-dir", type=Path, default=Path("phases/forensic/data/genimage_train/train")
    )
    p.add_argument("--val-dir", type=Path, default=Path("phases/forensic/data/genimage_train/val"))
    p.add_argument("--out-dir", type=Path, default=Path("phases/forensic/outputs/rgb"))
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "phases/forensic/external/FakeImageDetection/checkpoints/mask_15/rn50ft_fouriermask.pth"
        ),
    )
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-batches", type=int, default=None, help="Smoke test: cap dataset size")
    args = p.parse_args()

    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_rgb] device: {device}")
    print(
        f"[train_rgb] max_epochs={args.epochs} batch={args.batch_size} workers={args.num_workers}"
    )

    # data
    print(f"[train_rgb] loading train: {args.train_dir}")
    train_ds = datasets.ImageFolder(str(args.train_dir), transform=make_train_transform())
    val_ds = datasets.ImageFolder(str(args.val_dir), transform=make_eval_transform())
    if args.limit_batches is not None:
        n = args.limit_batches * args.batch_size
        train_ds = torch.utils.data.Subset(train_ds, range(min(n, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(n, len(val_ds))))

    print(f"[train_rgb] train n={len(train_ds)}  val n={len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # model
    model, init_method = build_rgb_model(args.checkpoint, device)

    # loss + optimizer + scheduler (F.10)
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    # --- training loop with early-stop -------------------------------------
    best_path = args.out_dir / "best.pt"
    latest_path = args.out_dir / "latest.pt"
    forensic_named_path = args.out_dir / "forensic_rgb_model.pth"
    history_path = args.out_dir / "training_history.csv"

    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_ap", "val_acc", "lr", "patience_left", "is_best"])

    best_ap = -float("inf")
    best_epoch = 0
    patience_left = EARLY_STOP_PATIENCE
    start = time.time()
    epoch = 0  # ensure defined if no epochs run

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_ap, val_acc = eval_one_epoch(model, val_loader, device)

        scheduler.step(val_ap)
        is_best = val_ap > best_ap
        if is_best:
            best_ap = val_ap
            best_epoch = epoch
            patience_left = EARLY_STOP_PATIENCE
        else:
            patience_left -= 1

        with open(history_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_ap:.6f}",
                    f"{val_acc:.6f}",
                    f"{optimizer.param_groups[0]['lr']:.2e}",
                    patience_left,
                    int(is_best),
                ]
            )

        # Checkpoint
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "val_ap": val_ap,
            "val_acc": val_acc,
            "init_method": init_method,
            "best_ap": best_ap,
            "best_epoch": best_epoch,
        }
        torch.save(payload, latest_path)
        if is_best:
            torch.save(payload, best_path)

        print(
            f"epoch={epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_ap={val_ap:.4f}  val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}  "
            f"patience_left={patience_left}  best={'*' if is_best else ' '}"
        )

        if patience_left <= 0:
            print(f"Early-stop at epoch {epoch} (patience={EARLY_STOP_PATIENCE} on val AP)")
            break

    # F.11: copy best to doctor-spec filename
    if best_path.exists():
        shutil.copyfile(best_path, forensic_named_path)

    elapsed = time.time() - start
    summary = {
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "init_method": init_method,
        "best_ckpt": str(best_path),
        "forensic_named_ckpt": str(forensic_named_path),
        "history_csv": str(history_path),
        "elapsed_s": round(elapsed, 1),
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "seed": args.seed,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[train_rgb] DONE  best_ap={best_ap:.4f} at epoch {best_epoch}  ({elapsed:.0f}s)")
    print(f"[train_rgb] checkpoint: {forensic_named_path}")


if __name__ == "__main__":
    main()

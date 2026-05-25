"""MMFakeBench transfer probe for the forensic detectors.

Quick external-validation probe: does the forensic detector (trained on
GenImage_v2) generalize to MMFakeBench's val set?

MMFakeBench has multiple "fake" subcategories that mix AI-generation,
image-tampering, and text-only attacks. We treat them all as label=1 and the
real subcategories as label=0, then compute per-subcategory mean p(fake)
plus overall AP/Acc/AUC.

Expected behavior:
  - Pure AI generators (antifact_image_generation, fever_AI, gossipcop_match,
    chatgpt_match, llm_* etc.): forensic SHOULD score high p(fake)
  - Pure tampered photos (coco_image_edit, Fakeddit_photo_edit, DGM4):
    forensic may or may not catch (tampering != AI gen)
  - Text-only attacks (DGM4_text_edit_senti, NewsCLIPpings): image IS real,
    forensic SHOULD score low p(fake) -> "wrong" by MMFakeBench label,
    "right" from a forensic-only perspective.

Usage:
    python phases/forensic/scripts/transfer_probe_mmfakebench.py \\
        --ckpt phases/forensic/outputs/rgb/forensic_rgb_model.pth \\
        --approach rgb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

EXTERNAL_ROOT = Path("phases/forensic/external/FakeImageDetection")
if EXTERNAL_ROOT.exists():
    sys.path.insert(0, str(EXTERNAL_ROOT.resolve()))


class _FlatImageDataset(Dataset):
    """Walks a directory recursively for .jpg/.png/.jpeg/.gif. Returns (tensor, label, path)."""

    def __init__(self, root: Path, label: int, transform):
        self.root = root
        self.label = label
        self.transform = transform
        suffixes = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        self.paths = [p for p in root.rglob("*") if p.suffix.lower() in suffixes]
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        return x, self.label, str(self.paths[idx])


class _DctOnTheFlyDataset(Dataset):
    """Walks a directory and computes dual-DCT on the fly (module-level so it pickles)."""

    def __init__(self, root: Path, label: int, mean: float, std: float):
        self.root = root
        self.label = label
        self.mean = mean
        self.std = std
        suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        self.paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in suffixes])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from forensic.preprocessing.dual_dct import compute_dual_dct

        t = compute_dual_dct(str(self.paths[idx]))
        t = (t - self.mean) / (self.std + 1e-8)
        return t, self.label, str(self.paths[idx])


def make_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_rgb_model(ckpt: Path):
    from networks.resnet import resnet50

    model = resnet50(pretrained=False)
    model.change_output(1)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    sd = {k.replace("module.", ""): v for k, v in sd.items()} if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True)
    return model


def _load_dct_model(ckpt: Path):
    from forensic.models.dct_resnet50 import build_dct_resnet50

    model = build_dct_resnet50(pretrained=False)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(sd, strict=True)
    return model


@torch.no_grad()
def predict_dir(
    model,
    root: Path,
    label: int,
    device,
    batch_size: int,
    num_workers: int,
    approach: str,
):
    """Returns ({path -> prob}, [labels]) for one subdir."""
    if approach == "rgb":
        ds = _FlatImageDataset(root, label, make_eval_transform())
    else:
        from forensic.data.dct_dataset import load_dct_stats

        stats_path = Path("phases/forensic/data/dct_stats.json")
        if not stats_path.exists():
            return {}, []
        mu, sd_ = load_dct_stats(stats_path)
        ds = _DctOnTheFlyDataset(root, label, mu, sd_)

    if len(ds) == 0:
        return {}, []

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval().to(device)
    probs: dict[str, float] = {}
    labels = []
    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(-1)
        prob = torch.sigmoid(logits).cpu().numpy()
        for p_, pth in zip(prob, paths, strict=False):
            probs[pth] = float(p_)
        labels.extend([int(yi) for yi in y])
    return probs, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--approach", choices=("rgb", "dct"), default="rgb")
    p.add_argument(
        "--mm-root",
        type=Path,
        default=Path("data/raw/MMFakeBench/MMFakeBench_val"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("phases/forensic/outputs/mmfakebench_transfer.json"),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    if not args.mm_root.exists():
        print(f"FATAL: {args.mm_root} not found", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mm_probe] device: {device}  approach: {args.approach}  ckpt: {args.ckpt}")

    if args.approach == "rgb":
        if not EXTERNAL_ROOT.exists():
            print(f"FATAL: {EXTERNAL_ROOT} not found for RGB approach", file=sys.stderr)
            sys.exit(2)
        model = _load_rgb_model(args.ckpt)
    else:
        model = _load_dct_model(args.ckpt)

    fake_root = args.mm_root / "fake"
    real_root = args.mm_root / "real"

    per_cat: dict[str, dict] = {}
    all_probs = []
    all_labels = []

    print("\n[mm_probe] fake subcategories (label=1):")
    for sub in sorted(fake_root.iterdir() if fake_root.exists() else []):
        if not sub.is_dir():
            continue
        probs_dict, labels = predict_dir(
            model, sub, 1, device, args.batch_size, args.num_workers, args.approach
        )
        if not probs_dict:
            continue
        probs_list = list(probs_dict.values())
        per_cat[sub.name] = {
            "label": "fake",
            "n": len(probs_list),
            "mean_p_fake": float(mean(probs_list)),
            "frac_above_0.5": float(np.mean([p >= 0.5 for p in probs_list])),
        }
        all_probs.extend(probs_list)
        all_labels.extend(labels)
        print(
            f"  {sub.name:45s} n={len(probs_list):4d}  "
            f"mean p(fake)={per_cat[sub.name]['mean_p_fake']:.4f}  "
            f"frac>=0.5={per_cat[sub.name]['frac_above_0.5']:.4f}"
        )

    print("\n[mm_probe] real subcategories (label=0):")
    for sub in sorted(real_root.iterdir() if real_root.exists() else []):
        if not sub.is_dir():
            continue
        probs_dict, labels = predict_dir(
            model, sub, 0, device, args.batch_size, args.num_workers, args.approach
        )
        if not probs_dict:
            continue
        probs_list = list(probs_dict.values())
        per_cat[sub.name] = {
            "label": "real",
            "n": len(probs_list),
            "mean_p_fake": float(mean(probs_list)),
            "frac_above_0.5": float(np.mean([p >= 0.5 for p in probs_list])),
        }
        all_probs.extend(probs_list)
        all_labels.extend(labels)
        print(
            f"  {sub.name:45s} n={len(probs_list):4d}  "
            f"mean p(fake)={per_cat[sub.name]['mean_p_fake']:.4f}  "
            f"frac<0.5={1.0 - per_cat[sub.name]['frac_above_0.5']:.4f}"
        )

    aggregate = {}
    if all_labels:
        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)
        if len(set(labels_arr.tolist())) >= 2:
            aggregate["overall_ap"] = float(average_precision_score(labels_arr, probs_arr))
            aggregate["overall_auc"] = float(roc_auc_score(labels_arr, probs_arr))
            aggregate["overall_acc_at_0.5"] = float(
                ((probs_arr >= 0.5).astype(int) == labels_arr).mean()
            )
        aggregate["n_total"] = int(len(all_labels))
        aggregate["n_fake"] = int(labels_arr.sum())
        aggregate["n_real"] = int(len(all_labels) - labels_arr.sum())

    print("\n[mm_probe] overall (MMFakeBench label vs forensic prediction):")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "approach": args.approach,
                "mm_root": str(args.mm_root),
                "per_category": per_cat,
                "aggregate": aggregate,
                "note": (
                    "MMFakeBench's 'fake' label includes AI-generation, tampering, and "
                    "text-only attacks. The forensic detector classifies real-vs-AI-image, "
                    "so categories where the image is real (e.g. DGM4_text_edit_senti, "
                    "NewsCLIPpings_*) should correctly score low p(fake) by the detector, "
                    "even though MMFakeBench labels them 'fake'."
                ),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n[mm_probe] wrote: {args.out_json}")


if __name__ == "__main__":
    main()

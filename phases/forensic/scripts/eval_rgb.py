"""Evaluate a trained forensic_rgb_model.pth across all 8 generators.

Output: outputs/eval_rgb.json + outputs/eval_table_rgb.md (doctor's required F.25 format).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import FakeImageDetection model
EXTERNAL_ROOT = Path("phases/forensic/external/FakeImageDetection")
if not EXTERNAL_ROOT.exists():
    print(f"FATAL: {EXTERNAL_ROOT} not found.", file=sys.stderr)
    sys.exit(2)
sys.path.insert(0, str(EXTERNAL_ROOT.resolve()))

try:
    from networks.resnet import resnet50
except Exception as e:
    print(f"FATAL: could not import: {e}", file=sys.stderr)
    sys.exit(2)

# Re-use the canonical 8-generator taxonomy from the DCT eval module
from forensic.evaluation.per_generator import (  # noqa: E402
    ALL_GENERATORS,
    aggregate_metrics,
)
from forensic.evaluation.table import write_eval_table  # noqa: E402


def make_eval_transform() -> transforms.Compose:
    """F.9 spec-exact: Resize 224x224 bilinear, ToTensor, Normalize ImageNet stats."""
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


@torch.no_grad()
def eval_one_generator(
    model, gen_root: Path, device: torch.device, batch_size: int, num_workers: int
) -> dict:
    """Per-gen AP/Accuracy/AUC on phases/forensic/data/genimage_test/<gen>/{0_real,1_fake}/*.jpg.

    Doctor's spec F.26: eval mode + no_grad + sigmoid.
    """
    ds = datasets.ImageFolder(str(gen_root), transform=make_eval_transform())
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model.eval().to(device)
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

    if len(set(labels.tolist())) < 2:
        return {
            "n_samples": int(len(labels)),
            "n_real": int((labels == 0).sum()),
            "n_fake": int((labels == 1).sum()),
            "ap": float("nan"),
            "accuracy": float(((probs >= 0.5).astype(int) == labels).mean()),
            "auc": float("nan"),
            "warning": "only one class present",
        }

    return {
        "n_samples": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "ap": float(average_precision_score(labels, probs)),
        "accuracy": float(((probs >= 0.5).astype(int) == labels).mean()),
        "auc": float(roc_auc_score(labels, probs)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--test-root", type=Path, default=Path("phases/forensic/data/genimage_test"))
    p.add_argument("--out-json", type=Path, default=Path("phases/forensic/outputs/eval_rgb.json"))
    p.add_argument(
        "--out-table",
        type=Path,
        default=Path("phases/forensic/outputs/eval_table_rgb.md"),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--only", nargs="+", default=None, help="Subset of generators to eval (default: all 8)"
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_rgb] device: {device}")

    # Build model + load weights
    print(f"[eval_rgb] loading model from {args.ckpt}")
    model = resnet50(pretrained=False)
    model.change_output(1)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    sd = {k.replace("module.", ""): v for k, v in sd.items()} if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    gens = tuple(args.only) if args.only else ALL_GENERATORS
    print(f"[eval_rgb] evaluating generators: {gens}")

    per_gen: dict[str, dict] = {}
    for g in gens:
        gen_root = args.test_root / g
        if not gen_root.exists():
            per_gen[g] = {
                "generator": g,
                "skipped": True,
                "reason": f"test folder not found at {gen_root}",
            }
            continue
        res = eval_one_generator(model, gen_root, device, args.batch_size, args.num_workers)
        per_gen[g] = {"generator": g, **res}

    aggregates = aggregate_metrics(per_gen)

    print("\n[eval_rgb] per-generator results:")
    for g in gens:
        row = per_gen.get(g, {})
        if row.get("skipped"):
            print(f"  {g:12s} SKIPPED: {row.get('reason')}")
            continue
        print(
            f"  {g:12s} AP={row.get('ap', float('nan')):.4f}  "
            f"Acc={row.get('accuracy', float('nan')):.4f}  "
            f"AUC={row.get('auc', float('nan')):.4f}  "
            f"n={row.get('n_samples', 0)}"
        )
    print("\n[eval_rgb] aggregates:")
    for k, v in aggregates.items():
        print(f"  {k}: {v}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "approach": "1 (RGB + Fourier mask)",
                "per_generator": per_gen,
                "aggregates": aggregates,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n[eval_rgb] wrote: {args.out_json}")

    write_eval_table(
        per_gen,
        aggregates,
        out_path=args.out_table,
        title="Forensic Image Detector - Approach 1 (RGB + Fourier Mask) - Per-Generator Eval",
        notes=f"Checkpoint: `{args.ckpt}`",
    )
    print(f"[eval_rgb] wrote: {args.out_table}")


if __name__ == "__main__":
    main()

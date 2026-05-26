"""Run forensic inference on one or more images.

Loads either the Approach 1 (RGB+Fourier) or Approach 2 (DCT) checkpoint and
prints `p(fake)` plus the rounded decision for each input image. Convenient for
ad-hoc demos to the doctor.

Examples:
    # Default: Approach 2 on a single image
    python phases/forensic/scripts/infer.py path/to/img.jpg

    # Approach 1 (RGB + Fourier) on multiple images
    python phases/forensic/scripts/infer.py --approach rgb img1.jpg img2.png

    # Use a custom checkpoint
    python phases/forensic/scripts/infer.py --ckpt my_model.pth img.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Approach 2 (DCT)


def _build_dct(ckpt: Path):
    """Build + load the Approach 2 detector. Returns (model, preprocess_fn)."""
    from forensic.data.dct_dataset import load_dct_stats
    from forensic.models.dct_resnet50 import build_dct_resnet50
    from forensic.preprocessing.dual_dct import compute_dual_dct

    model = build_dct_resnet50(pretrained=False)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(sd, strict=True)

    stats_path = Path("phases/forensic/data/dct_stats.json")
    if not stats_path.exists():
        print(
            f"FATAL: dct_stats.json not found at {stats_path}. "
            "Download from FerasMad/forensic-dct-v1 on HF Hub or generate via "
            "phases/forensic/scripts/compute_dct_stats.py.",
            file=sys.stderr,
        )
        sys.exit(2)
    mean, std = load_dct_stats(stats_path)

    def _preprocess(path: Path) -> torch.Tensor:
        t = compute_dual_dct(str(path))  # [1, 224, 224] float32 (unnormalized)
        t = (t - mean) / (std + 1e-8)
        return t.unsqueeze(0)  # [1, 1, 224, 224]

    return model, _preprocess


# Approach 1 (RGB + Fourier)


def _build_rgb(ckpt: Path):
    """Build + load the Approach 1 detector. Returns (model, preprocess_fn)."""
    external_root = Path("phases/forensic/external/FakeImageDetection")
    if not external_root.exists():
        print(
            f"FATAL: {external_root} not found. Clone first:\n"
            f"  git clone https://github.com/chandlerbing65nm/FakeImageDetection.git "
            f"{external_root}",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.path.insert(0, str(external_root.resolve()))
    from networks.resnet import resnet50

    model = resnet50(pretrained=False)
    model.change_output(1)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    sd = {k.replace("module.", ""): v for k, v in sd.items()} if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True)

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def _preprocess(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return tf(img).unsqueeze(0)  # [1, 3, 224, 224]

    return model, _preprocess


# main


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("images", nargs="+", type=Path, help="Image paths to classify")
    p.add_argument(
        "--approach",
        choices=("dct", "rgb"),
        default="dct",
        help="Which detector to use (dct = Approach 2, rgb = Approach 1)",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=(
            "Override checkpoint path (default: "
            "outputs/dct/forensic_dct_model.pth for dct, "
            "outputs/rgb/forensic_rgb_model.pth for rgb)"
        ),
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")
    args = p.parse_args()

    if args.ckpt is None:
        if args.approach == "dct":
            args.ckpt = Path("phases/forensic/outputs/dct/forensic_dct_model.pth")
        else:
            args.ckpt = Path("phases/forensic/outputs/rgb/forensic_rgb_model.pth")

    if not args.ckpt.exists():
        print(f"FATAL: checkpoint not found at {args.ckpt}", file=sys.stderr)
        print(
            "Tip: download from HF Hub:\n"
            "  hf download FerasMad/forensic-dct-v1 forensic_dct_model.pth "
            "--local-dir phases/forensic/outputs/dct/\n"
            "  hf download FerasMad/forensic-rgb-v1 forensic_rgb_model.pth "
            "--local-dir phases/forensic/outputs/rgb/",
            file=sys.stderr,
        )
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device: {device}  approach: {args.approach}  ckpt: {args.ckpt}")

    if args.approach == "dct":
        model, preprocess = _build_dct(args.ckpt)
    else:
        model, preprocess = _build_rgb(args.ckpt)
    model.to(device).eval()

    print()
    print(f"{'Image':<60s} {'p(fake)':>10s}  decision")
    print("-" * 85)
    with torch.no_grad():
        for img_path in args.images:
            if not img_path.exists():
                print(f"{img_path!s:<60s} {'MISSING':>10s}  -")
                continue
            try:
                x = preprocess(img_path).to(device)
                logit = model(x).squeeze().item()
                prob = float(torch.sigmoid(torch.tensor(logit)))
                decision = "FAKE" if prob >= args.threshold else "REAL"
                print(f"{img_path!s:<60s} {prob:>10.4f}  {decision}")
            except Exception as e:
                print(f"{img_path!s:<60s} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

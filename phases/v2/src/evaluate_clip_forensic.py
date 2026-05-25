"""V2 evaluation: test split + MMFakeBench transfer for the CLIP+forensic pipeline."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from models.clip_forensic_pipeline import ClipForensicPipeline
from train_clip_forensic import (CachedDataset, collate, main_label_to_aux,
                                 path_hash, pick_device)
from precompute_clip_features import feature_hash
from evaluate_forensic import (compute_dct_runtime, load_mmfakebench,
                               metrics_block, save_roc_curves, LABEL_NAMES)


class FullRuntimeDataset(Dataset):
    """For MMFakeBench: compute v_semantic on the fly with raw CLIP."""

    def __init__(self, df, clip_model, clip_proc, device,
                 dct_cache=None):
        self.df = df.reset_index(drop=True)
        self.clip = clip_model
        self.proc = clip_proc
        self.device = device
        self.dct_cache = Path(dct_cache) if dct_cache else None

    def __len__(self):
        return len(self.df)

    def _v_sem(self, text, pil):
        enc = self.proc(images=pil, text=text, return_tensors="pt",
                        padding="max_length", truncation=True, max_length=77)
        with torch.no_grad():
            v_out = self.clip.vision_model(
                pixel_values=enc["pixel_values"].to(self.device))
            img_pooled = v_out.pooler_output if hasattr(v_out, "pooler_output") \
                else v_out.last_hidden_state[:, 0]
            img_feat = self.clip.visual_projection(img_pooled)

            t_out = self.clip.text_model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device))
            txt_pooled = t_out.pooler_output if hasattr(t_out, "pooler_output") \
                else t_out.last_hidden_state[:, 0]
            txt_feat = self.clip.text_projection(txt_pooled)

            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            v = torch.cat([img_feat, txt_feat], dim=-1).cpu().squeeze(0)
        return v

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        text = str(row.get("text", ""))

        dct = None
        if self.dct_cache:
            dpt = self.dct_cache / f"{path_hash(path)}.pt"
            if dpt.exists():
                dct = torch.load(dpt, weights_only=False).float()
                if dct.dim() == 2:
                    dct = dct.unsqueeze(0)
        if dct is None:
            dct = compute_dct_runtime(path)

        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (224, 224), 0)
        v = self._v_sem(text, pil)

        return {
            "dct": dct,
            "v_semantic": v,
            "main_label": torch.tensor(int(row["label"]), dtype=torch.long),
            "aux_label": torch.tensor(main_label_to_aux(int(row["label"])),
                                      dtype=torch.long),
        }


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="eval"):
        out = model(
            dct=batch["dct"].to(device),
            v_semantic=batch["v_semantic"].to(device),
        )
        all_logits.append(out["main_logits"].cpu())
        all_labels.append(batch["main_label"])
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    proba = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    return labels, preds, proba


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", default="data/processed/forensic_3class_v2.csv")
    p.add_argument("--dct-cache", default="data/processed/dct_cache")
    p.add_argument("--clip-cache", default="data/processed/clip_features")
    p.add_argument("--mmfakebench", default="data/raw/MMFakeBench")
    p.add_argument("--mmfb-split", default="val")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="outputs/clip_forensic")
    p.add_argument("--skip-mmfb", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = ClipForensicPipeline(
        clip_dim=cfg.get("model", {}).get("clip_dim", 1024),
        forensic_feat_dim=cfg.get("model", {}).get("forensic_feat_dim", 768),
        fusion_proj_dim=cfg.get("model", {}).get("fusion_proj_dim", 512),
        num_classes=3,
        fusion_heads=cfg.get("model", {}).get("fusion_heads", 8),
        fusion_dropout=cfg.get("model", {}).get("fusion_dropout", 0.1),
        forensic_dropout=cfg.get("model", {}).get("forensic_dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Test split
    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"]
    test_set = CachedDataset(test_df, args.dct_cache, args.clip_cache,
                             clip_dim=cfg.get("model", {}).get("clip_dim", 1024))
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, collate_fn=collate)
    print(f"\nTest set: {len(test_set)} samples")
    y, pr, pb = run_eval(model, test_loader, device)
    test_m = metrics_block(y, pr, pb)
    print("\n=== Test split ===")
    for k, v in test_m.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    save_roc_curves(y, pb, out_dir / "test_roc.png")
    with open(out_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump(test_m, f)

    # MMFakeBench
    if not args.skip_mmfb:
        print(f"\nLoading MMFakeBench {args.mmfb_split}...")
        try:
            mmfb = load_mmfakebench(args.mmfakebench, split=args.mmfb_split)
        except Exception as e:
            print(f"  Failed: {e}")
            mmfb = None
        if mmfb is not None and len(mmfb) > 0:
            mmfb = mmfb[mmfb["image_path"].apply(os.path.exists)]
            print(f"  Usable samples: {len(mmfb)} "
                  f"(class dist: {mmfb['label'].value_counts().to_dict()})")

            print("  Loading CLIP for runtime v_semantic...")
            clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
            for pp in clip.parameters():
                pp.requires_grad = False
            proc = CLIPProcessor.from_pretrained(args.clip_model)

            mmfb_set = FullRuntimeDataset(mmfb, clip, proc, device,
                                          dct_cache=args.dct_cache)
            mmfb_loader = DataLoader(mmfb_set, batch_size=8, shuffle=False,
                                     num_workers=0, collate_fn=collate)
            y, pr, pb = run_eval(model, mmfb_loader, device)
            mmfb_m = metrics_block(y, pr, pb)
            print("\n=== MMFakeBench transfer (no fine-tuning) ===")
            for k, v in mmfb_m.items():
                if isinstance(v, list):
                    print(f"  {k}: {v}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            save_roc_curves(y, pb, out_dir / "mmfb_roc.png")
            with open(out_dir / "mmfb_metrics.yaml", "w") as f:
                yaml.safe_dump(mmfb_m, f)

    print(f"\nAll metrics + ROC curves written to {out_dir}/")


if __name__ == "__main__":
    main()

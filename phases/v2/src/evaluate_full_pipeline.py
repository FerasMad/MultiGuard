"""Evaluate the Step-2 full pipeline on the test split AND on MMFakeBench
(no fine-tuning) per V2 §12.

Reuses the cached v_semantic features for the test split and computes them
on the fly for MMFakeBench (where no precompute has run).
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from models.fnd_clip import FNDCLIP
from models.full_pipeline import FullPipeline
from train_full_pipeline import (CachedFeatureDataset, build_model, collate,
                                 main_label_to_aux, path_hash, pick_device)
from precompute_fnd_features import feature_hash
from evaluate_forensic import (load_mmfakebench, save_roc_curves,
                               compute_dct_runtime, LABEL_NAMES,
                               metrics_block as _metrics_block)


class FullRuntimeDataset(Dataset):
    """For MMFakeBench: compute v_semantic and DCT on the fly. Loads BERT/CLIP
    tokenizers + RGB image preprocessing. Slower than cached but only runs
    on the ~400-1000 MMFakeBench samples at eval time."""

    def __init__(self, df, fnd_clip, device,
                 dct_cache=None, fnd_cache=None,
                 bert_model="bert-base-uncased",
                 clip_model="openai/clip-vit-base-patch32", max_text_len=128):
        self.df = df.reset_index(drop=True)
        self.fnd = fnd_clip
        self.device = device
        self.dct_cache = Path(dct_cache) if dct_cache else None
        self.fnd_cache = Path(fnd_cache) if fnd_cache else None
        self.bert_tok = BertTokenizer.from_pretrained(bert_model)
        self.clip_proc = CLIPProcessor.from_pretrained(clip_model)
        self.max_text_len = max_text_len
        self.image_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def _v_semantic(self, text, pil):
        image = self.image_tf(pil).unsqueeze(0).to(self.device)
        bert = self.bert_tok(text, padding="max_length", truncation=True,
                             max_length=self.max_text_len, return_tensors="pt")
        clip = self.clip_proc(images=pil, text=text, return_tensors="pt",
                              padding="max_length", truncation=True,
                              max_length=77)
        with torch.no_grad():
            v = self.fnd.forward_semantic(
                image=image,
                bert_ids=bert["input_ids"].to(self.device),
                bert_mask=bert["attention_mask"].to(self.device),
                clip_pixels=clip["pixel_values"].to(self.device),
                clip_ids=clip["input_ids"].to(self.device),
                clip_mask=clip["attention_mask"].to(self.device),
            ).cpu().squeeze(0)
        return v

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        text = str(row.get("text", ""))

        # DCT
        dct = None
        if self.dct_cache:
            dpt = self.dct_cache / f"{path_hash(path)}.pt"
            if dpt.exists():
                dct = torch.load(dpt, weights_only=False).float()
                if dct.dim() == 2:
                    dct = dct.unsqueeze(0)
        if dct is None:
            dct = compute_dct_runtime(path)

        # v_semantic
        v = None
        if self.fnd_cache:
            fpt = self.fnd_cache / f"{feature_hash(text, path)}.pt"
            if fpt.exists():
                v = torch.load(fpt, weights_only=False).float()
        if v is None:
            try:
                pil = Image.open(path).convert("RGB")
            except Exception:
                pil = Image.new("RGB", (224, 224), 0)
            v = self._v_semantic(text, pil)

        return {
            "dct": dct,
            "v_semantic": v,
            "main_label": torch.tensor(int(row["label"]), dtype=torch.long),
            "aux_label": torch.tensor(main_label_to_aux(int(row["label"])),
                                      dtype=torch.long),
        }


def metrics_block(y_true, y_pred, y_proba, labels=(0, 1, 2)):
    return _metrics_block(y_true, y_pred, y_proba, labels=labels)


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
    p.add_argument("--csv", default="data/processed/forensic_3class.csv")
    p.add_argument("--dct-cache", default="data/processed/dct_cache")
    p.add_argument("--fnd-cache", default="data/processed/fnd_features")
    p.add_argument("--mmfakebench", default="data/raw/MMFakeBench")
    p.add_argument("--mmfb-split", default="val")
    p.add_argument(
        "--fnd-clip-ckpt",
        default=None,
        help="Optional FND-CLIP checkpoint to load before runtime "
             "v_semantic computation on MMFakeBench. Default: None "
             "(fresh FND-CLIP — pretrained sub-modules only). Loading "
             "outputs/v1_ooc/best.pt here would re-introduce the v1_ooc "
             "feature leak documented in HANDOFF.md, so it is opt-in. "
             "Whichever choice you make here MUST match what was used "
             "for `precompute_fnd_features.py` so test-split features "
             "and MMFakeBench features come from the same encoder.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="outputs/full_pipeline")
    p.add_argument("--skip-mmfb", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    pipeline = build_model(cfg, device)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    pipeline.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Test split
    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"]
    test_set = CachedFeatureDataset(
        test_df, args.dct_cache, args.fnd_cache,
        feat_dim=cfg.get("model", {}).get("fnd_feat_dim", 512))
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2,
                             collate_fn=collate)
    print(f"\nTest set: {len(test_set)} samples")
    y, pr, pb = run_eval(pipeline, test_loader, device)
    test_m = metrics_block(y, pr, pb)
    print("\n=== Test split ===")
    for k, v in test_m.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    save_roc_curves(y, pb, out_dir / "test_roc.png",
                    title="Step 2 / Test split (one-vs-rest)")
    with open(out_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump(test_m, f)

    # MMFakeBench transfer
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

            # Build a real FND-CLIP for runtime feature extraction.
            # Default: fresh (no task ckpt) -> consistent with the leak-free
            # precompute_fnd_features.py default. Pass --fnd-clip-ckpt
            # explicitly to opt back into a task-tuned backbone (and accept
            # the leakage trade-off documented in HANDOFF.md).
            print("  Loading FND-CLIP for on-the-fly v_semantic...")
            fnd_feat = cfg.get("model", {}).get("fnd_feat_dim", 512)
            fnd = FNDCLIP(feat_dim=fnd_feat, num_classes=1)
            ckpt_arg = args.fnd_clip_ckpt
            if ckpt_arg is None or str(ckpt_arg).lower() in ("", "none", "null"):
                print("    --fnd-clip-ckpt not provided -> fresh FND-CLIP "
                      "(leak-free; matches the precompute default).")
            else:
                ck = torch.load(ckpt_arg, map_location="cpu",
                                weights_only=False)
                state = ck["model_state"]
                fnd_state = fnd.state_dict()
                compat = {k: v for k, v in state.items()
                          if k in fnd_state and fnd_state[k].shape == v.shape}
                fnd.load_state_dict(compat, strict=False)
                print(f"    loaded {len(compat)}/{len(fnd_state)} tensors "
                      f"from {ckpt_arg}")
                print("    WARNING: loading a task-tuned checkpoint here "
                      "can leak labels into v_semantic — see HANDOFF.md.")
            fnd = fnd.to(device).eval()
            for pp in fnd.parameters():
                pp.requires_grad = False

            mmfb_set = FullRuntimeDataset(
                mmfb, fnd, device,
                dct_cache=args.dct_cache, fnd_cache=None,
                bert_model="bert-base-uncased",
                clip_model="openai/clip-vit-base-patch32")
            mmfb_loader = DataLoader(mmfb_set, batch_size=8, shuffle=False,
                                     num_workers=0, collate_fn=collate)
            y, pr, pb = run_eval(pipeline, mmfb_loader, device)
            mmfb_m = metrics_block(y, pr, pb)
            print("\n=== MMFakeBench transfer (no fine-tuning) ===")
            for k, v in mmfb_m.items():
                if isinstance(v, list):
                    print(f"  {k}: {v}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            save_roc_curves(y, pb, out_dir / "mmfb_roc.png",
                            title=f"Step 2 / MMFakeBench {args.mmfb_split} "
                                  "(one-vs-rest, zero-shot)")
            with open(out_dir / "mmfb_metrics.yaml", "w") as f:
                yaml.safe_dump(mmfb_m, f)

    print(f"\nAll metrics + ROC curves written to {out_dir}/")


if __name__ == "__main__":
    main()

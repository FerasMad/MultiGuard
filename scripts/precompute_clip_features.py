"""Precompute raw CLIP features (frozen, off-the-shelf) for v_semantic.

For each (text, image) pair we cache `[img_feat | txt_feat]` (1024-dim,
both L2-normalized) — exactly the OOC-relevant cross-modal signal.

Using openai/clip-vit-base-patch32 directly. NOT FND-CLIP, NOT v1_ooc —
no risk of leaking labels through a fine-tuned checkpoint.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def feature_hash(text: str, image_path: str) -> str:
    payload = f"{text}||{os.path.abspath(image_path)}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()


class CLIPInputDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        path = row["image_path"]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (224, 224), 0)
        enc = self.processor(images=pil, text=text,
                             return_tensors="pt",
                             padding="max_length", truncation=True,
                             max_length=77)
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "key": feature_hash(text, path),
        }


def collate(batch):
    keys = [b["key"] for b in batch]
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "keys": keys,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/forensic_3class_v2.csv")
    p.add_argument("--cache-dir", default="data/processed/clip_features")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    print("Scanning for already-cached keys...")
    keys_in_csv = [feature_hash(str(r["text"]), r["image_path"])
                   for _, r in df.iterrows()]
    todo_mask = [not (cache / f"{k}.pt").exists() for k in keys_in_csv]
    skipped = len(df) - sum(todo_mask)
    df_todo = df[todo_mask].reset_index(drop=True)
    print(f"  cached: {skipped}   to compute: {len(df_todo)}")
    if len(df_todo) == 0:
        print("Nothing to do.")
        return

    print(f"Loading raw CLIP ({args.clip_model})...")
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad = False
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    ds = CLIPInputDataset(df_todo, processor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    n_done = 0
    pbar = tqdm(loader, desc="CLIP features")
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            vision_out = model.vision_model(pixel_values=pixel_values)
            img_pooled = vision_out.pooler_output if hasattr(vision_out, "pooler_output") \
                else vision_out.last_hidden_state[:, 0]
            img_feat = model.visual_projection(img_pooled)

            text_out = model.text_model(input_ids=input_ids,
                                        attention_mask=attention_mask)
            txt_pooled = text_out.pooler_output if hasattr(text_out, "pooler_output") \
                else text_out.last_hidden_state[:, 0]
            txt_feat = model.text_projection(txt_pooled)

            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            v_sem = torch.cat([img_feat, txt_feat], dim=-1).cpu()  # [B, 1024]

            for i, key in enumerate(batch["keys"]):
                torch.save(v_sem[i].clone(), cache / f"{key}.pt")
                n_done += 1
            pbar.set_postfix(done=n_done)

    print(f"\nDone. computed={n_done}  total cache size:", end=" ")
    print(f"{len(list(cache.glob('*.pt')))} files")


if __name__ == "__main__":
    main()

"""Precompute and cache FND-CLIP semantic vectors (v_semantic [feat_dim]).

FND-CLIP is frozen, so v_semantic for a given (text, image) pair never
changes across epochs. Computing it once and loading from disk during
training removes the heaviest forward-pass cost from the training loop —
critical for the "computational cost" constraint in V2 §global.

Cached files are keyed by md5(text || image_path) so the cache is
deterministic and stable across re-runs.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.fnd_clip import FNDCLIP


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def feature_hash(text: str, image_path: str) -> str:
    """Stable cache key — independent of CSV row order."""
    payload = f"{text}||{os.path.abspath(image_path)}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()


class FNDFeatureDataset(Dataset):
    """Returns the inputs FND-CLIP needs plus the cache key."""

    def __init__(self, df, bert_model="bert-base-uncased",
                 clip_model="openai/clip-vit-base-patch32", max_text_len=128):
        self.df = df.reset_index(drop=True)
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        path = row["image_path"]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (224, 224), 0)
        image = self.image_tf(pil)
        bert = self.bert_tok(text, padding="max_length", truncation=True,
                             max_length=self.max_text_len, return_tensors="pt")
        clip = self.clip_proc(images=pil, text=text, return_tensors="pt",
                              padding="max_length", truncation=True,
                              max_length=77)
        return {
            "image": image,
            "bert_ids": bert["input_ids"].squeeze(0),
            "bert_mask": bert["attention_mask"].squeeze(0),
            "clip_pixels": clip["pixel_values"].squeeze(0),
            "clip_ids": clip["input_ids"].squeeze(0),
            "clip_mask": clip["attention_mask"].squeeze(0),
            "key": feature_hash(text, path),
        }


def collate(batch):
    keys = [b["key"] for b in batch]
    out = {k: torch.stack([b[k] for b in batch])
           for k in batch[0] if k != "key"}
    out["keys"] = keys
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/forensic_3class.csv")
    p.add_argument("--cache-dir", default="data/processed/fnd_features")
    p.add_argument(
        "--ckpt",
        default=None,
        help="path to a trained FND-CLIP checkpoint to load before "
             "extracting features. If omitted (the new default), FND-CLIP "
             "is initialised from its pretrained sub-modules only "
             "(BERT + ResNet50 ImageNet + frozen CLIP) and v_semantic "
             "carries no task-specific signal. Pass an explicit path "
             "ONLY if you understand the leakage risk — see HANDOFF.md.",
    )
    p.add_argument("--feat-dim", type=int, default=512)
    p.add_argument("--bert", default="bert-base-uncased")
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch-size", type=int, default=16)
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

    # Skip rows whose features are already cached, by computed key.
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

    print("Loading FND-CLIP...")
    fnd = FNDCLIP(feat_dim=args.feat_dim, num_classes=1)
    if args.ckpt is None or str(args.ckpt).lower() in ("", "none", "null"):
        print("  no --ckpt provided -> using fresh FND-CLIP "
              "(pretrained BERT/ResNet/CLIP only, no task signal). "
              "This is the leak-free default.")
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ck["model_state"]
        fnd_state = fnd.state_dict()
        compat = {k: v for k, v in state.items()
                  if k in fnd_state and fnd_state[k].shape == v.shape}
        fnd.load_state_dict(compat, strict=False)
        print(f"  loaded {len(compat)}/{len(fnd_state)} tensors from {args.ckpt}")
        print("  WARNING: loading a task-tuned checkpoint can leak labels "
              "into v_semantic — see HANDOFF.md for details.")
    fnd = fnd.to(device).eval()
    for p_ in fnd.parameters():
        p_.requires_grad = False

    ds = FNDFeatureDataset(df_todo, bert_model=args.bert, clip_model=args.clip)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    n_done = 0
    pbar = tqdm(loader, desc="FND-CLIP features")
    with torch.no_grad():
        for batch in pbar:
            v = fnd.forward_semantic(
                image=batch["image"].to(device),
                bert_ids=batch["bert_ids"].to(device),
                bert_mask=batch["bert_mask"].to(device),
                clip_pixels=batch["clip_pixels"].to(device),
                clip_ids=batch["clip_ids"].to(device),
                clip_mask=batch["clip_mask"].to(device),
            ).cpu()
            for i, key in enumerate(batch["keys"]):
                torch.save(v[i].clone(), cache / f"{key}.pt")
                n_done += 1
            pbar.set_postfix(done=n_done)

    print(f"\nDone. computed={n_done}  total cache size:", end=" ")
    total_files = len(list(cache.glob("*.pt")))
    print(f"{total_files} files")


if __name__ == "__main__":
    main()

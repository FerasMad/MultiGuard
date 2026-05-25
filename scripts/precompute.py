"""Precompute feature caches (one .pt per sample_id per modality).

Modalities:
  v_imgfor_dct   - patch-DCT [1, 224, 224]            CPU
  v_semantic_fnd - FND-CLIP semantic [768]            GPU
  v_textfor_qwen - Qwen2-7B last hidden mean [3584]   GPU
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from v4.core.logging import get_logger
from v4.core.paths import ensure_dir
from v4.core.registry import build_encoder, import_all
from v4.data.preprocessing.patch_dct import compute_patch_dct

log = get_logger(__name__)


def run_precompute(
    *,
    modality: str,
    csv_path: str,
    cache_root: str = "cache/v4",
    config_path: str | None = None,
    limit: int | None = None,
) -> str:
    import_all()
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    out_dir = Path(cache_root) / modality
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modality == "v_imgfor_dct":
        _precompute_dct(df, out_dir)
    elif modality == "v_semantic_fnd":
        _precompute_semantic(df, out_dir, config_path, device)
    elif modality == "v_textfor_qwen":
        _precompute_qwen(df, out_dir, config_path, device)
    else:
        raise ValueError(f"unknown modality: {modality}")
    return str(out_dir)


def _precompute_dct(df: pd.DataFrame, out_dir: Path) -> None:
    pbar = tqdm(df.itertuples(index=False), total=len(df), desc="patch-DCT")
    for row in pbar:
        sid = str(row.sample_id)
        shard = out_dir / f"{sid}.pt"
        if shard.exists():
            continue
        try:
            t = compute_patch_dct(row.image_path)
            torch.save(t, shard)
        except Exception as e:
            log.warning("DCT fail %s: %s", sid, e)


def _precompute_semantic(
    df: pd.DataFrame, out_dir: Path, config_path: str | None, device: torch.device
) -> None:
    import yaml
    from PIL import Image

    from v4.data.preprocessing.tokenizers import prepare_fnd_inputs

    if not config_path:
        raise ValueError("v_semantic_fnd requires --config")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc = build_encoder(dict(cfg["encoders"]["semantic"])).to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False

    pbar = tqdm(df.itertuples(index=False), total=len(df), desc="FND-CLIP")
    for row in pbar:
        sid = str(row.sample_id)
        shard = out_dir / f"{sid}.pt"
        if shard.exists():
            continue
        try:
            pil = Image.open(row.image_path).convert("RGB")
            inputs = prepare_fnd_inputs(str(row.text or ""), pil)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                v = enc(inputs).squeeze(0).cpu().float()
            torch.save(v, shard)
        except Exception as e:
            log.warning("FND-CLIP fail %s: %s", sid, e)


def _precompute_qwen(
    df: pd.DataFrame, out_dir: Path, config_path: str | None, device: torch.device
) -> None:
    import yaml

    if not config_path:
        raise ValueError("v_textfor_qwen requires --config")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    text_spec = dict(cfg["encoders"]["text"])
    text_spec["load_backbone"] = True
    enc = build_encoder(text_spec).to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False

    pbar = tqdm(df.itertuples(index=False), total=len(df), desc="Qwen")
    for row in pbar:
        sid = str(row.sample_id)
        shard = out_dir / f"{sid}.pt"
        if shard.exists():
            continue
        text = str(row.text or "")
        try:
            with torch.no_grad():
                pooled = enc.encode_text(text).squeeze(0).cpu().float()
            torch.save(pooled, shard)
        except Exception as e:
            log.warning("Qwen fail %s: %s", sid, e)

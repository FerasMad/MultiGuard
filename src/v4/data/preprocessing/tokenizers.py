"""Tokenizer + image-transform helpers for FND-CLIP, BERT, CLIP, Qwen.

Singleton per model_id so that re-instantiation doesn't re-download from HF.
"""
from __future__ import annotations

from functools import lru_cache

from v4.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=4)
def get_bert_tokenizer(model_name: str = "bert-base-uncased"):
    from transformers import BertTokenizer
    log.info("Loading BERT tokenizer: %s", model_name)
    return BertTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=4)
def get_clip_processor(model_name: str = "openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor
    log.info("Loading CLIP processor: %s", model_name)
    return CLIPProcessor.from_pretrained(model_name)


@lru_cache(maxsize=2)
def get_qwen_tokenizer(model_name: str = "Qwen/Qwen2-7B-Instruct"):
    from transformers import AutoTokenizer
    log.info("Loading Qwen tokenizer: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"   # decoder-only: real tokens flush right
    return tok


def make_image_transform():
    """Standard ImageNet transform used by FND-CLIP's visual stream."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def prepare_fnd_inputs(text: str, pil_img, *, bert_max_len: int = 128, clip_max_len: int = 77) -> dict:
    """Build the inputs FND-CLIP.forward_semantic expects from raw (text, PIL image)."""
    bert_tok = get_bert_tokenizer()
    clip_proc = get_clip_processor()
    image_tf = make_image_transform()

    img_tensor = image_tf(pil_img)
    bert = bert_tok(text, padding="max_length", truncation=True,
                    max_length=bert_max_len, return_tensors="pt")
    clip = clip_proc(images=pil_img, text=text, return_tensors="pt",
                     padding="max_length", truncation=True, max_length=clip_max_len)
    return {
        "image": img_tensor.unsqueeze(0),
        "bert_ids": bert["input_ids"],
        "bert_mask": bert["attention_mask"],
        "clip_pixels": clip["pixel_values"],
        "clip_ids": clip["input_ids"],
        "clip_mask": clip["attention_mask"],
    }


def masked_mean_pool(hidden_states, attention_mask):
    """V3.1 §4 mean pooling: average across sequence dim, EXCLUDING padding.

    Args:
        hidden_states: [B, S, H] from a transformer layer
        attention_mask: [B, S] binary (1=real token, 0=pad)

    Returns:
        [B, H] pooled, dtype-preserved.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, S, 1]
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts

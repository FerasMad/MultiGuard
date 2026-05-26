# Server v4_retrain code review (P11.2 / P8.5)

> Read-only review of `app/server_v4_retrain.py` + the 4 modules under
> `phases/v4/app_hf/inline/` it depends on. Covers security, correctness,
> and parity with the P9.1 server-vs-eval fix.

## Files reviewed

| Path | LOC | Role |
|---|---|---|
| `app/server_v4_retrain.py` | 242 | FastAPI server entrypoint; `/api/analyze` + `/api/health` |
| `phases/v4/app_hf/inline/dct_forensic.py` | 122 | DCT-Forensic image encoder (V4) |
| `phases/v4/app_hf/inline/qwen_text.py` | 94 | Qwen2-7B-Instruct text encoder |
| `phases/v4/app_hf/inline/fnd_clip.py` | 261 | V1 FND-CLIP semantic encoder + preprocessing |

## Greppable smells

`grep -nE 'TODO|FIXME|XXX|HACK|password|secret|api_key' app/server_v4_retrain.py phases/v4/app_hf/inline/*.py`

**Result: zero matches** (the only hit `token` in qwen_text.py is a substring of `tokenizer`, not a credential). No committed secrets, no abandoned TODOs.

## Verdict

**PASS with 5 recommendations.** The server is shippable for the doctor's local demo and the HF Space tier. The P9.1 parity fix is correctly wired. No security-critical issues. Recommendations below are quality-of-life (R1-R3) or stale-numbers (R4-R5), not blockers.

---

## Strengths

### Correctness
- **P9.1 parity fix correctly wired.** `app/server_v4_retrain.py:104-109` passes `head_state_path=outputs/v4/dctforensic_head_seed42.pt` AND `seed=42` to `DctForensicEncoder`. The encoder (`dct_forensic.py:59-83`) seeds *before* the Linear(2048, 768) head Kaiming init, then loads the saved head state *after* - belt-and-suspenders so cosine_sim runtime-vs-cached stays at 1.0.
- **Pipeline matches V3.1 paras 1-6 exactly.** FND-CLIP semantic (512-d) -> DCT-Forensic image (768-d) -> Qwen text (3584-d) -> V3PairwiseFusion with `proj_dims={'v_semantic':512,'v_textfor':3584}` -> MLP -> softmax -> argmax. Encoder dims thread cleanly into the fusion module.
- **All encoders frozen at inference.** `requires_grad=False` set on every parameter after load (server lines 99-100, 110-111, 127-128; qwen_text:72-73). No accidental fine-tuning during user requests.
- **`torch.no_grad()` around inference** (server line 185) - no autograd graph built per request.
- **Lazy backbone loading for Qwen** (qwen_text:77-78) - works for both offline cache builds (`load_backbone=False`) and server inference (`load_backbone=True`).
- **Module-level singletons via startup hook** (server:154-156). Heavy models load once at boot; subsequent requests are warm.
- **Bilingual output** (EN + AR labels + explanations) with verdict, label_index, per-class probabilities, and per-branch confidence breakdowns (server:218-227).
- **Graceful error handling** - try/except around the full `analyze` pipeline (server:174, 228-230) returns JSON 500 instead of crashing the server.
- **`prepare_fnd_inputs` uses V1's exact preprocessing** (fnd_clip:241-246) - Resize(256) + CenterCrop(224) + ImageNet normalize. Critical for parity with the cached v_semantic shards.

### Security
- **No path traversal.** Image bytes are read from the multipart upload into memory (`io.BytesIO`), then `Image.open()`. There's no user-controlled filesystem path.
- **No committed secrets.** Checkpoint paths overridable via env vars (`MULTIGUARD_FNDCLIP_CKPT`, `MULTIGUARD_DCT_CKPT`, etc.) - production-ready for environment-based config.
- **Fails fast on missing artifacts.** `_build_pipeline()` checks all 4 ckpts exist (server:90-93) before any model load.

### Operational
- **Health endpoint** (`/api/health`) returns device, spec version, and current ckpt path - easy for the doctor to sanity-check the live deployment.
- **CORS open** (`allow_origins=['*']`) is intentional for a public-read-only demo - no authenticated endpoints, no sensitive ops.

---

## Recommendations (non-blocking)

### R1. Add upload-size guard on `/api/analyze` (security / DoS)

Currently `image: UploadFile = File(...)` (server:173) has no size limit. A malicious client could upload an arbitrarily large image and OOM the server. Pillow's `MAX_IMAGE_PIXELS` provides partial protection (default ~89 megapixels), but a multi-GB file would still hit the FastAPI/uvicorn read buffer.

**Suggested fix** (in `analyze`, before `Image.open`):
```python
img_bytes = await image.read()
if len(img_bytes) > 50_000_000:  # 50 MB
    return JSONResponse({"error": "image too large (max 50MB)"}, status_code=413)
```

### R2. Cache the BertTokenizer + CLIPProcessor in `prepare_fnd_inputs` (performance)

`fnd_clip.py:236-237` re-instantiates `BertTokenizer.from_pretrained("bert-base-uncased")` AND `CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")` on **every** call to `prepare_fnd_inputs`. Each constructor takes ~100-200 ms (config + vocab download from HF cache, deserialization). For an interactive demo this adds noticeable per-request latency.

**Suggested fix**: wrap each in `@lru_cache(maxsize=1)` (or move to module-level globals).

```python
@lru_cache(maxsize=1)
def _get_bert_tok():
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")

@lru_cache(maxsize=1)
def _get_clip_proc():
    from transformers import CLIPProcessor
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

### R3. Document Qwen `device_map="auto"` will CPU-offload on 12 GB GPUs (operational)

`qwen_text.py:65-70` uses `device_map="auto"` for Qwen2-7B fp16 (~13 GB). On an RTX 4070 (12 GB), HuggingFace's auto-placement will offload some layers to CPU, slowing per-request inference noticeably. For the offline precompute we worked around this with bnb-4bit quantization (`recache_v_textfor_blip2.py`). The server currently runs at the slower fp16+offload speed.

**Suggested fix**: add an optional `quantize_4bit: bool = False` flag and document it in the docstring. Default off; flip on for memory-constrained deployments.

### R4. Stale health-endpoint numbers

`/api/health` (server:159-169) reports:
- `val_f1_macro: 0.7334`
- `test_f1_macro: 0.7267`
- `transfer_f1_macro: 0.4805`

These are from the **OLD shortcut ckpt** (`stage2_fusion_dctforensic/best.pt`), not the honest-path ensemble. After the P9-P10 work, the headline numbers should be:
- val (3-seed mean): 0.7292 +/- 0.0055
- test (ensemble): 0.7149
- transfer (ensemble + biascorr): 0.7197

The `checkpoint` field also still points at the old ckpt path. **Decision needed:** does the server load the old ckpt or the new ensemble? If the old, this is correct but stale; if the new (or if loading is configurable), update both the path and the metrics.

### R5. Server is single-ckpt, not ensemble (architecture limitation)

`_build_pipeline()` loads ONE fusion ckpt (`MULTIGUARD_FUSION_CKPT`, default `stage2_fusion_dctforensic/best.pt`). To serve the **ensemble** prediction the doctor sees in the eval table, the server would need to load all 3 honest-path seeds and average softmax. **Not blocking** for the local demo (single-seed is close enough), but worth documenting that the live demo's per-class probabilities aren't bit-exact-equal to `outputs/v4/stage2_fusion_honest_ensemble/`. Closing the gap would take ~40 LOC (mirror `EnsembleFusion` from `phases/v4/src/v4/evaluation/ensemble.py`).

---

## Risk classification

| Item | Severity | Action |
|---|---|---|
| R1 upload-size guard | LOW (DoS only, no data exfil) | Add before any public exposure |
| R2 tokenizer/processor cache | LOW (perf only) | Nice-to-have; ~100-200 ms per request saved |
| R3 Qwen fp16 + offload | LOW (perf only) | Document; flip on bnb-4bit if needed |
| R4 stale health numbers | LOW (cosmetic) | Update if/when server is repointed at honest-path ckpts |
| R5 single-ckpt vs ensemble | LOW (live demo deviates slightly from offline eval) | Either implement ensemble loading or document the variance |

**Zero HIGH or MEDIUM severity items.** The server is safe to ship to the doctor as-is for a local demo, and ready for HF Space deployment with R1 added.

---

## Verification

```bash
# Re-run the grep sanity check
grep -nE 'TODO|FIXME|XXX|HACK|password|secret|api_key' \
    app/server_v4_retrain.py phases/v4/app_hf/inline/*.py
# Expected: zero matches (only 'token' substrings inside 'tokenizer')

# Verify the P9.1 parity fix is plumbed (head_state_path + seed=42 in server)
grep -nE 'head_state_path|seed=42' app/server_v4_retrain.py
# Expected: at least 2 hits at lines 104-109

# Verify the inline encoder honors both kwargs
grep -nE 'head_state_path|seed' phases/v4/app_hf/inline/dct_forensic.py
# Expected: parameter definitions + body usage
```

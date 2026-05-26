# Inference flow -- what happens when you click "Analyze Article"

End-to-end trace of the V3.1 honest-path pipeline running on
`huggingface.co/spaces/FerasMad/multiguard-demo`. Tensor shapes and module
names taken straight from `phases/v4/app_hf/app.py` + `inline/*.py`.

```
USER CLICKS "Analyze Article"  (with text + image)
        |
        v
+--------------------------------------------------------------------+
| Step 1.  Browser -> HF Space (Gradio frontend)                     |
|                                                                    |
|   POST /queue/join                                                 |
|   payload: { image: <base64 jpg>, text: "Breaking news..." }       |
|   Gradio queues the call, returns event_id                         |
+--------------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------------+
| Step 2.  HF ZeroGPU allocates an A10G GPU for this request         |
|                                                                    |
|   @spaces.GPU(duration=120) wakes up the analyze() function.       |
|   Cold-start: ~5-10s waiting for GPU slot.                         |
|   Warm (within 120s of last call): ~0s wait.                       |
+--------------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------------+
| Step 3.  Move models from CPU to cuda (idempotent)                 |
|                                                                    |
|   _state["fndclip"].to("cuda")        # ~580 MB                    |
|   _state["dct_forensic"].to("cuda")   # ~280 MB                    |
|   for fusion in _state["fusions"]:    # 3 x ~80 MB                 |
|       fusion.to("cuda")                                            |
|   Qwen2-7B (~14 GB fp16) loads lazily on first encode_text call.   |
|   First-ever call: ~30-60s. Subsequent calls within window: ~0s.   |
+--------------------------------------------------------------------+
        |
        +-------------------+-------------------+-------------------+
        | (parallel-ish)    | (parallel-ish)    | (parallel-ish)    |
        v                   v                   v
+---------------------+ +---------------------+ +---------------------+
| Step 4a. SEMANTIC   | | Step 4b. IMAGE      | | Step 4c. TEXT       |
| (FND-CLIP V1)       | | (DCT-Forensic)      | | (Qwen2-7B-Instruct) |
|                     | |                     | |                     |
| prepare_fnd_inputs()| | compute_dual_dct()  | | tokenizer(text,     |
|   Resize 256 + CC224| |   YCbCr -> Y chan   | |     padding=left,   |
|   ImageNet norm     | |   8x8 patches DCT   | |     truncate=512)   |
|   BERT tok(128)     | |   16x16 patches DCT | |                     |
|   CLIP processor    | |   log scale         | | qwen_model(...,     |
|                     | |   avg -> [1,224,224]| |   output_hidden_    |
| forward_semantic:   | |                     | |   states=True)      |
|   ResNet50 image    | | z-score normalize:  | |                     |
|   BERT text         | |   (t-mean)/(std+eps)| | h = hidden_states   |
|   CLIP fused        | |                     | |       [-1]          |
|   modality attn     | | DctForensicEncoder: | |                     |
|     (3-way weighted | |   1-ch conv1 RN50   | | masked-mean pool:   |
|      sum)           | |   backbone -> 2048  | |   (h * mask).sum    |
|                     | |   head: Drop+Lin+   | |   / mask.sum        |
| output:             | |    GELU -> 768      | |                     |
|  v_semantic [B,512] | |                     | | output:             |
|                     | | output:             | |  v_textfor [B,3584] |
|                     | |  v_imgfor [B,768]   | |                     |
+---------------------+ +---------------------+ +---------------------+
        |                       |                       |
        |   <--- collected into a single dict --->      |
        |                       |                       |
        v                       v                       v
+--------------------------------------------------------------------+
| Step 5.  V3 Pairwise Fusion (run 3 times, once per seed)           |
|                                                                    |
|   for fusion in [seed_42, seed_1337, seed_2024]:                   |
|                                                                    |
|     5.1  Projections to common dim (768)                           |
|          sem_proj:  Linear(512, 768) + GELU       [B,512]->[B,768] |
|          img_proj:  Identity                      [B,768]->[B,768] |
|          text_proj: Linear(3584, 768) + GELU      [B,3584]->[B,768]|
|                                                                    |
|     5.2  LayerNorm per signal                                      |
|          s = LN(v_semantic)                                        |
|          i = LN(v_imgfor)                                          |
|          t = LN(v_textfor)                                         |
|                                                                    |
|     5.3  Three pairwise cross-attentions (8 heads, batch_first)    |
|          For each pair (X, Y):                                     |
|              dir1 = MHA(query=X, key=Y, value=Y)                   |
|              dir2 = MHA(query=Y, key=X, value=X)                   |
|              pair_out = LayerNorm(dir1 + dir2)    <-- ELEMENT-SUM  |
|          r1 = pair(s, i)                                           |
|          r2 = pair(s, t)                                           |
|          r3 = pair(i, t)                                           |
|                                                                    |
|     5.4  Stack -> Conv1d stack -> AdaptiveAvgPool                  |
|          stacked = stack([r1, r2, r3], dim=1)   -> [B, 3, 768]     |
|          permute to [B, 768, 3]                                    |
|          Conv1d(768, 768, k=3, pad=1) + GELU                       |
|          Conv1d(768, 1024, k=1)       + GELU                       |
|          AdaptiveAvgPool1d(1).squeeze -> [B, 1024]  (fused vector) |
|                                                                    |
|     5.5  MLP classifier (V3.1 sec 6, strict)                       |
|          Linear(1024, 512) + BN + GELU + Dropout(0.5)              |
|          Linear(512,  256) + GELU                                  |
|          Linear(256,    5) -> main_logits [B, 5]                   |
|                                                                    |
|     5.6  softmax -> probs_seed [B, 5]                              |
|                                                                    |
|     (aux head Linear(768,2) on detached v_imgfor                   |
|      runs at train time only; ignored at inference)                |
+--------------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------------+
| Step 6.  Ensemble: softmax average across 3 seeds                  |
|                                                                    |
|   probs = stack([probs_42, probs_1337, probs_2024]).mean(dim=0)    |
|                                                                    |
|   final probs (example): [0.02, 0.04, 0.06, 0.85, 0.03]            |
|                          [Real,OOC, Manip,AI-T, FullFab]           |
+--------------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------------+
| Step 7.  Build response                                            |
|                                                                    |
|   pred = argmax(probs) = 3                                         |
|   confidence = round(probs[3] * 100, 1) = 85.0                     |
|                                                                    |
|   modules (derived from probs):                                    |
|     text_ai       = probs[3] + probs[4]              = 0.88        |
|     text_patterns = 0.5*text_ai + 0.5*probs[1]       = 0.46        |
|     image_manip   = probs[2] + probs[4]              = 0.09        |
|     cross_modal   = probs[1]                         = 0.04        |
|     overall_risk  = 1 - probs[0]                     = 0.98        |
|                                                                    |
|   verdict_en = LABELS[3]      = "AI-Text"                          |
|   verdict_ar = LABELS_AR[3]   = (Arabic equivalent)                |
|   explanation_en/ar = LABEL_EXPLANATIONS[3] / _AR                  |
+--------------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------------+
| Step 8.  Render Gradio outputs (HTML strings)                      |
|                                                                    |
|   verdict_out (gr.HTML): colored verdict card + confidence bar     |
|                          + explanation paragraph                   |
|   probs_out (gr.HTML):   5 horizontal probability bars per class   |
|   modules_out (gr.HTML): module breakdown table (5 rows)           |
+--------------------------------------------------------------------+
        |
        v
+--------------------------------------------------------------------+
| Step 9.  HF Space pushes results back via /queue/data SSE          |
|                                                                    |
|   Browser receives 3 HTML strings, renders them into the result    |
|   panels next to the Analyze Article button.                       |
|                                                                    |
|   Total wallclock typical: 5-15s warm, 30-60s cold start.          |
+--------------------------------------------------------------------+
```

## File-to-step mapping

| File | Steps |
|---|---|
| `phases/v4/app_hf/app.py` | 1-3, 6-9 (Gradio entrypoint, ensemble averaging, response shaping) |
| `phases/v4/app_hf/inline/fnd_clip.py` | 4a (FND-CLIP V1 semantic encoder) |
| `phases/v4/app_hf/inline/dual_dct.py` | 4b preprocessing (dual-DCT) |
| `phases/v4/app_hf/inline/dct_forensic.py` | 4b encoder (1-ch RN50 + head) |
| `phases/v4/app_hf/inline/qwen_text.py` | 4c (Qwen2-7B masked-mean pooling) |
| `phases/v4/app_hf/inline/v3_pairwise.py` | 5.1-5.4 (3-pair MHA-SUM + Conv1d stack) |
| `phases/v4/app_hf/inline/mlp_head.py` | 5.5 (V3.1 sec 6 strict MLP) |
| `phases/v4/app_hf/inline/class_map.py` | Step 7 labels + explanations |

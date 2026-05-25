#!/bin/bash
set -e
cd /c/Desktop/Multimodal-fake-news-detection
PY="/c/Users/FSOS/AppData/Local/Programs/Python/Python313/python.exe"
VPY=".venv-qwen-py312/Scripts/python.exe"

echo "=========== Step 1/3: RoBERTa v_textfor c4fix (~3 min) ============"
date
"$PY" -u v3/scripts/precompute_textfor_deberta.py \
    --csv data/processed/forensic_5class_c4fix.csv \
    --cache-dir v3/cache/v_textfor_c4fix \
    --model-name roberta-base \
    --batch-size 16

echo ""
echo "=========== Step 2/3: FND-CLIP v_semantic c4fix (~10 min) ========"
date
"$PY" -u v3/scripts/precompute_v_semantic.py \
    --csv data/processed/forensic_5class_c4fix.csv \
    --cache-dir v3/cache/v_semantic_c4fix \
    --ckpt outputs/v1_leakfree/best.pt \
    --batch-size 16

echo ""
echo "=========== Step 3/3: Qwen v_textfor_qwen c4fix (~15 min) ========"
date
"$VPY" -u v3/scripts/precompute_qwen_textfor_v3_1.py \
    --csv data/processed/forensic_5class_c4fix.csv \
    --cache-dir v3/cache/v_textfor_qwen_c4fix \
    --batch-size 4 --max-length 256 \
    --no-quant --gpu-mem-gib 10

echo ""
echo "=========== ALL THREE C4FIX RECACHE STEPS DONE ==================="
date
for d in v_semantic_c4fix v_textfor_c4fix v_textfor_qwen_c4fix; do
    echo "  $d: $(ls v3/cache/$d/ | wc -l) shards"
done

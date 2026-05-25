#!/bin/bash
set -e
cd /c/Desktop/Multimodal-fake-news-detection
PY="/c/Users/FSOS/AppData/Local/Programs/Python/Python313/python.exe"

echo "========== Step 1/2: eval RoBERTa c4fix (test + MMFB ~10min) =========="
date
"$PY" -u v3/src/evaluate_v3_pipeline.py \
    --checkpoint v3/outputs/v3_pipeline_c4fix/best.pt \
    --csv data/processed/forensic_5class_c4fix.csv \
    --cache-root v3/cache \
    --out-dir v3/outputs/v3_pipeline_c4fix \
    --probe-mmfb \
    --mmfb-root data/raw/MMFakeBench \
    --images-root data/raw/MMFakeBench/images \
    --batch-size 16 2>&1 | grep -v "^eval:" | grep -v "MMFakeBench probe:" | grep -v "^Loading weights:"

echo ""
echo "========== Step 2/2: eval Qwen c4fix (test + MMFB ~30min) ============="
date
"$PY" -u v3/src/evaluate_v3_pipeline.py \
    --checkpoint v3/outputs/v3_pipeline_qwen_c4fix/best.pt \
    --csv data/processed/forensic_5class_c4fix.csv \
    --cache-root v3/cache \
    --out-dir v3/outputs/v3_pipeline_qwen_c4fix \
    --probe-mmfb \
    --mmfb-root data/raw/MMFakeBench \
    --images-root data/raw/MMFakeBench/images \
    --batch-size 8 2>&1 | grep -v "^eval:" | grep -v "MMFakeBench probe:" | grep -v "^Loading weights:"

echo ""
echo "========== BOTH C4FIX EVALS DONE ====================================="
date

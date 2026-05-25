#!/bin/bash
set -e
cd /c/Desktop/Multimodal-fake-news-detection
PY="/c/Users/FSOS/AppData/Local/Programs/Python/Python313/python.exe"

echo "=========== Step 1/2: train RoBERTa c4fix (~15 min) ============"
date
"$PY" -u v3/src/train_v3_pipeline.py \
    --config v3/configs/v3_pipeline_c4fix.yaml \
    > v3/outputs/v3_pipeline_c4fix/train.log 2>&1
echo "RoBERTa c4fix done"
grep -E "Best val F1|Early stopping" v3/outputs/v3_pipeline_c4fix/train.log | tail -3

echo ""
echo "=========== Step 2/2: train Qwen c4fix (~15 min) ==============="
date
"$PY" -u v3/src/train_v3_pipeline.py \
    --config v3/configs/v3_pipeline_qwen_c4fix.yaml \
    > v3/outputs/v3_pipeline_qwen_c4fix/train.log 2>&1
echo "Qwen c4fix done"
grep -E "Best val F1|Early stopping" v3/outputs/v3_pipeline_qwen_c4fix/train.log | tail -3

echo ""
echo "=========== BOTH C4FIX TRAININGS DONE ============================"
date

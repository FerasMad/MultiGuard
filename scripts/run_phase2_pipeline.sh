#!/usr/bin/env bash
# End-to-end Phase 2 pipeline.
# Builds the 3-class dataset, precomputes DCT + FND-CLIP feature caches,
# trains Step 1 (forensic baseline) and Step 2 (full pipeline), and
# evaluates both on the test split + MMFakeBench transfer.
#
# Run from the repo root:
#
#     bash scripts/run_phase2_pipeline.sh
#
# Pre-requisites:
#   pip install -r requirements.txt    (or requirements-v1.txt + extras)
#   data/raw/DGM4/                     (rshaojimmy/DGM4)
#   data/raw/NewsCLIPpings/            (g-luo/news_clippings)
#   data/raw/MMFakeBench/              (liuxuannan/MMFakeBench, optional)
#   data/processed/balanced_dataset.csv (the V1 OOC CSV — already in repo
#                                       outputs if V1 has been built once)
#
# Outputs:
#   data/processed/forensic_3class.csv
#   data/processed/dct_cache/
#   data/processed/fnd_features/
#   outputs/forensic_baseline/best.pt + metrics yaml + ROC pngs
#   outputs/full_pipeline/best.pt     + metrics yaml + ROC pngs

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"

LOG_DIR="outputs/phase2_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pipeline.log"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }
fail() { log "ERROR: $1"; exit 1; }

require_dir() {
    if [ ! -d "$1" ]; then
        fail "Required directory missing: $1
$2"
    fi
}

log "=== Phase 2 pipeline start ==="
log "Repo root: $ROOT"
log "Log dir:   $LOG_DIR"

# -----------------------------------------------------------
# 0. Sanity: required input data + Python deps
# -----------------------------------------------------------
log "Checking input data..."
require_dir "data/raw/DGM4" \
    "Download from https://huggingface.co/datasets/rshaojimmy/DGM4 to data/raw/DGM4"
require_dir "data/raw/NewsCLIPpings" \
    "Download from https://github.com/g-luo/news_clippings to data/raw/NewsCLIPpings"

if [ ! -f "data/processed/balanced_dataset.csv" ]; then
    fail "data/processed/balanced_dataset.csv missing.
The Phase 2 dataset builder reuses NewsCLIPpings OOC pairs from V1's
balanced_dataset.csv. Run the V1 dataset builder first:

    python scripts/build_balanced_dataset.py --dgm4 data/raw/DGM4 \\
        --newsclippings data/raw/NewsCLIPpings \\
        --output data/processed/balanced_dataset.csv"
fi

if ! python -c "import torch, torchvision, cv2, pandas, sklearn, scipy, transformers" \
     >/dev/null 2>&1; then
    fail "Python deps missing. Run: pip install -r requirements.txt"
fi
log "  ok"

# -----------------------------------------------------------
# 1. Build 3-class CSV
# -----------------------------------------------------------
if [ ! -f "data/processed/forensic_3class.csv" ]; then
    log "Step 1: building 3-class dataset CSV..."
    python scripts/build_forensic_3class.py 2>&1 | tee -a "$LOG"
else
    log "Step 1: data/processed/forensic_3class.csv exists, skipping."
fi

# -----------------------------------------------------------
# 2. Precompute DCT cache (uniform JPEG quality)
# -----------------------------------------------------------
if [ ! -d "data/processed/dct_cache" ] || \
   [ -z "$(ls -A data/processed/dct_cache 2>/dev/null)" ]; then
    log "Step 2: precomputing DCT cache (uniform JPEG q=85)..."
    python scripts/precompute_dct.py 2>&1 | tee -a "$LOG"
else
    log "Step 2: data/processed/dct_cache populated, skipping."
    log "       (rm -rf data/processed/dct_cache to force rebuild.)"
fi

# -----------------------------------------------------------
# 3. Precompute FND-CLIP features (leak-free, fresh backbone)
# -----------------------------------------------------------
if [ ! -d "data/processed/fnd_features" ] || \
   [ -z "$(ls -A data/processed/fnd_features 2>/dev/null)" ]; then
    log "Step 3: precomputing FND-CLIP v_semantic cache (leak-free)..."
    python scripts/precompute_fnd_features.py 2>&1 | tee -a "$LOG"
else
    log "Step 3: data/processed/fnd_features populated, skipping."
    log "       (rm -rf data/processed/fnd_features to force rebuild.)"
fi

# -----------------------------------------------------------
# 4. Train Step 1 (forensic baseline)
# -----------------------------------------------------------
if [ ! -f "outputs/forensic_baseline/best.pt" ]; then
    log "Step 4: training Step 1 forensic baseline..."
    python src/train_forensic.py 2>&1 | tee -a "$LOG"
else
    log "Step 4: outputs/forensic_baseline/best.pt exists, skipping."
    log "       (rm to force retrain.)"
fi

# -----------------------------------------------------------
# 5. Train Step 2 (full pipeline)
# -----------------------------------------------------------
if [ ! -f "outputs/full_pipeline/best.pt" ]; then
    log "Step 5: training Step 2 full pipeline..."
    python src/train_full_pipeline.py 2>&1 | tee -a "$LOG"
else
    log "Step 5: outputs/full_pipeline/best.pt exists, skipping."
    log "       (rm to force retrain.)"
fi

# -----------------------------------------------------------
# 6. Evaluate both
# -----------------------------------------------------------
log "Step 6: evaluating Step 1..."
python src/evaluate_forensic.py \
    --checkpoint outputs/forensic_baseline/best.pt 2>&1 | tee -a "$LOG"

log "Step 7: evaluating Step 2..."
python src/evaluate_full_pipeline.py \
    --checkpoint outputs/full_pipeline/best.pt 2>&1 | tee -a "$LOG"

log "=== Phase 2 pipeline complete ==="
log "Step 1 metrics:  outputs/forensic_baseline/{test,mmfb}_metrics.yaml"
log "Step 2 metrics:  outputs/full_pipeline/{test,mmfb}_metrics.yaml"
log "ROC curves:      outputs/{forensic_baseline,full_pipeline}/{test,mmfb}_roc.png"
log "Pipeline log:    $LOG"

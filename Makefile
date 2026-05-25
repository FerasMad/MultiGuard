PY := python
PIP := $(PY) -m pip
PYTEST := $(PY) -m pytest

DATA_ROOT := data/raw
CACHE_ROOT := cache/v4
OUTPUTS := outputs/v4
# Post-unification (P2.5): configs live under phases/v4/configs/
CONFIG_QWEN := phases/v4/configs/v4_pipeline_qwen.yaml
CONFIG_STAGE0 := phases/v4/configs/v4_pipeline_stage0.yaml
CONFIG_STAGE1 := phases/v4/configs/v4_pipeline_stage1.yaml

.PHONY: help
help:
	@echo "MultiGuard - Makefile targets"
	@echo ""
	@echo "V4 multimodal pipeline:"
	@echo "  setup                Install package + dev deps"
	@echo "  data                 Build manifests + leakage audit"
	@echo "  precompute           Precompute feature caches"
	@echo "  stage0/1/2           Train each stage"
	@echo "  train                Run stage0 + stage1 + stage2 sequentially"
	@echo "  eval                 Full V3.1 section 7 evaluation"
	@echo "  server               Launch FastAPI on port 8081"
	@echo ""
	@echo "Forensic Image Detector (P5):"
	@echo "  forensic-data        Prep GenImage_v2 + splits + DCT cache + stats"
	@echo "  forensic-train-dct   Approach 2 (DCT) training"
	@echo "  forensic-train-rgb   Approach 1 (RGB+Fourier) training"
	@echo "  forensic-train       Both approaches sequentially"
	@echo "  forensic-eval        Both evals + combined F.25 table"
	@echo "  forensic-report      Build REPORT.md + REPORT.docx + plots"
	@echo "  forensic-all         Full chain: data -> train -> eval -> report"
	@echo "  forensic-download    Pull both trained ckpts from HuggingFace Hub"
	@echo ""
	@echo "Dev:"
	@echo "  test/lint/fmt        Dev workflows"
	@echo "  clean                Remove build artifacts"

.PHONY: setup
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[gpu,server,dev]"
	pre-commit install || true

.PHONY: data
data:
	$(PY) -m v4.cli build-manifest --source newsclippings
	$(PY) -m v4.cli build-manifest --source dgm4
	$(PY) -m v4.cli build-manifest --source mmfakebench
	$(PY) -m v4.cli merge-manifest
	$(PY) -m v4.cli build-manifest --source stage1
	$(PY) -m v4.cli leakage-audit
	$(PY) -m v4.cli verify-image-paths

.PHONY: precompute
precompute:
	$(PY) -m v4.cli precompute --modality v_imgfor_dct
	$(PY) -m v4.cli precompute --modality v_semantic_fnd --config $(CONFIG_QWEN)
	$(PY) -m v4.cli precompute --modality v_textfor_qwen --config $(CONFIG_QWEN)

.PHONY: stage0 stage1 stage2 train
stage0:
	$(PY) -m v4.cli train --config $(CONFIG_STAGE0)

stage1:
	$(PY) -m v4.cli train --config $(CONFIG_STAGE1)

stage2:
	@for SEED in 42 1337 2024; do \
		echo "=== Stage 2 - seed $$SEED ==="; \
		$(PY) -m v4.cli train --config $(CONFIG_QWEN) --seed $$SEED --out-dir $(OUTPUTS)/stage2_fusion/seed_$$SEED; \
	done

train: stage0 stage1 stage2

.PHONY: eval
eval:
	$(PY) -m v4.cli eval --config $(CONFIG_QWEN) --split test
	$(PY) -m v4.cli eval --config $(CONFIG_QWEN) --split mmfakebench-transfer

.PHONY: server
server:
	$(PY) -m v4.cli server --port 8081 --config app/server_config.yaml

.PHONY: test lint fmt
test:
	$(PYTEST) -v

lint:
	ruff check phases/v4/src phases/forensic/src phases/v4/tests phases/forensic/tests tests app scripts shared
	ruff format --check phases/v4/src phases/forensic/src phases/v4/tests phases/forensic/tests tests app scripts shared

fmt:
	ruff check --fix phases/v4/src phases/forensic/src phases/v4/tests phases/forensic/tests tests app scripts shared
	ruff format phases/v4/src phases/forensic/src phases/v4/tests phases/forensic/tests tests app scripts shared

.PHONY: forensic-data forensic-train-dct forensic-train-rgb forensic-train forensic-eval forensic-report forensic-all forensic-download
forensic-data:
	$(PY) phases/forensic/scripts/prepare_genimage_v2.py --target-per-gen 1750
	$(PY) phases/forensic/scripts/build_splits.py --train-per-gen 1250 --test-per-gen 500 --val-frac 0.20
	$(PY) phases/forensic/scripts/precompute_dct.py --workers 4
	$(PY) phases/forensic/scripts/compute_dct_stats.py

forensic-train-dct:
	$(PY) phases/forensic/scripts/train_dct.py --out-dir phases/forensic/outputs/dct

forensic-train-rgb:
	$(PY) phases/forensic/scripts/train_rgb_fourier.py --out-dir phases/forensic/outputs/rgb

forensic-train: forensic-train-dct forensic-train-rgb

forensic-eval:
	$(PY) phases/forensic/scripts/eval_dct.py --ckpt phases/forensic/outputs/dct/forensic_dct_model.pth
	$(PY) phases/forensic/scripts/eval_rgb.py --ckpt phases/forensic/outputs/rgb/forensic_rgb_model.pth
	$(PY) phases/forensic/scripts/build_combined_eval_table.py

forensic-report:
	$(PY) phases/forensic/scripts/plot_training.py \
	    --csv phases/forensic/outputs/dct/training_history.csv \
	    --out phases/forensic/outputs/training_curves.png
	$(PY) phases/forensic/scripts/plot_training.py \
	    --csv phases/forensic/outputs/rgb/training_history.csv \
	    --out phases/forensic/outputs/training_curves_rgb.png \
	    --title "Approach 1 (RGB + Fourier mask) training: train_loss + val_ap vs epoch"
	$(PY) phases/forensic/scripts/build_handoff_report.py

forensic-download:
	hf download FerasMad/forensic-dct-v1 forensic_dct_model.pth --local-dir phases/forensic/outputs/dct/
	hf download FerasMad/forensic-rgb-v1 forensic_rgb_model.pth --local-dir phases/forensic/outputs/rgb/
	hf download FerasMad/forensic-dct-v1 dct_stats.json --local-dir phases/forensic/data/

forensic-all: forensic-data forensic-train forensic-eval forensic-report

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf .coverage htmlcov/

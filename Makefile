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
	@echo "MultiGuard V4 - Makefile targets"
	@echo ""
	@echo "  setup           Install package + dev deps"
	@echo "  data            Build manifests + leakage audit"
	@echo "  precompute      Precompute feature caches"
	@echo "  stage0/1/2      Train each stage"
	@echo "  train           Run stage0 + stage1 + stage2 sequentially"
	@echo "  eval            Full V3.1 section 7 evaluation"
	@echo "  server          Launch FastAPI on port 8081"
	@echo "  test/lint/fmt   Dev workflows"
	@echo "  clean           Remove build artifacts"

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

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf .coverage htmlcov/

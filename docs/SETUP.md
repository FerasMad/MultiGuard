# Setup - Windows + Python 3.12 + CUDA 12.4

This document covers installation on the V4 implementation PC (Windows 11, dual RTX 4090, 48 GB total VRAM). For Linux, the same `pip install -e ".[gpu,server,dev]"` works; skip the Windows-specific notes.

For raw-dataset acquisition (DGM4, MMFakeBench, VisualNews, GenImage, Qwen2-7B weights), see Step 6 below. The exact mechanism (HuggingFace, manual download, file transfer from another PC) is environment-specific and not committed to the repo.

---

## 1. Prerequisites

| Requirement | Verify |
|-------------|--------|
| Windows 11 (or Linux x86_64) | `winver` |
| Python 3.12.x | `python --version` (must NOT be the MS Store stub - see below) |
| Git >= 2.40 | `git --version` |
| CUDA 12.4 driver | `nvidia-smi` (driver version >= 550) |
| >= 150 GB free disk | `Get-PSDrive` (PowerShell) |

### Microsoft Store python stub (Windows gotcha)

`where python` may show `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe` - that's the stub that opens Microsoft Store. Disable it:

1. Settings -> Apps -> Advanced app settings -> App execution aliases.
2. Turn OFF both `python.exe` and `python3.exe`.
3. Install real Python from https://www.python.org/downloads/ (3.12.x) and tick "Add to PATH".

Verify:
```powershell
where python
python --version
# Expected: C:\Python312\python.exe (or similar real path) + 3.12.x
```

---

## 2. Clone the repo

```powershell
cd C:\
git clone https://github.com/FerasMad/MultiGuard.git
cd MultiGuard
```

---

## 3. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks the activation script:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 4. Install dependencies

For everything (training, server, dev tools):

```powershell
pip install -e ".[gpu,server,dev]"
```

Or via the Makefile (requires `make` - Git Bash ships with it):

```bash
make setup
```

### 4.1 Install torch + CUDA 12.4 wheel manually

`pyproject.toml` pins `torch>=2.6.0`. On Windows this MUST come from the PyTorch index, not PyPI's CPU wheel:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
```

Verify CUDA is wired up:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: 2.6.0+cu124 True 2   (or 1 if you're testing on a single-GPU host)
```

---

## 5. HuggingFace authentication

Required for MMFakeBench + BERT/CLIP auto-download. DGM4 download from HF requires accepting terms in the web UI first; if you have a copy of DGM4 from another environment, transferring it directly avoids the HF terms dance.

```powershell
huggingface-cli login
# Paste a READ-scope token from https://huggingface.co/settings/tokens.
# The token persists in %USERPROFILE%\.cache\huggingface\token and is gitignored.
```

---

## 6. Data acquisition

Acquire the raw datasets via whichever mechanism fits your environment (HuggingFace downloads, manual web UIs, file transfer from another machine). At minimum the new PC needs:

| Location | Contents |
|----------|----------|
| `data/raw/DGM4/` | ~80 GB |
| `data/raw/MMFakeBench/` | ~3 GB |
| `data/raw/NewsCLIPpings/` | ~200 MB annotations |
| `data/raw/visualnews/origin/` | ~50 GB (or subset) |
| `data/raw/GenImage/` | ~10 GB (8 generators x 1250) |
| `data/raw/pretrained/blur_jpg_prob0.pth` | ~100 MB |
| `%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2-7B-Instruct\` | ~15 GB |

---

## 7. Build manifests + audit

```powershell
make data
```

Which expands to:
```powershell
python -m v4.cli build-manifest --source newsclippings
python -m v4.cli build-manifest --source dgm4
python -m v4.cli build-manifest --source mmfakebench
python -m v4.cli merge-manifest
python -m v4.cli build-manifest --source stage1
python -m v4.cli leakage-audit
python -m v4.cli verify-image-paths
```

Expected output: `data/processed/forensic_5class_v4.csv` (~15,000 rows) and `data/processed/univfd_stage1.csv` (~30,000 rows). Leakage audit MUST pass before training.

---

## 8. Precompute feature caches

```powershell
make precompute
```

Caches:
- `cache/v4/v_imgfor_dct/` - patch-DCT (CPU, ~30 min)
- `cache/v4/v_semantic_fnd/` - FND-CLIP semantic (GPU, ~1.5 h)
- `cache/v4/v_textfor_qwen/` - Qwen2-7B last layer (GPU, ~4 h)

---

## 9. Train + evaluate

```powershell
make train     # Stage 0 (FND-CLIP) -> Stage 1 (UnivFD) -> Stage 2 (Fusion, 3 seeds)
make eval      # V3.1 section 7 in-distribution + MMFakeBench transfer probe
```

Stage 2 produces three checkpoints under `outputs/v4/stage2_fusion/seed_{42,1337,2024}/`. Eval reports mean and std across seeds.

---

## 10. Launch the FastAPI server

```powershell
make server
# Server listens on http://localhost:8081
# Health: GET /api/health
# Analyze: POST /api/analyze (multipart form: text=..., image=@...)
```

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `torch` import shows DLL error | Wrong CUDA wheel | Reinstall via the cu124 index URL (Step 4.1) |
| `cv2.dct` segfault | `cv2` imported before `torch` | Always `import torch` first (see `CONTRIBUTING.md`) |
| `cuda.is_available()` is False | Driver < 550 or wrong wheel | `nvidia-smi` -> update driver |
| MS Store opens when running `python` | Python stub still enabled | Step 1 disable + reinstall real Python |
| `huggingface-cli` "401 Unauthorized" | Token expired | `huggingface-cli login` with a fresh READ token |
| `pre-commit install` "command not found" | dev extras not installed | `pip install -e ".[dev]"` then re-run |
| Stage 2 OOM on 4090 | bf16 disabled or batch > 64 | Confirm `train.precision: bf16` in YAML; reduce `batch_size` |

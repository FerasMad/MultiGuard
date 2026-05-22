# MultiGuard V4 — Dataset Acquisition Recipe

**TeamViewer-hybrid strategy.** Run this on the new PC BEFORE any V4 code lands. The companion plan document is `docs/V4_PLAN.md` (this file is Appendix A of that plan, extracted for standalone use).

**Strategy:** TeamViewer File Transfer from the current PC (which already has all the bulk data) for the big/manual datasets (DGM4, VisualNews, GenImage, Qwen HF cache, blur_jpg). HuggingFace auto-download for the small/clean ones (MMFakeBench, BERT, CLIP). Saves manual web-UI steps and avoids Google Drive quota issues; requires the current PC to be awake during transfer (~9 hours overnight) plus ~10 min of HF downloads.

## A.0 — Prerequisites

| Requirement | How to verify |
|---|---|
| Python ≥ 3.10 on new PC PATH | `python --version` |
| ≥ 150 GB free disk on new PC data drive | `Get-PSDrive C` (PowerShell) or `df -h .` (Linux) |
| TeamViewer connected to current PC with **File Transfer** mode | TeamViewer → Files & Extras → File Transfer |
| Current PC stays AWAKE during the bulk transfer | Settings → Power → Screen and Sleep → Never on current PC |
| HuggingFace account (still needed for MMFakeBench + BERT/CLIP) | https://huggingface.co/join |
| Internet ≥ 10 Mbps on new PC (small HF downloads only) | `speedtest-cli` |

## A.1 — Install tools (one-time)

```bash
pip install --upgrade "huggingface_hub[cli]" tqdm
```

## A.2 — HuggingFace authentication

```bash
huggingface-cli login
# Paste a token with READ access. Create at https://huggingface.co/settings/tokens.
# The token persists in ~/.cache/huggingface/token. Do NOT commit it.
```

## A.3 — TeamViewer File Transfer setup

1. On the new PC, open TeamViewer.
2. Enter the current PC's TeamViewer ID and choose **"File Transfer"** (NOT Remote Control). Or: connect normally, then Files & Extras → Open File Transfer.
3. In the file-transfer window: **left pane = new PC (destination)**, **right pane = current PC (source)**.
4. Source root for all bulk transfers below: `C:\Desktop\Multimodal-fake-news-detection\data\raw\` on the current PC.
5. Destination root: `C:\path\to\MultiGuard\data\raw\` on the new PC (the cloned V4 repo).

**Throughput tips:**
- TeamViewer relay tops out at ~5 MB/s. Direct P2P (LAN or same NAT) reaches 20+ MB/s — TeamViewer auto-negotiates this.
- Start one big folder transfer and leave it overnight. Disable sleep on BOTH PCs.
- TeamViewer File Transfer supports resume — if interrupted, restart and accept "Resume" when prompted.

**Note:** because we're transferring DGM4 directly from the current PC, **DGM4 HuggingFace terms acceptance is NOT required** in this hybrid recipe. Skip it.

## A.4 — Create the workspace

```bash
mkdir -p data/raw
cd data/raw
```

After this section runs, `data/raw/` will contain:

```
data/raw/
├── DGM4/                  ~80 GB
├── MMFakeBench/           ~3 GB
├── NewsCLIPpings/         ~200 MB (annotations)
├── visualnews/            ~3-50 GB (images, subset or full)
├── GenImage/              ~10 GB (10k subset across 8 generators)
└── pretrained/
    └── blur_jpg_prob0.pth ~100 MB

# Plus, outside data/raw/ but referenced:
~/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/  ~15 GB
```

## A.5 — Per-dataset downloads (can run in parallel in separate terminals)

### A.5.1 — NewsCLIPpings annotations (~200 MB) — TeamViewer or manual

The current PC may already have these from V3. **TeamViewer-transfer if found** at e.g. `C:\Desktop\Multimodal-fake-news-detection\data\raw\NewsCLIPpings\`.

Fallback (if not on current PC):
1. Visit https://github.com/g-luo/news_clippings
2. Follow the **"Annotations"** section's Google Drive link
3. Download `news_clippings.tar.gz`
4. Extract to `data\raw\NewsCLIPpings\`:
   ```bash
   cd data/raw/NewsCLIPpings
   tar -xzf news_clippings.tar.gz
   ```

Expected: `NewsCLIPpings/news_clippings/data/{merged_balanced,...}.json`

### A.5.2 — DGM4 (~80 GB) — TeamViewer, ~4–9 hours

- **Source on current PC:** `C:\Desktop\Multimodal-fake-news-detection\data\raw\DGM4\`
- **Destination on new PC:** `C:\path\to\MultiGuard\data\raw\DGM4\`

Action:
1. In TeamViewer File Transfer, navigate source to the `DGM4` directory.
2. Drag the entire `DGM4` directory from source pane → destination pane.
3. Leave overnight. ~4-9 hours depending on TeamViewer throughput.
4. If interrupted, restart TeamViewer File Transfer — it will offer to resume.

Verify after transfer (PowerShell on new PC):
```powershell
$size = (Get-ChildItem data\raw\DGM4 -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("DGM4: {0:N1} GB" -f $size)   # expect ~80
(Get-ChildItem data\raw\DGM4\manipulation\HFGI | Measure-Object).Count   # > 50000
```

### A.5.3 — MMFakeBench (HF, public) — ~3 GB, ~5 min

```bash
huggingface-cli download liuxuannan/MMFakeBench \
    --repo-type dataset \
    --local-dir MMFakeBench \
    --local-dir-use-symlinks False
```

Verify:
```bash
du -sh MMFakeBench/             # ~3 GB
ls MMFakeBench/                 # MMFakeBench_val/  MMFakeBench_test/
ls MMFakeBench/MMFakeBench_val/fake/ | head     # subset directories
```

### A.5.4 — GenImage 10k subset — TeamViewer + HF/Drive fallback

- **Source on current PC:** `C:\Desktop\Multimodal-fake-news-detection\data\raw\GenImage\`
- **Destination on new PC:** `C:\path\to\MultiGuard\data\raw\GenImage\`

Action:
1. TeamViewer-transfer the `GenImage` directory.
2. Verify which generators came across:
   ```powershell
   Get-ChildItem data\raw\GenImage | Select-Object Name
   # Goal: 8 subdirectories — sd14, sd15, midjourney, adm, glide, vqdm, biggan, wukong
   ```
3. V3 used MidJourney heavily; the current PC may only have some of the 8 generators on disk. For **any missing generators**, fall back to one of:
   - Official GenImage benchmark Google Drive at https://github.com/GenImage-Dataset/GenImage
   - HuggingFace community mirror (verify URL before running):
     ```bash
     huggingface-cli download <community/genimage-<generator>-mirror> \
         --repo-type dataset --local-dir data/raw/GenImage/<generator>
     ```
4. V4 builder (Day 1) will subsample to 1,250 per generator for the 10k Stage 1 set.

Expected:
```bash
du -sh data/raw/GenImage/       # ~10 GB total once all 8 generators are present
```

### A.5.5 — VisualNews images (~50 GB) — TeamViewer, ~3–8 hours

- **Source on current PC:** `C:\Desktop\Multimodal-fake-news-detection\data\raw\visualnews\`
- **Destination on new PC:** `C:\path\to\MultiGuard\data\raw\visualnews\`

Action:
1. TeamViewer-transfer the entire `visualnews` directory.
2. **Performance warning:** VisualNews has hundreds of thousands of small JPEGs. Per-file overhead on TeamViewer is high.
3. **Recommended tip — zip first:** on the current PC, compress to a single `.zip`, then transfer one file:
   ```powershell
   # On current PC (PowerShell), before starting TeamViewer:
   Compress-Archive -Path C:\Desktop\Multimodal-fake-news-detection\data\raw\visualnews `
                    -DestinationPath C:\temp\visualnews.zip
   ```
4. TeamViewer-transfer `C:\temp\visualnews.zip` (one big file, much faster than 100k small files).
5. On new PC:
   ```powershell
   Expand-Archive -Path C:\temp\visualnews.zip -DestinationPath data\raw\
   ```

Expected after extract:
```powershell
Get-ChildItem data\raw\visualnews\origin | Select-Object Name
# Should include: guardian  usa_today  bbc  washington_post  ...
```

### A.5.6 — blur_jpg_prob0.pth (~100 MB) — TeamViewer or download

The current PC may have this from V3 — search for `blur_jpg_prob0.pth` in `outputs\`, `pretrained\`, or near `univfd_genimage\`. If found, **TeamViewer-transfer** to `data\raw\pretrained\blur_jpg_prob0.pth`.

Fallback if not on current PC, download from CNNDetection releases:
```powershell
mkdir data\raw\pretrained -Force
cd data\raw\pretrained
Invoke-WebRequest -Uri "https://github.com/PeterWang512/CNNDetection/releases/download/v1.0/blur_jpg_prob0.pth" -OutFile blur_jpg_prob0.pth
```

If the release-asset URL doesn't resolve, see https://github.com/PeterWang512/CNNDetection README for the current location.

Verify:
```powershell
Get-Item data\raw\pretrained\blur_jpg_prob0.pth | Select-Object Length   # ~100 MB
```

### A.5.7 — Qwen2-7B-Instruct (~15 GB) — TeamViewer (Option A) or HF (Option B)

**Option A — TeamViewer copy from current PC's HF cache (faster):**

- **Source on current PC:** `C:\Users\FSOS\.cache\huggingface\hub\models--Qwen--Qwen2-7B-Instruct\` (whole directory tree)
- **Destination on new PC:** `C:\Users\<your-username>\.cache\huggingface\hub\models--Qwen--Qwen2-7B-Instruct\`

⚠️ **Windows symlink warning:** HuggingFace cache uses symlinks from `snapshots/<hash>/*.safetensors` to `blobs/*`. TeamViewer File Transfer **may not preserve Windows symlinks**. After copying, smoke-test:
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct'); print('OK')"
```
If you get "file not found", the symlinks broke — use Option B.

**Option B — HF download on new PC (cleaner, slower, ~25 min):**
```bash
huggingface-cli download Qwen/Qwen2-7B-Instruct
```

**Recommended order:** try Option A first; fall back to Option B if symlinks fail.

Verify:
```powershell
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen2-7B-Instruct" -Recurse | Measure-Object -Property Length -Sum | ForEach-Object { "{0:N1} GB" -f ($_.Sum/1GB) }
# ~15 GB
```

### A.5.8 — BERT + CLIP (auto-downloaded on first use, no action needed)

- `bert-base-uncased` (~440 MB)
- `openai/clip-vit-base-patch32` (~600 MB)

These download to `~/.cache/huggingface/hub/` the first time the V4 FND-CLIP encoder is instantiated. No pre-download required, but you can warm them now:

```bash
python -c "from transformers import AutoTokenizer, CLIPProcessor; \
           AutoTokenizer.from_pretrained('bert-base-uncased'); \
           CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
```

## A.6 — Master verification

After all downloads complete, run this one-liner to sanity-check the layout:

```bash
cd /path/to/project/root

for d in DGM4 MMFakeBench NewsCLIPpings visualnews GenImage pretrained; do
  if [ -d "data/raw/$d" ]; then
    size=$(du -sh "data/raw/$d" 2>/dev/null | cut -f1)
    echo "  OK    data/raw/$d   $size"
  else
    echo "  MISS  data/raw/$d"
  fi
done

[ -f data/raw/pretrained/blur_jpg_prob0.pth ] && echo "  OK    blur_jpg_prob0.pth" || echo "  MISS  blur_jpg_prob0.pth"
[ -d ~/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct ] && echo "  OK    Qwen2-7B-Instruct in HF cache" || echo "  MISS  Qwen2-7B-Instruct"
```

Expected output: 7 lines all starting with `OK`.

## A.7 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| TeamViewer file transfer freezes mid-way | Network instability or sleep mode on either PC | Disable sleep on BOTH PCs. Reconnect TeamViewer File Transfer and accept "Resume". |
| TeamViewer < 1 MB/s | Going through relay (not P2P) | Verify both PCs on same network if possible; try different time of day. Or accept it — overnight transfers still finish. |
| Current PC slept during transfer | Default power plan triggered | Settings → Power → Screen and Sleep → "Never" while connected to AC. Optionally `powercfg /requestsoverride DISPLAY SYSTEM` |
| Qwen2-7B "files not found" after TeamViewer copy | HF cache symlinks broke in transit | Switch to A.5.7 Option B (HF download) |
| GenImage doesn't have all 8 generators on current PC | V3 only used some generators | Fall back to official GenImage Google Drive or community HF mirror for missing ones (A.5.4) |
| VisualNews transfer takes forever (per-file overhead) | Many tiny JPEGs over TeamViewer | Use the zip-first approach in A.5.5 |
| Out of disk during DGM4 transfer | DGM4 is ~80 GB | Free disk first. If desperate, drop `face_swap`/`face_edit` subdirs (keep `face_attribute`/HFGI for class 2) |
| HF download "401 Unauthorized" on MMFakeBench | HF token expired/missing | `huggingface-cli login` again with a fresh READ token |
| `blur_jpg_prob0.pth` 404 | CNNDetection release URL changed | Check the current URL at https://github.com/PeterWang512/CNNDetection README |

## A.8 — When you're done

You've successfully completed A.0–A.7. The new PC now has:
- All 5 raw datasets in `data/raw/`
- Pretrained UnivFD weights at `data/raw/pretrained/blur_jpg_prob0.pth`
- Qwen2-7B-Instruct in the HF cache

You can now clone the V4 repo (which will be empty initially with just the plan + this download recipe) and start Day 0 of V4 implementation per §13 of the plan.

The V4 builder scripts (Day 1, §3 "Dataset acquisition procedure") will then read this `data/raw/` layout and produce the Stage-1 binary CSV and the 5-class `forensic_5class_v4.csv` manifest.

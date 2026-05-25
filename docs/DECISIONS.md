# Design Decisions Register

This file records the major engineering choices made across the project and *why*. Two sections:
- **Section 1 (U1-U15):** Unification mega-repo decisions (per active plan, P3 deliverable, NEWEST)
- **Section 2 (D1-D15):** V4 rebuild decisions (per V4 plan section 17)

---

# Section 1 — Unification Decisions (U-series, P3 deliverable)

These were locked during the mega-repo restructure (P0-P6 in the active plan).

## U1 - Rename old folder to MultiGuard

**Date:** 2026-05-24 (restructure day)
**Decision:** Renamed `C:\Desktop\Multimodal-fake-news-detection\` -> `C:\Desktop\MultiGuard\` after first renaming the existing `MultiGuard\` (V4 clone) -> `MultiGuard-v4-original\` (preserved as backup until P6 verification passes).

**Why:** Single canonical folder name matches the GitHub repo. Avoids two parallel folders. Old V4 clone preserved during transition so we can roll back if needed.

## U2 - Phase subfolder layout

**Date:** 2026-05-24
**Decision:** `phases/{v1, v2, v3, v4, forensic}/` — one subfolder per project phase. Each is self-contained with its own `src/`, `configs/`, `scripts/` (where applicable), `README.md`, and links to phase-specific outputs.

**Why:** User's explicit directive: "subfolders from each phases we have done/what are we doing". Lets the doctor and future students navigate by phase. V4 is the active codebase; V1/V2/V3 are frozen historical reference.

## U3 - Hoist V4 tooling to root

**Date:** 2026-05-24
**Decision:** V4's `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `CHANGELOG.md` hoisted to MultiGuard root. V4's `scripts/`, `app/`, `docs/` merged into root scripts/, app/, docs/.

**Why:** V4 is the canonical going-forward codebase. Its tooling becomes the project-wide standard. Old V1/V2/V3 ad-hoc setup (requirements.txt, no Makefile, no pre-commit, no CI) is archived as `docs/requirements_v1v2v3_legacy.txt`.

## U4 - Update path resolution for unified root

**Date:** 2026-05-24
**Decision:** Updated `phases/v4/src/v4/core/paths.py`: `CONFIGS_ROOT = PROJECT_ROOT / "phases" / "v4" / "configs"` (was `configs/`). Added `V4_PHASE_ROOT` constant. `PROJECT_ROOT`, `DATA_ROOT`, etc. resolve correctly because `find_project_root()` walks up and finds the hoisted root pyproject.toml naturally.

**Why:** V4's path-resolution logic relies on walking up to find pyproject.toml. After hoist, the root MultiGuard/pyproject.toml IS the anchor. V4 configs moved to phase folder, so CONFIGS_ROOT had to be updated.

## U5 - pyproject.toml packages.find for v4 + forensic

**Date:** 2026-05-24
**Decision:** `[tool.setuptools.packages.find] where = ["phases/v4/src", "phases/forensic/src"] include = ["v4*", "forensic*"]`.

**Why:** Both V4 (active) and forensic (current sprint) packages need to be importable as `from v4 import ...` and `from forensic import ...`. Setuptools supports multiple `where` paths.

## U6 - Heavy artifacts gitignored, not deleted

**Date:** 2026-05-24
**Decision:** V3 cache (6.9 GB) -> `cache/v3/_features/`, V3 outputs (3.2 GB) -> `outputs/v3/_checkpoints/`, Qwen venv -> `phases/v3/.venv-qwen/`, articles.tar.gz (478 MB) -> stays at root. All gitignored.

**Why:** Files preserved on disk (research PC, no space pressure). Git stays clean. V3 reproducibility intact — caches + checkpoints accessible if anyone needs to re-evaluate.

## U7 - Mirror fnd_clip.py into each V1/V2/V3 phase

**Date:** 2026-05-24
**Decision:** V1's `fnd_clip.py` (FND-CLIP encoder) is mirrored into `phases/v1/src/models/`, `phases/v2/src/models/`, `phases/v3/src/models/`. The original top-level `src/models/fnd_clip.py` was deleted.

**Why:** Each phase is now self-contained. Modifying fnd_clip.py in V1 doesn't break V2/V3 (they have independent copies). V4 has its own canonical fnd_clip.py at `phases/v4/src/v4/models/encoders/fnd_clip.py` with the sem_proj 512->768 adapter (deviation F1).

## U8 - V1/V2 leftover scripts -> phases/v2/scripts/

**Date:** 2026-05-24
**Decision:** Leftover top-level scripts (build_balanced_dataset, llava_*, mps_smoke_test, etc.) moved to `phases/v2/scripts/` as V2-era helpers.

**Why:** Most of them are LLaVa diagnostics or V2-era dataset builders. None are V4 dependencies. Phase-2 folder is the most appropriate archive home.

## U9 - src/{corruption,generation,preprocess,scraping,utils} -> shared/legacy_utils/

**Date:** 2026-05-24
**Decision:** Top-level src/ utility subfolders moved into `shared/legacy_utils/`. The empty src/ shell deleted.

**Why:** These utilities are not part of any specific phase but were imported by V1/V2/V3 codebases. shared/ is the natural home for cross-phase reference. V4 has its own utilities under `phases/v4/src/v4/core/`.

## U10 - app/server.py = V4 modular; old V3 preserved as app/server_v3_legacy.py

**Date:** 2026-05-24
**Decision:** Renamed old `app/server.py` (V3.1 5-class FastAPI) -> `app/server_v3_legacy.py` BEFORE copying V4's modular `app/server.py` over it. Old `app/_debug_startup.py` -> `app/_debug_startup_v3_legacy.py` similarly.

**Why:** V4 server is canonical going forward (registry-driven, swappable encoders). V3 server preserved for visual parity reference + as a fallback if V4 server breaks.

## U11 - Push canonical to FerasMad/MultiGuard, preserve old Rashidbm history locally

**Date:** 2026-05-24
**Decision:** The unified repo will push to `https://github.com/FerasMad/MultiGuard.git` (user's repo). The old `https://github.com/Rashidbm/Multimodal-fake-news-detection.git` remote (collaborator's repo) is **not** force-pushed to; it stays at its pre-unification state. Local history preserved on `legacy-multimodal` branch.

**Why:** Old repo is shared with collaborator Rashidbm; can't disturb. User owns FerasMad/MultiGuard outright — safe to push the unified tree there. Tag `pre-unification-old` (on old remote) + tag `pre-unification-v4` (on FerasMad remote) snapshot pre-restructure state for rollback.

## U12 - Doctor PDFs in docs/doctor-briefs/

**Date:** 2026-05-24
**Decision:** All 5 doctor briefs copied to `docs/doctor-briefs/`:
- `V1_Dataset_Instructions.pdf`
- `V2_Implementation_Guidelines.pdf`
- `V3.1_Implementation_Guidelines.pdf`
- `Forensic_Image_Detector_En.pdf`
- `IMAGE_ONLY_METRICS.pdf`

**Why:** Single source of truth for the doctor's specs. Future students don't have to hunt down which Downloads folder has which version. Doctor audit goes straight to this folder.

## U13 - MASTER_CHECKLIST.md as the audit-tool

**Date:** 2026-05-24
**Decision:** Created `docs/MASTER_CHECKLIST.md` with sections A-G mapping every doctor requirement (F.1-F.26 forensic, V3.1-V3.11, V2.1-V2.6, V1.1-V1.3, E.1-E.3, FL.1-FL.7, G.1-G.6) to its implementation location.

**Why:** The user explicitly asked: "I want you to review all the files 'pdfs' that I have sent you before recursovily. So we have a check list of everything we need." This checklist is the audit-tool the doctor uses to verify nothing is missing.

## U14 - Forensic detector lives at phases/forensic/

**Date:** 2026-05-24
**Decision:** New current sprint (binary AI-image detector, both approaches) goes at `phases/forensic/`. Scaffolded in P2.6; full implementation in P5.

**Why:** Treats forensic as a peer to v1/v2/v3/v4 phases. Trained `forensic_dct_model.pth` can later replace `blur_jpg_v0.pth` as V4 Stage-1 init (separate V4 plan revision).

## U15 - Per-phase READMEs documenting scope + status

**Date:** 2026-05-24
**Decision:** Each `phases/<name>/` has a `README.md` documenting: what the phase was, canonical metrics, how to re-run, why it was superseded (for V1/V2/V3) or what's pending (for V4/forensic).

**Why:** Doctor's reusability mandate: future students should understand each phase without spelunking. Per-phase README is the first thing they read after the root README's phase status table.

---

# Section 2 — V4 Rebuild Decisions (D-series)

## D1 - Solo implementer, 4-layer "as-if-4-students" structure

**Date:** Day 0 of V4
**Decision:** V4 is implemented by one developer, but `src/v4/` is organized as if the original V3 4-student team still owns one subpackage each.

**Why:** The doctor's mental model is "Student 1 owns UnivFD, Student 2 owns Qwen, Student 3 owns fusion+classifier, Student 4 owns data+training+evaluation+Stage 0". Keeping the layout makes future handoff to a 4-student team possible without restructuring.

---

## D2 - Windows + Python 3.12 + CUDA 12.4 on dual RTX 4090

**Date:** Day 0
**Decision:** V4 implementation PC is Windows 11 + Python 3.12 + CUDA 12.4 with dual 4090 (48 GB total VRAM).

**Why:** This matches the V3 dev environment. The Windows-specific gotchas (MS Store python stub, cv2.dct vs scipy, torch-before-cv2) are documented in `CONTRIBUTING.md`. Moving to Linux now would force re-debugging issues V3 already solved.

---

## D3 - Strict V3.1 section 6 classifier (no flexibility hooks)

**Date:** Day 0
**Decision:** `src/v4/models/classifier/mlp_head.py` hardcodes `Linear(1024, 512) + BatchNorm1d + GELU + Dropout(0.5) -> Linear(512, 256) + GELU -> Linear(256, 5)`. No dropout-rate parameter, no head-dim parameter.

**Why:** The doctor approved the V3.1 spec exactly. Adding parametrization would invite spec drift. If a future student needs a different head (e.g. 3-class), they must edit `mlp_head.py` directly and add a new spec-compliance test.

**Trade-off:** Reduces reusability slightly. Documented in `ARCHITECTURE.md` and `SPEC_COMPLIANCE_MAP.md`.

---

## D4 - Qwen2-7B-Instruct, last layer, hidden 3584 (deviation F2)

**Date:** Day 0
**Decision:** Use Qwen2-7B-Instruct (not Qwen1.5-7B), pull from layer `-1` (not 30), hidden size 3584 (not 4096), with `Linear(3584, 768) + GELU` adapter.

**Why:** V3.1 spec language ("layer 30, hidden 4096") matches Qwen1.5-7B, but V3 already proved Qwen2-7B-Instruct works on dual 4090. The mismatch is a forced deviation, not a violation of intent.

**Logged as:** Deviation F2 in `V4_PLAN.md` section 2 and `ARCHITECTURE.md` section 5.

---

## D5 - 3 seeds (42, 1337, 2024) with mean and std reporting

**Date:** Day 0
**Decision:** Stage 2 trains 3 seeds. Two run in parallel on the dual 4090 via `CUDA_VISIBLE_DEVICES`, the third sequential. Eval reports mean and std.

**Why:** Variance estimation is cheap on dual 4090 (~5 h total wall-clock) and gives the doctor confidence the headline number isn't a single-seed lottery win.

---

## D6 - bf16 mixed precision

**Date:** Day 0
**Decision:** All training uses bf16 autocast via `torch.cuda.amp.autocast(dtype=torch.bfloat16)`.

**Why:** RTX 4090 has native bf16 support. fp16 has range issues; fp32 wastes memory. bf16 matches Qwen2's training precision and avoids the loss-scale gymnastics fp16 requires.

---

## D7 - Stratified 70/15/15 split by (label, source)

**Date:** Day 0
**Decision:** `src/v4/data/builders/merge.py::_stratified_split` partitions by the tuple `(label, source)`. Each `(label, source)` combination is split 70/15/15.

**Why:** V3 had source-fingerprint leakage where MMFakeBench captions had distinctive prefixes. Stratifying by source ensures train/test see the same source distribution, so any fingerprint learned on train is also "fairly" represented in test - it doesn't fake the metric.

**Risk acknowledged:** This doesn't ELIMINATE shortcuts; it eliminates the per-source CHEATING-by-omission shortcut. Per-source F1 must still be inspected manually (R1/R2 in `EVAL_PROTOCOL.md`).

---

## D8 - TeamViewer-hybrid data acquisition (locked decision C9)

**Date:** Day 0 pre-flight
**Decision:** Bulk datasets (DGM4, GenImage, VisualNews, Qwen HF cache, blur_jpg) come from the V3 PC via TeamViewer File Transfer. Small / clean datasets (MMFakeBench, BERT, CLIP) auto-download from HuggingFace.

**Why:** DGM4 HF authentication is a manual web-UI dance. GenImage requires accepting per-generator terms. VisualNews has Google Drive quota issues. TeamViewer File Transfer sidesteps all three. Current PC must stay awake (~9 h overnight).

**Risk:** TeamViewer file transfer can interrupt; documented as R4 in V4 plan, with HF fallback.

**Document:** Operator-specific transfer recipe is kept locally outside the repo (not committed) since it references paths and host names that are not part of public project documentation.

---

## D9 - Registry pattern (no plugin discovery)

**Date:** Day 0
**Decision:** Encoders / fusion / datasets register themselves via `@register(REGISTRY, "<name>")` decorators. Module discovery is explicit through `import_all()` walking `src/v4/models/` and `src/v4/data/`. No `pkg_resources.iter_entry_points`, no setuptools plugins.

**Why:** Explicit imports are debuggable. Plugin discovery looks magic when it works and is impossible to debug when it doesn't. For a 4-student team, "if you forget to import your module, your encoder isn't registered" is a clearer mental model than "if the metadata is wrong somewhere, your plugin doesn't load."

---

## D10 - Class 4 augmented from DGM4 multimodal (cap 800)

**Date:** Day 0
**Decision:** Class 4 (Fully-Fabricated) takes ~2,200 rows from MMFakeBench AI/AI subsets + up to 800 from DGM4 multimodal-paired-fake (text_swap, text_attribute).

**Why:** Class 4 is the smallest in MMFakeBench. Without augmentation, it would be ~2k rows; we want 3k/class for balance with the other 4 classes. DGM4 multimodal-paired-fake is the only authorized non-MMFB source (per V1 Dataset PDF).

---

## D11 - Strict V3.1 section 7 evaluation (no extra surface)

**Date:** Day 0 (locked decision F)
**Decision:** Evaluation produces only what V3.1 section 7 mandates: Precision, Recall, F1-macro, confusion matrix, MMFakeBench transfer probe. No bootstrap CI, no ablations, no shortcut detector.

**Why:** The doctor's brief frames V4 as a structural rebuild for reusability. Extra eval scope expands the audit surface without changing the headline numbers the doctor cares about.

**Risks accepted:** R1 (no shortcut audit) and R2 (per-source bias undetectable beyond visual inspection). Documented in `EVAL_PROTOCOL.md`.

---

## D12 - bf16 + dual-4090 layout

**Date:** Day 4 plan
**Decision:** Two seeds (42, 1337) run in parallel via `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`. Third seed (2024) runs sequential after either finishes.

**Why:** Maximizes hardware utilization. Sequential third seed avoids checkpoint-save contention if both parallel seeds hit early-stop within seconds of each other.

---

## D13 - V4 server on port 8081 alongside V3 server on 8080 (Day 6)

**Date:** Day 6 (migration plan)
**Decision:** V4 FastAPI server lives at port 8081. V3 server stays on 8080 during the parity-check window. Cloudflare tunnel flips to 8081 only after `scripts/compare_servers.py` validates JSON-shape parity on 50 sample pairs.

**Why:** Atomic cutover with rollback. If V4 breaks in production, flip Cloudflare back to 8080.

---

## D14 - Provenance metadata in every checkpoint

**Date:** Day 0
**Decision:** Every `BaseTrainer.save_checkpoint` writes config_hash, data_hash, git_sha, seed, torch_version, cuda_version, spec_version, stage in the checkpoint payload.

**Why:** Reproducibility. ~30 LOC investment. Saves pain later when someone asks "what config produced this number?"

---

## D15 - GenImage in Stage 1 ONLY, never in 5-class data

**Date:** Day 0 (deviation A1)
**Decision:** Class 4 (Fully-Fabricated) does NOT use GenImage MidJourney as in V3. GenImage is reserved for Stage 1 UnivFD binary pretrain.

**Why:** V3 used GenImage in class 4 because of the BLIP-2 caption availability. V3.1 spec says class 4 = MMFakeBench AI-Image+AI-Text + DGM4 multimodal. Aligning with spec also removes one source of cross-stage data leakage.

---

## How to add a new decision

When you make a significant choice during V4 development:

1. Append a new `## D<n>` entry below.
2. Include: date, decision, why, trade-offs, link to relevant file or test.
3. If the decision changes a number in `SPEC_COMPLIANCE_MAP.md`, update that file too.
4. Commit both together with a `docs: D<n> - <short description>` message.

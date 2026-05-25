# Design Decisions Register

This file records the major engineering choices made during the V4 rebuild and *why*. It is the source of truth for "we did it this way because ___" questions that come up during doctor reviews or future-student handoff.

The full plan with locked decisions is in [`V4_PLAN.md`](V4_PLAN.md) section 17. This file extracts those decisions as standalone entries so they can be updated when implementation reality diverges from the original plan.

---

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

# Contributing to MultiGuard V4

This document covers the git workflow, lint/format rules, and the **MANDATORY Windows workarounds** carried over from V3 lessons. If you're new to the project, read [`ARCHITECTURE.md`](ARCHITECTURE.md) first and [`SETUP.md`](SETUP.md) for environment install.

---

## 1. Git workflow

- **Default branch:** `main`. The frozen V4 working branch is `v4` (during the Day 0-7 build).
- **Per-stage feature branches:** `feat/<stage>-<short-desc>`, e.g. `feat/stage2-aux-loss`.
- **PRs:** Squash-merge after the doctor (or whoever owns review) approves.
- **Commits:** Conventional-commit style. Examples:
  - `feat(v4): foundation + core + preprocessing + encoders`
  - `fix(univfd): conv1 kaiming init was applied after blur_jpg load`
  - `docs(setup): clarify torch CUDA wheel pinning`
- **Never** force-push to `main`. Force-push to feature branches is fine.

---

## 2. Lint + format

```bash
make lint     # ruff check + ruff format --check  (fails CI on diff)
make fmt      # ruff check --fix + ruff format    (writes changes)
```

The pre-commit hook (`.pre-commit-config.yaml`) runs ruff on every staged file. Install once:

```bash
pip install -e ".[dev]"
pre-commit install
```

Bypass for a single commit (last resort):

```bash
SKIP=ruff git commit -m "..."
```

CI (`.github/workflows/ci.yml`) enforces both lint and ruff format. PRs with red lint cannot merge.

---

## 3. MANDATORY Windows workarounds (V3 lessons)

These are **hard rules**. Violations caused multi-day debugging in V3.

### 3.1 `import torch` BEFORE cv2 / scipy / torchvision

`torch` on Windows ships with its own MKL DLLs. If `cv2` or `scipy.fftpack` is imported first, you get a silent DLL conflict and either a segfault or wrong numerical results.

Every entry point (CLI, server, scripts) starts with:

```python
import torch  # MUST be first; do not move
import cv2    # OK after torch
```

### 3.2 Use `cv2.dct`, never `scipy.fftpack.dct`

`scipy.fftpack.dct` segfaults on Windows when torch CUDA is active. Use `cv2.dct` everywhere. See `src/v4/data/preprocessing/patch_dct.py`.

### 3.3 Use the full python.exe path for detached / background processes

`Start-Process python -ArgumentList ...` runs the MS Store stub even after disabling aliases, because PowerShell may resolve via PATH lazily. Always pass the full interpreter path:

```powershell
Start-Process "C:\Python312\python.exe" -ArgumentList "-m","v4.cli","train","--config","..."
```

### 3.4 Disable the MS Store python stub

See [`SETUP.md`](SETUP.md) section 1. Without this, `python` silently invokes Microsoft Store and your training script never runs.

### 3.5 Activate the venv before every session

```powershell
.venv\Scripts\Activate.ps1
```

Or use the full path: `.venv\Scripts\python.exe -m v4.cli train ...`

### 3.6 Path separators in YAML configs

YAML lets you use forward slashes in paths on Windows; `Pathlib` handles both. Prefer forward slashes (`outputs/v4/stage2/best.pt`) so configs are portable to Linux.

---

## 4. Adding code: where it goes

| What | Where | See |
|------|-------|-----|
| New encoder | `src/v4/models/encoders/<name>.py` | [`ADD_ENCODER.md`](ADD_ENCODER.md) |
| New dataset / source | `src/v4/data/builders/build_<source>.py` | [`ADD_DATASET.md`](ADD_DATASET.md) |
| New fusion module | `src/v4/models/fusion/<name>.py` | [`ADD_FUSION.md`](ADD_FUSION.md) |
| New loss type | `src/v4/training/losses.py` | (extend `LossSpec` + `CompositeLoss._build`) |
| New scheduler | `src/v4/training/schedulers.py` | (extend `build_scheduler`) |
| Server endpoint | `app/server.py` | follow `/api/analyze` pattern |

---

## 5. Logging

- Use `from v4.core.logging import get_logger; log = get_logger(__name__)`.
- **NEVER** use `print()` in library code. CLI / script entry points may use `print()` for user-facing output only.
- Default level is INFO. Set `LOG_LEVEL=DEBUG` env var to see all DEBUG logs.

---

## 6. Type hints

- `from __future__ import annotations` at the top of every file.
- Type all public APIs (function args, return types, dataclass fields).
- No mypy strict mode; ruff catches the egregious type bugs.

---

## 7. Tests

Run the full suite:
```bash
make test
```

Layers:

| Layer | Path | Purpose |
|-------|------|---------|
| Spec compliance | `tests/spec_compliance/test_v3_1_section_*.py` | One per V3.1 section |
| Unit | `tests/unit/test_*.py` | Class invariants, schema validation |
| Smoke | `tests/smoke/test_*.py` | Imports + 1-epoch / 128-sample training pass |
| Integration | `tests/integration/test_*.py` | Manifest -> trainer -> eval on fixtures |

Spec-compliance tests are sacred. Touching `models/fusion/v3_pairwise.py` MUST keep `test_v3_1_section_5_3_pairwise_sum.py` green (output shape `[B, dim]`, not `[B, 2*dim]`).

---

## 8. Spec-compliance review checklist

Before merging any change to `models/`, `training/`, or `evaluation/`:

- [ ] `make test` green
- [ ] `make lint` green
- [ ] `docs/SPEC_COMPLIANCE_MAP.md` still cites the new code accurately
- [ ] If you changed loss / optimizer / scheduler defaults, update `docs/DECISIONS.md` with an entry explaining why
- [ ] If you added a new encoder/fusion/dataset, the corresponding `ADD_*.md` doc still works as a recipe

---

## 9. Releases

V4 is currently pre-1.0. Tagging convention:

```
v4.0.0-alpha.<N>   # internal milestones during the Day 0-7 build
v4.0.0             # doctor sign-off
```

After tagging, write a short entry in [`CHANGELOG.md`](../CHANGELOG.md).

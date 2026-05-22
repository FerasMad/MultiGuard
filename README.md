# MultiGuard

Multimodal fake-news detection — 5-class classifier (Real / Out-of-Context / Manipulated / AI-Text / Fully Fabricated) over text + image inputs, built per the V3.1 Implementation Guidelines.

This repo currently contains **the plan only**. Implementation has not started yet.

## Documents

- **[`docs/V4_PLAN.md`](docs/V4_PLAN.md)** — full V4 rebuild plan (20 sections + risk register + locked decisions). Read this first.
- **[`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md)** — TeamViewer-hybrid dataset acquisition recipe. Run this on the implementation PC before any code lands.

## Sequence

1. Read `docs/V4_PLAN.md` end-to-end (~30 min).
2. On the implementation PC, follow `docs/DATA_DOWNLOAD.md` steps A.0–A.8 to stage `data/raw/` (~9 hours overnight via TeamViewer + ~10 min HF downloads).
3. When master verification (A.6) prints 7 OK lines, the PC is ready for V4 Day 0.
4. Day 0 begins per `docs/V4_PLAN.md` §13 schedule.

## Spec sources (canonical)

- Implementation Guidelines V3.1 — drives the 5-class architecture
- Implementation Guidelines V2 — predecessor 3-class baseline
- Dataset + V1 Instructions — V1 baseline + dataset construction rules

The doctor's spec PDFs are the authoritative source. `docs/V4_PLAN.md` §2 audits V4 against the spec and lists every aligned / deviated point.

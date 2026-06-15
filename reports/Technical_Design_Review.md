# MultiGuard Pluggable Pipeline — Technical Design Review

> Scope: the dependency-injection pipeline under `phases/v4/src/v4/pipeline/`
> (5 single-class files). This is a comprehensive review: Architecture, a
> per-class reference, Data Flow, Control Flow, Component Integration, an
> Extensibility Guide, Evaluation Methodology, Verification, and Analysis —
> each with diagrams.

---

## 0. Summary

The pipeline was refactored from a "build-it-inside" design into a **pluggable,
dependency-injection** design. Every model is **passed in from the outside**; no
wrapper builds a model. Each wrapper **auto-infers its `out_dim`** (no hardcoded
dimensions). The four branch files are **fully decoupled** (they never import
each other); only `main_pipeline.py` wires them together.

| Goal | How it is met |
|------|---------------|
| Pluggable / any model | Models injected via `__init__`; wrappers are model-agnostic |
| No hardcoded dimensions | `out_dim` inferred from a probe forward (or a model attribute) |
| Strict decoupling | Branch files import only `torch` + stdlib; never each other |
| Single orchestrator | Only `MainPipeline` imports the four wrappers |
| Swap a model = edit one file | Build a different model, pass it to the matching wrapper |

```mermaid
flowchart LR
    CALLER["caller / server"] --> MP["MainPipeline"]
    MP --> W1["SemanticEncoder"] --> M1["injected model (FND-CLIP)"]
    MP --> W2["ForensicTextEncoder"] --> M2["injected model (Qwen2-7B)"]
    MP --> W3["ImageForensicEncoder"] --> M3["injected model (DCT ResNet50)"]
    MP --> W4["FusionModule"] --> M4["injected model (V3PairwiseFusion)"]
```

---

## 1. Architecture (المعمارية)

### 1.1 Principles

- **Dependency injection** — a wrapper receives a ready `nn.Module`; it raises
  `TypeError` for anything else. It never constructs a backbone.
- **Auto `out_dim`** — resolved once at construction time (probe forward = source
  of truth; model attribute = fallback). No dimension literal appears in a wrapper.
- **Strict decoupling** — enforced by imports: branch files import only `torch`
  and stdlib; only `main_pipeline.py` (+ `__init__.py` for re-export) import them.
- **Plain `nn.Module`** — the pipeline composes like any PyTorch module:
  `.to(device)`, `.eval()`, `state_dict()`, autograd all work unchanged.

### 1.2 Layered architecture

Two clean layers: the **wrapper layer** (this package, model-agnostic) and the
**model layer** (the real backbones, built elsewhere and injected).

```mermaid
flowchart TB
    subgraph L0["Layer 0 — caller"]
        C["app/server.py or training/eval"]
    end
    subgraph L1["Layer 1 — orchestrator"]
        MP["MainPipeline"]
    end
    subgraph L2["Layer 2 — pluggable wrappers (this package)"]
        SEM["SemanticEncoder"]
        TXT["ForensicTextEncoder"]
        IMG["ImageForensicEncoder"]
        FUS["FusionModule"]
    end
    subgraph L3["Layer 3 — injected models (built outside)"]
        FND["FND-CLIP"]
        QWEN["Qwen2-7B + proj"]
        DCT["DCT ResNet50"]
        V3["V3PairwiseFusion"]
    end
    C --> MP
    MP --> SEM --> FND
    MP --> TXT --> QWEN
    MP --> IMG --> DCT
    MP --> FUS --> V3
```

### 1.3 Decoupling & import graph

```mermaid
flowchart LR
    INIT["__init__.py"] --> SEM["semantic.py"]
    INIT --> TXT["forensic_text.py"]
    INIT --> IMG["image_forensic.py"]
    INIT --> FUS["fusion.py"]
    INIT --> MAIN["main_pipeline.py"]
    MAIN --> SEM
    MAIN --> TXT
    MAIN --> IMG
    MAIN --> FUS
    SEM -. "NEVER" .- TXT
    TXT -. "NEVER" .- IMG
    IMG -. "NEVER" .- FUS
```

Solid = allowed import; dotted "NEVER" = forbidden (verified by grep — no branch
file imports another).

### 1.4 Class diagram

```mermaid
classDiagram
    class SemanticEncoder {
        +model
        +out_dim
        +forward(batch)
        -infer_out_dim()
        -probe()
    }
    class ForensicTextEncoder {
        +model
        +out_dim
        +forward(batch)
        +forward_text(text)
    }
    class ImageForensicEncoder {
        +model
        +out_dim
        +forward(batch)
    }
    class FusionModule {
        +model
        +out_dim
        +out_key
        +forward(features)
        -select()
    }
    class MainPipeline {
        +semantic
        +text
        +image
        +fusion
        +out_dim
        +encoder_dims
        +forward(batch, text)
        +from_models()
    }
    MainPipeline o-- SemanticEncoder
    MainPipeline o-- ForensicTextEncoder
    MainPipeline o-- ImageForensicEncoder
    MainPipeline o-- FusionModule
```

---

## 2. Class Reference (per-class detail)

Every wrapper follows the same shape: inject a model, expose `out_dim`, delegate
`forward`. Below, each class lists its constructor, behaviour, and a mini flow.

### 2.1 `SemanticEncoder` (`semantic.py`)

- **Constructor:** `SemanticEncoder(model, example_input=None)`
- **forward:** `forward(batch) -> Tensor[B, out_dim]` — delegates to `model(batch)`.
- **out_dim:** probe `model(example_input)` then read `shape[-1]`; else read one of
  `("out_dim","output_dim","embed_dim","hidden_size")`.
- **Wraps (typical):** FND-CLIP (ResNet50 + BERT + CLIP, frozen), `v_semantic` 512→768.

```mermaid
flowchart LR
    B["batch: image + BERT + CLIP tensors"] --> M["injected semantic model"] --> V["v_semantic [B,768]"]
```

### 2.2 `ForensicTextEncoder` (`forensic_text.py`)

- **Constructor:** `ForensicTextEncoder(model, example_input=None)`
- **forward:** `forward(batch) -> Tensor` (cached features path).
- **forward_text:** `forward_text(text) -> Tensor` — delegates to the model's
  `runtime_forward` if present (raw-text inference); never used for probing so the
  heavy backbone is not triggered at construction.
- **Wraps (typical):** Qwen2-7B-Instruct (frozen) + `Linear(3584→768)`.

```mermaid
flowchart LR
    B["batch: v_textfor_qwen [B,3584]"] --> M["injected text model"] --> V["v_textfor [B,768]"]
    R["raw text"] -. "forward_text" .-> M
```

### 2.3 `ImageForensicEncoder` (`image_forensic.py`)

- **Constructor:** `ImageForensicEncoder(model, example_input=None)`
- **forward:** `forward(batch) -> Tensor[B, out_dim]`.
- **Wraps (typical):** DCT ResNet50 (1-channel conv), reads `v_imgfor_dct [B,1,224,224]`.

```mermaid
flowchart LR
    B["batch: v_imgfor_dct [B,1,224,224]"] --> M["injected image model"] --> V["v_imgfor [B,768]"]
```

### 2.4 `FusionModule` (`fusion.py`)

- **Constructor:** `FusionModule(model, example_features=None, out_key="main_logits")`
- **forward:** `forward(features) -> dict` (e.g. `main_logits`, `aux_logits`, `fused`).
- **out_dim:** probe with a feature dict (batch ≥ 2 for `BatchNorm1d`), select the
  `out_key` tensor, read `shape[-1]`; else read `("out_dim","output_dim","num_classes")`.
- **Wraps (typical):** V3PairwiseFusion (3-pair cross-attention + Conv1d + MLP).

```mermaid
flowchart LR
    F["features {v_semantic,v_imgfor,v_textfor}"] --> M["injected fusion model"] --> O["{main_logits[B,5], aux_logits[B,2], fused[B,1024]}"]
```

### 2.5 `MainPipeline` (`main_pipeline.py`)

- **Constructor:** `MainPipeline(semantic, image, text, fusion, feature_keys=…)`
- **Factory:** `from_models(semantic_model, image_model, text_model, fusion_builder, …)`
  — builds wrappers, reads encoder `out_dim`s, builds the fusion from them, probes it.
- **forward:** `forward(batch, text=None) -> dict`.
- **Helpers:** `out_dim` (delegates to fusion), `encoder_dims` (per-branch widths).

```mermaid
flowchart LR
    IN["batch (+ optional text)"] --> SEM & IMG & TXT
    SEM["semantic"] --> F["features dict"]
    IMG["image"] --> F
    TXT["text"] --> F
    F --> FUS["fusion"] --> OUT["logits dict"]
```

---

## 3. Data Flow (مسار البيانات)

### 3.1 End-to-end with shapes

```mermaid
flowchart LR
    IN["batch dict (+ optional raw text)"]
    IN --> SEM["SemanticEncoder"] --> VS["v_semantic [B,768]"]
    IN --> IMG["ImageForensicEncoder"] --> VI["v_imgfor [B,768]"]
    IN --> TXT["ForensicTextEncoder"] --> VT["v_textfor [B,768]"]
    VS --> FUS["FusionModule"]
    VI --> FUS
    VT --> FUS
    FUS --> OUT["main_logits [B,5]<br/>aux_logits [B,2]<br/>fused [B,1024]"]
```

### 3.2 Merged-batch key routing

A single batch dict drives all three encoders; each injected model selects only
the keys it needs, so `MainPipeline` needs no routing logic.

```mermaid
flowchart LR
    B["merged batch dict"]
    B --> K1["image, bert_ids, bert_mask,<br/>clip_pixels, clip_ids, clip_mask"] --> SEM["SemanticEncoder"]
    B --> K2["v_imgfor_dct"] --> IMG["ImageForensicEncoder"]
    B --> K3["v_textfor_qwen"] --> TXT["ForensicTextEncoder"]
```

### 3.3 Fusion internals (inside the injected V3PairwiseFusion)

```mermaid
flowchart TB
    A["v_semantic / v_imgfor / v_textfor (3x [B,768])"]
    A --> P["3x pairwise cross-attention (8 heads)"]
    P --> C1["Conv1d 768->768 (k=3)"]
    C1 --> C2["Conv1d 768->1024 (k=1)"]
    C2 --> PL["AdaptiveAvgPool1d -> fused [B,1024]"]
    PL --> MLP["MLP 1024->512->256->5"]
    MLP --> ML["main_logits [B,5]"]
    A --> AUX["aux head on v_imgfor.detach() -> aux_logits [B,2]"]
```

### 3.4 Shape table

| Stage | Input | Output |
|-------|-------|--------|
| `SemanticEncoder` | image + BERT/CLIP tensors | `v_semantic [B,768]` |
| `ForensicTextEncoder` | `v_textfor_qwen [B,3584]` or raw text | `v_textfor [B,768]` |
| `ImageForensicEncoder` | `v_imgfor_dct [B,1,224,224]` | `v_imgfor [B,768]` |
| `FusionModule` | 3 × `[B,768]` | `main_logits [B,5]`, `aux_logits [B,2]`, `fused [B,1024]` |

---

## 4. Control Flow (تسلسل التنفيذ)

### 4.1 Construction (`MainPipeline.from_models`)

```mermaid
sequenceDiagram
    participant U as caller
    participant MP as MainPipeline.from_models
    participant E as Encoder wrappers
    participant F as FusionModule
    U->>MP: raw models + fusion_builder + examples
    MP->>E: construct (probe out_dim)
    E-->>MP: out_dim (768 each)
    MP->>MP: fusion_builder(d_sem, d_img, d_txt)
    MP->>F: construct with probe features (B>=2)
    F-->>MP: out_dim (5)
    MP-->>U: ready MainPipeline
```

### 4.2 Inference (`MainPipeline.forward`)

```mermaid
sequenceDiagram
    participant U as caller
    participant MP as MainPipeline
    participant S as SemanticEncoder
    participant I as ImageForensicEncoder
    participant T as ForensicTextEncoder
    participant F as FusionModule
    U->>MP: forward(batch, text?)
    MP->>S: semantic(batch)
    S-->>MP: v_semantic [B,768]
    MP->>I: image(batch)
    I-->>MP: v_imgfor [B,768]
    MP->>T: text(batch) or forward_text(text)
    T-->>MP: v_textfor [B,768]
    MP->>F: fusion(3 features)
    F-->>MP: main_logits, aux_logits, fused
    MP-->>U: logits dict
```

### 4.3 `out_dim` resolution (decision flow)

```mermaid
flowchart TD
    S["construct wrapper"] --> Q1{"example given?"}
    Q1 -- yes --> P["probe: eval()+no_grad(), model(example)"] --> R["return out.shape[-1]"]
    Q1 -- no --> Q2{"model has a dim attribute?"}
    Q2 -- yes --> A["return that int"]
    Q2 -- no --> E["raise ValueError (clear message)"]
```

### 4.4 Probe lifecycle (no side effects)

```mermaid
stateDiagram-v2
    [*] --> SaveMode: capture training flag
    SaveMode --> Eval: set eval mode
    Eval --> Forward: no_grad forward on example
    Forward --> Restore: restore prior mode in finally
    Restore --> [*]: return last dim
```

---

## 5. Component Integration (تكامل المكوّنات)

### 5.1 Integration map

```mermaid
flowchart LR
    YAML["server_config.yaml / configs"] --> REG["registry.build_encoder / build_fusion"]
    REG --> RM["real models (FND-CLIP, Qwen2, DCT, V3PairwiseFusion)"]
    RM --> WR["pluggable wrappers"]
    WR --> MP["MainPipeline.from_models"]
    MP --> SRV["app/server.py /api/analyze"]
```

### 5.2 Contract each injected model must satisfy

| Wrapper | Model `forward` input | Model `forward` output |
|---------|----------------------|------------------------|
| `SemanticEncoder` | the batch object you pass | `Tensor [B, D]` |
| `ForensicTextEncoder` | the batch object you pass | `Tensor [B, D]` (+ optional `runtime_forward(text)`) |
| `ImageForensicEncoder` | the batch object you pass | `Tensor [B, D]` |
| `FusionModule` | feature dict | `Tensor` **or** dict containing a `Tensor` (default key `main_logits`) |

### 5.3 Server request flow (target integration)

```mermaid
sequenceDiagram
    participant CL as client
    participant API as /api/analyze
    participant PRE as preprocessing (DCT, tokenizers)
    participant PIPE as MainPipeline
    CL->>API: POST text + image
    API->>PRE: build batch (dct, fnd inputs)
    PRE-->>API: merged batch dict
    API->>PIPE: pipe(batch, text=text)
    PIPE-->>API: {main_logits, ...}
    API-->>CL: verdict + probabilities (softmax + argmax)
```

---

## 6. Extensibility Guide (دليل إضافة نموذج جديد)

### 6.1 Swap a model in an existing branch (one file)

```mermaid
flowchart LR
    NEW["build new nn.Module -> [B,D]"] --> WRAP["pass to the matching wrapper"]
    WRAP --> DIM["out_dim auto-inferred"]
    DIM --> REBUILD["MainPipeline.from_models -> fusion adapts to new dims"]
```

```python
my_image_model = MyNewImageNet(out_dim=1024)        # any nn.Module -> [B,1024]
img = ImageForensicEncoder(
    my_image_model,
    example_input={"v_imgfor_dct": torch.zeros(2, 1, 224, 224)},
)
assert img.out_dim == 1024
pipe = MainPipeline.from_models(
    semantic_model=..., image_model=my_image_model, text_model=...,
    fusion_builder=lambda ds, di, dt: build_fusion({"type": "v3_pairwise",
                                                    "feat_dim": ds, "num_classes": 5}),
    image_example={"v_imgfor_dct": torch.zeros(2, 1, 224, 224)},
)
```

### 6.2 Add a brand-new modality (a 4th branch)

This is the one case that touches more than one file (by design).

```mermaid
flowchart TB
    A["create audio_forensic.py (new wrapper, 1 class)"] --> B["edit fusion: accept the extra feature"]
    B --> C["edit main_pipeline: wire it + add the key"]
    C --> D["3 existing branch files stay untouched"]
```

### 6.3 Rules for a new model

1. It must be an `nn.Module`.
2. Its `forward` must return `Tensor [B, D]` (fusion may return a dict).
3. Either expose a dimension attribute **or** pass an `example_input`.
4. For a runtime/raw path, implement `runtime_forward(text)` (text branch).

---

## 7. Evaluation Methodology (منهجية التقييم)

### 7.1 Evaluation pipeline

```mermaid
flowchart LR
    DS["test set (2,475, balanced)"] --> PIPE["MainPipeline(batch)"]
    PIPE --> LOG["main_logits [B,5]"]
    LOG --> ARG["argmax"]
    ARG --> MET["classification report + confusion matrix + F1-macro"]
```

### 7.2 Task, dataset, metrics

- **Labels:** `0 Real · 1 Out-of-Context · 2 Manipulated · 3 AI-Text · 4 Fully-Fabricated`.
- **Test set:** 2,475 held-out, **balanced** (495 / class).
- **Primary metric:** **F1-macro**; also accuracy, per-class P/R/F1, confusion matrix.
- **Deployment:** 3-seed ensemble (42 / 1337 / 2024, softmax-averaged) + temperature calibration.
- **Transfer:** MMFakeBench zero-shot with log-prior bias correction.

### 7.3 Ablation methodology (two complementary methods)

```mermaid
flowchart TB
    subgraph TRAIN["Training-time branch ablation"]
        T1["semantic only -> 0.6681"]
        T2["+ text -> 0.7103"]
        T3["+ image -> 0.6975"]
        T4["all three -> 0.7157"]
    end
    subgraph INF["Inference-time leave-one-out"]
        I1["full -> 0.7149"]
        I2["disable image -> -2.0 pp"]
        I3["disable text -> -26.6 pp"]
        I4["semantic only -> -40.0 pp"]
    end
```

### 7.4 Results

**Headline (deployed ensemble):** F1-macro **0.7149**, accuracy **0.7305**.

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Real | 0.428 | 0.370 | 0.397 | 495 |
| Out-of-Context | 0.447 | 0.438 | 0.442 | 495 |
| Manipulated | 0.707 | 0.814 | 0.757 | 495 |
| AI-Text | 0.984 | 0.994 | 0.989 | 495 |
| Fully-Fabricated | 0.994 | 0.986 | 0.990 | 495 |
| macro avg | 0.726 | 0.731 | 0.7149 | 2475 |

![Confusion matrix (deployed 3-seed ensemble, test n=2,475)](confusion_matrix_5class_ensemble.png)

---

## 8. Verification status

All checks were run and pass:

- `python main_pipeline.py` → end-to-end demo with dummy models of **different**
  widths (768 / 512 / 256) → `out_dim` auto-inferred; `main_logits [4,5]`.
- Edge cases: attribute fallback; probe overrides a stale attribute;
  `BatchNorm1d` survives the probe (eval + batch 2); clear error when neither
  source is available.
- Decoupling: grep confirms the four branch files do not import each other; only
  `main_pipeline.py` / `__init__.py` import them.
- One class per branch file; no hardcoded model dimensions in the four wrappers.

---

## 9. Analysis & Notes (تحليلنا وملاحظاتنا)

### 9.1 Strengths
- **True dependency injection** — wrappers are model-agnostic and trivially testable.
- **No hardcoded dimensions** — widths flow from probe/attribute, so swapping a
  model can change dims without editing fusion/classifier by hand.
- **Decoupling is structural** — enforced by imports, verifiable with one grep.
- **Side-effect-free construction** — probe restores training mode; runs under `no_grad()`.

### 9.2 Deliberate trade-offs
- **Probe vs attribute ordering** — probe is truth, but we fall back to attributes
  so construction never forces a heavy backbone (e.g. Qwen2-7B) to run.
- **`BatchNorm1d` constraint** — probe forces `eval()` and batch ≥ 2.
- **`out_dim` logic duplicated across the four files** — to honour "exactly 5 files /
  no shared helper". Relaxing that constraint would allow a `_probe.py` leaf util.
- **Fusion has two widths** (`num_classes`, `fused_dim`); `out_dim` defaults to
  `main_logits`, switchable via `out_key`.

### 9.3 Limitations / risks
- **Shape assumption** — `out_dim = out.shape[-1]`; models returning `[B,T,D]` or
  tuples need an adapter.
- **"Edit one file" = swapping, not adding modalities** — a new branch necessarily
  touches `fusion` + `main_pipeline`.
- **Device ownership** — wrappers don't move the injected model to a device; the
  caller does (the probe only moves the example input).
- **Pass-through outputs** — `aux_logits` / `fused` are returned untouched.

### 9.4 Model-quality notes (separate from pipeline design)
- **Real ↔ OOC is at-chance** (F1 ≈ 0.40 / 0.44) — frozen FND-CLIP cross-modal ceiling.
- **Classes 3/4 near-perfect** (F1 ≈ 0.99) — Qwen text branch isolates AI text.
- **DCT image detector weak on MidJourney/VQDM** (AP ≈ 0.83).

### 9.5 Recommendations
1. Add the dummy-model checks as a committed `tests/` unit test.
2. Optionally support an `example_factory` (lazy callable) for device-aware probes.
3. Wire `MainPipeline` into `app/server.py` to retire the hand-written inference block.
4. Document the `[B, D]` output contract in each branch docstring (partly done).

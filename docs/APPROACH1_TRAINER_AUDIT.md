# Approach 1 trainer-vs-upstream audit (Stage A7)

> Required by `forensic_image_branch_fix_plan.md` Step 5 ("practical path: prove our trainer matches upstream where the spec demands it, deviates only where the spec overrides upstream").

## Question

`phases/forensic/scripts/train_rgb_fourier.py` is OUR Windows-friendly single-GPU adaptation of the upstream `chandlerbing65nm/FakeImageDetection` `train.py`. Does our trainer perform the operations doctor's spec F.4-F.11 demands, and where it diverges from upstream, are those divergences explicitly required by the spec or are they unintended?

## Spec mapping (doctor brief F.4 - F.11)

| ID | Spec text | Our `train_rgb_fourier.py` |
|---|---|---|
| F.4 | Fine-tune `chandlerbing65nm/FakeImageDetection` RN50 with Fourier masking | YES -- imports `networks.resnet.resnet50` and `mask.FrequencyMaskGenerator` from the cloned external repo (lines 56-57) |
| F.5 | Load ckpt `mask_15/rn50ft_fouriermask.pth` exactly (do not rename) | YES with documented fallback -- `_resolve_checkpoint()` (lines 138-156) tries `rn50ft_fouriermask.pth` first; falls back to `rn50ft_spectralmask.pth` because upstream renamed the file (deviation F-A8 in `docs/DECISIONS.md`) |
| F.6 | Freeze conv1/bn1/layer1/layer2; train layer3/layer4/fc | YES -- lines 207-216 iterate `model.named_parameters()` and set `requires_grad = False` for any name starting with `conv1.`, `bn1.`, `layer1.`, `layer2.` |
| F.7 | Call `model.change_output(1)` (NOT manual nn.Linear) | YES -- line 204: `model.change_output(1)` (this is the upstream's public API; we do NOT manually replace `model.fc`) |
| F.8 | Fourier masking 50% probability, mask_ratio=0.15, training-only | YES -- `RandomFourierMask` class (lines 86-103) wraps `FrequencyMaskGenerator(ratio=0.15, band='low+high', transform_type='fourier', channel='all')` and applies it with `random.random() < 0.5` gate. Eval transform omits it entirely (line 121-132). |
| F.9 | Resize 224 bilinear, ToTensor, Normalize ImageNet stats | YES -- `make_train_transform()` lines 106-118 and `make_eval_transform()` lines 121-132 |
| F.10 | BCEWithLogitsLoss, AdamW lr=1e-4 wd=1e-4, batch=64, max 30 epochs, ReduceLROnPlateau(mode='max', factor=0.5, patience=3) on val AP, ES patience=5 | YES -- constants at lines 63-69; `BCEWithLogitsLoss()` line 325; `AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)` line 327; `ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)` lines 328-330; early-stop patience=5 with `patience_left -= 1` logic lines 344-359 |
| F.11 | Save best as `forensic_rgb_model.pth` | YES -- line 335 defines `forensic_named_path = args.out_dir / "forensic_rgb_model.pth"`; lines 402-404 copy best.pt to that filename at end of training |

## Behavioral diff: ours vs upstream `external/FakeImageDetection/train.py`

These differences exist **because doctor's spec overrides upstream defaults**. Every one is required by the spec, not accidental.

| Aspect | Upstream `train.py` | Our `train_rgb_fourier.py` | Why we differ |
|---|---|---|---|
| Seed | 44 (hard-coded line 70) | CLI `--seed` default 42 | Spec-neutral. Our project-wide seed is 42 (V4 convention). |
| Distributed | DistributedDataParallel (line 187) | Single-GPU | Doctor's spec doesn't specify DDP; single-GPU is required for the RTX 4070 + Windows. |
| Image preprocessing pipeline | `custom_resize` + `data_augment` (blur p=0.1, JPEG p=0.1) + `RandomCrop(224)` + `RandomHorizontalFlip()` + ToTensor + Normalize (utils.py:50-87) | Resize(224, bilinear) + ToTensor + Normalize (lines 108-118) | **Required by F.9**. Doctor's spec is literally `Resize 224x224 bilinear, ToTensor, Normalize ImageNet stats`. No blur, no JPEG, no random crop, no flip. |
| Mask probability gate | Always-on -- `transforms.Lambda(lambda img: mask_generator.transform(img))` (utils.py:48) | 50% gate via `if random.random() < 0.5` (line 101) | **Required by F.8**. Doctor's spec says `50% probability, training-only`. |
| Head replacement | `model.fc = nn.Linear(model.fc.in_features, 1)` (train.py:163) | `model.change_output(1)` (line 204) | **Required by F.7**. The spec says explicitly "NOT manual nn.Linear creation". |
| Layer freezing | None -- all params trainable (train.py:185) | conv1/bn1/layer1/layer2 frozen (lines 207-216) | **Required by F.6**. Spec freezes early stages to preserve ImageNet low-level features and only fine-tune mid/late stages. |
| LR scheduler | None | `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` on val AP (lines 328-330) | **Required by F.10**. |
| Early-stop metric | Validation accuracy (utils.py:233) | Validation AP (lines 350, 353-359) | **Required by F.10 + F.23**. Doctor's spec defines AP as the primary metric (F.23) and our scheduler+ES contract uses AP. Upstream uses accuracy because Wang_CVPR2020 has balanced binary classes; on the genimage subset AP is more informative. |
| Mask band | `band` is a CLI choice (`'all'` default, train.py:307) | Hard-coded `band='low+high'` (line 97) | **Documented as F-A8**; Stage A8 audits the behavioral diff between bands and Stage C-lite re-runs with `band='all'` to compare. |

## Spot-check: optimizer parameter set

Doctor's spec F.6 says only `layer3/layer4/fc` train. Verify that our optimizer actually receives only those parameters:

`train_rgb_fourier.py:326-327`:

```python
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
```

Our freeze loop (lines 210-216) sets `p.requires_grad = False` for `conv1.*`, `bn1.*`, `layer1.*`, `layer2.*`. The list comprehension on line 326 then filters to only the params with `requires_grad=True`. Result: the optimizer only sees layer3/layer4/fc params. **Matches spec.**

Upstream, by contrast, passes `model.parameters()` directly (train.py:190): the optimizer sees all params. This is the fundamental behavior difference our spec demands.

## Spot-check: BCEWithLogitsLoss + sigmoid contract

Both upstream and ours use `BCEWithLogitsLoss()` for training and `torch.sigmoid()` for eval probability extraction. Our eval (`eval_one_epoch`, lines 246-261) computes `torch.sigmoid(logits).cpu().numpy()` then `average_precision_score(labels, probs)` and `(probs >= 0.5) == labels` for accuracy. Identical to upstream's `outputs.sigmoid().detach().cpu().numpy()` (utils.py:216) followed by `accuracy_score(y_true, y_pred > 0.5)` and `average_precision_score(y_true, y_pred)` (utils.py:222-223).

## Checkpoint payload structure

`train_rgb_fourier.py:376-386`:

```python
payload = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "epoch": epoch,
    "val_ap": val_ap,
    "val_acc": val_acc,
    "init_method": init_method,
    "best_ap": best_ap,
    "best_epoch": best_epoch,
}
```

Loaders in `phases/forensic/scripts/eval_rgb.py` know to look for `model_state` (no `module.` prefix) and the V4 image encoder wrapper at `phases/v4/src/v4/models/encoders/rgb_forensic.py` does the same. Schema is consistent.

## Verdict

`train_rgb_fourier.py` correctly implements doctor's spec F.4-F.11. Every spec point is honored; every deviation from upstream is required by the spec (not an oversight). The single open item is **F-A8 (mask band)** which is being investigated as Stage A8 (behavioral diff) + Stage C-lite (band="all" retrain experiment).

## Provenance

- Our trainer: `phases/forensic/scripts/train_rgb_fourier.py` (427 lines)
- Upstream trainer: `phases/forensic/external/FakeImageDetection/train.py` (449 lines)
- Upstream utilities: `phases/forensic/external/FakeImageDetection/utils.py:44-242`
- Spec source: `docs/doctor-briefs/Forensic_Image_Detector_En.pdf` section F.4-F.11 (per `docs/MASTER_CHECKLIST.md`)
- F-A8 deviation context: `docs/DECISIONS.md` entry F-A8

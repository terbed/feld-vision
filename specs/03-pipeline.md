# 03 — Training Pipeline

Config-driven (YAML), model-agnostic, ClearML(and local)-logged. Runs on a GPU server or from any local machine with clearml config.

## Dataset / dataloader

`ChipDataset(chips_split.parquet, split, basemap_cog, mask_cog, taxonomy, cfg)`:

- Holds open rasterio handles to `basemap_z15.tif` and `mask_z15.tif` (one per
  worker; COG internal tiling makes windowed reads ~ms).
- `__getitem__`:
  1. take chip row → `(row_off, col_off)` (+ bounded random jitter for train only).
  2. read RGB window from basemap, class-id window from mask.
  3. **remap** raw ids → contiguous active training labels; taxonomy-ignored raw
     ids → `255`.
  4. build the detail input and, for dual-stream models, the context input.
  5. apply augmentation (train only).
  6. return `detail[3,H,W]`, optional `context[3,H,W]`,
     `target[H,W]`, and `meta(chip_id, style_id)`.

### Detail and context inputs

The detail input is always the native 256×256 RGB chip. Dual-stream models also
receive one wider RGB context crop:

- Read a `(256·context_scale)×(256·context_scale)` window with the same center as
  the detail chip and resize it to 256×256. Default `context_scale=4`.
- Detail and context remain separate 3-channel tensors. They are not concatenated
  into a 6-channel pseudo-image.
- The target and prediction always correspond only to the native-resolution detail
  chip. The context branch has no separate segmentation target.
- The single-stream baseline does not read the context window, avoiding unnecessary
  I/O.

### Augmentation (train)

Flips, 90° rotations, small brightness/contrast/hue jitter (style robustness),
and a train-only spatial jitter capped at 64 px per axis. Library:
`albumentations` (applied identically to image; geometric ops also to mask,
photometric only to image). For dual-stream inputs, use the same sampled geometric
and photometric transform parameters for detail and context so their centers and
appearance remain consistent. Configurable.

## Model registry (plug-and-play)

```
@register_model("segformer_b2_shared_context")
def build(cfg, num_classes) -> nn.Module: ...
```

- A model is any `nn.Module` with signature
  `forward(detail, context=None) -> logits[B, num_classes, H, W]`.
- `build_model(cfg)` looks up `cfg.model.name` and passes `num_classes` from the
  taxonomy. Every encoder keeps the normal 3-channel RGB patch embedding, preserving
  direct use of ImageNet-pretrained weights.
- **Variant 1 — `segformer_b2_single`:** standard SegFormer MiT-B2 encoder and
  decoder operating only on the detail crop. This is the baseline.
- **Variant 2 — `segformer_b2_shared_context`:** run detail and context separately
  through the same MiT-B2 encoder weights. At each of the four encoder stages,
  concatenate the detail and context feature maps and project them back to the
  stage's normal channel width with a learned 1×1 projection. The standard
  SegFormer decoder consumes the four fused feature maps and predicts the detail
  target.
- **Variant 3 — `segformer_b2_separate_context`:** the same feature-fusion and
  decoder design, but detail and context use independent pretrained MiT-B2
  encoders. This allows scale-specific representations at roughly twice the encoder
  parameters of the shared-weight variant.
- The shared and separate variants use the same fusion modules and decoder so the
  comparison isolates whether encoder weight sharing is beneficial.
- Both dual-stream variants require approximately twice the encoder compute and
  activation memory of the single-stream baseline; the separate-encoder variant
  additionally carries the second encoder's parameters and optimizer state.
- Custom models: drop a file in `feldvision/models/`, decorate with
  `@register_model`, reference by name in config. No pipeline changes.

## Training loop

- Optimizer, LR schedule, and early stopping are fully config-driven. Default:
  AdamW (`lr=1e-4`, `weight_decay=0.01`), cosine decay after 5 warmup epochs to
  `min_lr=1e-6`, mixed precision (`bf16`/`amp`), and gradient clipping at 1.0.
- Loss: combined CE + Dice, class weights + `ignore_index=255` (specs/02).
- Sampler: `WeightedRandomSampler` from the imbalance-aware weights (specs/02).
- Per epoch: train → validate (mIoU, per-class IoU, per-`style_id` IoU) → checkpoint
  on best val mIoU. Default early stopping monitors val mIoU after epoch 10 with
  `patience=15` and `min_delta=0.001`.
- Everything (lr, losses, metrics, config) logged to ClearML scalars.

## Test = full-sheet reconstruction

For each held-out test `style_id`:

1. Compute the sheet's pixel bounding box in the COGs.
2. Sweep it with a regular grid (stride = `chip_size`, or overlapping with averaging
   for seam-free output — config `recon_overlap`).
3. Run inference per window; **accumulate logits** into a full-sheet canvas
   (overlap-averaged), then argmax → predicted class map.
4. Mask the canvas to the sheet polygon (ignore pixels outside).
5. Compute metrics over the sheet (vs `mask_z15.tif` within the polygon).
6. Export both a georeferenced class-id GeoTIFF and a standard colorized PNG
   prediction image (transparent outside the sheet).
7. **Log to ClearML** the standalone prediction PNG and the triptych
   `basemap | basemap overlayed with ground-truth | basemap overlayed with
   prediction` as images, plus the sheet's per-class IoU table.

This is the headline qualitative deliverable: you see the model's predicted map
sheet next to the real one. Reconstruction code lives in `feldvision/reconstruct/`
and is reusable for **unlabeled inference regions** (predict-only, no metrics) for
generalization demos.

## ClearML logging

- One **Task** per run; `connect(cfg)` captures the full config.
- Scalars: losses, lr, mIoU, per-class IoU, per-style IoU.
- Images: reconstructed test sheets (per epoch or at end), a few train/val chip
  triptychs for sanity.
- Artifacts: best checkpoint model, the resolved config, `splits.summary.json`.
- Project name / server configurable; defaults in `configs/default.yaml`.
- Also logged in local experiment-run folder for local eval convenience

## Config surface (single YAML per experiment)

```yaml
data:
  basemap: ../dataprep/data/basemap_z15.tif
  mask:    ../dataprep/data/mask_z15.tif
  chips:   ../dataprep/data/chips_split.parquet
taxonomy: configs/taxonomy_default.yaml
split:
  test_style_ids: [ ... ]
  val_fraction: 0.20
  val_cell_size_px: 1024
  buffer_steps: derived
  seed: 0
sampler:
  strategy: rarest_class
  alpha: 0.5
  w_background: 0.01
  max_class_oversample_factor: 10.0
  min_sampling_chips: 100
augmentation:
  jitter: true
  max_jitter_px: 64
loss:
  type: ce_dice
  class_weight_beta: 0.5
  max_class_weight_ratio: 5.0
  ignore_index: 255
model:
  name: segformer_b2_single    # or segformer_b2_shared_context / segformer_b2_separate_context
  pretrained: true
  context_scale: 4             # used only by dual-stream variants
optim:
  name: adamw
  lr: 1e-4
  weight_decay: 0.01
  epochs: 100
  amp: bf16
  gradient_clip_norm: 1.0
scheduler:
  name: cosine
  warmup_epochs: 5
  min_lr: 1e-6
early_stopping:
  monitor: val/miou
  mode: max
  start_epoch: 10
  patience: 15
  min_delta: 0.001
clearml: {project: feld-vision, task: segformer_b2_baseline}
```

## Build order

1. `scripts/build_splits.py` — read `chips.parquet`, write `chips_split.parquet` +
   `splits.summary.json` (specs/02). *(Lives here or in dataprep; reads the index.)*
2. `feldvision/data/` — dataset, taxonomy remap, augmentation, sampler.
3. `feldvision/models/` — registry + single-stream, shared-context, and
   separate-context SegFormer variants.
4. `feldvision/train/` — loop, losses, metrics.
5. `feldvision/reconstruct/` — full-sheet inference + ClearML triptychs.
6. `scripts/train.py`, `scripts/test.py` — entrypoints wiring config → run.

Smoke-test each stage on a single sheet / a few hundred chips before scaling.

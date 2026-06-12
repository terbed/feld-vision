# 00 — Project Overview & Plan

## Objective

Train and evaluate semantic segmentation models that predict water/wetland classes
from crops of the First Military Survey map. Build it so **any model is plug-and-play**
and experiments are reproducible and tracked in ClearML.

## Locked decisions (from planning)

| Decision | Choice |
|---|---|
| **Taxonomy (default v1)** | 6 active classes + background: river(10), stream(11), creek(12), wetland(13), wet forest(14), lake(16). Oxbow(15) and currently under-supported canal(17) → **ignore**. Mapping is config-driven so canal can be enabled when its annotations are sufficiently representative. |
| **Test set** | Whole held-out `style_id`s (sheet = spatial block). No sheet leaks into train. |
| **Validation set** | Whole 1024×1024 raster-aligned macro-cells selected to approximate **20%** of the training region while matching per-style, background, and active-class distributions. A jitter-aware guard excludes nearby training chips whose target windows could overlap validation targets. |
| **Chip grid** | Stride-128, 256×256 windows over the **union of the 164 sheet polygons**. Parquet holds all center-in-union positions incl. pure-background, with per-class counts; boundary/overlap chips that cannot be assigned to exactly one sheet get `style_id=null` and are excluded from splits/training. The ~3.7% labels in the western strip outside any sheet are dropped. |
| **Models / input** | Compare three SegFormer variants: (1) single-stream RGB baseline, (2) detail + context with a shared-weight two-stream encoder, and (3) detail + context with separate encoders. Dual-stream models fuse encoder features; detail and context RGB are never concatenated as input channels. |
| **Imbalance** | Handled twice: (a) imbalance-aware **chip sampling** using per-class counts in the index; (b) class-weighted loss + ignore index. |

## Class taxonomy mapping (v1)

Raw mask id → training label id (output channel). `ignore_index = 255`.

```
0  background   -> 0
10 river        -> 1
11 stream       -> 2
12 creek        -> 3
13 wetland      -> 4
14 wet forest   -> 5
16 lake         -> 6
15 oxbow        -> 255
17 canal        -> 255
```

7 output channels (0–6) for the default taxonomy. Pixels mapped to `255` are
excluded from loss and metrics, but the surrounding chip remains usable. The
mapping lives in config (`taxonomy.yaml`), so enabling canal, grouping classes, or
using a wetland-only taxonomy is a one-file change; the output-channel count is
derived from the active mapping.

## Phases

1. **Chip index** (in `../dataprep`): produce `chips.parquet`. → `specs/01-chip-index.md`
2. **Splits & sampling** (in `feld-vision`): test sheet holdout, val carve-out with
   buffer exclusion, weighted sampler. → `specs/02-splits-and-sampling.md`
3. **Pipeline**: dataset/loader, model registry, train/val/test, full-sheet
   reconstruction, ClearML logging. → `specs/03-pipeline.md`

## Inputs this project consumes (from `../dataprep/data/`)

- `basemap_z15.tif` — RGB COG (model input)
- `mask_z15.tif` — class-id COG (target), co-registered
- `qgis/stylesheets.shp` — 164 sheet polygons w/ `style_id`
- `mask_z15.json` — class stats + `label_version`

All EPSG:3857; the two rasters share size/bounds/transform exactly.

## Non-goals (v1)

- Oxbow prediction, and canal prediction until canal has sufficient representative
  training support.
- Predicting outside surveyed sheets (stylesheets).
- Vectorizing predictions back to polygons (post-processing, later).

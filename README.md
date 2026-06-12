# feld-vision

Semantic segmentation of **hand-drawn 18th-century military survey maps** (the
Habsburg *First Military Survey*, ~1780s) covering the Carpathian Basin. The goal
is to train a neural network that, given a crop of the old map, predicts where the
cartographers painted **water and wetland features** — rivers, streams, marshes,
lakes, etc.

This repo (`feld-vision`) is the **deep-learning pipeline**: data loading, model
training, validation, testing, and experiment logging. It is designed to run on a
GPU server with **ClearML** for experiment tracking and visual result inspection.

The **data preparation** lives in a sibling repo, `../dataprep`, which scraped the
map tiles and rasterized the human annotations. This README explains what that
produced so an engineer (or agent) can pick up the project cold.

---

## The big picture

- The base imagery is the Arcanum/Mapire scan of the First Military Survey, fetched
  as XYZ map tiles at **zoom 15** (~4.78 m/pixel) and stitched into one big
  georeferenced raster.
- Domain experts hand-digitized water features in QGIS as polygons (the *pocsolya*
  — Hungarian for "puddle/marsh" — shapefile). Each polygon has a class `id`.
- The map was drawn over many years by **different cartographers**, so drawing
  style varies sheet to sheet (brush weight, colors, hatching). To handle this,
  the surveyed area is partitioned into **164 sheet polygons**, each with a unique
  `style_id`. These double as spatial blocks for leakage-free train/test splitting.

The core ML task is **multi-class semantic segmentation with extreme class
imbalance** (~98% of pixels are background) and **domain shift across sheets**.

---

## Data produced by `../dataprep`

All paths below are under `../dataprep/data/` unless noted. Everything shares one
coordinate system: **EPSG:3857 (Web Mercator)**, and the two rasters are
**pixel-for-pixel co-registered** (same size, bounds, transform).

| File | What it is | Size |
|---|---|---|
| `basemap_z15.tif` | RGB Cloud-Optimized GeoTIFF of the old map. 256,768 × 191,232 px, internally tiled 256×256, 5 overview levels. The model **input**. | ~15 GB |
| `mask_z15.tif` | uint8 GeoTIFF, one band, pixel value = class id (see legend). Co-registered with the basemap. The model **target**. | ~78 MB |
| `mask_z15.json` | Metadata sidecar: bounds, per-class pixel counts/fractions, and `label_version` (a hash of the source shapefile — lets you detect a stale mask). | small |
| `qgis/stylesheets.shp` | 164 polygons, attribute `style_id` (unique per polygon). Defines the map sheets / spatial blocks. EPSG:4326. | small |
| `pocsolya_2026-06-02/pocsolya.shp` | Source vector annotations (11,685 polygons). Not needed at train time — already rasterized into the mask — but kept for reference and re-rasterization. | ~42 MB |

### Class legend (raw mask values)

| id | feature | notes |
|---|---|---|
| 0 | background | ~97.8% of pixels |
| 10 | river | linear |
| 11 | stream (ér) | linear |
| 12 | creek (patak) | linear, very rare |
| 13 | wetland (marsh) | most common non-bg class (~1.7%) |
| 14 | wet forest (wooded marsh) | |
| 15 | oxbow (holtág) | **dropped for v1** — too few examples, needs global river context; mapped to *ignore* in training |
| 16 | lake (tó) | rare |
| 17 | canal (csatorna) | rarest (~0.002%) |

The pixel grid of `mask_z15.tif` is aligned to the z=15 tile grid, so a 256×256
window read at the same `(col_off, row_off)` from both rasters is guaranteed to
correspond.

### Resolution & geometry quick-reference

- 1 pixel ≈ 4.78 m on the ground.
- A 256×256 training chip ≈ 1.22 km across.
- The surveyed (annotated) area is ~144,000 km²; ~385k stride-128 chip positions
  fall inside the sheet polygons.

---

## How `feld-vision` uses the data

1. **Chip index** (`../dataprep` builds `chips.parquet`): a table of every
   striding ROI (region-of-interest window) inside the sheet polygons, annotated
   with its pixel offset, `style_id`, and per-class pixel counts. This is the
   sampling index — it lets us pick chips by label content without scanning the
   rasters. See `specs/01-chip-index.md`.
2. **Splits**: chosen `style_id`s are held out entirely as the **test** set;
   validation is a stratified 20% carved from the training region with adjacent
   overlapping chips blocked from train. See `specs/02-splits-and-sampling.md`.
3. **Dataloader**: reads chip windows on the fly from the COGs (image + mask),
   remaps raw class ids to the training taxonomy, applies augmentation. Supports a
   configurable **dual-resolution context** input (detail + downsampled context),
   with a plain single-scale RGB mode for baselines.
4. **Models**: a registry of pluggable segmentation models (SegFormer baseline;
   any custom `nn.Module` can be dropped in).
5. **Train / val / test loops** with class-weighted loss and imbalance-aware
   sampling.
6. **Test = full-sheet reconstruction**: for each held-out sheet, the model sweeps
   the whole sheet, predictions are stitched back into a full-sheet raster, and the
   georeferenced prediction GeoTIFF, colorized PNG, and
   `basemap | ground-truth | prediction` triptych are produced. The PNG and
   triptych are logged to ClearML for visual evaluation. See
   `specs/03-pipeline.md`.

---

## Tech stack

- **uv** for environment/dependency management (`uv sync`, `uv run ...`).
- **PyTorch** + **rasterio** (windowed COG reads) + **geopandas** (sheet geometry).
- **ClearML** for experiment tracking, metrics, and logging the reconstructed test
  maps.
- Designed to be **moved to a DL server**: data paths are configurable; the repo
  ships code + configs, the large COGs are synced separately.

## Repo layout

```
feld-vision/
  README.md
  specs/               <- design contracts
  pyproject.toml       <- uv project and CLI entry points
  configs/             <- experiment and taxonomy YAML
  feldvision/
    data/              <- index validation, splits, sampling, raster datasets
    models/            <- registry and three SegFormer variants
    train/             <- losses, metrics, scheduler, trainer
    reconstruct/       <- full-sheet inference, export, previews
    cli/               <- split, train, and test entry points
  scripts/             <- direct Python wrappers for the CLI entry points
  tests/               <- synthetic unit and integration tests
```

## Status

The v1 pipeline described by the specs is implemented. The default held-out test
stylesheets are `53, 54, 22, 18, 121`. Generated chip-index, split, run, and
reconstruction artifacts are intentionally ignored by Git.

```bash
uv sync
uv run pytest
uv run feldvision-build-index --config configs/default.yaml
uv run feldvision-build-splits --config configs/default.yaml --input data/chips.parquet
uv run feldvision-train --config configs/default.yaml
uv run feldvision-test --config configs/default.yaml --checkpoint runs/<run>/checkpoints/best.pt
```

For a bounded CPU smoke test using real raster windows:

```bash
uv run feldvision-train --config configs/smoke.yaml
uv run feldvision-test --config configs/smoke.yaml \
  --checkpoint runs/production-smoke/checkpoints/best.pt \
  --style-id 53 --max-size-px 256 --overlap 0
```

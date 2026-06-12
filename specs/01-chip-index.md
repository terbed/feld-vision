# 01 — Chip Index (`chips.parquet`)

The chip index is the backbone of training: a flat table where **one row = one
candidate 256×256 ROI** (a striding window). It records *where* the chip is and
*what labels it contains*, so the dataloader/sampler can pick chips by content
without ever scanning the 15 GB raster. Built once in `../dataprep`; consumed by
`feld-vision`. The pipeline is suitable for quick reconstruction, because annotations and stylesheets updates arriving iteratively (ultimately, covering all Carpathian basin).

> Format: **Parquet** (not CSV). Same columns, but ~10× smaller and typed; pandas/
> polars read it instantly. A `chips.csv` mirror can be emitted for eyeballing.

## Inputs

- `mask_z15.tif` — class-id raster (source of per-class pixel counts).
- `qgis/stylesheets.shp` — 164 sheet polygons (`style_id`); reprojected to EPSG:3857.
- Grid params: `chip_size=256`, `stride=128` (config).

The basemap is **not** read here — only geometry/labels matter for the index.
Co-registration guarantees a chip window in the mask matches the same window in the
basemap at load time.

## Grid definition

- Origin = top-left of the raster `(row_off=0, col_off=0)`.
- Positions: `row_off ∈ {0, 128, 256, …}`, `col_off ∈ {0, 128, …}` such that the
  256-window stays in bounds.
- Emit a position **iff** its window center falls inside the union of sheet
  polygons (surveyed area). ~385k positions qualify.
- Assign `style_id` only when the full 256×256 window is unambiguously inside
  exactly one sheet polygon. Boundary/overlap windows are still emitted, but with
  `style_id=null`; split/training code prunes them later.

## Extraction algorithm

```
load stylesheets -> reproject 3857 -> build STRtree spatial index
open mask_z15.tif (windowed reads, block-aligned for speed)

for each 4096-block of the raster that intersects the sheet union:
    read the block once (cheap: mask is tiny / sparse)
    for each (row_off, col_off) stride position whose 256-window lies in this block:
        center_xy = pixel_to_world(row_off+128, col_off+128)
        if center_xy is outside sheet union: continue
        window_geom = pixel_window_to_world_polygon(row_off, col_off, 256)
        sheets = sheets_intersecting(window_geom)
        style_id = sheets[0].style_id if len(sheets) == 1 and sheets[0].covers(window_geom) else null
        window = mask[row_off:row_off+256, col_off:col_off+256]
        counts = bincount(window over class ids 0,10..17)
        emit row(...)
```

Reading the mask block-by-block (4096²) and slicing 256-windows out of memory keeps
this to a few minutes. (Most blocks are 100% background and can be counted in O(1).)

Boundary chips are intentionally kept in the index for diagnostics and reconstruction
bookkeeping, but `style_id=null` marks them as unavailable for split assignment and
training. This also neutralizes minor stylesheet overlaps without needing to repair
the source polygons before v1.

## Schema (one row per chip)

| column                 | type    | meaning                                                                   |
|---                     |---      |---                                                                        |
| `chip_id`              | str     | `r{row_off}_c{col_off}` — stable, unique                                  |
| `row_off`, `col_off`   | int32   | top-left pixel offset into the COGs                                       |
| `size`                 | int16   | window size (256); stored for future multi-size support                   |
| `center_x`, `center_y` | float64 | chip center in EPSG:3857                                                  |
| `lon`, `lat`           | float32 | chip center in EPSG:4326 (for plotting/QGIS)                              |
| `style_id`             | int32?  | sheet id from `stylesheets.shp`; null for ambiguous boundary/overlap chips |
| `px_bg`                | int32   | background pixel count (raw id 0)                                         |
| `px_10` … `px_17`      | int32   | per-class pixel counts (one column per raw id 10–17, incl. `px_15` oxbow) |
| `n_label_px`           | int32   | non-background pixels = `size² − px_bg`                                   |
| `dominant_class`       | int16   | raw id of the most frequent non-bg class (−1 if none)                     |
| `label_version`        | str     | copied from `mask_z15.json`; detects a stale index                        |

Notes:
- Keeping **raw** per-class counts (not remapped) means the index is taxonomy-
  agnostic: switching to a 5-class grouping never requires rebuilding it.
- ~385k rows × ~18 columns ≈ a few MB Parquet.

## Derived helper columns (computed at load, not stored)

The loader/sampler derives booleans cheaply from the count columns, e.g.
`has_river = px_10>0`, `has_lake = px_16>0`, `n_active_px`,
`is_active_background = n_active_px==0`, `dominant_active_class`, and
`rarest_present`. These use only raw ids active in the selected taxonomy, so ignored
classes never drive stratification or sampling. Keeping them out of the file avoids
rebuilds when the taxonomy changes.

## Validation of the index

After build, assert:
- every non-null `style_id` in the index exists in `stylesheets.shp`;
- rows with `style_id=null` are excluded from train/val/test split assignment;
- `px_bg + Σ px_1x == size²` for every row;
- `label_version` matches the current mask;
- a few random rows: re-read the mask window and recompute counts → must match.

Emit a short `chips.summary.json`: row count, rows-per-sheet/null-style histogram,
label-bearing row count, per-class chip counts (how many chips contain each class).
This is the data we use to design the sampler and pick test sheets.

## Rerun
When new annotations and styleid sheets arrive the new `chips.parquet` file should be easily regenerated.

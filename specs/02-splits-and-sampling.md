# 02 — Splits & Imbalance-Aware Sampling

Operates on `chips.parquet`. Produces a `split` assignment per chip and a sampling
strategy for training. All deterministic given a `seed`.

## Split taxonomy

Each chip is assigned exactly one of: `test`, `val`, `train`, `excluded`.
Rows with `style_id=null` are boundary/overlap chips from the index and are assigned
`excluded` before any test/validation sampling.

### 1. Test — held-out sheets (spatial blocks)

- Config lists `test_style_ids: [...]`.
- Every chip whose `style_id ∈ test_style_ids` → `test`. Removed from all training
  consideration.
- Because a sheet is a contiguous ~500 km² block, this is leakage-free by
  construction and supports **full-sheet reconstruction** at test time (specs/03).
- Picking the test sheets: choose a handful that together cover the class and style
  diversity (use `chips.summary.json` per-sheet class histograms). Keep this list in
  the experiment config so runs are comparable.

### 2. Validation — representative macro-cells + buffer exclusion

The remaining non-null-style, non-test chips are the **training region**. From it:

1. **Group spatially.** Assign each chip to a raster-aligned macro-cell:
   `cell_row = row_off // val_cell_size_px`,
   `cell_col = col_off // val_cell_size_px`, with
   `val_cell_size_px=1024` by default. The validation group key is
   `(style_id, cell_row, cell_col)`, so a group never mixes styles.
2. **Summarize groups.** Aggregate chip count, active-class pixel/chip counts, and
   active-background chip count for every group using the selected taxonomy.
3. **Select whole groups.** With a fixed seed, choose complete validation groups
   until approximately `val_fraction=0.20` of eligible chips is reached. Selection
   minimizes deviation between validation and training-region distributions for:
   - chips per `style_id`;
   - active-background versus label-bearing chips;
   - per-active-class chip presence and pixel fractions.
   Require every active class to occur in both validation and train when its support
   spans enough groups; otherwise report the unsatisfied coverage constraint.
4. **Buffer exclusion (anti-leakage).** Validation chips are not jittered. A
   training chip is excluded if its 256×256 target window, after any allowed
   training jitter, could overlap a validation target window. The Chebyshev grid
   guard is derived rather than chosen independently:

   ```
   buffer_steps = ceil((chip_size + max_jitter_px) / stride) - 1
   ```

   With `chip_size=256`, `stride=128`, and `max_jitter_px=64`, the default is
   `buffer_steps=2`.
   - Implementation: chips live on a regular grid → neighbor lookup is integer math
     on `(row_off//stride, col_off//stride)`, no geometry needed.
5. Everything still in the training region after removing val + excluded → `train`.

The downsampled context input is deliberately not treated as target leakage: it may
include imagery from validation areas, but no validation labels or validation target
pixels are used in the training loss. Only target-window overlap determines the
guard.

Knobs: `val_fraction=0.20`, `val_cell_size_px=1024`,
`max_jitter_px=64`, derived `buffer_steps=2`, `seed`.

### 3. Train — the remainder

Everything not `test`, `val`, or `excluded`.

Persist the assignment as a `split` column (write `chips_split.parquet`, or a small
sidecar keyed by `chip_id`) plus a `splits.summary.json` reporting counts per split,
per-sheet, and per-class. Also report the achieved validation fraction, selected
macro-cell counts, validation-vs-training-region distribution deltas, derived
buffer size, and any active class that could not be represented in both train and
validation.

## Imbalance-aware training sampler

Background dominates (~98% of pixels; most chips are pure background). Two layered
mechanisms, both config-driven:

### (a) Chip-level weighted sampling

Each train chip gets a sampling weight from its taxonomy-active class content.
Ignored raw classes never affect the weight. Default **"rarest-class" weighting**:

```
f_ref = frequency of the most common active foreground class
for each chip:
    present = active classes c with px_c > 0
    if not present:                      # pure active-background chip
        w = w_background                 # very small constant, e.g. 0.01 (active chips contain background as well)
    else:
        for c in present:
            score[c] = min((f_ref / global_active_freq[c]) ** alpha,
                           max_class_oversample_factor)
            if support_chip_count[c] < min_sampling_chips:
                score[c] = 1.0           # unstable class does not drive oversampling
        w = max_{c in present} score[c]
weights normalized; torch WeightedRandomSampler draws an epoch of N chips.
```

- `global_active_freq[c]` = fraction of active pixels that are class c (from the
  training split). `alpha=0.5` (sqrt-inverse) is the default.
- `max_class_oversample_factor=10.0` caps how strongly any rare active class can
  increase chip sampling.
- `min_sampling_chips=100` prevents classes with too few supporting training chips
  from driving oversampling; emit a warning when this guard activates.
- `w_background` keeps enough negatives in the mix (model must learn "no feature").

Alternative strategies selectable by config: `uniform`, `inverse_freq`,
`class_balanced_targeted` (pick a target class, then a random chip containing it).

### (b) Loss-level weighting + ignore

- The taxonomy explicitly maps unsupported raw classes to `ignore_index=255`.
  Ignored pixels contribute to neither CE, Dice, sampling statistics, nor metrics;
  other pixels in the same chip remain usable.
- Per-class loss weights are proportional to `(f_ref / freq) ** beta`, with
  `beta=0.5` and `max_class_weight_ratio=5.0` by default. Normalize after clipping
  while preserving the capped max/min ratio.
- Combined **CE + Dice** (Dice is robust to imbalance; CE gives stable gradients).
  Loss is pluggable.

### Augmentation: random jitter around grid positions (without leakage)

For training only, the loader may offset a sampled chip independently in each axis
by an integer in `[-max_jitter_px, +max_jitter_px]`, with
`max_jitter_px=64` by default. Validation and test chips are never jittered. The
limit keeps the actual crop near the nominal chip whose class counts determined its
sampling weight; the split guard above accounts for the maximum displacement.

## Metrics

- Primary: **mean IoU** and **per-class IoU** over taxonomy-active classes.
- Reported **globally and per `style_id`** on val/test → exposes cross-style
  generalization (the whole reason sheets are tracked).
- Background IoU reported separately (it will be ~0.99 and shouldn't flatter mIoU).

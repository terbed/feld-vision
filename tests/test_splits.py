import pandas as pd

from feldvision.config import SplitConfig
from feldvision.data.splits import build_splits
from feldvision.taxonomy import ClassDefinition, Taxonomy


def taxonomy() -> Taxonomy:
    return Taxonomy(
        classes=(
            ClassDefinition("background", 0, (0,), (0, 0, 0)),
            ClassDefinition("river", 1, (10,), (0, 0, 255)),
            ClassDefinition("lake", 2, (16,), (0, 100, 255)),
        ),
        ignored_raw_ids=frozenset({15, 17}),
    )


def chip_grid() -> pd.DataFrame:
    rows = []
    size = 16
    stride = 8
    for style_id, col_base in ((1, 0), (2, 96), (3, 192)):
        for row_index in range(8):
            for col_index in range(8):
                row_off = row_index * stride
                col_off = col_base + col_index * stride
                river = 4 if (row_index + col_index) % 3 == 0 else 0
                lake = 2 if row_index in (1, 5) else 0
                rows.append(
                    {
                        "chip_id": f"r{row_off}_c{col_off}",
                        "row_off": row_off,
                        "col_off": col_off,
                        "size": size,
                        "style_id": style_id,
                        "px_bg": size**2 - river - lake,
                        "px_10": river,
                        "px_16": lake,
                    }
                )
    rows.append(
        {
            "chip_id": "r80_c80",
            "row_off": 80,
            "col_off": 80,
            "size": size,
            "style_id": None,
            "px_bg": size**2,
            "px_10": 0,
            "px_16": 0,
        }
    )
    return pd.DataFrame(rows)


def test_splits_are_deterministic_group_atomic_and_buffered() -> None:
    config = SplitConfig(
        test_style_ids=(3,),
        val_fraction=0.25,
        val_cell_size_px=32,
        chip_size=16,
        stride=8,
        seed=7,
    )
    first = build_splits(chip_grid(), taxonomy(), config, max_jitter_px=4)
    second = build_splits(chip_grid(), taxonomy(), config, max_jitter_px=4)

    assert first.chips["split"].tolist() == second.chips["split"].tolist()
    assert set(first.chips.loc[first.chips["style_id"].eq(3), "split"]) == {"test"}
    assert first.chips.loc[first.chips["style_id"].isna(), "split"].eq("excluded").all()
    assert first.summary["buffer_steps"] == 2
    assert set(first.summary["counts_per_style"]) == {"1", "2", "3"}
    assert set(first.summary["distribution_deltas"]) == {
        "style_fraction",
        "active_background_fraction",
        "class_chip_fraction",
        "class_active_pixel_fraction",
    }

    eligible = first.chips.loc[first.chips["style_id"].isin([1, 2])].copy()
    eligible["cell"] = list(
        zip(
            eligible["style_id"],
            eligible["row_off"] // config.val_cell_size_px,
            eligible["col_off"] // config.val_cell_size_px,
            strict=True,
        )
    )
    for _, group in eligible.groupby("cell"):
        assert group["split"].eq("val").all() or not group["split"].eq("val").any()

    validation_grid = {
        (row // config.stride, col // config.stride)
        for row, col in first.chips.loc[
            first.chips["split"].eq("val"), ["row_off", "col_off"]
        ].itertuples(index=False, name=None)
    }
    for row, col in first.chips.loc[
        first.chips["split"].eq("train"), ["row_off", "col_off"]
    ].itertuples(index=False, name=None):
        grid = (row // config.stride, col // config.stride)
        assert all(max(abs(grid[0] - val[0]), abs(grid[1] - val[1])) > 2 for val in validation_grid)

    assert 0.15 <= first.summary["achieved_val_fraction"] <= 0.35


def test_validation_contains_whole_macro_cells() -> None:
    config = SplitConfig(
        val_fraction=0.25,
        val_cell_size_px=32,
        chip_size=16,
        stride=8,
        seed=3,
    )
    result = build_splits(chip_grid(), taxonomy(), config, max_jitter_px=0)
    chips = result.chips.loc[result.chips["style_id"].notna()].copy()
    chips["cell_row"] = chips["row_off"] // config.val_cell_size_px
    chips["cell_col"] = chips["col_off"] // config.val_cell_size_px

    for _, group in chips.groupby(["style_id", "cell_row", "cell_col"]):
        assert group["split"].eq("val").all() or not group["split"].eq("val").any()


def test_supported_class_is_represented_in_validation_and_train() -> None:
    rows = []
    for cell in range(10):
        lake = 1 if cell in {0, 9} else 0
        rows.append(
            {
                "chip_id": f"r0_c{cell * 16}",
                "row_off": 0,
                "col_off": cell * 16,
                "size": 8,
                "style_id": 1,
                "px_bg": 64 - lake,
                "px_10": 0,
                "px_16": lake,
            }
        )
    config = SplitConfig(
        val_fraction=0.1,
        val_cell_size_px=16,
        chip_size=8,
        stride=8,
        seed=11,
    )

    result = build_splits(pd.DataFrame(rows), taxonomy(), config, max_jitter_px=0)

    for split in ("train", "val"):
        subset = result.chips.loc[result.chips["split"].eq(split)]
        assert subset["px_16"].sum() > 0
    assert "lake:train" not in result.summary["coverage_failures"]
    assert "lake:val" not in result.summary["coverage_failures"]

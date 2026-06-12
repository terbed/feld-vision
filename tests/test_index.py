import pandas as pd
import pytest

from feldvision.data.index import ChipIndexError, validate_chip_index


def make_index() -> pd.DataFrame:
    data = {
        "chip_id": ["r0_c0", "r0_c128"],
        "row_off": [0, 0],
        "col_off": [0, 128],
        "size": [4, 4],
        "style_id": pd.Series([1, None], dtype="Int64"),
        "label_version": ["version", "version"],
        "px_bg": [10, 16],
    }
    for raw_id in range(10, 18):
        data[f"px_{raw_id}"] = [6 if raw_id == 10 else 0, 0]
    return pd.DataFrame(data)


def test_index_validation_accepts_nullable_styles_and_reports_summary() -> None:
    summary = validate_chip_index(
        make_index(),
        expected_label_version="version",
        known_style_ids={1},
        expected_stride=128,
    )

    assert summary.row_count == 2
    assert summary.null_style_count == 1
    assert summary.label_bearing_count == 1
    assert summary.chips_per_raw_class["10"] == 1
    assert summary.rows_per_style == {"1": 1, "null": 1}


def test_index_validation_rejects_incorrect_pixel_total() -> None:
    chips = make_index()
    chips.loc[0, "px_bg"] = 9

    with pytest.raises(ChipIndexError, match="size\\^2"):
        validate_chip_index(chips)

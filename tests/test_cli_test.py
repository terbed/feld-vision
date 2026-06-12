from shapely.geometry import box

from feldvision.cli.test import _bounded_geometry


def test_bounded_geometry_limits_extent_and_stays_inside_sheet() -> None:
    sheet = box(0, 0, 100, 100)

    bounded = _bounded_geometry(
        sheet,
        pixel_width=2,
        pixel_height=2,
        max_size_px=20,
    )

    assert bounded.bounds == (30, 30, 70, 70)
    assert sheet.covers(bounded)

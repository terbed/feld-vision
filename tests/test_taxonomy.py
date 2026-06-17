from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feldvision.taxonomy import ClassDefinition, Taxonomy, TaxonomyError, load_taxonomy

ROOT = Path(__file__).parents[1]


def test_default_taxonomy_remaps_active_and_ignored_ids() -> None:
    taxonomy = load_taxonomy(ROOT / "configs/taxonomy_default.yaml")
    raw = np.array([[0, 10, 11, 12, 13, 14, 15, 16, 17, 99]], dtype=np.uint8)

    remapped = taxonomy.remap(raw)

    assert taxonomy.num_classes == 7
    assert taxonomy.ignored_raw_ids == frozenset({15, 17})
    assert remapped.tolist() == [[0, 1, 2, 3, 4, 5, 255, 6, 255, 255]]


def test_count_frame_groups_raw_ids_by_active_class() -> None:
    taxonomy = Taxonomy(
        classes=(
            ClassDefinition("background", 0, (0,), (0, 0, 0)),
            ClassDefinition("water", 1, (10, 11), (0, 0, 255)),
        ),
        ignored_raw_ids=frozenset({15}),
    )
    chips = pd.DataFrame({"px_bg": [10, 8], "px_10": [2, 0], "px_11": [1, 4]})

    counts = taxonomy.count_frame(chips)

    assert counts.to_dict("list") == {"background": [10, 8], "water": [3, 4]}
    assert taxonomy.active_pixel_counts(chips).tolist() == [3, 4]


def test_non_contiguous_train_ids_are_rejected() -> None:
    with pytest.raises(TaxonomyError, match="contiguous"):
        Taxonomy(
            classes=(
                ClassDefinition("background", 0, (0,), (0, 0, 0)),
                ClassDefinition("river", 2, (10,), (0, 0, 255)),
            ),
            ignored_raw_ids=frozenset(),
        )

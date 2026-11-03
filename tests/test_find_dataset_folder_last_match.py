"""find_dataset_folder anchors on the last dataset_name component before derivatives."""

import pytest

from champollion_pipeline.utils.lib import find_dataset_folder


def test_single_occurrence_returns_parent():
    assert find_dataset_folder("/data/TEST01/derivatives/cortical_tiles/crops/2mm", "TEST01") == "/data"


def test_repeated_name_before_derivatives_uses_last_occurrence():
    assert find_dataset_folder("/data/UKB/runs/UKB/derivatives/cortical_tiles", "UKB") == "/data/UKB/runs"


def test_name_after_derivatives_is_ignored():
    path = "/data/UKB/derivatives/UKB/crops"
    assert find_dataset_folder(path, "UKB") == "/data"


def test_without_derivatives_uses_last_occurrence():
    assert find_dataset_folder("/a/UKB/b/UKB/c", "UKB") == "/a/UKB/b"


def test_missing_name_raises():
    with pytest.raises(ValueError):
        find_dataset_folder("/data/OTHER/derivatives/x", "UKB")


def test_name_only_after_derivatives_raises():
    with pytest.raises(ValueError):
        find_dataset_folder("/data/derivatives/UKB", "UKB")

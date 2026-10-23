#!/usr/bin/env python3
"""
model_set.json written by HuggingFaceStrategy after a Hugging Face download (MODELVER epic).

After snapshot_download of model version subfolder <V>, HuggingFaceStrategy.fetch writes
<V>/model_set.json (model_version <V>, masks <V>: the Hugging Face sets name their masks
after the model version) unless that directory already holds one, which stays byte-for-byte
unchanged (a marker shipped by the repo, or one written by an earlier fetch of the cache).
Without a subfolder nothing is written. A marker that cannot be written only warns: the
download still returns the model set directory.

No network: huggingface_hub.snapshot_download is replaced by a fake that lays out files.
"""

import os
import stat
import warnings
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_embeddings import (
    MODEL_SET_FILE_NAME,
    HuggingFaceStrategy,
    read_model_set,
)

pytestmark = pytest.mark.unit  # noqa: V107

REPO_ID = "neurospin/Champollion_V1"
VERSION = "canonical_corrected_26_1"
SHIPPED_MARKER = '{"model_version": "shipped_set",\n  "masks": "shipped_masks", "checkpoint_epoch": 12}\n'


def _fake_snapshot_download(shipped_marker: str | None = None):
    """Return a snapshot_download stand-in that creates <local_dir>/<subfolder>/weights.pt
    (and the shipped marker, if given) for the requested allow_patterns, then returns local_dir."""

    def fake(repo_id, local_dir, allow_patterns=None, **_options):
        os.makedirs(local_dir, exist_ok=True)
        for pattern in allow_patterns or ["*"]:
            folder = os.path.join(local_dir, pattern.rstrip("*").rstrip("/"))
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "weights.pt"), "w", encoding="utf-8") as f:
                f.write("weights")
            if shipped_marker is not None:
                with open(os.path.join(folder, MODEL_SET_FILE_NAME), "w", encoding="utf-8") as f:
                    f.write(shipped_marker)
        return local_dir

    return fake


def _fetch(cache_dir, subfolder=VERSION, shipped_marker=None) -> str:
    strategy = HuggingFaceStrategy(subfolder=subfolder)
    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download(shipped_marker)):
        return strategy.fetch(REPO_ID, str(cache_dir))


class TestHfMarkerWritten:
    """REQ-MODELVER-BDRABCZUK-2AAC6E49C8FE: the marker names the downloaded subfolder."""

    def test_marker_written_with_subfolder_as_version_and_masks(self, tmp_path):
        model_set_dir = _fetch(tmp_path)
        assert os.path.isfile(os.path.join(model_set_dir, MODEL_SET_FILE_NAME)), (
            f"no {MODEL_SET_FILE_NAME} written in {model_set_dir}"
        )
        model_set = read_model_set(model_set_dir)
        assert model_set.model_version == VERSION
        assert model_set.masks == VERSION


class TestHfMarkerKept:
    """REQ-MODELVER-BDRABCZUK-1B05640B1F64: an existing marker is left byte-for-byte unchanged."""

    def test_shipped_marker_left_unchanged(self, tmp_path):
        model_set_dir = _fetch(tmp_path, shipped_marker=SHIPPED_MARKER)
        with open(os.path.join(model_set_dir, MODEL_SET_FILE_NAME), encoding="utf-8") as f:
            assert f.read() == SHIPPED_MARKER

    def test_second_fetch_from_cache_keeps_written_marker(self, tmp_path):
        first_dir = _fetch(tmp_path)
        marker = os.path.join(first_dir, MODEL_SET_FILE_NAME)
        assert os.path.isfile(marker), f"first fetch wrote no {MODEL_SET_FILE_NAME} in {first_dir}"
        with open(marker, "rb") as f:
            first_bytes = f.read()
        second_dir = _fetch(tmp_path)
        assert second_dir == first_dir
        with open(os.path.join(second_dir, MODEL_SET_FILE_NAME), "rb") as f:
            assert f.read() == first_bytes


class TestHfMarkerNoSubfolder:
    """REQ-MODELVER-BDRABCZUK-CF747A6F0C54: no subfolder, no marker."""

    def test_no_marker_without_subfolder(self, tmp_path):
        snapshot_root = _fetch(tmp_path, subfolder=None)
        assert not os.path.exists(os.path.join(snapshot_root, MODEL_SET_FILE_NAME))


class TestHfMarkerWriteFailure:
    """REQ-MODELVER-BDRABCZUK-F4A6841ABAD3: an OSError on the marker write only warns."""

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
    def test_write_failure_warns_and_returns_directory(self, tmp_path):
        cache = tmp_path / "cache"
        local_dir = cache / f"{REPO_ID.split('/')[-1]}_{VERSION}"
        model_set_dir = local_dir / VERSION
        model_set_dir.mkdir(parents=True)
        (model_set_dir / "weights.pt").write_text("weights")
        model_set_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        def read_only_snapshot(repo_id, local_dir, allow_patterns=None, **_options):
            return local_dir

        strategy = HuggingFaceStrategy(subfolder=VERSION)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch("huggingface_hub.snapshot_download", side_effect=read_only_snapshot):
                    result = strategy.fetch(REPO_ID, str(cache))
        finally:
            model_set_dir.chmod(stat.S_IRWXU)
        assert result == str(model_set_dir)
        marker_path = str(model_set_dir / MODEL_SET_FILE_NAME)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any(marker_path in message for message in messages), (
            f"no UserWarning naming {marker_path}; got {messages!r}"
        )

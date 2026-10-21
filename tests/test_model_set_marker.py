#!/usr/bin/env python3
"""
model_set.json marker of a local model set directory (MODELVER epic).

A model set directory may carry model_set.json, a JSON object with
- model_version (non-empty string): the model-set name,
- masks (non-empty string): the cortical tiles mask version the models were trained on,
- checkpoint_epoch (optional, non-negative integer): the training epoch of the weights.

read_model_set(<dir>) returns those values (checkpoint_epoch None when absent) and raises
a ValueError naming the file path on an invalid marker. When models_path is a local
directory whose marker masks differs from the resolved --masks, the generate_embeddings
parser exits with status 2 naming both, unless --allow-version-mismatch is given, in
which case it emits a UserWarning naming both (policy of the mismatch-check Task).

No model, no evaluate.py, no network: only the reader and the real GenerateEmbeddings parser.
"""

import json
import warnings

import pytest

from champollion_pipeline.generate_embeddings import (
    MODEL_SET_FILE_NAME,
    GenerateEmbeddings,
    read_model_set,
)

pytestmark = pytest.mark.unit  # noqa: V107

MARKER_MASKS = "canonical_corrected_26_1"
MASKS = "canonical_25"


def _write_marker(directory, content) -> str:
    """Write content (a dict, serialised to JSON, or raw text) as directory/model_set.json; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_SET_FILE_NAME
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return str(path)


def _local_model_set(tmp_path, masks: str = MARKER_MASKS):
    models = tmp_path / "models"
    _write_marker(models, {"model_version": masks, "masks": masks, "checkpoint_epoch": 80})
    return models


def _datasets_root(tmp_path) -> str:
    return str(tmp_path / "TEST01")


class TestReadModelSet:
    """REQ-MODELVER-BDRABCZUK-B1075B1E7493: the reader returns the marker values."""

    def test_returns_all_fields(self, tmp_path):
        model_dir = tmp_path / "models"
        _write_marker(model_dir, {"model_version": "cc26_1", "masks": MARKER_MASKS, "checkpoint_epoch": 80})
        model_set = read_model_set(str(model_dir))
        assert model_set.model_version == "cc26_1"
        assert model_set.masks == MARKER_MASKS
        assert model_set.checkpoint_epoch == 80

    def test_checkpoint_epoch_absent_is_none(self, tmp_path):
        model_dir = tmp_path / "models"
        _write_marker(model_dir, {"model_version": MARKER_MASKS, "masks": MARKER_MASKS})
        model_set = read_model_set(str(model_dir))
        assert model_set.masks == MARKER_MASKS
        assert model_set.checkpoint_epoch is None


class TestReadModelSetInvalid:
    """REQ-MODELVER-BDRABCZUK-A81A073CF4EC and REQ-MODELVER-BDRABCZUK-F289C4CEA620: ValueError naming the path."""

    @pytest.mark.parametrize(
        "content",
        [
            "{not json",
            "[]",
            {"masks": MARKER_MASKS},
            {"model_version": MARKER_MASKS},
            {"model_version": "", "masks": MARKER_MASKS},
            {"model_version": MARKER_MASKS, "masks": 25},
        ],
        ids=["not_json", "not_object", "no_model_version", "no_masks", "empty_model_version", "masks_not_string"],
    )
    def test_invalid_marker_raises_value_error_naming_path(self, tmp_path, content):
        """REQ-MODELVER-BDRABCZUK-A81A073CF4EC."""
        model_dir = tmp_path / "models"
        path = _write_marker(model_dir, content)
        with pytest.raises(ValueError) as exc_info:
            read_model_set(str(model_dir))
        assert path in str(exc_info.value), str(exc_info.value)

    @pytest.mark.parametrize("epoch", [-1, 1.5, "80", True], ids=["negative", "float", "string", "bool"])
    def test_bad_checkpoint_epoch_raises_value_error_naming_path(self, tmp_path, epoch):
        """REQ-MODELVER-BDRABCZUK-F289C4CEA620."""
        model_dir = tmp_path / "models"
        path = _write_marker(
            model_dir, {"model_version": MARKER_MASKS, "masks": MARKER_MASKS, "checkpoint_epoch": epoch}
        )
        with pytest.raises(ValueError) as exc_info:
            read_model_set(str(model_dir))
        assert path in str(exc_info.value), str(exc_info.value)


class TestLocalMarkerMasksMismatch:
    """REQ-MODELVER-BDRABCZUK-953E3185D61B: marker masks <K> != resolved --masks <M> exits 2 naming both."""

    def _expect_usage_error(self, argv, capsys) -> str:
        script = GenerateEmbeddings()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(SystemExit) as exc_info:
                script.parse_args(argv)
        assert exc_info.value.code == 2
        return capsys.readouterr().err

    def test_marker_masks_mismatch_exits_2_naming_both(self, tmp_path, capsys):
        models = _local_model_set(tmp_path)
        stderr = self._expect_usage_error([str(models), _datasets_root(tmp_path), "--masks", MASKS], capsys)
        assert MASKS in stderr, stderr
        assert MARKER_MASKS in stderr, stderr

    def test_marker_masks_mismatch_with_default_masks_exits_2(self, tmp_path, capsys):
        """Neither --masks nor --model-version: the resolved --masks is the canonical_25 default."""
        models = _local_model_set(tmp_path)
        stderr = self._expect_usage_error([str(models), _datasets_root(tmp_path)], capsys)
        assert MASKS in stderr, stderr
        assert MARKER_MASKS in stderr, stderr


class TestLocalMarkerMasksMismatchAllowed:
    """REQ-MODELVER-BDRABCZUK-8FA8D4F8599A: --allow-version-mismatch turns it into a UserWarning naming both."""

    def test_override_warns_naming_both(self, tmp_path):
        models = _local_model_set(tmp_path)
        argv = [str(models), _datasets_root(tmp_path), "--masks", MASKS, "--allow-version-mismatch"]
        script = GenerateEmbeddings()
        with pytest.warns(UserWarning) as record:
            try:
                script.parse_args(argv)
            except SystemExit as exc:
                pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
        messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
        assert any(MASKS in m and MARKER_MASKS in m for m in messages), messages


class TestLocalMarkerMissing:
    """REQ-MODELVER-BDRABCZUK-FF80E381F552: a local directory <D> without model_set.json gives a UserWarning naming
    <D>/model_set.json and the resolved --masks <M>, instead of exiting."""

    def _parse_expect_warning(self, argv) -> list[str]:
        """Parse argv with the real parser; fail on SystemExit; return the UserWarning messages emitted."""
        script = GenerateEmbeddings()
        with pytest.warns(UserWarning) as record:
            try:
                script.parse_args(argv)
            except SystemExit as exc:
                pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
        return [str(w.message) for w in record if issubclass(w.category, UserWarning)]

    def test_missing_marker_warns_naming_path_and_masks(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        marker_path = str(models / MODEL_SET_FILE_NAME)
        messages = self._parse_expect_warning([str(models), _datasets_root(tmp_path), "--masks", MARKER_MASKS])
        assert any(marker_path in m and MARKER_MASKS in m for m in messages), messages

    def test_missing_marker_with_default_masks_warns(self, tmp_path):
        """Neither --masks nor --model-version: the resolved --masks is the canonical_25 default."""
        models = tmp_path / "models"
        models.mkdir()
        marker_path = str(models / MODEL_SET_FILE_NAME)
        messages = self._parse_expect_warning([str(models), _datasets_root(tmp_path)])
        assert any(marker_path in m and MASKS in m for m in messages), messages

#!/usr/bin/env python3
"""
generate_embeddings: --masks default for a local models path (DEF-MODELVER-BDRABCZUK-B69D9AC66796).

With a local models directory and neither --model-version nor --masks, the parser takes --masks
from the last folder name of that directory when it is a Mask version (canonical_25,
canonical_corrected_26_1), so a local .../canonical_corrected_26_1 model set reads
crops/canonical_corrected_26_1/2mm, not crops/canonical_25/2mm. Any other folder name: the
parser exits with status 2 asking for --masks.

No model, no evaluate.py, no network: only the real GenerateEmbeddings parser on a tmp_path
models directory without model_set.json.
"""

import os
from os.path import join

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.embeddings_parse_helpers import masks_of, parse_or_fail

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("ignore::UserWarning")]  # noqa: V107

MASK_VERSIONS = ["canonical_25", "canonical_corrected_26_1"]
UNKNOWN_FOLDER_NAME = "my_models"


def _models_dir(tmp_path, name: str) -> str:
    """A local models directory named <name> without model_set.json."""
    path = tmp_path / "models" / name
    path.mkdir(parents=True)
    return str(path)


class TestLocalFolderNamesMaskVersion:
    """REQ (local folder named after a Mask version <V>, neither version option): masks <V>."""

    @pytest.mark.parametrize("version", MASK_VERSIONS)
    def test_folder_named_mask_version_yields_masks(self, tmp_path, version):
        script = parse_or_fail([_models_dir(tmp_path, version), str(tmp_path / "TEST01")])
        assert masks_of(script) == version, vars(script.args)

    def test_trailing_slash_folder_yields_masks(self, tmp_path):
        models_dir = _models_dir(tmp_path, "canonical_corrected_26_1") + os.sep
        script = parse_or_fail([models_dir, str(tmp_path / "TEST01")])
        assert masks_of(script) == "canonical_corrected_26_1", vars(script.args)

    def test_folder_named_mask_version_selects_crops_dir(self, tmp_path):
        datasets_root = str(tmp_path / "TEST01")
        script = parse_or_fail([_models_dir(tmp_path, "canonical_corrected_26_1"), datasets_root])
        expected = join(
            datasets_root, "derivatives", script.args.cortical_version, "crops", "canonical_corrected_26_1", "2mm"
        )
        assert script._get_crops_2mm_dir(datasets_root) == expected


class TestLocalFolderUnknownName:
    """REQ (local folder not named after a Mask version, neither version option): exit 2 naming --masks."""

    def test_unknown_folder_name_exits_2(self, tmp_path):
        argv = [_models_dir(tmp_path, UNKNOWN_FOLDER_NAME), str(tmp_path / "TEST01")]
        with pytest.raises(SystemExit) as exc_info:
            GenerateEmbeddings().parse_args(argv)
        assert exc_info.value.code == 2

    def test_unknown_folder_name_error_names_masks(self, tmp_path, capsys):
        argv = [_models_dir(tmp_path, UNKNOWN_FOLDER_NAME), str(tmp_path / "TEST01")]
        with pytest.raises(SystemExit):
            GenerateEmbeddings().parse_args(argv)
        assert "--masks" in capsys.readouterr().err

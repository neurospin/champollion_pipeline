#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/upload_models_to_hf.py

The huggingface_hub dependency is imported lazily inside ``run()``, so every
test injects a stub module into ``sys.modules`` instead of contacting the Hub.
"""

import sys
from unittest.mock import MagicMock

import pytest

import upload_models_to_hf
from upload_models_to_hf import UploadModelsToHF

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(argv: list) -> UploadModelsToHF:
    script = UploadModelsToHF()
    script.args = script.parse_args(argv)
    return script


@pytest.fixture
def hf_stub(monkeypatch):
    """Install a stub ``huggingface_hub`` module exposing a mock ``HfApi``."""
    api = MagicMock()
    module = MagicMock()
    module.HfApi = MagicMock(return_value=api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module, api


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_required_arguments(self, tmp_path):
        script = _make_script([str(tmp_path), "neurospin/champollion", "--masks-version", "canonical_25"])
        assert script.args.models_dir == str(tmp_path)
        assert script.args.repo_id == "neurospin/champollion"
        assert script.args.masks_version == "canonical_25"

    def test_token_defaults_to_none(self, tmp_path):
        script = _make_script([str(tmp_path), "repo", "--masks-version", "v1"])
        assert script.args.token is None

    def test_private_defaults_to_false(self, tmp_path):
        script = _make_script([str(tmp_path), "repo", "--masks-version", "v1"])
        assert script.args.private is False

    def test_optional_arguments_are_read(self, tmp_path):
        script = _make_script(
            [str(tmp_path), "repo", "--masks-version", "v1", "--token", "hf_xxx", "--private"],
        )
        assert script.args.token == "hf_xxx"
        assert script.args.private is True

    def test_masks_version_is_required(self, tmp_path):
        script = UploadModelsToHF()
        with pytest.raises(SystemExit):
            script.parse_args([str(tmp_path), "repo"])


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_returns_one_when_huggingface_hub_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        script = _make_script([str(tmp_path), "repo", "--masks-version", "v1"])
        assert script.run() == 1
        assert "huggingface_hub is required" in capsys.readouterr().out

    def test_returns_one_when_models_dir_missing(self, tmp_path, hf_stub, capsys):
        missing = tmp_path / "nope"
        script = _make_script([str(missing), "repo", "--masks-version", "v1"])
        assert script.run() == 1
        assert "models_dir not found" in capsys.readouterr().out

    def test_creates_repo_and_uploads_folder(self, tmp_path, hf_stub):
        module, api = hf_stub
        script = _make_script([str(tmp_path), "neurospin/champollion", "--masks-version", "canonical_25"])

        assert script.run() == 0

        module.HfApi.assert_called_once_with(token=None)
        api.create_repo.assert_called_once_with(
            repo_id="neurospin/champollion",
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        api.upload_folder.assert_called_once_with(
            repo_id="neurospin/champollion",
            folder_path=str(tmp_path.resolve()),
            path_in_repo="canonical_25",
            repo_type="model",
        )

    def test_token_and_private_are_forwarded(self, tmp_path, hf_stub):
        module, api = hf_stub
        script = _make_script([str(tmp_path), "repo", "--masks-version", "v1", "--token", "hf_abc", "--private"])

        assert script.run() == 0

        module.HfApi.assert_called_once_with(token="hf_abc")
        assert api.create_repo.call_args.kwargs["private"] is True

    def test_prints_browse_url(self, tmp_path, hf_stub, capsys):
        script = _make_script([str(tmp_path), "neurospin/champollion", "--masks-version", "canonical_25"])
        script.run()
        out = capsys.readouterr().out
        assert "https://huggingface.co/neurospin/champollion/tree/main/canonical_25" in out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_builds_and_runs(self, tmp_path, hf_stub, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["upload_models_to_hf.py", str(tmp_path), "repo", "--masks-version", "v1"],
        )
        monkeypatch.setattr(
            "champollion_utils.script_builder.check_for_updates",
            lambda *a, **k: None,
            raising=False,
        )
        assert upload_models_to_hf.main() == 0

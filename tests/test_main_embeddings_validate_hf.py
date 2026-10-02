#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that main.py's embeddings stage validates HuggingFace mode on
dataset.hf_repo_id rather than on a local config.models_path
(REQ-HFVALID-01, REQ-HFVALID-02), keeps the models_path check outside HF
mode (REQ-HFVALID-03), and lets the orchestrator reach execute() in HF mode
(REQ-HFVALID-04).

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in
test_main_stage_argv.py. No network: execute() is never run for real.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
LOGGER_NAME = "test_main_embeddings_validate_hf"
REPO_ID = "neurospin/champollion"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_validate_hf_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _stage(main_module, config):
    return main_module.GenerateEmbeddingsStage("generate_embeddings", config, logging.getLogger(LOGGER_NAME))


class TestHfEnabledValidate:
    """REQ-HFVALID-01/02: in HF mode, validate() checks hf_repo_id, not models_path."""

    def test_validate_passes_with_repo_id_and_missing_models_path(self, main_module, tmp_path):
        """REQ-HFVALID-01: hf_enabled + non-empty repo id + nonexistent models_path -> True."""
        missing = tmp_path / "no_such_models_dir"
        config = main_module.PipelineConfig(models_path=str(missing))
        config.dataset.hf_enabled = True  # noqa: V101
        config.dataset.hf_repo_id = REPO_ID  # noqa: V101

        assert _stage(main_module, config).validate() is True, (
            f"REQ-HFVALID-01: validate() rejected HF mode (repo {REPO_ID!r}) as models_path {missing} is absent"
        )

    @pytest.mark.parametrize("empty_repo_id", [None, ""])
    def test_validate_fails_without_repo_id(self, main_module, tmp_path, caplog, empty_repo_id):
        """REQ-HFVALID-02: hf_enabled + empty repo id -> False with an error naming hf_repo_id."""
        config = main_module.PipelineConfig(models_path=str(tmp_path))  # exists: isolate the repo-id check
        config.dataset.hf_enabled = True  # noqa: V101
        config.dataset.hf_repo_id = empty_repo_id  # noqa: V101

        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            ok = _stage(main_module, config).validate()

        assert ok is False, f"REQ-HFVALID-02: validate() accepted HF mode with hf_repo_id={empty_repo_id!r}"
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("hf_repo_id" in m for m in errors), (
            f"REQ-HFVALID-02: no error log mentioning 'hf_repo_id'; error logs were {errors!r}"
        )


class TestHfDisabledValidate:
    """REQ-HFVALID-03: outside HF mode, a missing models_path still fails validation."""

    def test_validate_fails_with_missing_models_path(self, main_module, tmp_path):
        """REQ-HFVALID-03: hf_disabled + nonexistent models_path -> False."""
        config = main_module.PipelineConfig(models_path=str(tmp_path / "no_such_models_dir"))
        config.dataset.hf_enabled = False  # noqa: V101
        config.dataset.hf_repo_id = REPO_ID  # noqa: V101  # a stray repo id must not bypass the local check

        assert _stage(main_module, config).validate() is False


class TestOrchestratorHfFlow:
    """REQ-HFVALID-04: run() reaches GenerateEmbeddingsStage.execute() in HF mode."""

    def test_run_reaches_execute_in_hf_mode(self, main_module, tmp_path):
        """REQ-HFVALID-04: only generate_embeddings enabled, HF mode, missing models_path -> execute() called."""
        config = main_module.PipelineConfig(
            models_path=str(tmp_path / "no_such_models_dir"),
            outputs_path=str(tmp_path / "outputs"),
            log_to_file=False,
            log_to_console=False,
        )
        config.stages = {name: name == "generate_embeddings" for name in config.stages}  # noqa: V101
        config.dataset.hf_enabled = True  # noqa: V101
        config.dataset.hf_repo_id = REPO_ID  # noqa: V101

        ok = main_module.StageResult(stage_name="generate_embeddings", success=True, message="stubbed")
        with patch.object(main_module.GenerateEmbeddingsStage, "execute", return_value=ok) as execute:
            main_module.PipelineOrchestrator(config).run()

        assert execute.call_count == 1, (
            "REQ-HFVALID-04: PipelineOrchestrator.run() never called GenerateEmbeddingsStage.execute() in HF mode"
        )

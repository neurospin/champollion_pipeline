"""Stage 4 model source fallback cascade (TASK-MODELSRC-BDRABCZUK-EFFEE1006BFE).

User decision 2026-11-20: a run with no model source option finds or fetches its models, from local to Hugging Face
as the last try. User decision 2026-11-23: an explicit --models-path (or YAML models_path) that does not exist is an
error with no download; only the default <checkout>/models falls through.

An empty PipelineConfig.models_path means "not set" (no --models-path, no YAML models_path). <root_path> is
PipelineConfig.root_path (the checkout); <models_cache> is dataset.models_cache, else
<datasets_root>/derivatives/champollion_V1/models_cache; <masks> is dataset.masks_version.

Requirements pinned here:

* REQ-MODELSRC-BDRABCZUK-12EB03CDC1DD (stakeholder) - a run without a model source option obtains its model set.
* REQ-MODELSRC-BDRABCZUK-74BDF84B5789 (system) - GenerateEmbeddingsStage passes the cascade's source.
* REQ-MODELSRC-BDRABCZUK-867178EFF7A0 (system) - validate() accepts a missing default models folder.
* REQ-MODELSRC-BDRABCZUK-2F35B98A4C14 (system) - a failed Hugging Face fallback logs one ERROR naming --models-path
  and the models cache folder.
* REQ-MODELSRC-BDRABCZUK-10AE61B5046B (system) - the replay file of such a run keeps models_path empty.
* REQ-MODELSRC-BDRABCZUK-9B3315506639 (system) - docs/usage.md Model sources lists the cascade order.
* REQ-HFVALID-BDRABCZUK-F8B6064B543E (system, narrowed) - a missing explicit models_path fails validate().
* REQ-MODELSRC-BDRABCZUK-52458C8BA60B (system) - with hf_enabled false the config token is not exported as HF_TOKEN
  (user decision 2026-11-23: the fallback downloads the public repository without the config token).
* REQ-STREAMING-BDRABCZUK-775E5D8B97B9 (system) - streaming passes the stage 4 models source.

The component Requirements of resolve_models_source are tested in tests/test_models_source_resolver.py.

No network: the Hugging Face hub module seen by champollion_pipeline.models.fetch is a fake that fails the test on
any call. Every folder lives under tmp_path.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from champollion_pipeline import cli
from champollion_pipeline.config import PipelineConfig
from champollion_pipeline.pipeline_stages import GenerateEmbeddingsStage
from champollion_pipeline.replay_config import write_replay_config
from champollion_pipeline.streaming import runner as streaming_runner
from tests.test_models_source_resolver import (
    HF_REPO,
    MASKS,
    make_cache_entry,
    make_checkout,
    make_default_models,
    refuse_hub,
)

pytestmark = pytest.mark.unit  # noqa: V107

STAGE_LOGGER = "champollion_pipeline.test_models_source_cascade"
USAGE = Path(__file__).resolve().parents[1] / "docs" / "usage.md"


@pytest.fixture(autouse=True)  # noqa: V103
def _no_network(monkeypatch):
    """No Hugging Face hub call reaches the network in this module."""
    refuse_hub(monkeypatch)


def _cli_config(tmp_path: Path, *argv: str) -> PipelineConfig:
    """The PipelineConfig main.py builds for argv, its checkout moved to tmp_path/checkout, its cache under tmp_path."""
    args = cli.ChampollionPipeline().parse_args(list(argv))
    config = cli._load_config(args)
    cli._apply_cli_overrides(config, args, from_default=not args.config)
    config.root_path = str(make_checkout(tmp_path))  # noqa: V101
    config.outputs_path = str(tmp_path / "outputs")  # noqa: V101
    dataset = config.dataset
    dataset.datasets_root = str(tmp_path / "data" / "DS")  # noqa: V101
    dataset.models_cache = str(tmp_path / "models_cache")  # noqa: V101
    dataset.masks_version = MASKS  # noqa: V101
    dataset.hf_enabled = False  # noqa: V101
    return config


def _stage(config: PipelineConfig) -> GenerateEmbeddingsStage:
    return GenerateEmbeddingsStage("generate_embeddings", config, logging.getLogger(STAGE_LOGGER))


def _models_source(config: PipelineConfig) -> str:
    """The models source GenerateEmbeddingsStage passes generate_embeddings: its first argument."""
    return _stage(config).build_args()[0]


class _FailedScript:
    """A stage 4 script whose run exits with status 1, as generate_embeddings does when no model set is reachable."""

    def run(self) -> int:
        return 1


class TestNominalRun:
    """REQ-MODELSRC-BDRABCZUK-12EB03CDC1DD (stakeholder)."""

    def test_run_without_model_option_obtains_models(self, tmp_path):
        """No --models-path, no --config, nothing local: stage 4 validates and names a source, without a prompt."""
        config = _cli_config(tmp_path)

        with patch("builtins.input", side_effect=AssertionError("user input asked")):
            stage = _stage(config)
            valid = stage.validate()
            source = stage.build_args()[0]

        assert valid is True, "stage 4 validation failed on a run with no model source option"
        assert source == HF_REPO, f"stage 4 models source {source!r}; expected the Hugging Face fallback {HF_REPO!r}"


class TestStageCascade:
    """REQ-MODELSRC-BDRABCZUK-74BDF84B5789: the order of the cascade, through GenerateEmbeddingsStage.build_args."""

    def test_default_models_folder_first(self, tmp_path):
        config = _cli_config(tmp_path)
        default = make_default_models(tmp_path)
        make_cache_entry(Path(config.dataset.models_cache))

        assert _models_source(config) == str(default)

    def test_models_cache_entry_second(self, tmp_path):
        config = _cli_config(tmp_path)
        entry = make_cache_entry(Path(config.dataset.models_cache))

        assert _models_source(config) == str(entry)

    def test_models_cache_entry_under_default_cache(self, tmp_path):
        """dataset.models_cache unset: the entry is looked up under <datasets_root>/derivatives/champollion_V1."""
        config = _cli_config(tmp_path)
        config.dataset.models_cache = None  # noqa: V101
        default_cache = Path(config.dataset.datasets_root) / "derivatives" / "champollion_V1" / "models_cache"
        entry = make_cache_entry(default_cache)

        assert _models_source(config) == str(entry)

    def test_hugging_face_repository_last(self, tmp_path):
        config = _cli_config(tmp_path)

        assert _models_source(config) == HF_REPO


class TestStageValidate:
    """REQ-MODELSRC-BDRABCZUK-867178EFF7A0 and the narrowed REQ-HFVALID-BDRABCZUK-F8B6064B543E."""

    def test_validate_passes_without_default_models_folder(self, tmp_path):
        config = _cli_config(tmp_path)

        assert _stage(config).validate() is True, "validate() failed on a missing default <root_path>/models"

    def test_validate_fails_on_missing_explicit_models_path(self, tmp_path):
        """An explicit --models-path that does not exist stays an error, with no fallback (guard)."""
        typo = tmp_path / "no_such_models"
        config = _cli_config(tmp_path, "--models-path", str(typo))
        make_default_models(tmp_path)
        make_cache_entry(Path(config.dataset.models_cache))

        assert _stage(config).validate() is False, "validate() accepted a missing explicit --models-path"
        assert _models_source(config) == str(typo), "a missing explicit --models-path was replaced by a fallback"


class TestOfflineMessage:
    """REQ-MODELSRC-BDRABCZUK-2F35B98A4C14."""

    def test_failed_fallback_error_names_option_and_cache(self, tmp_path, pipeline_caplog):
        config = _cli_config(tmp_path)
        stage = _stage(config)
        assert stage.build_args()[0] == HF_REPO, "precondition: the Hugging Face fallback is the models source"

        status = stage.run_script(_FailedScript())

        assert status != 0
        cache = config.dataset.models_cache
        errors = [r.getMessage() for r in pipeline_caplog.records if r.levelno >= logging.ERROR]
        naming = [m for m in errors if "--models-path" in m and cache in m]
        assert len(naming) == 1, f"expected one ERROR naming --models-path and {cache}; ERROR records: {errors!r}"


class _TokenProbe:
    """A stage 4 script that records the HF_TOKEN environment variable it sees while it runs."""

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    def run(self) -> int:
        self.seen.append(os.environ.get("HF_TOKEN"))
        return 0


class TestFallbackToken:
    """REQ-MODELSRC-BDRABCZUK-52458C8BA60B (guard: pins the existing run_script behaviour, user decision 2026-11-23)."""

    @pytest.mark.parametrize("ambient", [None, "hf_from_environment"])
    def test_config_token_not_exported_without_hf_enabled(self, tmp_path, monkeypatch, ambient):
        """hf_enabled false + a config token: GenerateEmbeddings sees the ambient HF_TOKEN, unset or not."""
        if ambient is None:
            monkeypatch.delenv("HF_TOKEN", raising=False)
        else:
            monkeypatch.setenv("HF_TOKEN", ambient)
        config = _cli_config(tmp_path)
        config.dataset.hf_token = "hf_from_config"  # noqa: V101
        stage = _stage(config)
        assert stage.build_args()[0] == HF_REPO, "precondition: the Hugging Face fallback is the models source"
        probe = _TokenProbe()

        status = stage.run_script(probe)

        assert status == 0
        assert probe.seen == [ambient], f"HF_TOKEN seen by the stage 4 script {probe.seen!r}; expected [{ambient!r}]"
        assert os.environ.get("HF_TOKEN") == ambient, "HF_TOKEN changed after the stage 4 script ran"


class TestReplay:
    """REQ-MODELSRC-BDRABCZUK-10AE61B5046B."""

    def test_replay_of_default_run_keeps_models_path_empty(self, tmp_path):
        config = _cli_config(tmp_path)
        config.log_dir = str(tmp_path / "logs")  # noqa: V101

        replay = yaml.safe_load(Path(write_replay_config(config)).read_text(encoding="utf-8"))

        assert replay["models_path"] == "", f"replay models_path {replay['models_path']!r}; expected empty"


class TestStreaming:
    """REQ-STREAMING-BDRABCZUK-775E5D8B97B9: streaming passes the stage 4 models source."""

    def test_streaming_passes_stage_models_source(self, tmp_path, monkeypatch):
        config = _cli_config(tmp_path)
        entry = make_cache_entry(Path(config.dataset.models_cache))
        config.mode = "streaming"  # noqa: V101
        config.dataset.embeddings_only = True  # noqa: V101
        config.dataset.morphologist_graphs = str(tmp_path / "graphs")  # noqa: V101
        calls: list[str] = []

        def fake_run_parallel_pipeline(_settings, _dataset, models_path: str) -> int:
            calls.append(models_path)
            return 0

        monkeypatch.setattr(streaming_runner, "run_parallel_pipeline", fake_run_parallel_pipeline)
        args = argparse.Namespace(dry_run=True, config=None, datasets_root=None)

        cli._run_streaming_mode(config, args, run_log_path=None)

        assert calls == [_models_source(config)], f"streaming models_path {calls!r} differs from stage 4's"
        assert calls == [str(entry)], f"streaming models_path {calls!r}; expected the models cache entry {entry}"


class TestUsageDoc:
    """REQ-MODELSRC-BDRABCZUK-9B3315506639."""

    def test_model_sources_lists_cascade_order(self):
        text = USAGE.read_text(encoding="utf-8")
        match = re.search(r"^###\s+Model sources\s*$(.*?)(?=^##)", text, re.MULTILINE | re.DOTALL)
        assert match, "no '### Model sources' subsection in docs/usage.md"
        steps = (r"--models-path", r"<(?:root_path|checkout)>/models", r"models_cache", re.escape(HF_REPO))
        for paragraph in re.split(r"\n\s*\n", match.group(1)):
            if "champollion-pipeline" not in paragraph:
                continue
            found = [re.search(step, paragraph) for step in steps]
            if all(found) and [m.start() for m in found] == sorted(m.start() for m in found):
                return
        pytest.fail(
            "no paragraph of the Model sources subsection names champollion-pipeline and lists, in order, "
            "--models-path, <root_path>/models, the models_cache folder and " + HF_REPO
        )

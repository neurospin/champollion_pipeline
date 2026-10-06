#!/usr/bin/env python3
"""
--use_last_checkpoint opt-out of TASK-176's best-weights default, and removal of
the dead dataset.use_best_model config field (REQ-LASTCKPT-01..14, TASK-209).

The generate_embeddings tests reuse test_generate_embeddings_best_weights.py's
model-directory builders and FakeEvaluate: _run_per_region runs with
execute_command replaced by a fake that inspects the ``-m`` directory the way
evaluate.py would. No real evaluate.py runs.

The main.py tests reuse test_main_stage_argv.py's argv capture: the stage runs
with GenerateEmbeddings mocked, and its argv is parsed with the real parser.
"""

import dataclasses
import logging
import re
from pathlib import Path

import pytest
import yaml

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.test_generate_embeddings_best_weights import (
    BEST_STATE,
    NATIVE_STATE,
    REGION,
    FakeEvaluate,
    assert_evaluate_sees_only,
    lines_naming,
    make_model_dir,
    snapshot_tree,
    write_native_ckpt,
    write_weights,
)
from tests.test_main_stage_argv import _base_config, _embeddings_namespace, _load_main_module

pytestmark = pytest.mark.unit  # noqa: V107

FLAG = "--use_last_checkpoint"
README = Path(__file__).resolve().parent.parent / "README.md"
_ABSENT = "<no use_last_checkpoint attribute>"


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def parse_embeddings_args(argv: list[str]):
    """Parse argv with the real GenerateEmbeddings parser; fail (not error) on rejection."""
    try:
        return GenerateEmbeddings().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")


def run_region_with_flag(tmp_path: Path) -> FakeEvaluate:
    """Run _run_per_region over tmp_path/models with --use_last_checkpoint and the fake evaluate."""
    models = tmp_path / "models"
    script = GenerateEmbeddings()
    try:
        script.parse_args([str(models), str(tmp_path / "dataset"), FLAG])
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {FLAG} (SystemExit {exc.code})")
    fake = FakeEvaluate()
    script.execute_command = fake
    code = script._run_per_region("evaluate.py", str(tmp_path / "crops"), str(tmp_path / "out"))
    assert code == 0
    assert len(fake.calls) == 1
    return fake


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-01 / 02: dead use_best_model field removed
# --------------------------------------------------------------------------- #


class TestDatasetConfigHasNoUseBestModel:
    """REQ-LASTCKPT-01: main.py DatasetConfig declares no use_best_model field."""

    def test_dataset_config_declares_no_use_best_model_field(self, main_module):
        names = {f.name for f in dataclasses.fields(main_module.DatasetConfig)}
        assert "use_best_model" not in names, "DatasetConfig still declares the dead use_best_model field"


class TestSavedConfigHasNoUseBestModel:
    """REQ-LASTCKPT-02: save_to_yaml writes no dataset.use_best_model key."""

    def test_saved_dataset_mapping_has_no_use_best_model_key(self, main_module, tmp_path):
        output = tmp_path / "saved.yaml"
        main_module.ConfigLoader.save_to_yaml(main_module.PipelineConfig(), str(output))
        with open(output, "r") as f:
            written = yaml.safe_load(f)
        assert "use_best_model" not in written["dataset"], "save_to_yaml still writes dataset.use_best_model"


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-03: the flag
# --------------------------------------------------------------------------- #


class TestUseLastCheckpointFlag:
    """REQ-LASTCKPT-03: --use_last_checkpoint parses true when given, false when omitted."""

    def test_flag_given_parses_true(self):
        namespace = parse_embeddings_args(["/models", "/data/MYDATASET", FLAG])
        assert vars(namespace).get("use_last_checkpoint", _ABSENT) is True

    def test_flag_omitted_parses_false(self):
        namespace = parse_embeddings_args(["/models", "/data/MYDATASET"])
        assert vars(namespace).get("use_last_checkpoint", _ABSENT) is False


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-04..06: flag behaviour with best weights beside a native ckpt
# --------------------------------------------------------------------------- #


class TestLastCheckpointPassesRegionDir:
    """REQ-LASTCKPT-04: with the flag, -m is the region model dir itself (no best-weights mirror)."""

    def test_region_dir_is_model_arg_with_last_epoch_native_ckpt(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        fake = run_region_with_flag(tmp_path)

        assert fake.calls[0]["model_arg"] == str(model_dir)

    def test_region_dir_is_model_arg_with_wrapped_best_weights(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE, wrapped=True)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        fake = run_region_with_flag(tmp_path)

        assert fake.calls[0]["model_arg"] == str(model_dir)


class TestLastCheckpointLeavesModelDirUnchanged:
    """REQ-LASTCKPT-05: with the flag, no file in the region model dir is added, removed or rewritten."""

    def test_model_dir_unchanged(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
        before = snapshot_tree(model_dir)

        run_region_with_flag(tmp_path)

        assert snapshot_tree(model_dir) == before


class TestLastCheckpointLogsNativeCkpt:
    """REQ-LASTCKPT-06: with the flag, one console line names the region and the native .ckpt."""

    def test_native_ckpt_named(self, tmp_path, capsys):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        native = write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        run_region_with_flag(tmp_path)

        out = capsys.readouterr().out
        assert len(lines_naming(out, REGION, str(native))) == 1, out


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-07 / 08: main.py wiring
# --------------------------------------------------------------------------- #


class TestStageForwardsUseLastCheckpoint:
    """REQ-LASTCKPT-07: dataset.use_last_checkpoint true -> argv parses with use_last_checkpoint true."""

    def test_true_config_argv_parses_with_use_last_checkpoint_true(self, main_module):
        config = _base_config(main_module)
        config.dataset.use_last_checkpoint = True  # noqa: V101
        namespace = _embeddings_namespace(main_module, config)
        assert vars(namespace).get("use_last_checkpoint", _ABSENT) is True


class TestStageDefaultUseLastCheckpoint:
    """REQ-LASTCKPT-08: default DatasetConfig -> argv parses with use_last_checkpoint false."""

    def test_default_config_argv_parses_with_use_last_checkpoint_false(self, main_module):
        config = _base_config(main_module)
        namespace = _embeddings_namespace(main_module, config)
        assert vars(namespace).get("use_last_checkpoint", _ABSENT) is False


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-09: README
# --------------------------------------------------------------------------- #


def _section_5_option_flags() -> set[str]:
    """First-cell backticked flags of the table rows in README section 5 (Generate Embeddings)."""
    text = README.read_text(encoding="utf-8")
    match = re.search(r"^## 5\. Generate Embeddings$(.*?)^## ", text, flags=re.MULTILINE | re.DOTALL)
    assert match, "sanity check: README has no '## 5. Generate Embeddings' section"
    return set(re.findall(r"^\|\s*`(--[A-Za-z][A-Za-z0-9_-]*)`\s*\|", match.group(1), flags=re.MULTILINE))


class TestReadmeDocumentsUseLastCheckpoint:
    """REQ-LASTCKPT-09: README section 5's options table has a --use_last_checkpoint row."""

    def test_section_5_options_table_has_use_last_checkpoint_row(self):
        flags = _section_5_option_flags()
        assert "--overwrite" in flags, "sanity check: section 5 options table not found"
        assert FLAG in flags, f"README section 5 options table lists {sorted(flags)}, not {FLAG}"


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-10 / 13 / 14: loading a YAML that still carries use_best_model
# --------------------------------------------------------------------------- #

# A dataset mapping with only live DatasetConfig fields.
_LIVE_DATASET = {"input_path": "/raw", "datasets_root": "/data/MYDATASET"}


def _write_yaml(tmp_path: Path, name: str, dataset: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"models_path": "/models", "dataset": dataset}))
    return str(path)


class TestLoadIgnoresRemovedUseBestModel:
    """REQ-LASTCKPT-10: a dataset.use_best_model key loads to the same PipelineConfig as without it."""

    @pytest.mark.parametrize("value", [False, True])
    def test_use_best_model_key_loads_equal_to_config_without_it(self, main_module, tmp_path, value):
        without = _write_yaml(tmp_path, "without.yaml", dict(_LIVE_DATASET))
        with_key = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_best_model": value})

        expected = main_module.ConfigLoader.load_from_yaml(without)
        try:
            loaded = main_module.ConfigLoader.load_from_yaml(with_key)
        except TypeError as exc:
            pytest.fail(f"load_from_yaml rejected dataset.use_best_model: {value}: {exc}")

        assert loaded == expected


class TestLoadWarnsOnRemovedUseBestModel:
    """REQ-LASTCKPT-13: loading dataset.use_best_model logs one WARNING record naming dataset.use_best_model."""

    def test_one_warning_names_dataset_use_best_model(self, main_module, tmp_path, caplog):
        path = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_best_model": False})
        # Also listen on the pipeline logger directly, in case another test turned propagation off.
        pipeline_logger = logging.getLogger("champollion_pipeline")
        pipeline_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING), caplog.at_level(logging.WARNING, logger="champollion_pipeline"):
                try:
                    main_module.ConfigLoader.load_from_yaml(path)
                except TypeError as exc:
                    pytest.fail(f"load_from_yaml rejected dataset.use_best_model: {exc}")
        finally:
            pipeline_logger.removeHandler(caplog.handler)

        unique = {id(r): r for r in caplog.records}.values()
        naming = [r for r in unique if r.levelno == logging.WARNING and "dataset.use_best_model" in r.getMessage()]
        assert len(naming) == 1, [(r.levelname, r.getMessage()) for r in unique]


class TestLoadRejectsOtherUnknownDatasetKey:
    """REQ-LASTCKPT-14: any other unknown dataset key still raises TypeError (characterization)."""

    def test_unknown_dataset_key_raises_type_error(self, main_module, tmp_path):
        path = _write_yaml(tmp_path, "typo.yaml", {**_LIVE_DATASET, "use_last_chekpoint": True})
        with pytest.raises(TypeError):
            main_module.ConfigLoader.load_from_yaml(path)


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-11 / 12: flag with best weights and no native ckpt in version_0
# --------------------------------------------------------------------------- #


class TestLastCheckpointFallsBackToBestWeights:
    """REQ-LASTCKPT-11: with the flag and no native ckpt in version_0, -m is the region dir holding the best weights."""

    def test_region_dir_is_model_arg_holding_converted_best_weights(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)

        fake = run_region_with_flag(tmp_path)

        assert fake.calls[0]["model_arg"] == str(model_dir)
        assert_evaluate_sees_only(fake.calls[0], BEST_STATE)


class TestLastCheckpointIgnoredLogged:
    """REQ-LASTCKPT-12: that fallback prints one weights line naming the .pt and "--use_last_checkpoint ignored"."""

    def test_weights_line_names_pt_and_says_flag_ignored(self, tmp_path, capsys):
        model_dir = make_model_dir(tmp_path)
        pt = write_weights(model_dir, BEST_STATE)

        run_region_with_flag(tmp_path)

        out = capsys.readouterr().out
        weights_lines = [line for line in out.splitlines() if f"[Region {REGION}] weights:" in line]
        assert len(weights_lines) == 1, out
        assert lines_naming(out, REGION, str(pt)) == weights_lines, out
        assert "--use_last_checkpoint ignored" in weights_lines[0], out

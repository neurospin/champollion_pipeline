#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that main.py's orchestrator stages hand their target script an argument
list that script's own parser accepts (REQ-STAGEARGV-01 .. REQ-STAGEARGV-07), and that
main.py itself resolves those script classes when loaded (REQ-STAGEARGV-08).

Each stage builds an argv list and calls ``<Script>().parse_args(argv)``. Here
the script class referenced by main.py is replaced by a MagicMock to capture
that argv without running anything, then the argv is parsed with the *real*
script class's parser. An argparse rejection surfaces as ``SystemExit``.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in
test_main_morphologist_stage.py.
"""

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_stage_argv_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _base_config(main_module):
    """PipelineConfig with every path the six stages read set to a distinct value."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.input_path = "/raw"  # noqa: V101
    dataset.morphologist_graphs = "/graphs"  # noqa: V101
    dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    dataset.crops_path = "/crops"  # noqa: V101
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.snapshots_path = "/snapshots"  # noqa: V101
    return config


def _captured_argv(main_module, stage_class_name, script_attr, config):
    """Run ``main_module.<stage_class_name>.execute()`` with ``script_attr`` mocked; return its argv."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, script_attr, script_cls, create=True):
        stage_cls = getattr(main_module, stage_class_name)
        stage = stage_cls(stage_class_name, config, logging.getLogger("test_main_stage_argv"))
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return list(script_cls.return_value.parse_args.call_args[0][0])


def _parse_with_real_parser(script_cls, argv):
    """Parse ``argv`` with ``script_cls``'s real parser; fail (not error) on an argparse rejection."""
    try:
        return script_cls().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"{script_cls.__name__} parser rejected the stage argv {argv!r} (SystemExit {exc.code})")


def _embeddings_namespace(main_module, config):
    argv = _captured_argv(main_module, "GenerateEmbeddingsStage", "GenerateEmbeddings", config)
    return _parse_with_real_parser(GenerateEmbeddings, argv)


@pytest.mark.unit
class TestGenerateEmbeddingsStageArgv:
    """GenerateEmbeddingsStage argv vs GenerateEmbeddings' real parser."""

    def test_local_models_argv_parses_with_models_and_datasets_root(self, main_module):
        """REQ-STAGEARGV-01: hf disabled -> parses; models_path/datasets_root carry the config values."""
        config = _base_config(main_module)
        config.dataset.hf_enabled = False  # noqa: V101
        namespace = _embeddings_namespace(main_module, config)
        assert namespace.models_path == "/models"
        assert namespace.datasets_root == "/data/MYDATASET"

    def test_hf_enabled_argv_parses_with_repo_id_as_models_path(self, main_module):
        """REQ-STAGEARGV-02: hf enabled -> parses; models_path is config.dataset.hf_repo_id."""
        config = _base_config(main_module)
        config.dataset.hf_enabled = True  # noqa: V101
        config.dataset.hf_repo_id = "neurospin/champollion-models"  # noqa: V101
        namespace = _embeddings_namespace(main_module, config)
        assert namespace.models_path == "neurospin/champollion-models"

    def test_embeddings_path_argv_parses_as_output(self, main_module):
        """REQ-STAGEARGV-03: non-empty embeddings_path -> parses; output is config.dataset.embeddings_path."""
        config = _base_config(main_module)
        config.dataset.embeddings_path = "/data/MYDATASETembeddings"  # noqa: V101
        namespace = _embeddings_namespace(main_module, config)
        assert namespace.output == "/data/MYDATASETembeddings"

    def test_boolean_toggles_argv_parses_with_overwrite_and_cpu(self, main_module):
        """REQ-STAGEARGV-04: verbose/overwrite/cpu/embeddings_only/use_best_model true -> parses; overwrite, cpu set."""
        config = _base_config(main_module)
        config.verbose = True  # noqa: V101
        config.dataset.overwrite = True
        config.dataset.cpu = True
        config.dataset.embeddings_only = True  # noqa: V101
        config.dataset.use_best_model = True  # noqa: V101
        namespace = _embeddings_namespace(main_module, config)
        assert namespace.overwrite is True
        assert namespace.cpu is True

    def test_optional_fields_argv_parses(self, main_module):
        """REQ-STAGEARGV-05: hf_enabled with hf_token/splits_basedir/idx_region_evaluation/config_path set -> parses."""
        config = _base_config(main_module)
        config.dataset.hf_enabled = True  # noqa: V101
        config.dataset.hf_repo_id = "neurospin/champollion-models"  # noqa: V101
        config.dataset.hf_token = "hf_dummy_token"  # noqa: V101
        config.dataset.splits_basedir = "/splits"  # noqa: V101
        config.dataset.idx_region_evaluation = 3  # noqa: V101
        config.dataset.config_path = "/configs"  # noqa: V101
        _embeddings_namespace(main_module, config)


@pytest.mark.unit
class TestEmbeddingsRegionsArgv:
    """REQ-EMBREGIONS-01: dataset.regions (cortical_tiles names) reach GenerateEmbeddings as model names."""

    def test_regions_forwarded_as_model_names_both_hemispheres(self, main_module):
        """REQ-EMBREGIONS-01: each region -> dots removed + _left and + _right, exactly, in --regions."""
        config = _base_config(main_module)
        config.dataset.regions = ["S.C.-sylv.", "F.I.P.-F.I.P.Po.C.inf.", "Lobule_parietal_sup."]  # noqa: V101
        namespace = _embeddings_namespace(main_module, config)
        assert namespace.regions is not None, "GenerateEmbeddingsStage passed no --regions"
        assert sorted(namespace.regions) == sorted(
            [
                "SC-sylv_left",
                "SC-sylv_right",
                "FIP-FIPPoCinf_left",
                "FIP-FIPPoCinf_right",
                "Lobule_parietal_sup_left",
                "Lobule_parietal_sup_right",
            ]
        )


@pytest.mark.unit
class TestPutTogetherEmbeddingsStageArgv:
    """PutTogetherEmbeddingsStage argv vs PutTogetherEmbeddings' real parser."""

    def test_argv_parses_with_embeddings_path_as_source(self, main_module):
        """REQ-STAGEARGV-06: parses; embeddings_source is config.dataset.embeddings_path."""
        config = _base_config(main_module)
        config.dataset.embeddings_path = "/data/MYDATASETembeddings"  # noqa: V101
        argv = _captured_argv(main_module, "PutTogetherEmbeddingsStage", "PutTogetherEmbeddings", config)
        namespace = _parse_with_real_parser(PutTogetherEmbeddings, argv)
        assert namespace.embeddings_source == "/data/MYDATASETembeddings"


_COMBINED_EMBEDDINGS_SUBDIR = Path("derivatives") / "champollion_V1" / "embeddings"


@pytest.mark.unit
class TestCombinedEmbeddingsLocationArgv:
    """REQ-COMBOUT-01/02: combine writes, and snapshots reads, <datasets_root>/derivatives/champollion_V1/embeddings."""

    def test_combine_output_path_is_datasets_root_derivatives_embeddings(self, main_module):
        """REQ-COMBOUT-01: combine --output_path is datasets_root/derivatives/champollion_V1/embeddings."""
        config = _base_config(main_module)
        config.dataset.embeddings_path = "/data/MYDATASETembeddings"  # noqa: V101
        argv = _captured_argv(main_module, "PutTogetherEmbeddingsStage", "PutTogetherEmbeddings", config)
        namespace = _parse_with_real_parser(PutTogetherEmbeddings, argv)
        assert Path(namespace.output_path) == Path("/data/MYDATASET") / _COMBINED_EMBEDDINGS_SUBDIR

    def test_snapshots_embeddings_dir_is_datasets_root_derivatives_embeddings(self, main_module):
        """REQ-COMBOUT-02: snapshots --embeddings_dir is datasets_root/derivatives/champollion_V1/embeddings."""
        config = _base_config(main_module)
        config.dataset.embeddings_path = "/data/MYDATASETembeddings"  # noqa: V101
        argv = _captured_argv(main_module, "GenerateSnapshotsStage", "GenerateSnapshots", config)
        namespace = _parse_with_real_parser(GenerateSnapshots, argv)
        assert namespace.embeddings_dir is not None, f"no --embeddings_dir in {argv!r}"
        assert Path(namespace.embeddings_dir) == Path("/data/MYDATASET") / _COMBINED_EMBEDDINGS_SUBDIR


@pytest.mark.unit
class TestOtherStagesArgv:
    """REQ-STAGEARGV-07: the four remaining stages' argv parses with the receiving script's parser."""

    @pytest.mark.parametrize(
        ("stage_class_name", "script_cls"),
        [
            ("GenerateMorphologistGraphsStage", GenerateMorphologistGraphs),
            ("RunCorticalTilesStage", RunCorticalTiles),
            ("GenerateChampollionConfigStage", GenerateChampollionConfig),
            ("GenerateSnapshotsStage", GenerateSnapshots),
        ],
    )
    def test_stage_argv_parses_with_target_parser(self, main_module, stage_class_name, script_cls):
        """REQ-STAGEARGV-07: stage argv is accepted by the receiving script's real parser."""
        config = _base_config(main_module)
        argv = _captured_argv(main_module, stage_class_name, script_cls.__name__, config)
        _parse_with_real_parser(script_cls, argv)


# Loads main.py exactly as `python main.py` would see it (no conftest sys.path tweaks, no soma stubs),
# without running main(), and reports where each stage-script name resolved.
_LOAD_MAIN_SNIPPET = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("champollion_main_import_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
names = sys.argv[2:]
resolved = {}
for name in names:
    obj = getattr(module, name, None)
    resolved[name] = None if obj is None else obj.__module__ + "." + obj.__qualname__
print("RESOLVED=" + json.dumps(resolved))
"""

_STAGE_SCRIPT_CLASSES = {
    "GenerateMorphologistGraphs": "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs",
    "RunCorticalTiles": "champollion_pipeline.run_cortical_tiles.RunCorticalTiles",
    "GenerateChampollionConfig": "champollion_pipeline.generate_champollion_config.GenerateChampollionConfig",
    "GenerateEmbeddings": "champollion_pipeline.generate_embeddings.GenerateEmbeddings",
    "PutTogetherEmbeddings": "champollion_pipeline.put_together_embeddings.PutTogetherEmbeddings",
    "GenerateSnapshots": "champollion_pipeline.generate_snapshots.GenerateSnapshots",
}


@pytest.fixture(scope="module")
def loaded_main_report():
    """Load main.py in a fresh interpreter from the repo root; return (resolved names, combined output)."""
    completed = subprocess.run(
        [sys.executable, "-c", _LOAD_MAIN_SNIPPET, str(MAIN_PY), *_STAGE_SCRIPT_CLASSES],
        cwd=str(MAIN_PY.parent),
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, f"loading main.py failed:\n{output}"
    resolved_lines = [line for line in completed.stdout.splitlines() if line.startswith("RESOLVED=")]
    assert len(resolved_lines) == 1, f"probe printed no RESOLVED line:\n{output}"
    return json.loads(resolved_lines[0][len("RESOLVED=") :]), output


@pytest.mark.unit
class TestMainImportsStageScripts:
    """REQ-STAGEARGV-08: loading main.py binds each stage-script name to the champollion_pipeline class."""

    @pytest.mark.parametrize("name", list(_STAGE_SCRIPT_CLASSES))
    def test_stage_script_name_resolves_to_package_class(self, loaded_main_report, name):
        """REQ-STAGEARGV-08: main.<name> is champollion_pipeline.<module>.<name>."""
        resolved, output = loaded_main_report
        assert resolved[name] == _STAGE_SCRIPT_CLASSES[name], f"main.{name} -> {resolved[name]!r}\n{output}"

    def test_no_could_not_import_warning(self, loaded_main_report):
        """REQ-STAGEARGV-08: loading main.py prints no 'Could not import pipeline scripts' warning."""
        _, output = loaded_main_report
        assert "Could not import pipeline scripts" not in output

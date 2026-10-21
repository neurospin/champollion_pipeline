#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Characterization tests pinning the command-line surfaces that
champollion_sulcal_mcp calls, and the stage modules' main() entry points.

Safety net for the behaviour-preserving refactor (epic
TASK-REFACTOR-...-5D3A3F7B6C6D): these tests pass on today's code and must keep
passing after it. The MCP server (champollion_sulcal_mcp/tools/stages.py)
launches ``<python> src/champollion_pipeline/<script>.py <argv>`` by file path,
so a renamed file, a reordered positional argument, a renamed option or an
option turned required breaks it. New options are safe and are not pinned.

Per script the tests pin:

- the argv the MCP server builds, with every optional MCP argument set, parses
  into the values it carries (positional order, option names, nargs);
- the defaults given only the arguments the MCP server always passes;
- the exit status 2 when a required option is missing;
- the exit status 0 of ``--help`` when the file is run by path.

And for the packaging: the pyproject ``[project.scripts]`` targets and a
``main`` callable with no argument in each stage module.

The parsers are built in-process (conftest.py stubs ``soma``). The by-path runs
use a fresh interpreter with the remote update check disabled, so they do not
depend on the network.

Requirements (area MCPCLI, author BDRABCZUK):
7477D569050A 89E4F241000A 46888B709ACA 01852DAB6201 66374B760AAC 4572AE92E465
D45C694E8754 F288A4A5C45D 1157AE9A1732 MCP argv; 29CD3DC21560 0A9784F4F0DC
5FC9DCCFB22C D543F0EAEB4B C30221660399 1986D14C884B 131825D12989 B61DA5965AAD
defaults; 190321567F55 276BC23EA421 3CF0B16A7629 86440988A44A 5CF7C4F56B96
7206664A2841 93167F2FBB4E required options; 5DE36858B82B run by path;
CDCCFFFC2B72 console scripts; 526A041D63FF main callable.
"""

import importlib
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import tomllib

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _SRC_DIR / "champollion_pipeline"

# Script file (as the MCP server names it) -> ScriptBuilder class in that module.
_MCP_SCRIPTS = {
    "generate_morphologist_graphs": "GenerateMorphologistGraphs",
    "run_cortical_tiles": "RunCorticalTiles",
    "generate_champollion_config": "GenerateChampollionConfig",
    "generate_embeddings": "GenerateEmbeddings",
    "put_together_embeddings": "PutTogetherEmbeddings",
    "train_champollion": "TrainChampollion",
    "generate_snapshots": "GenerateSnapshots",
    "purge_subject": "PurgeSubject",
    "prune_failed_subjects": "PruneFailedSubjects",
}

_CONSOLE_SCRIPTS = {
    "champollion-morphologist": "generate_morphologist_graphs",
    "champollion-cortical-tiles": "run_cortical_tiles",
    "champollion-config": "generate_champollion_config",
    "champollion-embeddings": "generate_embeddings",
    "champollion-combine": "put_together_embeddings",
    "champollion-snapshots": "generate_snapshots",
    "champollion-train": "train_champollion",
    "champollion-prune": "prune_failed_subjects",
    "champollion-purge": "purge_subject",
    "champollion-labelling-qc": "generate_labelling_qc",
}

_CHILD_PRELUDE = textwrap.dedent(
    """
    import importlib.util
    import runpy
    import sys
    from unittest.mock import MagicMock
    script = sys.argv[1]
    if importlib.util.find_spec("soma") is None:
        for _name in ("soma", "soma.aims", "soma.aimsalgo", "soma.aimsalgo.sulci"):
            sys.modules[_name] = MagicMock()
    import champollion_utils.update_check as _uc
    _uc.check_for_updates = lambda *a, **k: None
    import champollion_utils.script_builder as _sb
    _sb.check_for_updates = lambda *a, **k: None
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, str(__import__("pathlib").Path(script).parent))
    runpy.run_path(script, run_name="__main__")
    """
)


def _parse(module_name, argv):
    """Parse ``argv`` with the real parser of the MCP-called script ``module_name``."""
    module = importlib.import_module(f"champollion_pipeline.{module_name}")
    script = getattr(module, _MCP_SCRIPTS[module_name])()
    return vars(script.parser.parse_args(argv))


def _assert_subset(namespace, expected):
    assert {key: namespace[key] for key in expected} == expected


def _assert_exit_2(module_name, argv):
    with pytest.raises(SystemExit) as excinfo:
        _parse(module_name, argv)
    assert excinfo.value.code == 2


# --- the argv the MCP server builds -------------------------------------------


def test_morphologist_accepts_mcp_argv():
    ns = _parse("generate_morphologist_graphs", ["/in", "/out", "--parallel", "--enable-sulcal-recognition"])
    _assert_subset(ns, {"input": "/in", "output": "/out", "parallel": True, "enable_sulcal_recognition": True})


def test_cortical_tiles_accepts_mcp_argv():
    argv = [
        "/in",
        "/out",
        "--path_to_graph",
        "t1mri/default_acquisition/default_analysis/folds/3.1",
        "--path_sk_with_hull",
        "t1mri/default_acquisition/default_analysis/segmentation",
        "--sk_qc_path",
        "/qc.tsv",
        "--njobs",
        "4",
        "--masks",
        "canonical_25",
        "--regions",
        "S.C.-sylv.",
        "F.Coll.-S.Rh.",
        "--labelling_session",
        "my_session",
        "--overwrite",
    ]
    _assert_subset(
        _parse("run_cortical_tiles", argv),
        {
            "input": "/in",
            "output": "/out",
            "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
            "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
            "sk_qc_path": "/qc.tsv",
            "njobs": 4,
            "masks": "canonical_25",
            "regions": ["S.C.-sylv.", "F.Coll.-S.Rh."],
            "labelling_session": "my_session",
            "overwrite": True,
        },
    )


def test_config_accepts_mcp_argv():
    argv = [
        "/crops",
        "--dataset",
        "MYDATASET",
        "--champollion_loc",
        "/champollion_V1",
        "--output",
        "/configs",
        "--external-config",
        "/configs/dataset_localization",
        "--external_crops",
        "--masks",
        "my_masks",
    ]
    _assert_subset(
        _parse("generate_champollion_config", argv),
        {
            "crop_path": "/crops",
            "dataset": "MYDATASET",
            "champollion_loc": "/champollion_V1",
            "output": "/configs",
            "external_config": "/configs/dataset_localization",
            "external_crops": True,
            "masks": "my_masks",
        },
    )


def test_embeddings_accepts_mcp_argv():
    # --masks-version is pinned as it stands on WIP (e56c584); the MODELVER merge
    # adds --model-version and must update this pin if it touches --masks-version.
    argv = [
        "/models",
        "/data/MYDATASET",
        "--cpu",
        "--overwrite",
        "--run-cka",
        "--masks",
        "my_masks",
        "--masks-version",
        "v2",
        "--output",
        "/embeddings",
        "--subjects",
        "/subjects.csv",
        "--regions",
        "S.C.-sylv.",
        "F.Coll.-S.Rh.",
        "--cortical_version",
        "cortical_tiles-2025",
        "--profiling",
        "--no-cache",
        "--legacy",
        "--checkpoint-epoch",
        "12",
        "--models-cache",
        "/cache",
    ]
    _assert_subset(
        _parse("generate_embeddings", argv),
        {
            "models_path": "/models",
            "datasets_root": "/data/MYDATASET",
            "cpu": True,
            "overwrite": True,
            "run_cka": True,
            "masks": "my_masks",
            "masks_version": "v2",
            "output": "/embeddings",
            "subjects": "/subjects.csv",
            "regions": ["S.C.-sylv.", "F.Coll.-S.Rh."],
            "cortical_version": "cortical_tiles-2025",
            "profiling": True,
            "no_cache": True,
            "legacy": True,
            "checkpoint_epoch": 12,
            "models_cache": "/cache",
        },
    )


def test_combine_accepts_mcp_argv():
    ns = _parse("put_together_embeddings", ["/embeddings", "--output_path", "/combined"])
    _assert_subset(ns, {"embeddings_source": "/embeddings", "output_path": "/combined"})


def test_training_accepts_mcp_argv():
    argv = [
        "--dataset",
        "MYDATASET",
        "--region",
        "S.C.-sylv._left",
        "--mode",
        "encoder",
        "--output_dir",
        "/models/S.C.-sylv._left",
        "--config-dir",
        "/configs",
        "--njobs",
        "4",
        "--cpu",
        "--overwrite",
    ]
    _assert_subset(
        _parse("train_champollion", argv),
        {
            "dataset": "MYDATASET",
            "region": "S.C.-sylv._left",
            "mode": "encoder",
            "output_dir": "/models/S.C.-sylv._left",
            "config_dir": "/configs",
            "njobs": 4,
            "cpu": True,
            "overwrite": True,
        },
    )


def test_snapshots_accepts_mcp_argv():
    argv = [
        "--output_dir",
        "/snapshots",
        "--morphologist_dir",
        "/graphs",
        "--embeddings_dir",
        "/embeddings",
        "--cortical_tiles_dir",
        "/tiles",
        "--subject",
        "sub-01",
        "--acquisition",
        "acq-01",
        "--sulcal-only",
        "--tiles-only",
        "--umap-only",
        "--umap_region",
        "S.C.-sylv.",
        "--champollion_data_root",
        "/data_root",
        "--reference_data_dir",
        "/reference",
    ]
    _assert_subset(
        _parse("generate_snapshots", argv),
        {
            "output_dir": "/snapshots",
            "morphologist_dir": "/graphs",
            "embeddings_dir": "/embeddings",
            "cortical_tiles_dir": "/tiles",
            "subject": "sub-01",
            "acquisition": "acq-01",
            "sulcal_only": True,
            "tiles_only": True,
            "umap_only": True,
            "umap_region": "S.C.-sylv.",
            "champollion_data_root": "/data_root",
            "reference_data_dir": "/reference",
        },
    )


def test_purge_accepts_mcp_argv():
    ns = _parse("purge_subject", ["/derivatives", "--subject", "sub-01", "--dry-run"])
    _assert_subset(ns, {"derivatives": "/derivatives", "subject": "sub-01", "dry_run": True})


def test_prune_accepts_mcp_argv():
    ns = _parse("prune_failed_subjects", ["/out", "--qc", "/qc.tsv", "--dry-run"])
    _assert_subset(ns, {"output": "/out", "qc": "/qc.tsv", "dry_run": True})


# --- defaults given only the arguments the MCP server always passes -----------


def test_morphologist_defaults():
    ns = _parse("generate_morphologist_graphs", ["/in", "/out"])
    _assert_subset(ns, {"parallel": False, "enable_sulcal_recognition": False, "bids": False})


def test_cortical_tiles_defaults():
    ns = _parse("run_cortical_tiles", ["/in", "/out", "--path_to_graph", "g", "--path_sk_with_hull", "s"])
    _assert_subset(
        ns,
        {
            "sk_qc_path": "",
            "labelling_session": "deepcnn_session_auto",
            "njobs": None,
            "masks": None,
            "regions": None,
            "overwrite": False,
        },
    )


def test_config_defaults():
    ns = _parse("generate_champollion_config", ["/crops", "--dataset", "MYDATASET"])
    assert Path(ns.pop("champollion_loc")).resolve() == (_REPO_ROOT / "external" / "champollion_V1").resolve()
    _assert_subset(
        ns,
        {
            "masks": "canonical_25",
            "localization": "local",
            "output": None,
            "external_config": None,
            "external_crops": False,
        },
    )


def test_embeddings_defaults():
    # Since the MODELVER merge --masks has no parser default: GenerateEmbeddings.parse_args fills it from
    # --model-version, else canonical_25 (pinned in tests/test_generate_embeddings_masks_default.py).
    ns = _parse("generate_embeddings", ["/models", "/data/MYDATASET"])
    _assert_subset(
        ns,
        {
            "masks": None,
            "model_version": None,
            "cortical_version": "cortical_tiles-2026",
            "checkpoint_epoch": 80,
            "masks_version": None,
            "output": None,
            "subjects": None,
            "regions": None,
            "models_cache": None,
            "legacy": False,
        },
    )


def test_training_defaults():
    ns = _parse("train_champollion", ["--dataset", "MYDATASET", "--region", "S.C.-sylv._left"])
    _assert_subset(
        ns,
        {
            "mode": "encoder",
            "localization": "local",
            "config_dir": None,
            "output_dir": None,
            "njobs": None,
            "cpu": False,
            "overwrite": False,
            "swf": False,
        },
    )


def test_snapshots_defaults():
    ns = _parse("generate_snapshots", ["--output_dir", "/snapshots"])
    _assert_subset(
        ns,
        {
            "width": 800,
            "height": 600,
            "tiles_level": 1,
            "reference_data_dir": None,
            "champollion_data_root": None,
            "sulcal_only": False,
            "tiles_only": False,
            "umap_only": False,
        },
    )


def test_purge_defaults():
    _assert_subset(_parse("purge_subject", ["/derivatives", "--subject", "sub-01"]), {"dry_run": False})


def test_prune_defaults():
    _assert_subset(_parse("prune_failed_subjects", ["/out", "--qc", "/qc.tsv"]), {"dry_run": False})


# --- required options -----------------------------------------------------------


def test_cortical_tiles_required_options():
    _assert_exit_2("run_cortical_tiles", ["/in", "/out", "--path_sk_with_hull", "s"])
    _assert_exit_2("run_cortical_tiles", ["/in", "/out", "--path_to_graph", "g"])


def test_config_required_options():
    _assert_exit_2("generate_champollion_config", ["/crops"])


def test_combine_required_options():
    _assert_exit_2("put_together_embeddings", ["/embeddings"])


def test_training_required_options():
    _assert_exit_2("train_champollion", ["--region", "S.C.-sylv._left"])
    _assert_exit_2("train_champollion", ["--dataset", "MYDATASET"])


def test_snapshots_required_options():
    _assert_exit_2("generate_snapshots", [])


def test_purge_required_options():
    _assert_exit_2("purge_subject", ["/derivatives"])


def test_prune_required_options():
    _assert_exit_2("prune_failed_subjects", ["/out"])


# --- the files the MCP server runs by path --------------------------------------


@pytest.mark.parametrize("module_name", sorted(_MCP_SCRIPTS))
def test_stage_script_runs_by_path_with_help(module_name):
    script = _SCRIPTS_DIR / f"{module_name}.py"
    assert script.is_file()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC_DIR), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PRELUDE, str(script), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout


# --- packaging entry points ------------------------------------------------------


def test_console_scripts_target_main():
    scripts = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    expected = {name: f"champollion_pipeline.{module}:main" for name, module in _CONSOLE_SCRIPTS.items()}
    assert {name: scripts.get(name) for name in expected} == expected


@pytest.mark.parametrize("module_name", sorted(_CONSOLE_SCRIPTS.values()))
def test_stage_module_defines_main(module_name):
    main = importlib.import_module(f"champollion_pipeline.{module_name}").main
    assert inspect.isfunction(main)
    inspect.signature(main).bind()

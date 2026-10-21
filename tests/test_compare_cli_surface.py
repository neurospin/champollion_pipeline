#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Characterization tests pinning the command-line surface of the compare tools.

Safety net for the compare.py dedupe and split (epic TASK-REFACTOR-...-628C6B3E0F00):
these tests pass on today's code and must keep passing after the refactor. They
pin, for src/compare.py (subcommands masks, cortical_tiles, crops, databases,
embeddings) and for the scripts src/compare_masks.py,
src/compare_cortical_tiles.py and src/compare_databases.py:

- the options their --help text lists, exactly;
- which options are required (argparse exit status 2 when missing);
- the defaults given only the required options;
- the choices of --metric, --input_type and --side;
- the exit status of --help (0) and of a missing --set_a directory (1);
- the public names the other tests import from each module;
- the pixi ``compare-databases`` task command.

Each script run happens in a fresh Python subprocess (``runpy`` with
``run_name="__main__"``, i.e. ``python src/<script>.py``), with ``soma``
stubbed the way conftest.py stubs it in-process and the remote update check
disabled, so the results do not depend on PyAIMS or on the network.

Requirements (area COMPARE, author BDRABCZUK):
1BCAE707D4F5 subcommand set, 0001B4BA200B no subcommand, 6890D56963CE --help exit,
25766A5C0F8A 5E573581E6D1 55C9624283C1 75FFC4D00D3A 78DD0EF742C1 subcommand options,
4B6CAF044462 9948C646D2E5 required options, 117DA8854F73 176B8286C589
344D566E476B 7B6A038CF0A3 06A9B555A5BF subcommand defaults, B8A47BB26FE8
1BA70A85A01E EA954461B742 choices, 941405980FD7 missing --set_a, 5EA1F9FAA661
compare names, 2A0C618A7734 E39EB582D5D5 622864743E4C script options,
509D22CD1937 script required options, 3D27AA1B3C0D AE6339FE8D98 4AC071AD83CC
script defaults, E6ABDDDFD058 script --metric choices, A43D0BA27200 script
missing --set_a, 200C6AFF041C 4E8737605582 B3921FE6A4DF script names,
8A738586FB83 pixi task.
"""

import argparse
import functools
import importlib
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import tomllib

import compare
import compare_cortical_tiles
import compare_databases
import compare_masks

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

_CHILD_PRELUDE = textwrap.dedent(
    f"""
    import runpy
    import sys
    from unittest.mock import MagicMock
    sys.path.insert(0, {str(_SRC_DIR)!r})
    for _name in ("soma", "soma.aims", "soma.aimsalgo", "soma.aimsalgo.sulci"):
        sys.modules[_name] = MagicMock()
    import champollion_utils.script_builder as _sb
    _sb.check_for_updates = lambda *a, **k: None
    """
)

_SUBCOMMANDS = ("masks", "cortical_tiles", "crops", "databases", "embeddings")
_SET_SUBCOMMANDS = ("masks", "cortical_tiles", "crops", "embeddings")
_MASK_SCRIPTS = ("compare_masks", "compare_cortical_tiles")
_DB_REQUIRED = {
    "--labeled_subjects_dir": "/subjects",
    "--path_to_graph_a": "graph/a",
    "--path_to_graph_b": "graph/b",
}

_MASK_OPTIONS = {
    "--set_a",
    "--set_b",
    "--output",
    "--metric",
    "--bucket_step",
    "--xor_dir",
    "--visualisation",
    "--help",
}
_DB_OPTIONS = {
    "--labeled_subjects_dir",
    "--path_to_graph_a",
    "--path_to_graph_b",
    "--label_a",
    "--label_b",
    "--side",
    "--masks_a",
    "--masks_b",
    "--output",
    "--njobs",
    "--help",
}

_COMPARE_OPTIONS = {
    "masks": _MASK_OPTIONS,
    "cortical_tiles": _MASK_OPTIONS | {"--pattern"},
    "crops": {
        "--set_a",
        "--set_b",
        "--output",
        "--input_type",
        "--xor_dir",
        "--regions",
        "--side",
        "--top_k",
        "--njobs",
        "--help",
    },
    "databases": _DB_OPTIONS,
    "embeddings": {
        "--set_a",
        "--set_b",
        "--output",
        "--regions",
        "--side",
        "--k",
        "--njobs",
        "--umap_a",
        "--umap_b",
        "--covariate",
        "--covariate_column",
        "--help",
    },
}

_SCRIPT_OPTIONS = {
    "compare_masks": {"--set_a", "--set_b", "--output", "--metric", "--bucket_step", "--help"},
    "compare_cortical_tiles": {"--set_a", "--set_b", "--output", "--pattern", "--metric", "--bucket_step", "--help"},
    "compare_databases": _DB_OPTIONS,
}

_PUBLIC_NAMES = {
    "compare": (
        "Compare",
        "main",
        "bucket_label",
        "load_mask_vol",
        "save_xor_vol",
        "sort_buckets",
        "visualise_mask_diffs",
        "voxel_diff",
        "wasserstein_distance",
    ),
    "compare_masks": (
        "main",
        "parse_args",
        "bucket_label",
        "find_masks",
        "load_mask_vol",
        "voxel_diff",
        "wasserstein_distance",
    ),
    "compare_cortical_tiles": (
        "CompareCorticalTiles",
        "main",
        "bucket_label",
        "find_masks",
        "load_mask_vol",
        "voxel_diff",
        "wasserstein_distance",
    ),
    "compare_databases": ("CompareDatabases", "main"),
}

# An option line of argparse help starts with exactly two spaces, then "-x, --name" or "--name".
_OPTION_LINE = re.compile(r"^  (?:-\w, )?(--[\w-]+)", re.MULTILINE)


def _run_script(script: str, argv: list, cwd: Path) -> subprocess.CompletedProcess:
    """Run src/<script>.py as __main__ in a fresh interpreter and return the result."""
    path = _SRC_DIR / f"{script}.py"
    code = _CHILD_PRELUDE + textwrap.dedent(
        f"""
        sys.argv = [{str(path)!r}] + {argv!r}
        runpy.run_path({str(path)!r}, run_name="__main__")
        """
    )
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd), capture_output=True, text=True, timeout=180)


@functools.lru_cache(maxsize=None)
def _help(script: str, subcommand: str | None = None) -> subprocess.CompletedProcess:
    argv = ([subcommand] if subcommand else []) + ["--help"]
    return _run_script(script, argv, _REPO_ROOT)


def _help_options(text: str) -> set:
    return set(_OPTION_LINE.findall(text))


def _help_choices(text: str, option: str) -> set:
    match = re.search(re.escape(option) + r" \{([^}]*)\}", text)
    assert match, f"{option} shows no choice set in help:\n{text}"
    return set(match.group(1).split(","))


def _compare_parser():

    return compare.Compare().parser


def _parse_compare(argv: list):
    return _compare_parser().parse_args(argv)


def _set_args(tmp_path: Path) -> list:
    return ["--set_a", str(tmp_path / "a"), "--set_b", str(tmp_path / "b")]


def _db_args() -> list:
    return [token for pair in _DB_REQUIRED.items() for token in pair]


# --------------------------------------------------------------------------- #
# src/compare.py
# --------------------------------------------------------------------------- #


def test_compare_offers_exactly_five_subcommands():
    """REQ-COMPARE-BDRABCZUK-1BCAE707D4F5."""
    actions = [a for a in _compare_parser()._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(actions) == 1
    assert set(actions[0].choices) == set(_SUBCOMMANDS)


def test_compare_without_subcommand_exits_2(tmp_path):
    """REQ-COMPARE-BDRABCZUK-0001B4BA200B."""
    result = _run_script("compare", [], tmp_path)
    assert result.returncode == 2, result.stderr


@pytest.mark.parametrize("subcommand", _SUBCOMMANDS)
def test_compare_subcommand_help_exits_0(subcommand):
    """REQ-COMPARE-BDRABCZUK-6890D56963CE."""
    result = _help("compare", subcommand)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage:")
    assert subcommand in result.stdout.splitlines()[0]


@pytest.mark.parametrize("subcommand", _SUBCOMMANDS)
def test_compare_help_lists_exact_options(subcommand):
    """REQ-COMPARE-BDRABCZUK-25766A5C0F8A, -5E573581E6D1, -55C9624283C1, -75FFC4D00D3A, -78DD0EF742C1."""
    result = _help("compare", subcommand)
    assert result.returncode == 0, result.stderr
    assert _help_options(result.stdout) == _COMPARE_OPTIONS[subcommand]


@pytest.mark.parametrize("subcommand", _SET_SUBCOMMANDS)
@pytest.mark.parametrize("missing", ["--set_a", "--set_b"])
def test_compare_set_options_required(subcommand, missing, tmp_path):
    """REQ-COMPARE-BDRABCZUK-4B6CAF044462."""
    argv = [subcommand] + _set_args(tmp_path)
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit) as excinfo:
        _parse_compare(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize("missing", list(_DB_REQUIRED))
def test_databases_graph_options_required(missing):
    """REQ-COMPARE-BDRABCZUK-9948C646D2E5 (src/compare.py databases and src/compare_databases.py)."""

    argv = _db_args()
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit) as excinfo:
        _parse_compare(["databases"] + argv)
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        compare_databases.CompareDatabases().parser.parse_args(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize("subcommand", ["masks", "cortical_tiles"])
def test_compare_mask_modes_defaults(subcommand, tmp_path):
    """REQ-COMPARE-BDRABCZUK-117DA8854F73."""
    args = _parse_compare([subcommand] + _set_args(tmp_path))
    assert args.output == "comparison_report.json"
    assert args.metric == "diff"
    assert args.bucket_step == 1.0
    assert args.xor_dir is None
    assert args.visualisation is False


def test_compare_cortical_tiles_pattern_default(tmp_path):
    """REQ-COMPARE-BDRABCZUK-176B8286C589."""
    args = _parse_compare(["cortical_tiles"] + _set_args(tmp_path))
    assert args.pattern == "*mask_skeleton.nii.gz"


def test_compare_crops_defaults(tmp_path):
    """REQ-COMPARE-BDRABCZUK-344D566E476B."""
    args = _parse_compare(["crops"] + _set_args(tmp_path))
    assert args.output == "crops_comparison"
    assert args.input_type == "skeleton"
    assert args.xor_dir is None
    assert args.regions is None
    assert args.side == "both"
    assert args.top_k == 5
    assert args.njobs == 1


def test_compare_databases_defaults():
    """REQ-COMPARE-BDRABCZUK-7B6A038CF0A3."""
    args = _parse_compare(["databases"] + _db_args())
    assert args.label_a == "A"
    assert args.label_b == "B"
    assert args.side == "both"
    assert args.masks_a is None
    assert args.masks_b is None
    assert args.output == "db_comparison.csv"
    assert args.njobs is None


def test_compare_embeddings_defaults(tmp_path):
    """REQ-COMPARE-BDRABCZUK-06A9B555A5BF."""
    args = _parse_compare(["embeddings"] + _set_args(tmp_path))
    assert args.output == "embeddings_comparison"
    assert args.side == "both"
    assert args.k == 15
    assert args.njobs == 1
    assert args.regions is None
    assert args.umap_a is None
    assert args.umap_b is None
    assert args.covariate is None
    assert args.covariate_column is None


@pytest.mark.parametrize("subcommand", ["masks", "cortical_tiles"])
def test_compare_metric_choices(subcommand, tmp_path):
    """REQ-COMPARE-BDRABCZUK-B8A47BB26FE8."""
    expected = {"wasserstein", "diff", "both"}
    assert _help_choices(_help("compare", subcommand).stdout, "--metric") == expected
    for value in expected:
        assert _parse_compare([subcommand] + _set_args(tmp_path) + ["--metric", value]).metric == value
    with pytest.raises(SystemExit) as excinfo:
        _parse_compare([subcommand] + _set_args(tmp_path) + ["--metric", "bogus"])
    assert excinfo.value.code == 2


def test_compare_crops_input_type_choices(tmp_path):
    """REQ-COMPARE-BDRABCZUK-1BA70A85A01E."""
    expected = {"skeleton", "label", "extremities", "distmap", "distbottom"}
    assert _help_choices(_help("compare", "crops").stdout, "--input_type") == expected
    for value in expected:
        assert _parse_compare(["crops"] + _set_args(tmp_path) + ["--input_type", value]).input_type == value


@pytest.mark.parametrize("subcommand", ["crops", "databases", "embeddings"])
def test_compare_side_choices(subcommand, tmp_path):
    """REQ-COMPARE-BDRABCZUK-EA954461B742."""
    expected = {"L", "R", "both"}
    assert _help_choices(_help("compare", subcommand).stdout, "--side") == expected
    base = [subcommand] + (_db_args() if subcommand == "databases" else _set_args(tmp_path))
    for value in expected:
        assert _parse_compare(base + ["--side", value]).side == value


@pytest.mark.parametrize("subcommand", _SET_SUBCOMMANDS)
def test_compare_missing_set_a_exits_1(subcommand, tmp_path):
    """REQ-COMPARE-BDRABCZUK-941405980FD7."""
    (tmp_path / "b").mkdir()
    result = _run_script("compare", [subcommand] + _set_args(tmp_path), tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# src/compare_masks.py, src/compare_cortical_tiles.py, src/compare_databases.py
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", list(_SCRIPT_OPTIONS))
def test_script_help_lists_exact_options(script):
    """REQ-COMPARE-BDRABCZUK-2A0C618A7734, -E39EB582D5D5, -622864743E4C."""
    result = _help(script)
    assert result.returncode == 0, result.stderr
    assert _help_options(result.stdout) == _SCRIPT_OPTIONS[script]


def _parse_mask_script(script: str, argv: list):
    if script == "compare_masks":
        return compare_masks.parse_args(argv)

    return compare_cortical_tiles.CompareCorticalTiles().parser.parse_args(argv)


@pytest.mark.parametrize("script", _MASK_SCRIPTS)
@pytest.mark.parametrize("missing", ["--set_a", "--set_b"])
def test_mask_scripts_set_options_required(script, missing, tmp_path):
    """REQ-COMPARE-BDRABCZUK-509D22CD1937."""
    argv = _set_args(tmp_path)
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit) as excinfo:
        _parse_mask_script(script, argv)
    assert excinfo.value.code == 2


def test_compare_masks_script_defaults(tmp_path):
    """REQ-COMPARE-BDRABCZUK-3D27AA1B3C0D."""
    args = _parse_mask_script("compare_masks", _set_args(tmp_path))
    assert args.output == "mask_diff_report.json"
    assert args.metric == "wasserstein"
    assert args.bucket_step == 1.0


def test_compare_cortical_tiles_script_defaults(tmp_path):
    """REQ-COMPARE-BDRABCZUK-AE6339FE8D98."""
    args = _parse_mask_script("compare_cortical_tiles", _set_args(tmp_path))
    assert args.output == "cortical_tiles_diff_report.json"
    assert args.pattern == "*mask_skeleton.nii.gz"
    assert args.metric == "diff"
    assert args.bucket_step == 10


def test_compare_databases_script_defaults():
    """REQ-COMPARE-BDRABCZUK-4AC071AD83CC."""

    args = compare_databases.CompareDatabases().parser.parse_args(_db_args())
    assert args.label_a == "A"
    assert args.label_b == "B"
    assert args.side == "both"
    assert args.masks_a is None
    assert args.masks_b is None
    assert args.output == "./db_comparison.csv"
    assert args.njobs is None


@pytest.mark.parametrize("script", _MASK_SCRIPTS)
def test_mask_scripts_metric_choices(script, tmp_path):
    """REQ-COMPARE-BDRABCZUK-E6ABDDDFD058."""
    expected = {"wasserstein", "diff", "both"}
    assert _help_choices(_help(script).stdout, "--metric") == expected
    for value in expected:
        assert _parse_mask_script(script, _set_args(tmp_path) + ["--metric", value]).metric == value
    with pytest.raises(SystemExit) as excinfo:
        _parse_mask_script(script, _set_args(tmp_path) + ["--metric", "bogus"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("script", _MASK_SCRIPTS)
def test_mask_scripts_missing_set_a_exits_1(script, tmp_path):
    """REQ-COMPARE-BDRABCZUK-A43D0BA27200."""
    (tmp_path / "b").mkdir()
    result = _run_script(script, _set_args(tmp_path), tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# Public names and the pixi task
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module_name", list(_PUBLIC_NAMES))
def test_module_public_names(module_name):
    """REQ-COMPARE-BDRABCZUK-5EA1F9FAA661, -200C6AFF041C, -4E8737605582, -B3921FE6A4DF."""
    module = importlib.import_module(module_name)
    missing = [name for name in _PUBLIC_NAMES[module_name] if not hasattr(module, name)]
    assert missing == []


def test_pixi_compare_databases_task():
    """REQ-COMPARE-BDRABCZUK-8A738586FB83."""
    with open(_REPO_ROOT / "pixi.toml", "rb") as handle:
        task = tomllib.load(handle)["tasks"]["compare-databases"]
    command = task if isinstance(task, str) else task["cmd"]
    assert command == "python3 src/compare_databases.py"

"""Stage 1-2 subprocesses get cwd= and the process cwd never moves (TASK-PYTHONIC-BDRABCZUK-FEA9D87B8576).

User decision 2026-10-30: no chdir at all (no context-manager helper, no contextlib.chdir);
generate_morphologist_graphs and run_cortical_tiles pass the working directory of their
subprocess through ScriptBuilder.execute_command(cmd, shell, cwd), available from
champollion_utils 0.4.0. Reading the cwd (the LOG line of REQ-LOGGING-BDRABCZUK-205E7A01F3F1)
stays allowed.

REQ-PYTHONIC-BDRABCZUK-F7EE03ADDB6A, -E4EF4855A4EA: neither module source calls a function named chdir.
REQ-PYTHONIC-BDRABCZUK-AD5613FEB639: morphologist-cli runs with the input directory as cwd.
REQ-PYTHONIC-BDRABCZUK-41A36519D15F: generate_sulcal_regions.py runs with its own directory as cwd.
REQ-PYTHONIC-BDRABCZUK-83D824016E94, -ECF42A00512C: run() leaves the process cwd as found, on return and on raise.
REQ-PYTHONIC-BDRABCZUK-3B5E8BCCDFD1 (supersedes -C19B8BB3589F): the champollion-utils pin names a v0.4.1-or-later
release tag (0.4.1 adds champollion_utils.process.active_profiling, TASK-PROFILING-BDRABCZUK-0BB1760DF489).
"""

import ast
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.unit  # noqa: V107

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_DIR = _REPO_ROOT / "src" / "champollion_pipeline"
_RELEASE_TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")
_MINIMUM_UTILS = Version("0.4.1")
_GENERATE_SULCAL_REGIONS = (
    _REPO_ROOT / "external" / "cortical_tiles" / "cortical_tiles" / "brainvisa" / "generate_sulcal_regions.py"
)


class _SubprocessFailure(RuntimeError):
    """Stand-in for an exception escaping execute_command."""


def _chdir_calls(module: str) -> list[int]:
    """Line numbers of every call whose callee is named chdir (chdir(...), os.chdir(...), contextlib.chdir(...))."""
    tree = ast.parse((_PACKAGE_DIR / f"{module}.py").read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "chdir":
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("module", ["generate_morphologist_graphs", "run_cortical_tiles"])
def test_source_has_no_chdir_call(module):
    """REQ-PYTHONIC-BDRABCZUK-F7EE03ADDB6A (generate_morphologist_graphs), -E4EF4855A4EA (run_cortical_tiles)."""
    assert _chdir_calls(module) == [], f"{module}.py calls chdir at line(s) {_chdir_calls(module)}"


@pytest.fixture
def launch_dir(tmp_path, monkeypatch):
    """A process cwd distinct from every stage directory; the test starts there."""
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    return launch


def _morphologist(tmp_path: Path) -> GenerateMorphologistGraphs:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir)])
    return script


def _cortical_tiles(tmp_path: Path) -> RunCorticalTiles:
    input_dir = tmp_path / "graphs_in"
    output_dir = tmp_path / "tiles_out"
    input_dir.mkdir()
    output_dir.mkdir()
    # An existing pipeline config skips the template copy, so generate_sulcal_regions is the only command.
    (output_dir / "pipeline_loop_2mm.json").write_text("{}", encoding="utf-8")
    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            "graphs",
            "--path_sk_with_hull",
            "skeleton",
            "--njobs",
            "1",
        ]
    )
    return script


def _run_morphologist(script: GenerateMorphologistGraphs, **execute_command):
    with (
        patch.object(script, "_run_input_qc", return_value=["sub-01.nii.gz"]),
        patch.object(script, "execute_command", **execute_command) as mock_exec,
    ):
        return script.run(), mock_exec


def _run_cortical_tiles(script: RunCorticalTiles, **execute_command):
    """run() up to the generate_sulcal_regions call; a non-zero return code stops it there."""
    with (
        patch.object(script, "_preflight_check", return_value=True),
        patch.object(script, "execute_command", **execute_command) as mock_exec,
    ):
        return script.run(), mock_exec


def test_morphologist_cli_receives_input_directory_as_cwd(tmp_path, launch_dir):
    """REQ-PYTHONIC-BDRABCZUK-AD5613FEB639."""
    script = _morphologist(tmp_path)

    _, mock_exec = _run_morphologist(script, return_value=0)

    assert mock_exec.call_count == 1
    assert mock_exec.call_args.args[0][0] == "morphologist-cli"
    assert mock_exec.call_args.kwargs.get("cwd") == script.args.input


def test_generate_sulcal_regions_receives_script_directory_as_cwd(tmp_path, launch_dir):
    """REQ-PYTHONIC-BDRABCZUK-41A36519D15F."""
    script = _cortical_tiles(tmp_path)

    result, mock_exec = _run_cortical_tiles(script, return_value=1)

    assert result == 1
    assert mock_exec.call_count == 1
    assert str(_GENERATE_SULCAL_REGIONS) in mock_exec.call_args.args[0]
    assert mock_exec.call_args.kwargs.get("cwd") == str(_GENERATE_SULCAL_REGIONS.parent)


def test_morphologist_run_keeps_process_cwd_when_execute_command_raises(tmp_path, launch_dir):
    """REQ-PYTHONIC-BDRABCZUK-83D824016E94, raise path."""
    script = _morphologist(tmp_path)

    with pytest.raises(_SubprocessFailure):
        _run_morphologist(script, side_effect=_SubprocessFailure("morphologist-cli crashed"))

    assert os.getcwd() == str(launch_dir)


def test_morphologist_run_keeps_process_cwd_on_return(tmp_path, launch_dir):
    """REQ-PYTHONIC-BDRABCZUK-83D824016E94, return path."""
    script = _morphologist(tmp_path)

    result, _ = _run_morphologist(script, return_value=0)

    assert result == 0
    assert os.getcwd() == str(launch_dir)


def test_cortical_tiles_run_keeps_process_cwd_when_execute_command_raises(tmp_path, launch_dir):
    """REQ-PYTHONIC-BDRABCZUK-ECF42A00512C, raise path."""
    script = _cortical_tiles(tmp_path)

    with pytest.raises(_SubprocessFailure):
        _run_cortical_tiles(script, side_effect=_SubprocessFailure("generate_sulcal_regions crashed"))

    assert os.getcwd() == str(launch_dir)


def test_cortical_tiles_run_keeps_process_cwd_on_return(tmp_path, launch_dir):
    """REQ-PYTHONIC-BDRABCZUK-ECF42A00512C, return path."""
    script = _cortical_tiles(tmp_path)

    result, _ = _run_cortical_tiles(script, return_value=1)

    assert result == 1
    assert os.getcwd() == str(launch_dir)


def _utils_dependency() -> Requirement:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(dep) for dep in pyproject["project"]["dependencies"]]
    utils = [req for req in requirements if canonicalize_name(req.name) == "champollion-utils"]
    assert len(utils) == 1, f"expected one champollion-utils dependency, got {utils!r}"
    return utils[0]


def test_pyproject_champollion_utils_pins_release_tag_0_4_1_or_later():
    """REQ-PYTHONIC-BDRABCZUK-3B5E8BCCDFD1."""
    url = _utils_dependency().url
    assert url is not None and url.startswith("git+"), f"{url!r} is not a PEP 508 git direct reference"
    path = url.split("://", 1)[-1]
    assert "@" in path, f"{url} does not pin a git ref"
    ref = path.rsplit("@", 1)[1]
    tag = _RELEASE_TAG.match(ref)
    assert tag is not None, f"{url} pins {ref!r}, not a v<MAJOR>.<MINOR>.<PATCH> tag"
    assert Version(tag["version"]) >= _MINIMUM_UTILS, f"{url} pins {ref}, below v{_MINIMUM_UTILS}"

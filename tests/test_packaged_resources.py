#!/usr/bin/env python3
"""
Runtime data files live in the package, so a non-editable install finds them.

TASK-LAYOUT-BDRABCZUK-B741C309747E moves sulci_regions_champollion_V1.json, reference.yaml and
pipeline_loop_2mm.json from the repository root to src/champollion_pipeline/resources/.

The wheel tests build a real wheel with setuptools' PEP 517 backend, called directly on a staged copy of
pyproject.toml and src/ (no pip, no network, nothing installed, the checkout untouched). The stage tests then
unpack that wheel into an empty directory and run each stage from there in a child Python: only the wheel's
files are reachable, as in site-packages, and a file left outside the wheel cannot be found. Each stage is
stopped right after the read under test, so the tests depend on no code that follows it.

Requirements:
- REQ-LAYOUT-BDRABCZUK-7DC3A0D78810, -50520AE5222D, -15612D17F29A: the wheel contains each file.
- REQ-LAYOUT-BDRABCZUK-77FEAB50B31A, -6E194496B465: the move keeps each file's bytes.
- REQ-LAYOUT-BDRABCZUK-9609E7AB5B18 (supersedes -700FB107C68C): the template keeps its bytes minus the
  skip_distbottom line (epic NODISTB).
- REQ-LAYOUT-BDRABCZUK-73AB5FC0DAB8, -13578E19D70D, -7153BA94441D: the repository root keeps no copy.
- REQ-LAYOUT-BDRABCZUK-D9B4FB454285: GenerateMasks.run() reads the packaged region table.
- REQ-LAYOUT-BDRABCZUK-136516FF987E: GenerateChampollionConfig.run() copies the packaged reference.yaml.
- REQ-LAYOUT-BDRABCZUK-BFCE3774D58B: RunCorticalTiles.run() copies the packaged template.
- REQ-LAYOUT-BDRABCZUK-AF85B9A8B24A: RunCorticalTiles.run() writes brain_regions_json explicitly; cortical_tiles
  resolves "$local" to <repository root>/sulci_regions_champollion_V1.json, where the file no longer is.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import champollion_pipeline
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.stage2_layout import stage_dirs

pytestmark = pytest.mark.integration  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(champollion_pipeline.__file__).resolve().parent
RESOURCES_DIR = PACKAGE_DIR / "resources"

REGION_TABLE = "sulci_regions_champollion_V1.json"
REFERENCE_YAML = "reference.yaml"
TEMPLATE = "pipeline_loop_2mm.json"
# SHA-256 of each packaged resource: the repository-root file at commit d35966d, the last commit before the move,
# except reference.yaml, whose six distbottom and extremities keys commit c6bf8c31 dropped (epic NODISTB): its digest
# is the current file, pinned by REQ-LAYOUT-BDRABCZUK-E39834C1AE4D, which supersedes REQ-LAYOUT-BDRABCZUK-6E194496B465;
# and pipeline_loop_2mm.json, whose skip_distbottom line was then dropped (epic NODISTB).
RESOURCE_DIGESTS = {
    REGION_TABLE: "82585e8e531a1e11c9ee6a480444d1689444100686e8dfaf602487fafabf8500",
    REFERENCE_YAML: "a22fee84bcbd6e5adf08ae2c40c0580132c9302833eb92c26f1dfced8c1733ec",
    TEMPLATE: "105e7ff93fdaf3c02ecf7db2baf38fd5aa575e40a70f2b7e0ad329bb35a44f8c",
}
DATASET = "TEST01"
SUBPROCESS_TIMEOUT = 120

BUILD_WHEEL = "import sys; from setuptools import build_meta; build_meta.build_wheel(sys.argv[1])"

# The soma stubs mirror tests/conftest.py: the child imports the stages outside the brainvisa environment too.
PROBE_PREAMBLE = """
import json
import subprocess
import sys
from unittest.mock import MagicMock

for name in ("soma", "soma.aims", "soma.aimsalgo", "soma.aimsalgo.sulci", "soma.qt_gui", "soma.qt_gui.qt_backend"):
    sys.modules.setdefault(name, MagicMock())


class Reached(Exception):
    '''The stage got past the read under test.'''
"""

MASKS_PROBE = (
    PROBE_PREAMBLE
    + """
from champollion_pipeline import generate_masks

read_region_table = generate_masks.get_sulci_for_regions
sulci = []


def read_then_stop(*args, **kwargs):
    sulci.extend(read_region_table(*args, **kwargs))
    raise Reached


generate_masks.get_sulci_for_regions = read_then_stop
script = generate_masks.GenerateMasks()
script.parse_args(
    ["--labeled_subjects_dir", sys.argv[1], "--path_to_graph_supervised", "graphs", "--output_dir", sys.argv[1],
     "--regions", "S.C.-sylv.", "--sides", "L"]
)
try:
    script.run()
except Reached:
    pass
print(json.dumps({"module": generate_masks.__file__, "sulci": sorted(sulci)}))
"""
)

CONFIG_PROBE = (
    PROBE_PREAMBLE
    + """
from champollion_pipeline import generate_champollion_config

crop_path, configs_root, champollion_loc, dataset = sys.argv[1:5]
script = generate_champollion_config.GenerateChampollionConfig()
script.parse_args([crop_path, "--dataset", dataset, "--output", configs_root, "--champollion_loc", champollion_loc])
script._run_input_qc = script._run_output_qc = lambda *_args: 0  # no crop on disk: tests/conftest.py stubs them too
script.run()
print(json.dumps({"module": generate_champollion_config.__file__}))
"""
)

STAGE2_PROBE = (
    PROBE_PREAMBLE
    + """
import numpy

from champollion_pipeline import run_cortical_tiles


class Volume(numpy.ndarray):
    np = property(numpy.asarray)


run_cortical_tiles.aims.read = MagicMock(return_value=numpy.ones(1, dtype=numpy.int16).view(Volume))


def run_until_stage2(command, **_kwargs):
    if any("generate_sulcal_regions" in str(part) for part in command):
        raise Reached
    return subprocess.run(command, check=False).returncode


script = run_cortical_tiles.RunCorticalTiles()
script.parse_args([sys.argv[1], sys.argv[2], "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton"])
script.execute_command = run_until_stage2
try:
    script.run()
except Reached:
    pass
print(json.dumps({"module": run_cortical_tiles.__file__}))
"""
)


class _Stage2Reached(Exception):
    """RunCorticalTiles.run() launched generate_sulcal_regions.py."""


def _run_until_stage2(command, **_kwargs):
    """Stand-in for execute_command: run the setup commands, stop at the launch of stage 2."""
    if any("generate_sulcal_regions" in str(part) for part in command):
        raise _Stage2Reached
    return subprocess.run(command, check=False).returncode


def _run_stage2_from_checkout(input_dir, output_dir):
    """Run RunCorticalTiles.run() in this process up to the launch of generate_sulcal_regions.py."""
    script = RunCorticalTiles()
    script.parse_args([str(input_dir), str(output_dir), "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton"])
    script.execute_command = _run_until_stage2
    with pytest.raises(_Stage2Reached):
        script.run()


def _run_probe(site, probe, *args):
    """Run a probe in a child Python that imports champollion_pipeline from the unpacked wheel only."""
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", probe, *map(str, args)],
        cwd=site.parent,
        env={**os.environ, "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        check=False,
    )


def _probe_report(proc, site):
    """Return the JSON a probe printed last, once the probe ended well and ran the wheel's code."""
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-2000:]}"
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(report["module"]).resolve().is_relative_to(site.resolve()), report["module"]
    return report


def _assert_names_packaged_table(value, site_resources):
    """The config value is the absolute path of the region table in ``site_resources``."""
    assert Path(value).is_absolute(), f"brain_regions_json is not an absolute path: {value!r}"
    assert Path(value).resolve() == (site_resources / REGION_TABLE).resolve()
    assert Path(value).is_file()


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory):
    """Build a wheel from a staged copy of pyproject.toml and src/, leaving the checkout untouched."""
    stage = tmp_path_factory.mktemp("wheel_stage")
    shutil.copy(REPO_ROOT / "pyproject.toml", stage)
    shutil.copytree(REPO_ROOT / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", BUILD_WHEEL, "dist"],
        cwd=stage,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    return next((stage / "dist").glob("*.whl"))


@pytest.fixture(scope="module")
def installed_site(wheel_path, tmp_path_factory):
    """Unpack the wheel into an empty directory, the way an installer lays it out in site-packages."""
    site = tmp_path_factory.mktemp("installed") / "site-packages"
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(site)
    return site


@pytest.mark.parametrize("name", RESOURCE_DIGESTS)
def test_wheel_contains_resource(wheel_path, name):
    """The built wheel carries the file inside the package, so an installed package has it."""
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
    data_files = [entry for entry in names if not entry.endswith(".py") and ".dist-info/" not in entry]
    assert f"champollion_pipeline/resources/{name}" in names, f"non-Python files in the wheel: {data_files}"


@pytest.mark.parametrize("name", RESOURCE_DIGESTS)
def test_resource_bytes_unchanged(name):
    """The move changes no byte, so the crops stay the same."""
    path = RESOURCES_DIR / name
    assert path.is_file(), f"{path} does not exist"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RESOURCE_DIGESTS[name]


@pytest.mark.parametrize("name", RESOURCE_DIGESTS)
def test_repo_root_holds_no_copy(name):
    """A second copy at the root could drift from the packaged one."""
    assert not (REPO_ROOT / name).exists(), f"{name} is still at the repository root"


def test_generate_masks_reads_packaged_region_table(installed_site, tmp_path):
    """From an unpacked wheel, GenerateMasks.run() finds the region table and reads its sulci."""
    report = _probe_report(_run_probe(installed_site, MASKS_PROBE, tmp_path), installed_site)
    assert report["sulci"] == ["S.C.", "S.C.sylvian."]


def test_config_generation_copies_packaged_reference_yaml(installed_site, tmp_path):
    """From an unpacked wheel, GenerateChampollionConfig.run() writes the dataset reference.yaml."""
    crop_path = tmp_path / "data" / DATASET / "derivatives" / DERIVATIVES_FOLDER / "crops" / "canonical_25" / "2mm"
    crop_path.mkdir(parents=True)
    configs_root = tmp_path / "configs"
    proc = _run_probe(installed_site, CONFIG_PROBE, crop_path, configs_root, tmp_path / "champollion_V1", DATASET)
    _probe_report(proc, installed_site)
    written = configs_root / "dataset" / DATASET / REFERENCE_YAML
    assert written.is_file(), f"{written} was not written"
    assert f"{DATASET}/derivatives/{DERIVATIVES_FOLDER}/crops/" in written.read_text()


def test_stage2_copies_packaged_template(installed_site, tmp_path):
    """From an unpacked wheel, RunCorticalTiles.run() puts the template in an output directory without one."""
    input_dir, output_dir = stage_dirs(tmp_path)
    config_path = output_dir / TEMPLATE
    _probe_report(_run_probe(installed_site, STAGE2_PROBE, input_dir, output_dir), installed_site)
    assert config_path.is_file(), f"{config_path} was not written"
    assert json.loads(config_path.read_text())["skel_seed"] == 42


def test_stage2_config_names_packaged_table_from_wheel(installed_site, tmp_path):
    """From an unpacked wheel, the derivatives config names the wheel's region table."""
    input_dir, output_dir = stage_dirs(tmp_path)
    config_path = output_dir / TEMPLATE
    _probe_report(_run_probe(installed_site, STAGE2_PROBE, input_dir, output_dir), installed_site)
    assert config_path.is_file(), f"{config_path} was not written"
    value = json.loads(config_path.read_text())["brain_regions_json"]
    _assert_names_packaged_table(value, installed_site / "champollion_pipeline" / "resources")


def test_stage2_config_names_packaged_table_from_checkout(tmp_path):
    """From the checkout, the template's "$local" does not reach stage 2 in a freshly copied config."""
    input_dir, output_dir = stage_dirs(tmp_path)
    config_path = output_dir / TEMPLATE
    _run_stage2_from_checkout(input_dir, output_dir)
    _assert_names_packaged_table(json.loads(config_path.read_text())["brain_regions_json"], RESOURCES_DIR)


@pytest.mark.parametrize(
    "existing",
    [
        pytest.param("$local", id="local"),
        pytest.param("/old/install/sulci_regions_champollion_V1.json", id="stale-install"),
    ],
)
def test_stage2_config_replaces_existing_value(tmp_path, existing):
    """A derivatives config left by an earlier run gets the path too: run() copies the template only when absent."""
    input_dir, output_dir = stage_dirs(tmp_path)
    config_path = output_dir / TEMPLATE
    config_path.write_text(json.dumps({"brain_regions_json": existing}))
    _run_stage2_from_checkout(input_dir, output_dir)
    _assert_names_packaged_table(json.loads(config_path.read_text())["brain_regions_json"], RESOURCES_DIR)

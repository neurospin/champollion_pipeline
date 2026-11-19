#!/usr/bin/env python3
"""
A read-only install: stages 2 and 3 keep working when the packaged resources grant no write permission.

DEF-ROINSTALL-BDRABCZUK-2EBB0B54060C: `cp` keeps the mode of the packaged file, so opening the copy with "w" raises
PermissionError [Errno 13] (IXI run on rosette, WIP ed4b833).

Nothing here mocks the copy or the write. Each test lays out a real copy of the package whose files really refuse
writing, and runs the stage from it in a child Python that reaches only that copy, as an install in site-packages
would. The probes come from tests/test_packaged_resources.py: stage 2 stops at the launch of generate_sulcal_regions.py,
stage 3 stubs only its crop QC, because no crop is on disk.

Root ignores file modes, so the tests that rely on the refused write are guarded for root; the tests on the mode of
the copy read the mode bits, and the leftover tests read its content too, so they hold for root.

The tests of a run stopping before the copy site run in this process, each stage only as far as that stop. Stage 3 runs
its real input QC there, so this file overrides the conftest fixture stubbing it; the probes stub it themselves.

Requirements:
- REQ-ROINSTALL-BDRABCZUK-AA50E19970B9: stage 2 writes its settings into its copy of the template.
- REQ-ROINSTALL-BDRABCZUK-A01AFB93B915: the file owner can write the stage 2 copy.
- REQ-ROINSTALL-BDRABCZUK-04737B848EA6: stage 3 writes the crop path into its copy of reference.yaml.
- REQ-ROINSTALL-BDRABCZUK-00E6E91AB0ED: the file owner can write the stage 3 copy.
- REQ-ROINSTALL-BDRABCZUK-8A9A6077D3FE: stage 2 replaces a read-only copy left by an earlier failed run, only after
  _preflight_check returns True.
- REQ-ROINSTALL-BDRABCZUK-AF2473682939: stage 3 replaces a read-only copy left by an earlier failed run, only after its
  input QC passes.
- REQ-ROINSTALL-BDRABCZUK-F2F91B184FAC: stage 2 keeps a writable copy left by an earlier run.
"""

import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.generate_champollion_config import FAILED_STATUS, GenerateChampollionConfig
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.stage2_layout import (
    GRAPH_DIR,
    SUBJECT,
    add_crop_marker,
    add_crops,
    parsed_script,
    stage_dirs,
    stage_dirs_on_template,
)
from tests.test_packaged_resources import (
    CONFIG_PROBE,
    DATASET,
    REFERENCE_YAML,
    REPO_ROOT,
    STAGE2_PROBE,
    TEMPLATE,
    _probe_report,
    _run_probe,
)

pytestmark = pytest.mark.integration  # noqa: V107

PACKAGE_SOURCE = REPO_ROOT / "src" / "champollion_pipeline"
READ_ONLY_DIR_MODE = 0o555
READ_ONLY_FILE_MODES = {
    "read-for-all": 0o444,
    "read-and-execute-for-all": 0o555,
    "read-for-owner-only": 0o400,
}
WRITABLE_FILE_MODES = {
    "write-for-owner": 0o644,
    "write-for-owner-only": 0o600,
}

EARLIER_SKEL_SEED = 7
STALE_REFERENCE = "left_by: an earlier failed run\n"

STAGE2_INPUT_QC_FAILED = 1
STAGE2_OUTPUT_QC_PASSED = 0
MARKED_REGION = "S.C.-sylv."

needs_refused_write = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores file modes: the write under test is never refused"
)
each_read_only_mode = pytest.mark.parametrize("mode", READ_ONLY_FILE_MODES.values(), ids=READ_ONLY_FILE_MODES.keys())


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


@dataclass(frozen=True)
class ReadOnlySite:
    """A site-packages folder holding a read-only copy of the package."""

    path: Path
    file_mode: int

    def resource(self, name):
        """Path of the packaged resource ``name``."""
        return self.path / "champollion_pipeline" / "resources" / name

    def assert_resource_mode_kept(self, name):
        """The stage worked on its copy: the packaged file keeps the mode it had before the run."""
        assert stat.S_IMODE(self.resource(name).stat().st_mode) == self.file_mode


def _set_tree_modes(root, file_mode, dir_mode):
    """Give every file under ``root`` ``file_mode`` and every directory ``dir_mode``, ``root`` included."""
    for folder, _subfolders, files in os.walk(root):
        for name in files:
            os.chmod(Path(folder) / name, file_mode)
    for folder, _subfolders, _files in os.walk(root, topdown=False):
        os.chmod(folder, dir_mode)


def _unlock_tree(root):
    """Open the directories under ``root`` again, so pytest can delete its temporary folder."""
    os.chmod(root, stat.S_IRWXU)
    for folder, subfolders, _files in os.walk(root):
        for name in subfolders:
            os.chmod(Path(folder) / name, stat.S_IRWXU)


def _assert_write_refused(path):
    """The test is only worth something if this file really refuses writing to this process."""
    if os.geteuid() != 0:
        with pytest.raises(PermissionError), path.open("a"):
            pass


def _config_crop_path(root):
    """Create the crop folder of the dataset under ``root`` and return it."""
    crop_path = root / "data" / DATASET / "derivatives" / DERIVATIVES_FOLDER / "crops" / DEFAULT_MASKS_VERSION / "2mm"
    crop_path.mkdir(parents=True)
    return crop_path


def _leave_read_only_copy(copy, text, mode):
    """Put the read-only ``copy`` an earlier failed run would have left, holding ``text``."""
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text)
    copy.chmod(mode)
    _assert_write_refused(copy)


def _run_stage2(site, root, leftover=None):
    """Run stage 2 from ``site``, over a read-only ``leftover`` text when given.

    Return the child process, the input folder and the path of the working copy.
    """
    input_dir, output_dir = stage_dirs(root)
    copy = output_dir / TEMPLATE
    if leftover is not None:
        _leave_read_only_copy(copy, leftover, site.file_mode)
    return _run_probe(site.path, STAGE2_PROBE, input_dir, output_dir), input_dir, copy


def _run_config_generation(site, root, leftover=None):
    """Run stage 3 from ``site``, over a read-only ``leftover`` text when given.

    Return the child process and the path of the working copy.
    """
    configs_root = root / "configs"
    copy = configs_root / "dataset" / DATASET / REFERENCE_YAML
    if leftover is not None:
        _leave_read_only_copy(copy, leftover, site.file_mode)
    proc = _run_probe(site.path, CONFIG_PROBE, _config_crop_path(root), configs_root, root / "champollion_V1", DATASET)
    return proc, copy


def _stage2_over_read_only_copy(root, mode):
    """Lay stage 2 folders over a read-only copy of the template; return them, the copy and its text."""
    input_dir, output_dir, copy = stage_dirs_on_template(root, {"skel_seed": EARLIER_SKEL_SEED})
    text = copy.read_text()
    _leave_read_only_copy(copy, text, mode)
    return input_dir, output_dir, copy, text


def _run_stage2_to_its_stop(input_dir, output_dir):
    """Run stage 2 in this process and return its result; the run must stop before any command."""
    script = parsed_script(input_dir, output_dir)
    with patch.object(script, "execute_command", return_value=0) as mock_exec:
        result = script.run()
    mock_exec.assert_not_called()
    return result


def _assert_left_as_found(copy, text, mode):
    """``copy`` still holds ``text`` and still has ``mode``."""
    assert copy.read_text() == text
    assert stat.S_IMODE(copy.stat().st_mode) == mode


def _assert_owner_can_write(copy, proc):
    """``copy`` exists and its mode bits grant the owner write permission."""
    assert copy.is_file(), f"{copy} was not written:\n{proc.stderr[-2000:]}"
    mode = stat.S_IMODE(copy.stat().st_mode)
    assert mode & stat.S_IWUSR, f"{copy} has mode {oct(mode)}: its owner cannot write it"


@pytest.fixture(scope="module", params=READ_ONLY_FILE_MODES.values(), ids=READ_ONLY_FILE_MODES.keys())
def read_only_site(request, tmp_path_factory):
    """Copy the package of this checkout into an empty site-packages folder, then lock every file and directory."""
    site = tmp_path_factory.mktemp("read_only_install") / "site-packages"
    shutil.copytree(
        PACKAGE_SOURCE, site / "champollion_pipeline", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    _set_tree_modes(site, request.param, READ_ONLY_DIR_MODE)
    locked = ReadOnlySite(site, request.param)
    _assert_write_refused(locked.resource(TEMPLATE))
    _assert_write_refused(locked.resource(REFERENCE_YAML))
    yield locked
    _unlock_tree(site)


@needs_refused_write
def test_stage2_writes_settings_from_read_only_package(read_only_site, tmp_path):
    """Stage 2 reaches the crop launch with the input folder of the run in its copy of the template."""
    proc, input_dir, copy = _run_stage2(read_only_site, tmp_path)
    _probe_report(proc, read_only_site.path)
    assert json.loads(copy.read_text())["graphs_dir"] == os.path.abspath(input_dir)
    read_only_site.assert_resource_mode_kept(TEMPLATE)


def test_stage2_copy_is_owner_writable(read_only_site, tmp_path):
    """The stage 2 copy does not inherit the read-only mode of the packaged template."""
    proc, _input_dir, copy = _run_stage2(read_only_site, tmp_path)
    _assert_owner_can_write(copy, proc)


@needs_refused_write
def test_config_generation_writes_crop_path_from_read_only_package(read_only_site, tmp_path):
    """Stage 3 ends well with the crop path of the run in its copy of reference.yaml."""
    proc, copy = _run_config_generation(read_only_site, tmp_path)
    _probe_report(proc, read_only_site.path)
    text = copy.read_text()
    assert f"{DATASET}/derivatives/{DERIVATIVES_FOLDER}/crops/" in text
    assert "TESTXX" not in text
    read_only_site.assert_resource_mode_kept(REFERENCE_YAML)


def test_config_generation_copy_is_owner_writable(read_only_site, tmp_path):
    """The stage 3 copy does not inherit the read-only mode of the packaged reference.yaml."""
    proc, copy = _run_config_generation(read_only_site, tmp_path)
    _assert_owner_can_write(copy, proc)


def test_stage2_replaces_read_only_leftover(read_only_site, tmp_path):
    """A read-only copy left by an earlier failed run gives way to the packaged template, then to the run's settings."""
    packaged = json.loads(read_only_site.resource(TEMPLATE).read_text())
    leftover = json.dumps({**packaged, "skel_seed": EARLIER_SKEL_SEED})
    proc, input_dir, copy = _run_stage2(read_only_site, tmp_path, leftover=leftover)
    _probe_report(proc, read_only_site.path)
    config = json.loads(copy.read_text())
    assert config["skel_seed"] == packaged["skel_seed"]
    assert config["graphs_dir"] == os.path.abspath(input_dir)
    _assert_owner_can_write(copy, proc)


def test_config_generation_replaces_read_only_leftover(read_only_site, tmp_path):
    """A read-only copy left by an earlier failed run gives way to the packaged reference.yaml."""
    proc, copy = _run_config_generation(read_only_site, tmp_path, leftover=STALE_REFERENCE)
    _probe_report(proc, read_only_site.path)
    text = copy.read_text()
    assert STALE_REFERENCE not in text
    assert f"{DATASET}/derivatives/{DERIVATIVES_FOLDER}/crops/" in text
    _assert_owner_can_write(copy, proc)


@pytest.mark.parametrize("mode", WRITABLE_FILE_MODES.values(), ids=WRITABLE_FILE_MODES.keys())
def test_stage2_keeps_writable_copy(read_only_site, tmp_path, mode):
    """A copy its owner can write stays: run() sets its own settings there, the others keep their value."""
    input_dir, output_dir, copy = stage_dirs_on_template(tmp_path, {"skel_seed": EARLIER_SKEL_SEED})
    copy.chmod(mode)
    _probe_report(_run_probe(read_only_site.path, STAGE2_PROBE, input_dir, output_dir), read_only_site.path)
    config = json.loads(copy.read_text())
    assert config["skel_seed"] == EARLIER_SKEL_SEED
    assert config["graphs_dir"] == os.path.abspath(input_dir)


@each_read_only_mode
def test_stage2_input_qc_failure_leaves_read_only_copy(tmp_path, mode):
    """No subject passes the input QC: stage 2 stops before the copy site, so the copy stays as found."""
    input_dir, output_dir, copy, text = _stage2_over_read_only_copy(tmp_path, mode)
    shutil.rmtree(input_dir / SUBJECT / GRAPH_DIR)
    assert _run_stage2_to_its_stop(input_dir, output_dir) == STAGE2_INPUT_QC_FAILED
    _assert_left_as_found(copy, text, mode)


@each_read_only_mode
def test_stage2_marked_crops_leave_read_only_copy(tmp_path, mode):
    """Marked crops make _preflight_check return False: stage 2 stops before the copy site, so the copy stays."""
    input_dir, output_dir, copy, text = _stage2_over_read_only_copy(tmp_path, mode)
    crops_dir = compute_versioned_crops_dir(str(output_dir / DERIVATIVES_FOLDER), DEFAULT_MASKS_VERSION)
    add_crops(crops_dir, [SUBJECT], [MARKED_REGION])
    add_crop_marker(crops_dir)
    assert _run_stage2_to_its_stop(input_dir, output_dir) == STAGE2_OUTPUT_QC_PASSED
    _assert_left_as_found(copy, text, mode)


@each_read_only_mode
def test_config_generation_input_qc_failure_leaves_read_only_copy(tmp_path, mode):
    """No subject passes the input QC: stage 3 stops before the copy site, so the copy stays as found."""
    configs_root = tmp_path / "configs"
    copy = configs_root / "dataset" / DATASET / REFERENCE_YAML
    _leave_read_only_copy(copy, STALE_REFERENCE, mode)
    script = GenerateChampollionConfig()
    script.parse_args(
        [
            str(_config_crop_path(tmp_path)),
            "--dataset",
            DATASET,
            "--output",
            str(configs_root),
            "--champollion_loc",
            str(tmp_path / "champollion_V1"),
        ]
    )
    assert script.run() == FAILED_STATUS
    _assert_left_as_found(copy, STALE_REFERENCE, mode)

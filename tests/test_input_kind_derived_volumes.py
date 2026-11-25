"""Morphologist derived volumes are never T1 files: the T1 file list leaves them out, an input holding only them stops
with exit status 2 before stage 1, and derived volumes beside T1 files give one WARNING.

Defect DEF-ARGDISC-BDRABCZUK-49CD772113A6; behaviour TASK-ARGDISC-BDRABCZUK-8D7C8EEA6AD8 (epic
TASK-ARGDISC-BDRABCZUK-D91B3C7ACD31). Requirements: one per test below (alm links on the Task root).
Trees: names only, from tests/morphologist_layout_fixtures.py or built here; no file holds data.
Harness: tests/main_input_autodetect_harness.py.
"""

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from champollion_pipeline.utils.input_kind import find_t1_files
from tests.main_input_autodetect_harness import (
    STAGE1,
    _batch,
    _error_messages,
    _main_module_under_test,
    _run_stages,
    _stop_config,
)
from tests.morphologist_layout_fixtures import (
    CLASSIC_SUBJECTS,
    CLI_SUBJECTS,
    build_classic_tree,
    build_segmentation_volumes_tree,
    build_suffix_volumes_tree,
)

EXTENSIONS = (".nii.gz", ".nii")
PLAIN_SUFFIXES = ("brain", "skull_stripped", "nobias", "hfiltered", "variance", "edges", "voronoi", "whiteridge")
HEMI_VOLUMES = ("skeleton", "cortex", "grey_white", "gw_interface", "roots", "csf")
DERIVED_SUFFIXES = (
    *PLAIN_SUFFIXES,
    *HEMI_VOLUMES,
    *(f"hemi-{hemi}_{volume}" for hemi in ("L", "R") for volume in HEMI_VOLUMES),
)
DERIVED_PREFIXES = (
    *(f"{name}_" for name in PLAIN_SUFFIXES),
    *(f"{hemi}{volume}_" for hemi in ("L", "R") for volume in HEMI_VOLUMES),
)
RAW_SCAN = "IXI001"
RAW_T1 = f"{RAW_SCAN}.nii.gz"
CLI_SEGMENTATION = Path(CLI_SUBJECTS[0], "ses-1", "run-0", "ana-0", "segmentation")
CLASSIC_SEGMENTATION = Path(CLASSIC_SUBJECTS[0], "t1mri", "default_acquisition", "default_analysis", "segmentation")
MORPHOLOGIST_OUTPUT = "Morphologist output"
INPUT_PATH_OPTION = "--input-path"
HEADER_READERS = (
    "champollion_pipeline.utils.nifti_check.check_nifti_input",
    "champollion_pipeline.generate_morphologist_graphs.check_nifti_input",
)


def _touch(root: Path, *names: str) -> Path:
    """Create each relative name below root as an empty file; return root."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def _cli_segmentation_root(tmp_path: Path) -> Path:
    return build_segmentation_volumes_tree(tmp_path / "in")


def _cli_segmentation_folder(tmp_path: Path) -> Path:
    return build_segmentation_volumes_tree(tmp_path / "in") / CLI_SEGMENTATION


def _suffix_root(tmp_path: Path) -> Path:
    return build_suffix_volumes_tree(tmp_path / "in")


def _suffix_subject(tmp_path: Path) -> Path:
    return build_suffix_volumes_tree(tmp_path / "in") / CLI_SUBJECTS[0]


def _classic_segmentation_folder(tmp_path: Path) -> Path:
    return build_classic_tree(tmp_path / "in") / CLASSIC_SEGMENTATION


def _plain_segmentation_folder(tmp_path: Path) -> Path:
    return _touch(tmp_path / "in" / "segmentation", "scan01.nii.gz", "scan02.nii")


DERIVED_ONLY_INPUTS: dict[str, Callable[[Path], Path]] = {
    "cli_segmentation_root": _cli_segmentation_root,
    "cli_segmentation_folder": _cli_segmentation_folder,
    "suffix_root": _suffix_root,
    "suffix_subject": _suffix_subject,
    "classic_segmentation_folder": _classic_segmentation_folder,
}


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    return _main_module_under_test(monkeypatch, tmp_path)


@pytest.fixture
def header_reads(monkeypatch) -> list[str]:
    """Replace the NIfTI header reader where main.py and stage 1 reach it; return the paths it was asked to read."""
    reads: list[str] = []

    def record_read(path, *_args, **_kwargs):
        reads.append(str(path))
        return ()

    for target in HEADER_READERS:
        monkeypatch.setattr(target, record_read)
    return reads


def _derived_names(input_path: Path) -> list[str]:
    """File names of the volumes below input_path, each a derived volume in the trees of DERIVED_ONLY_INPUTS."""
    return sorted(path.name for path in input_path.rglob("*.nii*"))


# ====================== The T1 file list ======================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build", "relative_input", "expected"),
    [
        (lambda tmp: _touch(tmp / "in", RAW_T1, "segmentation/scan01.nii.gz", "segmentation/scan02.nii"), "", [RAW_T1]),
        (_plain_segmentation_folder, None, []),
        (_cli_segmentation_folder, None, []),
        (_classic_segmentation_folder, None, []),
    ],
    ids=["segmentation_below_input", "input_is_segmentation", "cli_segmentation", "classic_segmentation"],
)
def test_volume_in_segmentation_folder_is_left_out_of_t1_list(tmp_path, build, relative_input, expected):
    # REQ: a .nii.gz or .nii file whose own folder is named segmentation is not in the T1 file list
    built = build(tmp_path)
    input_path = built / relative_input if relative_input else built
    assert find_t1_files(str(input_path)) == expected


@pytest.mark.unit
@pytest.mark.parametrize("extension", EXTENSIONS)
@pytest.mark.parametrize("suffix", DERIVED_SUFFIXES)
def test_volume_with_derived_suffix_is_left_out_of_t1_list(tmp_path, suffix, extension):
    # REQ: a name ending with a morphologist-cli derived-volume suffix is not in the T1 file list; the raw T1 stays
    input_path = _touch(tmp_path / "subject", RAW_T1, f"{RAW_SCAN}_{suffix}{extension}")
    assert find_t1_files(str(input_path)) == [RAW_T1]


@pytest.mark.unit
@pytest.mark.parametrize("extension", EXTENSIONS)
@pytest.mark.parametrize("prefix", DERIVED_PREFIXES)
def test_volume_with_derived_prefix_is_left_out_of_t1_list(tmp_path, prefix, extension):
    # REQ: a name starting with a classic BrainVISA derived-volume prefix is not in the T1 file list; the raw T1 stays
    input_path = _touch(tmp_path / "subject", RAW_T1, f"{prefix}{RAW_SCAN}{extension}")
    assert find_t1_files(str(input_path)) == [RAW_T1]


@pytest.mark.unit
def test_suffix_volumes_fixture_subject_has_empty_t1_list(tmp_path):
    # REQ (name rule) on the fixture tree: the subject folder of the suffix tree, read as the input, lists no T1 file
    assert find_t1_files(str(_suffix_subject(tmp_path))) == []


# ====================== The stop on derived volumes only ======================


@pytest.mark.unit
@pytest.mark.parametrize("dataset", sorted(DERIVED_ONLY_INPUTS))
def test_only_derived_volumes_end_with_exit_2_before_stage_1(main_module, monkeypatch, tmp_path, header_reads, dataset):
    # REQ: derived volumes as the only .nii.gz/.nii files, no .arg: exit status 2, no stage runs, no header read
    _batch(monkeypatch)
    input_path = DERIVED_ONLY_INPUTS[dataset](tmp_path)
    status, argv = _run_stages(main_module, monkeypatch, _stop_config(main_module, tmp_path, input_path))
    assert status == 2, f"main() exit status {status}, expected 2"
    assert not argv, f"stages ran before the stop: {sorted(argv)}"
    assert not header_reads, f"NIfTI headers read before the stop: {header_reads[:3]}"


@pytest.mark.unit
@pytest.mark.parametrize("dataset", sorted(DERIVED_ONLY_INPUTS))
def test_only_derived_volumes_log_one_error_naming_layout_and_option(
    main_module, monkeypatch, tmp_path, pipeline_caplog, dataset
):
    # REQ: one ERROR record naming the input folder, the Morphologist output layout, one derived volume, --input-path
    _batch(monkeypatch)
    input_path = DERIVED_ONLY_INPUTS[dataset](tmp_path)
    _run_stages(main_module, monkeypatch, _stop_config(main_module, tmp_path, input_path))
    messages = _error_messages(pipeline_caplog)
    assert len(messages) == 1, f"{len(messages)} ERROR records, expected 1: {messages}"
    message = messages[0]
    for text in (str(input_path), MORPHOLOGIST_OUTPUT, INPUT_PATH_OPTION):
        assert text in message, f"{text!r} not in the ERROR record: {message}"
    derived = _derived_names(input_path)
    assert any(name in message for name in derived), f"no derived volume of {derived} named in: {message}"


# ====================== Derived volumes beside T1 files ======================


MIXED_DERIVED = (
    f"{RAW_SCAN}/nobias_{RAW_SCAN}.nii.gz",
    f"{RAW_SCAN}/{RAW_SCAN}_hfiltered.nii.gz",
    f"{RAW_SCAN}/segmentation/brain_{RAW_SCAN}.nii.gz",
    f"{RAW_SCAN}/segmentation/Lskeleton_{RAW_SCAN}.nii.gz",
    f"{RAW_SCAN}/segmentation/scan.nii",
)


@pytest.mark.unit
def test_derived_volumes_beside_t1_files_log_one_warning_with_count(
    main_module, monkeypatch, tmp_path, pipeline_caplog, header_reads
):
    # REQ: no .arg, T1 files and derived volumes: one WARNING record giving the number of derived volumes left out
    _batch(monkeypatch)
    input_path = _touch(tmp_path / "in", f"{RAW_SCAN}/{RAW_T1}", *MIXED_DERIVED)
    status, argv = _run_stages(main_module, monkeypatch, _stop_config(main_module, tmp_path, input_path))
    assert STAGE1 in argv, f"stage 1 did not run on the T1 file (exit status {status})"
    count = re.compile(rf"\b{len(MIXED_DERIVED)}\b")
    warnings = [
        record.getMessage().replace(str(input_path), "")
        for record in pipeline_caplog.records
        if record.levelno == logging.WARNING
    ]
    counted = [message for message in warnings if count.search(message)]
    assert len(counted) == 1, f"{len(counted)} WARNING records give the count {len(MIXED_DERIVED)}: {warnings}"

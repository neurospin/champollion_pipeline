"""Tests for the per-scan producer runners of a streaming run (TASK-STREAMING-BDRABCZUK-976F253B3EF1).

Morphologist and cortical_tiles never run: the script classes are replaced by
recording stand-ins, and every input is a synthetic directory tree.

Requirements:
- REQ-STREAMING-BDRABCZUK-46989274D669 single-scan directory
- REQ-STREAMING-BDRABCZUK-2E2105DAB2D5 per-scan tiles directory
- REQ-STREAMING-BDRABCZUK-5D6608194C0A per-scan crops root
- REQ-STREAMING-BDRABCZUK-12550265A4F0 tiles input is the single-scan directory
- REQ-STREAMING-BDRABCZUK-2C273B5F35B8 tiles output is the per-scan tiles directory
- REQ-STREAMING-BDRABCZUK-F6FF6E223B76 tiles base options
- REQ-STREAMING-BDRABCZUK-57C480955842 tiles --bids
- REQ-STREAMING-BDRABCZUK-F40CA50E003B tiles --regions
- REQ-STREAMING-BDRABCZUK-B2E6D8F46CBA tiles --sk_qc_path
- REQ-STREAMING-BDRABCZUK-864832B0CA24 tiles --skel-seed
- REQ-STREAMING-BDRABCZUK-4108D564774F tiles non-zero code raises
- REQ-STREAMING-BDRABCZUK-A4F61AE8D57D tiles removes the single-scan directory
- REQ-STREAMING-BDRABCZUK-CC2A9B2BB067 tiles missing scan input
- REQ-STREAMING-BDRABCZUK-E572A6C64A7D tiles QC warning on missing crops
- REQ-STREAMING-BDRABCZUK-A6BD1A4FB2E7 Morphologist single-scan input
- REQ-STREAMING-BDRABCZUK-FFBF0819F5CA Morphologist arguments
- REQ-STREAMING-BDRABCZUK-F1956D4BFE22 Morphologist non-zero code raises
- REQ-STREAMING-BDRABCZUK-D5A8352E93D7 Morphologist removes the single-scan directory
- REQ-STREAMING-BDRABCZUK-000ABB1CB055 Morphologist missing raw scan
- REQ-STREAMING-BDRABCZUK-686264DAB6F3 stages module leaves sys.path untouched
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.streaming import stages
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

STAGES_MODULE = "champollion_pipeline.streaming.stages"
PATH_TO_GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.3/base"
PATH_SK_WITH_HULL = "t1mri/default_acquisition/default_analysis/segmentation/mesh"


@dataclass(frozen=True)
class _Scan:
    """Synthetic stand-in for ScanId: path_prefix and str() as the ScanId requirements state."""

    subject: str
    session: str | None = None

    @property
    def path_prefix(self) -> str:
        return self.subject if self.session is None else f"{self.subject}/{self.session}"

    def __str__(self) -> str:
        return self.subject if self.session is None else f"{self.subject}_{self.session}"


class _ScriptRecorder:
    """Records how a runner drives a stage script, and what its input directory holds at run time."""

    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.argv: list[str] | None = None
        self.runs = 0
        self.input_entries: set[tuple[str, bool, str | None]] | None = None
        self.output_existed: bool | None = None

    def script_class(self) -> type:
        recorder = self

        class _RecordingScript:
            def parse_args(self, args: list[str] | None = None) -> None:
                recorder.argv = list(args or [])

            def run(self) -> int:
                recorder.runs += 1
                recorder.input_entries = _tree_entries(recorder.argv[0])
                recorder.output_existed = os.path.isdir(recorder.argv[1])
                return recorder.return_code

        return _RecordingScript


def _tree_entries(root: str) -> set[tuple[str, bool, str | None]]:
    """(relative path, is a symlink, link target resolved) for every entry below root; links are not followed."""
    entries = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            is_link = os.path.islink(path)
            target = os.path.realpath(path) if is_link else None
            entries.add((os.path.relpath(path, root), is_link, target))
    return entries


@pytest.fixture
def subjects_dir(tmp_path: Path) -> Path:
    """Synthetic Morphologist subjects directory: two subjects, the first with two sessions."""
    root = tmp_path / "subjects"
    for rel in ("sub-01/ses-1", "sub-01/ses-2", "sub-02/ses-1", "subj01"):
        (root / rel).mkdir(parents=True)
    return root


@pytest.fixture
def settings(tmp_path: Path, subjects_dir: Path) -> StreamingSettings:
    return StreamingSettings(
        subjects_dir=str(subjects_dir),
        output_dir=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        n_workers=2,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> DatasetConfig:
    morphologist = tmp_path / "morphologist"
    morphologist.mkdir()
    return DatasetConfig(
        name="synthetic",
        morphologist_graphs=str(morphologist),
        cortical_tiles_output=str(tmp_path / "derivatives"),
        path_to_graph=PATH_TO_GRAPH,
        path_sk_with_hull=PATH_SK_WITH_HULL,
        masks_version="synthetic_masks",
    )


@pytest.fixture
def raw_scan(tmp_path: Path) -> Path:
    """Synthetic raw T1 file next to another subject's file."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sub-02.nii.gz").write_bytes(b"synthetic")
    raw = raw_dir / "sub-01.nii.gz"
    raw.write_bytes(b"synthetic")
    return raw


@pytest.fixture
def tiles(monkeypatch: pytest.MonkeyPatch) -> _ScriptRecorder:
    recorder = _ScriptRecorder()
    monkeypatch.setattr(stages, "RunCorticalTiles", recorder.script_class())
    return recorder


@pytest.fixture
def morphologist(monkeypatch: pytest.MonkeyPatch) -> _ScriptRecorder:
    recorder = _ScriptRecorder()
    monkeypatch.setattr(stages, "GenerateMorphologistGraphs", recorder.script_class())
    return recorder


@pytest.fixture
def stages_records() -> list[logging.LogRecord]:
    """Records of the stages module logger, captured on the logger itself (propagation not needed)."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    stage_logger = logging.getLogger(STAGES_MODULE)
    handler = _ListHandler(level=logging.DEBUG)
    level = stage_logger.level
    stage_logger.addHandler(handler)
    stage_logger.setLevel(logging.DEBUG)
    yield records
    stage_logger.removeHandler(handler)
    stage_logger.setLevel(level)


def _tiles_namespace(argv: list[str]):
    """Parse the recorded argv with the real RunCorticalTiles parser."""
    return RunCorticalTiles().parse_args(argv)


# --- make_single_scan_dir -------------------------------------------------------------------------


def test_single_scan_dir_links_path_prefix(subjects_dir: Path) -> None:
    tmp = stages.make_single_scan_dir(str(subjects_dir), "subj01")
    try:
        assert _tree_entries(tmp.name) == {("subj01", True, os.path.realpath(subjects_dir / "subj01"))}
    finally:
        tmp.cleanup()


def test_single_scan_dir_links_session_only(subjects_dir: Path) -> None:
    tmp = stages.make_single_scan_dir(str(subjects_dir), "sub-01/ses-1")
    try:
        assert _tree_entries(tmp.name) == {
            ("sub-01", False, None),
            ("sub-01/ses-1", True, os.path.realpath(subjects_dir / "sub-01" / "ses-1")),
        }
    finally:
        tmp.cleanup()


# --- compute_scan_tiles_dir and find_scan_crops_root ----------------------------------------------


def test_compute_scan_tiles_dir() -> None:
    scan = _Scan("sub-01", "ses-1")
    assert stages.compute_scan_tiles_dir("/data/derivatives", scan) == os.path.join(
        "/data/derivatives", "scans", "sub-01_ses-1"
    )


def test_find_scan_crops_root_present(tmp_path: Path) -> None:
    crops = tmp_path / "scan" / f"cortical_tiles-{CORTICAL_TILES_VERSION}" / "crops" / "synthetic_masks" / "2mm"
    crops.mkdir(parents=True)
    assert stages.find_scan_crops_root(str(tmp_path / "scan"), "synthetic_masks") == str(crops)


def test_find_scan_crops_root_absent(tmp_path: Path) -> None:
    other_version = tmp_path / "scan" / f"cortical_tiles-{CORTICAL_TILES_VERSION}" / "crops" / "other" / "2mm"
    other_version.mkdir(parents=True)
    assert stages.find_scan_crops_root(str(tmp_path / "scan"), "synthetic_masks") is None


# --- run_cortical_tiles_for_scan: arguments -------------------------------------------------------


def test_tiles_input_is_single_scan_dir(tiles, settings, dataset, subjects_dir) -> None:
    stages.run_cortical_tiles_for_scan(_Scan("sub-01", "ses-1"), settings, dataset)
    assert tiles.runs == 1
    assert tiles.input_entries == {
        ("sub-01", False, None),
        ("sub-01/ses-1", True, os.path.realpath(subjects_dir / "sub-01" / "ses-1")),
    }


def test_tiles_output_is_existing_scan_tiles_dir(tiles, settings, dataset) -> None:
    scan = _Scan("sub-01", "ses-1")
    stages.run_cortical_tiles_for_scan(scan, settings, dataset)
    assert _tiles_namespace(tiles.argv).output == stages.compute_scan_tiles_dir(dataset.cortical_tiles_output, scan)
    assert tiles.output_existed is True


def test_tiles_base_options(tiles, settings, dataset) -> None:
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    args = _tiles_namespace(tiles.argv)
    assert args.path_to_graph == PATH_TO_GRAPH
    assert args.path_sk_with_hull == PATH_SK_WITH_HULL
    assert args.masks == "synthetic_masks"
    assert args.njobs == settings.njobs_per_worker


@pytest.mark.parametrize("bids", [True, False])
def test_tiles_bids_flag(tiles, settings, dataset, bids: bool) -> None:
    dataset.bids = bids
    stages.run_cortical_tiles_for_scan(_Scan("sub-01", "ses-1"), settings, dataset)
    assert ("--bids" in tiles.argv) is bids


def test_tiles_regions_forwarded(tiles, settings, dataset) -> None:
    dataset.regions = ["S.C.-sylv.", "F.I.P."]
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert _tiles_namespace(tiles.argv).regions == ["S.C.-sylv.", "F.I.P."]


def test_tiles_no_regions_when_empty(tiles, settings, dataset) -> None:
    dataset.regions = []
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert tiles.argv is not None
    assert "--regions" not in tiles.argv


def test_tiles_sk_qc_path_forwarded(tiles, settings, dataset) -> None:
    dataset.sk_qc_path = "/synthetic/qc.tsv"
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert _tiles_namespace(tiles.argv).sk_qc_path == "/synthetic/qc.tsv"


def test_tiles_no_sk_qc_path_when_empty(tiles, settings, dataset) -> None:
    dataset.sk_qc_path = ""
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert tiles.argv is not None
    assert not [arg for arg in tiles.argv if arg.startswith("--sk_qc_path")]


def test_tiles_skel_seed_forwarded(tiles, settings, dataset) -> None:
    dataset.skel_seed = 0
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert _tiles_namespace(tiles.argv).skel_seed == 0


def test_tiles_no_skel_seed_when_none(tiles, settings, dataset) -> None:
    dataset.skel_seed = None
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert tiles.argv is not None
    assert not [arg for arg in tiles.argv if arg.startswith("--skel-seed")]


# --- run_cortical_tiles_for_scan: failures, cleanup, input check and QC ---------------------------


def test_tiles_nonzero_code_raises(tiles, settings, dataset) -> None:
    tiles.return_code = 37
    with pytest.raises(RuntimeError) as excinfo:
        stages.run_cortical_tiles_for_scan(_Scan("sub-01", "ses-1"), settings, dataset)
    assert "sub-01_ses-1" in str(excinfo.value)
    assert re.search(r"\b37\b", str(excinfo.value))


def test_tiles_single_scan_dir_removed_after_success(tiles, settings, dataset) -> None:
    stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert tiles.argv is not None
    assert not os.path.exists(tiles.argv[0])


def test_tiles_single_scan_dir_removed_after_failure(tiles, settings, dataset) -> None:
    tiles.return_code = 1
    with pytest.raises(RuntimeError):
        stages.run_cortical_tiles_for_scan(_Scan("subj01"), settings, dataset)
    assert tiles.argv is not None
    assert not os.path.exists(tiles.argv[0])


def test_tiles_missing_scan_input_raises(tiles, settings, dataset, subjects_dir) -> None:
    scan = _Scan("sub-09", "ses-1")
    with pytest.raises(FileNotFoundError) as excinfo:
        stages.run_cortical_tiles_for_scan(scan, settings, dataset)
    assert str(subjects_dir / scan.path_prefix) in str(excinfo.value)
    assert tiles.runs == 0


def test_tiles_qc_missing_crops_warns_without_raising(tiles, settings, dataset, stages_records) -> None:
    result = stages.run_cortical_tiles_for_scan(_Scan("sub-01", "ses-1"), settings, dataset)
    assert result is None
    assert tiles.runs == 1
    warnings = [r for r in stages_records if r.levelno == logging.WARNING and "sub-01_ses-1" in r.getMessage()]
    assert warnings, [(r.levelname, r.getMessage()) for r in stages_records]


# --- run_morphologist_for_scan ---------------------------------------------------------------------


def test_morphologist_input_holds_only_scan_link(morphologist, settings, dataset, raw_scan) -> None:
    stages.run_morphologist_for_scan(str(raw_scan), settings, dataset)
    assert morphologist.runs == 1
    assert morphologist.input_entries == {("sub-01.nii.gz", True, os.path.realpath(raw_scan))}


@pytest.mark.parametrize("bids", [True, False])
def test_morphologist_arguments(morphologist, settings, dataset, raw_scan, bids: bool) -> None:
    dataset.bids = bids
    stages.run_morphologist_for_scan(str(raw_scan), settings, dataset)
    assert morphologist.argv is not None
    expected = [morphologist.argv[0], dataset.morphologist_graphs] + (["--bids"] if bids else [])
    assert morphologist.argv == expected
    assert morphologist.argv[0] != str(raw_scan.parent)


def test_morphologist_nonzero_code_raises(morphologist, settings, dataset, raw_scan) -> None:
    morphologist.return_code = 41
    with pytest.raises(RuntimeError) as excinfo:
        stages.run_morphologist_for_scan(str(raw_scan), settings, dataset)
    assert "sub-01.nii.gz" in str(excinfo.value)
    assert re.search(r"\b41\b", str(excinfo.value))


def test_morphologist_single_scan_dir_removed(morphologist, settings, dataset, raw_scan) -> None:
    morphologist.return_code = 1
    with pytest.raises(RuntimeError):
        stages.run_morphologist_for_scan(str(raw_scan), settings, dataset)
    assert morphologist.argv is not None
    assert not os.path.exists(morphologist.argv[0])
    assert raw_scan.is_file()


def test_morphologist_missing_raw_scan_raises(morphologist, settings, dataset, tmp_path: Path) -> None:
    missing = tmp_path / "raw" / "sub-09.nii.gz"
    with pytest.raises(FileNotFoundError) as excinfo:
        stages.run_morphologist_for_scan(str(missing), settings, dataset)
    assert str(missing) in str(excinfo.value)
    assert morphologist.runs == 0


# --- module source -----------------------------------------------------------------------------------


def test_stages_module_never_references_sys_path() -> None:
    source = Path(importlib.util.find_spec(STAGES_MODULE).origin).read_text()
    references = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    ]
    assert references == []

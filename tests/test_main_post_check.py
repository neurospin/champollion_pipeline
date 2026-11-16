"""Sequential post-stage output report in main.py (TASK-STREAMING-BDRABCZUK-EC9C2CF191D9).

After a sequential stage succeeds, PipelineOrchestrator.run() indexes the folder the stage's output_dir() names,
logs the build_output_report summary and saves the index beside the pre-stage one. The report is informational:
it never fails a stage (NTUI).

Each test goes through PipelineOrchestrator.run() on run_cortical_tiles (the stage whose output_dir() names a
folder), with the crop script mocked by a fake that writes the output files, and reads the records the user reads.
build_output_report is replaced, for the failure tests, wherever main.py may reach it from, so the tests hold
however main.py imports it.
"""

import importlib.util
import json
import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline import file_indexer
from champollion_pipeline.file_indexer import pipeline_checks
from champollion_pipeline.file_indexer.btree import BTree
from tests.log_assertions import OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

STAGE = "run_cortical_tiles"
POST_CHECK_PREFIX = f"Post-check {STAGE}:"
INDEX_FILE = f"index_post_{STAGE}.json"
TILE_FILES = ["crops/2mm/S.C.-sylv./mask/Rskeleton.nii.gz", "crops/2mm/S.C.-sylv./mask/Rskeleton_subject.csv", "a.json"]
REPORT_ERROR = "output folder vanished during the post-check"


@pytest.fixture
def main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_post_check_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(main_module, tmp_path):
    """Sequential run_cortical_tiles only, on an existing (empty) subjects folder, tiles under tmp_path/tiles."""
    subjects = tmp_path / "subjects"
    subjects.mkdir()
    pipeline_config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(tmp_path / "outputs"),
    )
    pipeline_config.stages = {STAGE: True}
    pipeline_config.log_to_file = False  # noqa: V101
    pipeline_config.log_dir = None  # noqa: V101
    pipeline_config.dataset.morphologist_graphs = str(subjects)  # noqa: V101
    pipeline_config.dataset.cortical_tiles_output = str(tmp_path / "tiles")  # noqa: V101
    return pipeline_config


class _FakeCropScript:
    """Replaces RunCorticalTiles.run: writes the tile files (when asked), notes the post-checks seen, returns a code."""

    def __init__(self, tiles: Path, capture, return_code: int = 0, write: bool = True):
        self.tiles = tiles
        self.capture = capture
        self.return_code = return_code
        self.write = write
        self.post_checks_seen_during_run: list[int] = []

    def __call__(self, *_args, **_kwargs) -> int:
        if self.write:
            for relative in TILE_FILES:
                path = self.tiles / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
        self.post_checks_seen_during_run.append(len(_post_checks(self.capture)))
        return self.return_code


class _RaisingReport:
    """Replaces build_output_report: counts its calls and raises."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise OSError(REPORT_ERROR)


def _post_checks(capture) -> list[str]:
    return [
        record.getMessage()
        for record in capture.records
        if record.levelno == OUTPUT and record.getMessage().startswith(POST_CHECK_PREFIX)
    ]


def _run(main_module, config, script: _FakeCropScript) -> int:
    with patch("champollion_pipeline.run_cortical_tiles.RunCorticalTiles.run", script):
        return main_module.PipelineOrchestrator(config).run()


def _install_raising_report(main_module, monkeypatch) -> _RaisingReport:
    """Replace build_output_report in every namespace main.py may call it through."""
    fake = _RaisingReport()
    monkeypatch.setattr(pipeline_checks, "build_output_report", fake)
    for namespace in (file_indexer, main_module):
        if hasattr(namespace, "build_output_report"):
            monkeypatch.setattr(namespace, "build_output_report", fake)
    return fake


def test_post_check_summary_logged_after_stage_script(main_module, config, tmp_path, pipeline_caplog):
    """REQ-STREAMING-BDRABCZUK-FE73FF5EDBFE.

    After the stage script returned, one 'Post-check run_cortical_tiles:' OUTPUT record on its folder.
    """
    script = _FakeCropScript(tmp_path / "tiles", pipeline_caplog)

    assert _run(main_module, config, script) == 0

    assert script.post_checks_seen_during_run == [0], "a Post-check record came before the stage script returned"
    post_checks = _post_checks(pipeline_caplog)
    assert len(post_checks) == 1, f"expected one {POST_CHECK_PREFIX!r} record, got {post_checks}"
    assert str(tmp_path / "tiles") in post_checks[0]
    assert re.search(rf"\b{len(TILE_FILES)}\b", post_checks[0]), "total file count of the tiles folder missing"


def test_post_check_index_written_under_outputs_path(main_module, config, tmp_path, pipeline_caplog):
    """REQ-STREAMING-BDRABCZUK-F27BE975046A.

    The output index of the tiles folder goes to <outputs_path>/index_post_run_cortical_tiles.json.
    """
    script = _FakeCropScript(tmp_path / "tiles", pipeline_caplog)

    assert _run(main_module, config, script) == 0

    index_path = Path(config.outputs_path) / INDEX_FILE
    assert index_path.is_file(), f"{INDEX_FILE} not written under outputs_path"
    tree = BTree.from_dict(json.loads(index_path.read_text()))
    assert sorted(key for key, _ in tree) == sorted(TILE_FILES)


def test_no_post_check_after_failed_stage(main_module, config, tmp_path, pipeline_caplog):
    """REQ-STREAMING-BDRABCZUK-38C9765FA68B.

    A failed stage gets no 'Post-check run_cortical_tiles:' record, though its folder holds files.
    """
    script = _FakeCropScript(tmp_path / "tiles", pipeline_caplog, return_code=1)

    assert _run(main_module, config, script) == 1

    assert _post_checks(pipeline_caplog) == []
    assert not (Path(config.outputs_path) / INDEX_FILE).exists()


def test_no_post_check_without_output_folder(main_module, config, tmp_path, pipeline_caplog):
    """REQ-STREAMING-BDRABCZUK-161DDE7DAE77.

    A successful stage whose output_dir() names no existing folder gets no Post-check record.
    """
    script = _FakeCropScript(tmp_path / "tiles", pipeline_caplog, write=False)

    assert _run(main_module, config, script) == 0

    assert not (tmp_path / "tiles").exists()
    assert _post_checks(pipeline_caplog) == []


def test_failing_output_report_keeps_stage_completed(main_module, config, tmp_path, pipeline_caplog, monkeypatch):
    """REQ-STREAMING-BDRABCZUK-7C1FBD187B4D.

    Build_output_report raising after the stage succeeded leaves the stage completed and the run at 0.
    """
    fake = _install_raising_report(main_module, monkeypatch)
    script = _FakeCropScript(tmp_path / "tiles", pipeline_caplog)

    exit_code = _run(main_module, config, script)

    assert fake.calls == 1, "PipelineOrchestrator.run() did not ask build_output_report for the post-check"
    assert exit_code == 0
    completed = [
        record.getMessage() for record in pipeline_caplog.records if record.getMessage().startswith("Completed stages:")
    ]
    assert completed == [f"Completed stages: {[STAGE]}"]


def test_failing_output_report_logs_warning(main_module, config, tmp_path, pipeline_caplog, monkeypatch):
    """REQ-STREAMING-BDRABCZUK-3CEBFA47EA81.

    Build_output_report raising after the stage succeeded gives one WARNING naming the stage and the error.
    """
    fake = _install_raising_report(main_module, monkeypatch)
    script = _FakeCropScript(tmp_path / "tiles", pipeline_caplog)

    _run(main_module, config, script)

    assert fake.calls == 1, "PipelineOrchestrator.run() did not ask build_output_report for the post-check"
    warnings = [
        record.getMessage()
        for record in pipeline_caplog.records
        if record.levelno == logging.WARNING and REPORT_ERROR in record.getMessage()
    ]
    assert len(warnings) == 1, f"expected one WARNING with {REPORT_ERROR!r}, got {warnings}"
    assert STAGE in warnings[0]

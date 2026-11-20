#!/usr/bin/env python3
"""
Stage 6 snapshots over model ids (TASK-MULTIMODEL-BDRABCZUK-62C933AD9A93, epic TASK-MULTIMODEL-BDRABCZUK-A02369CED8AE).

User decisions 2026-11-20: stage 5 writes one combined folder per model id,
<embeddings_dir>/<id>/<region>_embeddings.csv (the id may contain '/'), or today's flat
<embeddings_dir>/<region>_embeddings.csv when there are no ids. Stage 6 takes --model <id> to render the UMAP snapshots
of <embeddings_dir>/<id>/; without it a flat folder or a single id folder is used, and several id folders are refused,
listing the ids. main.py forwards dataset.model. UMAP stays opt-in on --reference_data_dir.

Stakeholder parent: REQ-MULTIMODEL-BDRABCZUK-23D430821417 (snapshots of one chosen model id).

The fixtures build the stage 5 layout by hand (synthetic embeddings, one region per CSV); the UMAP renderer is a mock.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_snapshots import GenerateSnapshots
from tests.main_stage_helpers import load_main_module, option_values
from tests.test_generate_snapshots import _embeddings_csv, _qc_rows, _umap_reference
from tests.test_generate_snapshots_output_qc import _rows

pytestmark = pytest.mark.unit  # noqa: V107

_RENDER_UMAP = "champollion_pipeline.generate_snapshots.generate_umap_snapshot"
SNAPSHOT_REGION = "S.Or."
OTHER_REGION = "F.C.M."
HEMI = "left"


def _id_folder(embeddings: Path, model_id: str, region: str = SNAPSHOT_REGION, subject: str = "sub-01") -> Path:
    """Write <embeddings>/<model_id>/<region>_<hemi>_embeddings.csv holding ``subject``; return the id folder."""
    folder = embeddings / model_id
    _embeddings_csv(folder, region, HEMI, [(subject, 0.1, 0.2)])
    return folder


def _reference(root: Path, regions=(SNAPSHOT_REGION,)) -> Path:
    reference = root / "reference"
    for region in regions:
        _umap_reference(reference, region, HEMI)
    return reference


def _umap_args(root: Path, embeddings: Path, reference: Path, model_id: str | None = None) -> list[str]:
    args = [
        "--output_dir",
        str(root / "out"),
        "--embeddings_dir",
        str(embeddings),
        "--reference_data_dir",
        str(reference),
        "--umap-only",
    ]
    return [*args, "--model", model_id] if model_id is not None else args


def _assert_model_option_known(root: Path) -> None:
    """Fail the test when the generate_snapshots parser does not accept --model."""
    try:
        GenerateSnapshots().parse_args(["--output_dir", str(root / "probe"), "--model", "probe"])
    except SystemExit as exc:
        pytest.fail(f"generate_snapshots rejects the --model option (SystemExit {exc.code})")


def _run(args: list[str]):
    """Run GenerateSnapshots with the UMAP renderer mocked; return (exit status, render mock).

    A SystemExit raised while parsing or running counts as the exit status.
    """
    with patch(_RENDER_UMAP, return_value=[]) as render:
        script = GenerateSnapshots()
        try:
            script.parse_args(args)
            status = script.run()
        except SystemExit as exc:
            status = exc.code
    return status, render


def _rendered_dir(render) -> str:
    """The embeddings directory handed to the single UMAP render call."""
    assert render.call_count == 1, render.call_args_list
    call = render.call_args
    return call.args[0] if call.args else call.kwargs["embeddings_dir"]


def _refusal_text(capture, capsys) -> str:
    """WARNING-and-above pipeline messages and stderr of the run, joined."""
    messages = [record.getMessage() for record in capture.records if record.levelno >= logging.WARNING]
    return "\n".join([*messages, capsys.readouterr().err])


class TestModelOptionRendersIdFolder:
    """REQ-MULTIMODEL-BDRABCZUK-2ACFDE275106: --model <id> renders the UMAP snapshots of <embeddings_dir>/<id>/."""

    def test_renders_named_id_folder(self, tmp_path):
        _assert_model_option_known(tmp_path)
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "run_a")
        chosen = _id_folder(embeddings, "run_b")
        _, render = _run(_umap_args(tmp_path, embeddings, _reference(tmp_path), "run_b"))
        assert Path(_rendered_dir(render)) == chosen

    def test_renders_nested_id_folder(self, tmp_path):
        _assert_model_option_known(tmp_path)
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "grp/run1")
        chosen = _id_folder(embeddings, "grp/run2")
        _, render = _run(_umap_args(tmp_path, embeddings, _reference(tmp_path), "grp/run2"))
        assert Path(_rendered_dir(render)) == chosen


class TestUnknownModelIdRefused:
    """REQ-MULTIMODEL-BDRABCZUK-C07BB4E95C63: a --model id naming no folder exits non-zero before any UMAP render."""

    def test_exit_non_zero_before_render(self, tmp_path):
        _assert_model_option_known(tmp_path)
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "run_a")
        _id_folder(embeddings, "run_b")
        status, render = _run(_umap_args(tmp_path, embeddings, _reference(tmp_path), "run_z"))
        assert status not in (0, None)
        assert not render.called


class TestUnknownModelIdMessageListsIds:
    """REQ-MULTIMODEL-BDRABCZUK-EF4213AF0941: the refusal of an unknown --model id names each model id folder."""

    def test_refusal_names_id_folders(self, tmp_path, pipeline_caplog, capsys):
        _assert_model_option_known(tmp_path)
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "run_a")
        _id_folder(embeddings, "grp/run1")
        _run(_umap_args(tmp_path, embeddings, _reference(tmp_path), "run_z"))
        text = _refusal_text(pipeline_caplog, capsys)
        assert "run_z" in text and "run_a" in text and "grp/run1" in text, text


class TestFlatFolderUnchanged:
    """REQ-MULTIMODEL-BDRABCZUK-AF217FDF5667: no --model, a flat folder is rendered as today (pin)."""

    def test_flat_folder_rendered(self, tmp_path):
        embeddings = tmp_path / "embeddings"
        _embeddings_csv(embeddings, SNAPSHOT_REGION, HEMI, [("sub-01", 0.1, 0.2)])
        _, render = _run(_umap_args(tmp_path, embeddings, _reference(tmp_path)))
        assert Path(_rendered_dir(render)) == embeddings


class TestSingleIdFolderUsed:
    """REQ-MULTIMODEL-BDRABCZUK-697B0D5D2000: without --model, the only model id folder is rendered."""

    def test_single_id_folder_rendered(self, tmp_path):
        embeddings = tmp_path / "embeddings"
        only = _id_folder(embeddings, "grp/run1")
        _, render = _run(_umap_args(tmp_path, embeddings, _reference(tmp_path)))
        assert Path(_rendered_dir(render)) == only


class TestSeveralIdFoldersRefused:
    """REQ-MULTIMODEL-BDRABCZUK-9638D9078CB8: no --model, several id folders exit non-zero before any render."""

    def test_exit_non_zero_before_render(self, tmp_path):
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "run_a")
        _id_folder(embeddings, "run_b")
        status, render = _run(_umap_args(tmp_path, embeddings, _reference(tmp_path)))
        assert status not in (0, None)
        assert not render.called


class TestSeveralIdFoldersMessageListsIds:
    """REQ-MULTIMODEL-BDRABCZUK-6B02C391B53C: the several-id-folders refusal names each id folder."""

    def test_refusal_names_id_folders(self, tmp_path, pipeline_caplog, capsys):
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "run_a")
        _id_folder(embeddings, "grp/run1")
        _run(_umap_args(tmp_path, embeddings, _reference(tmp_path)))
        text = _refusal_text(pipeline_caplog, capsys)
        assert "run_a" in text and "grp/run1" in text, text


class TestInputQcReadsChosenFolder:
    """REQ-MULTIMODEL-BDRABCZUK-35577BD52095: the snapshots input QC reads the chosen id folder."""

    def test_input_qc_lists_chosen_folder_subjects(self, tmp_path):
        _assert_model_option_known(tmp_path)
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "run_a", subject="sub-A1")
        _id_folder(embeddings, "run_b", subject="sub-B1")
        _run(_umap_args(tmp_path, embeddings, _reference(tmp_path), "run_b"))
        assert sorted(_qc_rows(str(tmp_path / "out"))) == ["sub-B1"]


class TestOutputQcChecksChosenFolder:
    """REQ-MULTIMODEL-BDRABCZUK-72C839684838: the snapshots output QC checks the chosen id folder."""

    def test_output_qc_rows_follow_chosen_folder(self, tmp_path):
        _assert_model_option_known(tmp_path)
        embeddings = tmp_path / "embeddings"
        _id_folder(embeddings, "run_a", region=OTHER_REGION)
        _id_folder(embeddings, "run_b", region=SNAPSHOT_REGION)
        reference = _reference(tmp_path, regions=(SNAPSHOT_REGION, OTHER_REGION))
        _run(_umap_args(tmp_path, embeddings, reference, "run_b"))
        assert sorted(_rows(str(tmp_path))) == [f"umap_{SNAPSHOT_REGION}_{HEMI}.png"]


def _snapshots_argv(tmp_path, model_id: str | None) -> list[str]:
    """argv built by main.py's GenerateSnapshotsStage, dataset.model set to ``model_id`` when given."""
    main_module = load_main_module("_champollion_main_snapshots_multi_model_under_test")
    config = main_module.PipelineConfig(models_path="/models")
    config.dataset.datasets_root = str(tmp_path / "MYDATASET")  # noqa: V101
    if model_id is not None:
        config.dataset.model = model_id  # noqa: V101
    stage = main_module.GenerateSnapshotsStage("generate_snapshots", config, logging.getLogger(__name__))
    return stage.build_args()


class TestMainForwardsModel:
    """REQ-MULTIMODEL-BDRABCZUK-77EC27DBD337: main.py passes dataset.model to generate_snapshots as --model."""

    def test_model_forwarded(self, tmp_path):
        argv = _snapshots_argv(tmp_path, "grp/run1")
        assert option_values(argv, "--model") == ["grp/run1"], argv


class TestMainNoModelNoOption:
    """REQ-MULTIMODEL-BDRABCZUK-2DB755133CB5: an empty dataset.model passes no --model (pins today's behaviour)."""

    def test_empty_model_not_forwarded(self, tmp_path):
        argv = _snapshots_argv(tmp_path, "")
        assert option_values(argv, "--model") == [], argv

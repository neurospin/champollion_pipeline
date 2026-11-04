#!/usr/bin/env python3
"""
Stage 2 input QC tests of run_cortical_tiles.py: input_qc.tsv, the --sk_qc_path file check and
the exit policy (user decision 2026-11-04, "Fail only if nothing usable"). Synthetic fixtures only.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.log_assertions import FAIL
from tests.stage2_layout import add_graphs

_LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"
_GRAPH_SUBPATH = "graphs/path"
_QC_FILE = "input_qc.tsv"
_CROP_SCRIPT = "generate_sulcal_regions.py"
_REFUSED = "invalid_sk_qc_file"
_SCOPE_OPTION = "--sk_qc_input_only"


def _sk_qc_file(temp_dir, rows, name="sk_qc.tsv", separator="\t", columns=("participant_id", "qc")):
    """Write a --sk_qc_path file under ``temp_dir`` (``rows``: tuples of cells) and return its path."""
    path = Path(temp_dir) / name
    lines = [separator.join(columns), *(separator.join(str(cell) for cell in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@dataclass
class _StageRun:
    """What run() did with the crop subprocess mocked."""

    result: int
    crop_commands: list[list[str]]
    report_at_crop: list[bool]
    qc_path: Path

    def rows(self):
        """Rows of input_qc.tsv, keyed by participant_id."""
        assert self.qc_path.is_file(), f"{self.qc_path} was not written"
        return {row["participant_id"]: row for row in read_qc_tsv(self.qc_path)}


def _run_stage(temp_dir, subjects, extra_args=(), *, crop_status=0, layout=(_GRAPH_SUBPATH, _GRAPH_SUBPATH)):
    """Run run() on ``subjects`` (``{subject: hemispheres holding a graph}``); the crop subprocess returns
    ``crop_status``, the mask, fusion and ventricle steps are mocked.

    ``layout`` is the --path_to_graph value and the folder under each subject holding the graphs.
    """
    path_to_graph, graph_dir = layout
    root = Path(temp_dir)
    input_dir, output_dir = root / "input", root / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    for subject, sides in subjects.items():
        add_graphs(input_dir, subject, graph_dir, sides)
    qc_path = output_dir / DERIVATIVES_FOLDER / _QC_FILE
    crop_commands, report_at_crop = [], []

    def execute(command, *_args, **_kwargs):
        if any(_CROP_SCRIPT in str(part) for part in command):
            crop_commands.append([str(part) for part in command])
            report_at_crop.append(qc_path.is_file())
        return crop_status

    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            path_to_graph,
            "--path_sk_with_hull",
            "skeleton/path",
            *extra_args,
        ]
    )
    with (
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", side_effect=execute),
        patch.object(script, "_generate_mask_npys"),
        patch("champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes"),
        patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle"),
    ):
        result = script.run()
    return _StageRun(result, crop_commands, report_at_crop, qc_path)


def _messages_at(level, pipeline_caplog):
    """Messages of the run_cortical_tiles logger records at exactly ``level``."""
    return [r.getMessage() for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == level]


def _reasons(row):
    """The reason cell of ``row`` as a set of codes."""
    return set(row["reason"].split(";"))


class TestInputQcReport:
    """input_qc.tsv: one row per input subject folder, written before the crop step."""

    def test_report_exists_at_the_crop_call(self, temp_dir):
        run = _run_stage(temp_dir, {"sub-01": "LR"})

        assert run.report_at_crop == [True]

    def test_one_row_per_subject_folder(self, temp_dir):
        input_dir = Path(temp_dir) / "input"
        (input_dir / ".hidden").mkdir(parents=True)
        (input_dir / "notes.txt").touch()

        run = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": ""})

        assert sorted(run.rows()) == ["sub-01", "sub-02"]

    def test_report_columns(self, temp_dir):
        run = _run_stage(temp_dir, {"sub-01": "LR"})

        assert run.qc_path.is_file()
        assert run.qc_path.read_text().splitlines()[0].split("\t") == ["participant_id", "qc", "reason"]

    def test_bids_rows_hold_the_subject_folder_name(self, temp_dir):
        run = _run_stage(
            temp_dir,
            {"sub-01": "LR"},
            ["--bids"],
            layout=("*/t1mri/default_analysis/folds/3.1", "ses-1/t1mri/default_analysis/folds/3.1"),
        )

        assert list(run.rows()) == ["sub-01"]


class TestGraphPresence:
    """A subject needs one Morphologist graph per hemisphere where cortical_tiles looks for it."""

    @pytest.mark.parametrize(("sides", "reason"), [("L", "missing_graph_R"), ("R", "missing_graph_L")])
    def test_missing_hemisphere_graph(self, temp_dir, sides, reason):
        rows = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": sides}).rows()

        assert rows["sub-02"]["qc"] == 0
        assert rows["sub-02"]["reason"] == reason

    def test_labelled_graph_is_not_a_graph_input(self, temp_dir):
        labelled = f"{_GRAPH_SUBPATH}/deepcnn_session_auto"
        add_graphs(Path(temp_dir) / "input", "sub-02", labelled)

        rows = _run_stage(temp_dir, {"sub-01": "LR"}).rows()

        assert rows["sub-02"]["qc"] == 0
        assert _reasons(rows["sub-02"]) == {"missing_graph_L", "missing_graph_R"}

    @pytest.mark.parametrize("extra_args", [[], ["--bids"]], ids=["plain", "bids"])
    def test_wildcard_path_to_graph(self, temp_dir, extra_args):
        rows = _run_stage(
            temp_dir,
            {"sub-01": "LR"},
            extra_args,
            layout=("t1mri/*/default_analysis/folds/3.1", "t1mri/acq1/default_analysis/folds/3.1"),
        ).rows()

        assert rows["sub-01"]["qc"] == 1


class TestSkQcIds:
    """The ids of a valid --sk_qc_path file are matched against the subject folders."""

    def test_subject_absent_from_the_file(self, temp_dir):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1)])

        rows = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert rows["sub-02"]["qc"] == 0
        assert rows["sub-02"]["reason"] == "not_in_sk_qc_file"

    def test_subject_excluded_by_the_file(self, temp_dir):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-02", 0)])

        rows = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert rows["sub-02"]["qc"] == 0
        assert rows["sub-02"]["reason"] == "excluded_by_sk_qc"

    @pytest.mark.parametrize("qc_value", [1, 0])
    def test_file_id_without_a_subject_folder(self, temp_dir, qc_value):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-09", qc_value)])

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert rows["sub-09"]["qc"] == 0
        assert rows["sub-09"]["reason"] == "no_subject_folder"


def _write_bytes(path, content):
    path.write_bytes(content)


_INVALID_FILES = [
    pytest.param(lambda path: None, id="missing"),
    pytest.param(lambda path: path.mkdir(), id="directory"),
    pytest.param(lambda path: _write_bytes(path, b""), id="empty_file"),
    pytest.param(lambda path: _write_bytes(path, b"\xff\xfe\x00\x01"), id="not_text"),
    pytest.param(lambda path: _write_bytes(path, b"subject\tqc\nsub-01\t1\n"), id="no_participant_id_column"),
    pytest.param(lambda path: _write_bytes(path, b"participant_id\tkeep\nsub-01\t1\n"), id="no_qc_column"),
    pytest.param(lambda path: _write_bytes(path, b"participant_id\tqc\nsub-01\t1\nsub-01\t0\n"), id="repeated_id"),
    pytest.param(lambda path: _write_bytes(path, b"participant_id\tqc\nsub-01\t1\n\t1\n"), id="empty_id"),
    pytest.param(lambda path: _write_bytes(path, b"participant_id\tqc\nsub-01\t2\n"), id="qc_out_of_range"),
    pytest.param(lambda path: _write_bytes(path, b"participant_id\tqc\nsub-01\tyes\n"), id="qc_not_a_number"),
]


class TestRefusedSkQcFile:
    """A --sk_qc_path file that cannot be trusted leaves no subject with a known QC status."""

    @pytest.mark.parametrize("lay_out", _INVALID_FILES)
    def test_invalid_file_is_refused(self, temp_dir, lay_out):
        sk_qc = Path(temp_dir) / "sk_qc.tsv"
        lay_out(sk_qc)

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", str(sk_qc)]).rows()

        assert _REFUSED in _reasons(rows["sub-01"])

    def test_valid_file_is_not_refused(self, temp_dir):
        columns = ("participant_id", "qc", "site", "reason")
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1, "A", "")], columns=columns)

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert rows["sub-01"]["qc"] == 1

    def test_refused_file_marks_each_subject(self, temp_dir):
        sk_qc = str(Path(temp_dir) / "absent.tsv")

        rows = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert [row["qc"] for row in rows.values()] == [0, 0]
        assert [row["reason"] for row in rows.values()] == [_REFUSED, _REFUSED]

    def test_refused_file_emits_one_error_record_naming_it(self, temp_dir, pipeline_caplog):
        sk_qc = str(Path(temp_dir) / "absent.tsv")

        _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc])

        messages = _messages_at(logging.ERROR, pipeline_caplog)
        assert len([message for message in messages if sk_qc in message]) == 1, messages


class TestSkQcSeparator:
    """The file is read the way cortical_tiles select_good_qc reads it: tab with .tsv in the path, else comma."""

    @pytest.mark.parametrize(
        ("name", "separator"),
        [("sk_qc.tsv", "\t"), ("sk_qc.tsv.bak", "\t"), ("sk_qc.csv", ","), ("sk_qc.txt", ",")],
    )
    def test_matching_separator_is_read(self, temp_dir, name, separator):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1)], name=name, separator=separator)

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert rows["sub-01"]["qc"] == 1

    @pytest.mark.parametrize(("name", "separator"), [("sk_qc.tsv", ","), ("sk_qc.csv", "\t")])
    def test_other_separator_is_refused(self, temp_dir, name, separator):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1)], name=name, separator=separator)

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert rows["sub-01"]["reason"] == _REFUSED


class TestPassingSubject:
    """A subject failing no input check keeps qc 1 and an empty reason."""

    def test_without_a_qc_file(self, temp_dir):
        rows = _run_stage(temp_dir, {"sub-01": "LR"}).rows()

        assert (rows["sub-01"]["qc"], rows["sub-01"]["reason"]) == (1, "")

    def test_with_a_qc_file(self, temp_dir):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1)])

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc]).rows()

        assert (rows["sub-01"]["qc"], rows["sub-01"]["reason"]) == (1, "")


class TestSeveralReasons:
    """Every failed check of a subject is named in its reason cell."""

    def test_both_hemispheres_missing(self, temp_dir):
        rows = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": ""}).rows()

        assert _reasons(rows["sub-02"]) == {"missing_graph_L", "missing_graph_R"}

    def test_graph_and_qc_file_reasons(self, temp_dir):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1)])

        rows = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "L"}, ["--sk_qc_path", sk_qc]).rows()

        assert _reasons(rows["sub-02"]) == {"missing_graph_R", "not_in_sk_qc_file"}


_NO_QC1_CASES = [
    pytest.param({"sub-01": "", "sub-02": "R"}, None, id="no_graph"),
    pytest.param({"sub-01": "L", "sub-02": "R"}, None, id="one_hemisphere_each"),
    pytest.param({"sub-01": "LR"}, [("sub-01", 0)], id="excluded_by_the_file"),
    pytest.param({"sub-01": "LR"}, [("sub-09", 1)], id="ids_do_not_match"),
    pytest.param({"sub-01": "LR", "sub-02": "LR"}, [("sub-01", 1), ("sub-02", "yes")], id="one_bad_qc_cell"),
]


def _run_no_qc1(temp_dir, subjects, sk_rows, options=()):
    """Run the stage on a case of ``_NO_QC1_CASES``, with ``options`` added."""
    extra_args = ["--sk_qc_path", _sk_qc_file(temp_dir, sk_rows)] if sk_rows else []
    return _run_stage(temp_dir, subjects, [*extra_args, *options])


class TestInputQcExitPolicy:
    """Bad subjects are reported per subject; the stage fails only when input_qc.tsv keeps no subject."""

    def test_one_error_record_per_qc0_row(self, temp_dir, pipeline_caplog):
        _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "L", "sub-03": "R"})

        messages = _messages_at(logging.ERROR, pipeline_caplog)
        assert len(messages) == 2, messages
        for subject, reason in (("sub-02", "missing_graph_R"), ("sub-03", "missing_graph_L")):
            assert len([m for m in messages if subject in m and reason in m]) == 1, messages

    def test_subject_excluded_by_the_file_is_a_warning_not_an_error(self, temp_dir, pipeline_caplog):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-02", 0)])

        _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "LR"}, ["--sk_qc_path", sk_qc])

        warnings = _messages_at(logging.WARNING, pipeline_caplog)
        assert len([m for m in warnings if "sub-02" in m and "excluded_by_sk_qc" in m]) == 1, warnings
        assert [m for m in _messages_at(logging.ERROR, pipeline_caplog) if "sub-02" in m] == []

    def test_excluded_subject_with_a_missing_graph_is_an_error_not_a_warning(self, temp_dir, pipeline_caplog):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-02", 0)])

        _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "L"}, ["--sk_qc_path", sk_qc])

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert len([m for m in errors if "sub-02" in m and "missing_graph_R" in m and "excluded_by_sk_qc" in m]) == 1
        assert [m for m in _messages_at(logging.WARNING, pipeline_caplog) if "sub-02" in m] == []

    def test_subject_absent_from_the_file_is_an_error_not_a_warning(self, temp_dir, pipeline_caplog):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1)])

        _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "LR"}, ["--sk_qc_path", sk_qc])

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert len([m for m in errors if "sub-02" in m and "not_in_sk_qc_file" in m]) == 1, errors
        assert [m for m in _messages_at(logging.WARNING, pipeline_caplog) if "sub-02" in m] == []

    def test_excluded_id_without_a_subject_folder_is_an_error_not_a_warning(self, temp_dir, pipeline_caplog):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-09", 0)])

        _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc])

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert len([m for m in errors if "sub-09" in m and "no_subject_folder" in m]) == 1, errors
        assert [m for m in _messages_at(logging.WARNING, pipeline_caplog) if "sub-09" in m] == []

    def test_no_error_record_when_all_pass(self, temp_dir, pipeline_caplog):
        run = _run_stage(temp_dir, {"sub-01": "LR"})

        assert run.qc_path.is_file()
        assert _messages_at(logging.ERROR, pipeline_caplog) == []

    @pytest.mark.parametrize(("subjects", "sk_rows"), _NO_QC1_CASES)
    def test_no_qc1_row_returns_1_without_the_crop_step(self, temp_dir, subjects, sk_rows):
        run = _run_no_qc1(temp_dir, subjects, sk_rows)

        assert run.result == 1
        assert run.crop_commands == []

    def test_empty_input_returns_1_without_the_crop_step(self, temp_dir):
        run = _run_stage(temp_dir, {})

        assert run.rows() == {}
        assert run.result == 1
        assert run.crop_commands == []

    @pytest.mark.parametrize(("subjects", "sk_rows"), _NO_QC1_CASES)
    def test_no_qc1_row_emits_one_fail_record(self, temp_dir, pipeline_caplog, subjects, sk_rows):
        run = _run_no_qc1(temp_dir, subjects, sk_rows)

        messages = _messages_at(FAIL, pipeline_caplog)
        assert len(messages) == 1, messages
        assert str(run.qc_path) in messages[0]

    def test_empty_input_emits_one_fail_record(self, temp_dir, pipeline_caplog):
        run = _run_stage(temp_dir, {})

        messages = _messages_at(FAIL, pipeline_caplog)
        assert len(messages) == 1, messages
        assert str(run.qc_path) in messages[0]

    def test_qc1_row_emits_no_fail_record(self, temp_dir, pipeline_caplog):
        run = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "L"})

        assert run.qc_path.is_file()
        assert _messages_at(FAIL, pipeline_caplog) == []

    @pytest.mark.parametrize("crop_status", [0, 2, 3])
    @pytest.mark.parametrize(
        "subjects", [{"sub-01": "LR"}, {"sub-01": "LR", "sub-02": "L"}], ids=["all_qc1", "partial"]
    )
    def test_qc1_row_returns_the_crop_status(self, temp_dir, subjects, crop_status):
        run = _run_stage(temp_dir, subjects, crop_status=crop_status)

        assert run.qc_path.is_file()
        assert run.result == crop_status


class TestCropArguments:
    """input_qc.tsv stays a report: it is never fed to cortical_tiles as a QC file (CT-M7)."""

    def test_no_sk_qc_option_without_the_flag(self, temp_dir):
        run = _run_stage(temp_dir, {"sub-01": "LR"})

        assert run.qc_path.is_file()
        assert len(run.crop_commands) == 1
        assert "--sk_qc_path" not in run.crop_commands[0]


_RELATIVE_SK_QC_PATHS = ["sk_qc.tsv", "./sk_qc.tsv", "nested/sk_qc.tsv", "nested/../sk_qc.tsv"]


def _run_from_launch_dir(temp_dir, monkeypatch, sk_qc_path, rows):
    """Run run() from <temp_dir>/launch with ``sk_qc_path`` relative to it, on a seeded pipeline config.

    Returns the run, the QC file and the config. The crop step's own directory stays untouched:
    what the child would read is asserted from the values it is given.
    """
    launch_dir = Path(temp_dir) / "launch"
    (launch_dir / "nested").mkdir(parents=True)
    qc_file = Path(_sk_qc_file(launch_dir, rows, name=sk_qc_path))
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir()
    config_path = output_dir / "pipeline_loop_2mm.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(launch_dir)
    run = _run_stage(temp_dir, {"sub-01": "LR", "sub-02": "LR"}, ["--sk_qc_path", sk_qc_path])
    return run, qc_file, config_path


class TestRelativeSkQcPath:
    """A relative --sk_qc_path names one file, under the process's current directory (user decision 2026-11-04)."""

    @pytest.mark.parametrize("sk_qc_path", _RELATIVE_SK_QC_PATHS)
    def test_crop_command_holds_the_absolute_path(self, temp_dir, monkeypatch, sk_qc_path):
        run, qc_file, _ = _run_from_launch_dir(temp_dir, monkeypatch, sk_qc_path, [("sub-01", 1), ("sub-02", 1)])

        command = run.crop_commands[0]
        value = Path(command[command.index("--sk_qc_path") + 1])
        assert value.is_absolute(), command
        assert value.samefile(qc_file), command

    @pytest.mark.parametrize("sk_qc_path", _RELATIVE_SK_QC_PATHS)
    def test_config_holds_the_absolute_path(self, temp_dir, monkeypatch, sk_qc_path):
        _, qc_file, config_path = _run_from_launch_dir(
            temp_dir, monkeypatch, sk_qc_path, [("sub-01", 1), ("sub-02", 1)]
        )

        value = Path(json.loads(config_path.read_text(encoding="utf-8"))["skel_qc_path"])
        assert value.is_absolute(), value
        assert value.samefile(qc_file), value

    @pytest.mark.parametrize("sk_qc_path", _RELATIVE_SK_QC_PATHS)
    def test_input_qc_reads_the_file_of_the_current_directory(self, temp_dir, monkeypatch, sk_qc_path):
        run, _, _ = _run_from_launch_dir(temp_dir, monkeypatch, sk_qc_path, [("sub-01", 1), ("sub-02", 0)])

        rows = run.rows()
        assert rows["sub-01"]["qc"] == 1
        assert rows["sub-02"]["qc"] == 0
        assert rows["sub-02"]["reason"] == "excluded_by_sk_qc"


# The file of each case holds sub-09, an id with no subject folder, as a cohort file does.
_SCOPE_CASES = [
    pytest.param({"sub-01": "LR", "sub-02": "LR"}, [("sub-01", 1), ("sub-02", 1), ("sub-09", 1)], id="all_pass"),
    pytest.param({"sub-01": "LR", "sub-02": "L"}, [("sub-01", 1), ("sub-02", 1), ("sub-09", 1)], id="missing_graph"),
    pytest.param({"sub-01": "LR", "sub-02": "LR"}, [("sub-01", 1), ("sub-09", 1)], id="not_in_the_file"),
    pytest.param({"sub-01": "LR", "sub-02": "LR"}, [("sub-01", 1), ("sub-02", 0), ("sub-09", 1)], id="excluded"),
    pytest.param({"sub-01": "LR", "sub-02": "R"}, [("sub-01", 1), ("sub-09", 1)], id="several_reasons"),
    pytest.param({"sub-01": "LR"}, [("sub-01", 1), ("sub-09", "yes")], id="bad_cell_of_an_id_with_no_folder"),
    pytest.param({"sub-01": "LR", "sub-02": "L"}, None, id="no_qc_file"),
]


class TestSkQcInputOnly:
    """--sk_qc_input_only limits the input QC to the subjects of the input and lowers no other check."""

    def test_file_id_without_a_subject_folder_gets_no_row(self, temp_dir):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-08", 1), ("sub-09", 0)])

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc, _SCOPE_OPTION]).rows()

        assert sorted(rows) == ["sub-01"]

    def test_file_id_without_a_subject_folder_gets_no_record(self, temp_dir, pipeline_caplog):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-08", 1), ("sub-09", 0)])

        run = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc, _SCOPE_OPTION])

        assert run.qc_path.is_file()
        messages = [r.getMessage() for r in pipeline_caplog.records if r.name == _LOGGER_NAME]
        assert [m for m in messages if "sub-08" in m or "sub-09" in m] == []

    @pytest.mark.parametrize(("subjects", "sk_rows"), _SCOPE_CASES)
    def test_subject_folder_rows_do_not_depend_on_the_option(self, temp_dir, subjects, sk_rows):
        sk_args = ["--sk_qc_path", _sk_qc_file(temp_dir, sk_rows)] if sk_rows else []

        plain = _run_stage(temp_dir, subjects, sk_args).rows()
        scoped = _run_stage(temp_dir, subjects, [*sk_args, _SCOPE_OPTION]).rows()

        assert scoped == {subject: plain[subject] for subject in subjects}

    def test_refused_file_stays_refused(self, temp_dir, pipeline_caplog):
        sk_qc = str(Path(temp_dir) / "absent.tsv")

        rows = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc, _SCOPE_OPTION]).rows()

        assert rows["sub-01"]["reason"] == _REFUSED
        messages = _messages_at(logging.ERROR, pipeline_caplog)
        assert len([message for message in messages if sk_qc in message]) == 1, messages

    @pytest.mark.parametrize(("subjects", "sk_rows"), _NO_QC1_CASES)
    def test_no_qc1_row_returns_1_without_the_crop_step(self, temp_dir, subjects, sk_rows):
        run = _run_no_qc1(temp_dir, subjects, sk_rows, [_SCOPE_OPTION])

        assert run.result == 1
        assert run.crop_commands == []

    def test_qc1_row_returns_the_crop_status(self, temp_dir):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-09", 1)])

        run = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc, _SCOPE_OPTION], crop_status=2)

        assert run.qc_path.is_file()
        assert run.result == 2

    def test_crop_command_does_not_depend_on_the_option(self, temp_dir):
        sk_qc = _sk_qc_file(temp_dir, [("sub-01", 1), ("sub-09", 1)])

        plain = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc])
        scoped = _run_stage(temp_dir, {"sub-01": "LR"}, ["--sk_qc_path", sk_qc, _SCOPE_OPTION])

        assert scoped.qc_path.is_file()
        assert len(scoped.crop_commands) == 1
        assert scoped.crop_commands == plain.crop_commands

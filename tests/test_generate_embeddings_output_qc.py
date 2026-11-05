#!/usr/bin/env python3
"""
Stage 4 output QC tests of generate_embeddings.py: output_qc.tsv over the full_embeddings.csv of the run's
region models. Synthetic fixtures only; evaluate.py is faked.

The width of a row comes from the .hydra/config.yaml of its region model, so these tables are _WIDTH wide, not 32.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION, compute_region_embeddings_dir
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.models.embeddings_qc import fetch_embeddings_csv_reasons
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.log_assertions import FAIL, assert_one_subject_error
from tests.stage2_layout import add_crops
from tests.test_generate_embeddings_any_failure import make_models, make_script, region_of

pytestmark = pytest.mark.unit  # noqa: V107

_LOGGER_NAME = "champollion_pipeline.generate_embeddings"
_WIDTH = 3
_CROPS = ("S.C.-sylv.", "S.T.s.")
_REGION = "SC-sylv_left"
_OTHER_REGION = "STs_right"
_REGIONS = (_REGION, _OTHER_REGION)
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_STRAY = "sub-99"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)
_NUMBER = "0.5"


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage4_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage4_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


def _line(subject, cells=(_NUMBER,) * _WIDTH):
    """A data row of an embeddings table."""
    return ",".join([subject, *cells])


def _table(*lines, width=_WIDTH):
    """The bytes of an embeddings table: an ID column, then ``width`` columns, as evaluate.py writes them."""
    header = ",".join(["ID", *(f"dim{index}" for index in range(1, width + 1))])
    return "\n".join([header, *lines, ""]).encode()


def _complete(subjects=_COHORT, width=_WIDTH):
    """A valid embeddings table of ``subjects``."""
    return _table(*(_line(subject, (_NUMBER,) * width) for subject in subjects), width=width)


_WITH_STRAY = _complete((*_COHORT, _STRAY))


def _config(width):
    """Text of a region .hydra/config.yaml holding ``width`` as its backbone_output_size."""
    return f"backbone_output_size: {width}\n"


def _lay_complete(crops_dir, subjects):
    add_crops(crops_dir, subjects, _CROPS)


def _lay_second_subject_excluded(crops_dir, subjects):
    """Crops where the last of ``subjects`` misses the last region: the input QC excludes that subject."""
    add_crops(crops_dir, subjects, _CROPS[:1])
    add_crops(crops_dir, subjects[:-1], _CROPS[1:])


@dataclass(frozen=True)
class _Stage:
    """A run of the stage over _REGIONS, each file valid and each config _WIDTH wide unless said otherwise."""

    tables: dict = field(default_factory=dict)  # region -> bytes evaluate.py writes; None: it writes nothing
    reused: dict = field(default_factory=dict)  # region -> bytes on disk before the run
    configs: dict = field(default_factory=dict)  # region -> text of .hydra/config.yaml; None: no such file
    subjects: tuple[str, ...] = _COHORT
    lay: object = _lay_complete
    extra_args: tuple[str, ...] = ()
    has_output_option: bool = True
    codes: dict = field(default_factory=dict)  # region -> non-zero status of its evaluate.py, which writes nothing
    per_region_status: int | None = None  # status _run_per_region gives instead of running


_DEFAULT_STAGE = _Stage()


@dataclass
class _Finished:
    """The files a run left and its status."""

    output_base: Path
    result: int

    @property
    def report(self):
        return self.output_base / "output_qc.tsv"

    def rows(self):
        """Rows of output_qc.tsv, keyed by participant_id."""
        assert self.report.is_file(), f"{self.report} was not written"
        return {row["participant_id"]: row for row in read_qc_tsv(self.report)}

    def qc_values(self):
        return {subject: row["qc"] for subject, row in self.rows().items()}

    def reasons(self):
        """The reason cell of each participant_id, as a Counter of its reasons."""
        return {subject: Counter(filter(None, row["reason"].split(";"))) for subject, row in self.rows().items()}


def _execute_as_evaluate(stage):
    """An execute_command that writes the table ``stage`` gives the region of the command, or returns its code."""

    def execute(command, **_kwargs):
        code = stage.codes.get(region_of(command), 0)
        content = stage.tables.get(region_of(command), _complete(stage.subjects))
        if content is not None and code == 0:
            target = Path(command[command.index("-s") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return code

    return execute


def _lay_configs(models, stage):
    """Write the .hydra/config.yaml of each region of ``stage``."""
    for region in _REGIONS:
        text = stage.configs.get(region, _config(_WIDTH))
        if text is not None:
            (models / region / ".hydra").mkdir()
            (models / region / ".hydra" / "config.yaml").write_text(text)


def _finish(root, stage=_DEFAULT_STAGE):
    """Run GenerateEmbeddings.run() under ``root`` as ``stage`` says and return what it left."""
    root = Path(root)
    dataset = root / "dataset"
    derivatives = dataset / "derivatives" / f"cortical_tiles-{CORTICAL_TILES_VERSION}"
    stage.lay(derivatives / "crops" / DEFAULT_MASKS_VERSION / "2mm", stage.subjects)
    models = make_models(root, list(_REGIONS))
    _lay_configs(models, stage)
    output_args = ["--output", str(root / "out")] if stage.has_output_option else []
    script = make_script([str(models), str(dataset), *output_args, *stage.extra_args])
    default_base = Path(compute_region_embeddings_dir(str(dataset), script.args.masks))
    output_base = root / "out" if stage.has_output_option else default_base
    for region, content in stage.reused.items():
        (output_base / region).mkdir(parents=True)
        (output_base / region / "full_embeddings.csv").write_bytes(content)
    script.execute_command = _execute_as_evaluate(stage)
    if stage.per_region_status is not None:
        script._run_per_region = lambda *_args: stage.per_region_status
    with patch.object(GenerateEmbeddings, "fetch_models", return_value=str(models)):
        result = script.run()
    return _Finished(output_base, result)


def _records_at(pipeline_caplog, level):
    """Messages of the stage logger at exactly ``level``."""
    return [item.getMessage() for item in pipeline_caplog.records if (item.name, item.levelno) == (_LOGGER_NAME, level)]


def _only(reason, region=_REGION):
    """The reasons of a subject holding ``reason`` of ``region`` alone."""
    return Counter({f"{reason}:{region}": 1})


def _assert_each_row_fails_on(tmp_path, stage, reason):
    """Run ``stage``: the rows of _COHORT and _STRAY are qc 0, each holding ``reason`` of _REGION once."""
    run = _finish(tmp_path, stage)

    assert run.qc_values() == dict.fromkeys((*_COHORT, _STRAY), 0)
    assert {subject: held[f"{reason}:{_REGION}"] for subject, held in run.reasons().items()} == dict.fromkeys(
        (*_COHORT, _STRAY), 1
    )


class TestReport:
    @pytest.mark.parametrize("has_output_option", [True, False], ids=["output_option", "default_folder"])
    def test_report_lands_beside_the_input_report(self, tmp_path, has_output_option):
        run = _finish(tmp_path, _Stage(has_output_option=has_output_option))

        assert run.report.is_file()
        assert (run.output_base / "input_qc.tsv").is_file()

    def test_report_has_the_shared_columns(self, tmp_path):
        run = _finish(tmp_path)

        assert run.report.read_text().splitlines()[0].split("\t") == ["participant_id", "qc", "reason"]


class TestRows:
    @pytest.mark.parametrize("subjects", [_COHORT[:1], _COHORT, (*_COHORT, "sub-03")], ids=["one", "two", "three"])
    def test_one_row_per_qc_1_subject_of_the_input_report(self, tmp_path, subjects):
        assert sorted(_finish(tmp_path, _Stage(subjects=subjects)).rows()) == sorted(subjects)

    def test_subject_excluded_by_the_input_qc_has_no_row(self, tmp_path):
        run = _finish(tmp_path, _Stage(lay=_lay_second_subject_excluded))

        assert run.qc_values() == {_SUBJECT: 1}


class TestRegions:
    @pytest.mark.parametrize("region", _REGIONS)
    def test_each_region_model_is_checked(self, tmp_path, region):
        run = _finish(tmp_path, _Stage(tables={region: _table(_line(_OTHER_SUBJECT))}))

        assert run.reasons()[_SUBJECT] == _only("missing_row", region)

    def test_reused_file_is_checked(self, tmp_path):
        run = _finish(tmp_path, _Stage(reused={_REGION: _table(_line(_OTHER_SUBJECT))}))

        assert run.reasons()[_SUBJECT] == _only("missing_row")

    def test_regions_option_leaves_the_other_regions_out(self, tmp_path):
        run = _finish(tmp_path, _Stage(tables={_OTHER_REGION: None}, extra_args=("--regions", _REGION)))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}


class TestUsableSubject:
    def test_complete_files_give_qc_1_rows(self, tmp_path):
        assert _finish(tmp_path).qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_usable_subject_has_an_empty_reason(self, tmp_path):
        assert _finish(tmp_path).rows()[_SUBJECT]["reason"] == ""

    @pytest.mark.parametrize("cell", ["0", "-1", "1.5e-03", "-0.25"])
    def test_any_finite_float_is_a_usable_cell(self, tmp_path, cell):
        table = _table(_line(_SUBJECT, (cell,) * _WIDTH), _line(_OTHER_SUBJECT))

        assert _finish(tmp_path, _Stage(tables={_REGION: table})).qc_values()[_SUBJECT] == 1


class TestRegionReasons:
    """A fault of a whole file reaches each subject of the report."""

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            (None, "missing_embeddings"),
            (b"\xff\xfe\x00", "unreadable_embeddings"),
            (b"", "unreadable_embeddings"),
            (b"dim1,dim2,dim3\n0.5,0.5,0.5\n", "unreadable_embeddings"),
            (_table(_line(_SUBJECT), _line(_OTHER_SUBJECT), _line("")), "unreadable_embeddings"),
        ],
        ids=["absent", "not_utf8", "empty_file", "no_id_column", "empty_id_cell"],
    )
    def test_each_subject_gets_the_reason_of_the_file(self, tmp_path, content, reason):
        run = _finish(tmp_path, _Stage(tables={_REGION: content}))

        assert run.qc_values() == {_SUBJECT: 0, _OTHER_SUBJECT: 0}
        assert run.reasons() == {_SUBJECT: _only(reason), _OTHER_SUBJECT: _only(reason)}

    @pytest.mark.parametrize(
        ("tables", "reason"),
        [
            ({_REGION: None, _OTHER_REGION: _WITH_STRAY}, "missing_embeddings"),
            ({_REGION: b"\xff\xfe\x00", _OTHER_REGION: _WITH_STRAY}, "unreadable_embeddings"),
            (
                {_REGION: _table(_line(_SUBJECT), _line(_OTHER_SUBJECT), _line(_STRAY), _line(""))},
                "unreadable_embeddings",
            ),
        ],
        ids=["absent_file", "unreadable_file", "empty_id_cell_beside_the_id"],
    )
    def test_unexpected_id_gets_the_reason_of_the_file(self, tmp_path, tables, reason):
        _assert_each_row_fails_on(tmp_path, _Stage(tables=tables), reason)


class TestUnexpectedIdText:
    """An ID of a file keeps its text through write_qc_tsv and read_qc_tsv."""

    @pytest.mark.parametrize(
        "stray", ["a\tb", "a\nb", "a\r\nb", "a\rb"], ids=["tab", "line_feed", "crlf", "carriage_return"]
    )
    def test_id_holding_a_separator_round_trips(self, tmp_path, stray):
        table = _table(_line(_SUBJECT), _line(_OTHER_SUBJECT), _line(f'"{stray}"'))

        run = _finish(tmp_path, _Stage(tables={_REGION: table}))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1, stray: 0}
        assert run.reasons()[stray] == _only("unexpected_row")


class TestSubjectReasons:
    """A fault of one subject leaves the other subject usable."""

    @pytest.mark.parametrize(
        ("lines", "reason"),
        [
            ((_line(_OTHER_SUBJECT),), "missing_row"),
            ((_line(_SUBJECT), _line(_SUBJECT), _line(_OTHER_SUBJECT)), "duplicate_row"),
            ((_line(_SUBJECT, (_NUMBER,) * (_WIDTH - 1)), _line(_OTHER_SUBJECT)), "wrong_width"),
            ((_line(_SUBJECT, (_NUMBER,) * (_WIDTH + 1)), _line(_OTHER_SUBJECT)), "wrong_width"),
            ((_line(_SUBJECT, (_NUMBER, "abc", _NUMBER)), _line(_OTHER_SUBJECT)), "non_numeric"),
        ],
        ids=["no_row", "two_rows", "short_row", "long_row", "text_cell"],
    )
    def test_row_fault_reaches_that_subject_alone(self, tmp_path, lines, reason):
        run = _finish(tmp_path, _Stage(tables={_REGION: _table(*lines)}))

        assert run.qc_values() == {_SUBJECT: 0, _OTHER_SUBJECT: 1}
        assert run.reasons()[_SUBJECT] == _only(reason)

    @pytest.mark.parametrize("cell", ["nan", "NaN", "inf", "-inf", "Infinity", ""], ids=lambda cell: cell or "empty")
    def test_non_finite_cell_reaches_that_subject_alone(self, tmp_path, cell):
        table = _table(_line(_SUBJECT, (_NUMBER, cell, _NUMBER)), _line(_OTHER_SUBJECT))

        run = _finish(tmp_path, _Stage(tables={_REGION: table}))

        assert run.qc_values() == {_SUBJECT: 0, _OTHER_SUBJECT: 1}
        assert run.reasons()[_SUBJECT] == _only("non_finite")

    def test_header_alone_gives_each_subject_a_missing_row(self, tmp_path):
        run = _finish(tmp_path, _Stage(tables={_REGION: _table()}))

        assert run.reasons() == {_SUBJECT: _only("missing_row"), _OTHER_SUBJECT: _only("missing_row")}

    def test_unexpected_id_gets_a_failing_row_of_its_own(self, tmp_path):
        table = _table(_line(_SUBJECT), _line(_OTHER_SUBJECT), _line(_STRAY))

        run = _finish(tmp_path, _Stage(tables={_REGION: table}))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1, _STRAY: 0}
        assert run.reasons()[_STRAY] == _only("unexpected_row")


class TestWidthSource:
    @pytest.mark.parametrize("width", [1, 4, 32])
    def test_table_of_the_config_width_is_usable(self, tmp_path, width):
        stage = _Stage(tables={_REGION: _complete(width=width)}, configs={_REGION: _config(width)})

        assert _finish(tmp_path, stage).qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    @pytest.mark.parametrize(("config_width", "table_width"), [(4, 3), (3, 4), (3, 32)])
    def test_table_of_another_width_fails(self, tmp_path, config_width, table_width):
        stage = _Stage(tables={_REGION: _complete(width=table_width)}, configs={_REGION: _config(config_width)})

        assert _finish(tmp_path, stage).reasons()[_SUBJECT] == _only("wrong_width")

    def test_each_region_keeps_the_width_of_its_config(self, tmp_path):
        stage = _Stage(
            tables={_REGION: _complete(width=4)}, configs={_REGION: _config(4), _OTHER_REGION: _config(_WIDTH)}
        )

        assert _finish(tmp_path, stage).qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_regions_option_reads_the_config_of_the_region(self, tmp_path):
        stage = _Stage(
            tables={_REGION: _complete(width=4)}, configs={_REGION: _config(4)}, extra_args=("--regions", _REGION)
        )

        assert _finish(tmp_path, stage).qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    @pytest.mark.parametrize(
        "text",
        [None, "a: [\n", "- 1\n", "other_key: 1\n", "backbone_output_size:\n"]
        + [_config(value) for value in (0, -1, "wide", "true", 3.5)],
        ids=["no_file", "unparsable", "list", "no_key", "no_value", "zero", "negative", "text", "boolean", "float"],
    )
    def test_config_without_a_width_reaches_each_subject(self, tmp_path, text):
        run = _finish(tmp_path, _Stage(configs={_REGION: text}))

        assert run.reasons() == {_SUBJECT: _only("unknown_width"), _OTHER_SUBJECT: _only("unknown_width")}

    @pytest.mark.parametrize("file_region", _REGIONS, ids=["same_file", "other_file"])
    def test_unexpected_id_gets_the_reason_of_the_config(self, tmp_path, file_region):
        stage = _Stage(tables={file_region: _WITH_STRAY}, configs={_REGION: None})

        _assert_each_row_fails_on(tmp_path, stage, "unknown_width")


def _sidecar(run, region):
    """The JSON object of the embedding_width.json of ``region``."""
    return json.loads((run.output_base / region / "embedding_width.json").read_text(encoding="utf-8"))


class TestWidthSidecar:
    """The width each region leaves for the stage 5 input QC, beside output_qc.tsv."""

    def test_each_region_gets_the_width_of_its_config(self, tmp_path):
        stage = _Stage(tables={_REGION: _complete(width=4)}, configs={_REGION: _config(4)})

        run = _finish(tmp_path, stage)

        assert _sidecar(run, _REGION) == {"width": 4}
        assert _sidecar(run, _OTHER_REGION) == {"width": _WIDTH}

    def test_reused_file_gets_its_width(self, tmp_path):
        run = _finish(tmp_path, _Stage(reused={_REGION: _complete()}))

        assert _sidecar(run, _REGION) == {"width": _WIDTH}

    def test_regions_option_leaves_the_other_regions_out(self, tmp_path):
        run = _finish(tmp_path, _Stage(extra_args=("--regions", _REGION)))

        assert _sidecar(run, _REGION) == {"width": _WIDTH}
        assert not (run.output_base / _OTHER_REGION / "embedding_width.json").exists()


class TestSeveralReasons:
    def test_reasons_of_two_regions_share_the_cell(self, tmp_path):
        tables = {_REGION: _table(_line(_OTHER_SUBJECT)), _OTHER_REGION: _table(_line(_OTHER_SUBJECT))}

        run = _finish(tmp_path, _Stage(tables=tables))

        assert run.reasons()[_SUBJECT] == _only("missing_row") + _only("missing_row", _OTHER_REGION)

    def test_reasons_of_one_file_share_the_cell(self, tmp_path):
        nan_row = _line(_SUBJECT, ("nan",) * _WIDTH)

        run = _finish(tmp_path, _Stage(tables={_REGION: _table(nan_row, nan_row, _line(_OTHER_SUBJECT))}))

        assert run.reasons()[_SUBJECT] == _only("duplicate_row") + _only("non_finite")

    def test_repeated_reason_is_written_once(self, tmp_path):
        table = _table(_line(_SUBJECT, ("nan",) * _WIDTH), _line(_OTHER_SUBJECT))

        assert _finish(tmp_path, _Stage(tables={_REGION: table})).rows()[_SUBJECT]["reason"] == f"non_finite:{_REGION}"


class TestRecords:
    def test_failing_subject_gets_one_error_record(self, tmp_path, pipeline_caplog):
        run = _finish(tmp_path, _Stage(tables={_REGION: _table(_line(_OTHER_SUBJECT))}))

        assert run.qc_values()[_SUBJECT] == 0
        assert_one_subject_error(_records_at(pipeline_caplog, logging.ERROR), _SUBJECT, f"missing_row:{_REGION}")

    def test_each_failing_subject_gets_its_error_record(self, tmp_path, pipeline_caplog):
        run = _finish(tmp_path, _Stage(tables={_REGION: None}))

        errors = _records_at(pipeline_caplog, logging.ERROR)
        assert len(errors) == len(run.rows())
        assert all(any(subject in error for error in errors) for subject in _COHORT)

    def test_usable_subjects_get_no_error_record(self, tmp_path, pipeline_caplog):
        run = _finish(tmp_path)

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}
        assert _records_at(pipeline_caplog, logging.ERROR) == []

    def test_no_usable_subject_gives_one_fail_record_naming_the_report(self, tmp_path, pipeline_caplog):
        run = _finish(tmp_path, _Stage(tables={_REGION: None}))

        failures = _records_at(pipeline_caplog, FAIL)
        assert len(failures) == 1
        assert str(run.report) in failures[0]

    def test_a_usable_subject_gives_no_fail_record(self, tmp_path, pipeline_caplog):
        run = _finish(tmp_path, _Stage(tables={_REGION: _table(_line(_OTHER_SUBJECT))}))

        assert run.qc_values()[_OTHER_SUBJECT] == 1
        assert _records_at(pipeline_caplog, FAIL) == []


_THREE = (*_COHORT, "sub-03")  # with _lay_second_subject_excluded: two qc 1 subjects, one qc 0
_LOST_ONE = _table(_line(_OTHER_SUBJECT))
_EVALUATE_FAILURE = 7


class TestExitStatus:
    """Status of run() by the rows of output_qc.tsv, which the report must hold first (user decision 2026-11-05)."""

    @pytest.mark.parametrize(
        ("stage", "qc_counts", "status"),
        [
            (_Stage(), {1: 2}, 0),
            (_Stage(tables={_REGION: _LOST_ONE}), {0: 1, 1: 1}, 3),
            (_Stage(tables={_REGION: None}), {0: 2}, 1),
            (_Stage(lay=_lay_second_subject_excluded), {1: 1}, 3),
            (
                _Stage(lay=_lay_second_subject_excluded, subjects=_THREE, tables={_REGION: _complete(_THREE[:1])}),
                {0: 1, 1: 1},
                3,
            ),
            (_Stage(lay=_lay_second_subject_excluded, tables={_REGION: None}), {0: 1}, 1),
        ],
        ids=["kept_usable", "kept_mixed", "kept_none", "partial_usable", "partial_mixed", "partial_none"],
    )
    def test_status_follows_the_report(self, tmp_path, stage, qc_counts, status):
        run = _finish(tmp_path, stage)

        assert Counter(run.qc_values().values()) == qc_counts
        assert run.result == status

    @pytest.mark.parametrize("lay", [_lay_complete, _lay_second_subject_excluded], ids=["kept", "partial"])
    def test_failed_evaluate_status_wins_over_a_lost_region(self, tmp_path, lay):
        run = _finish(tmp_path, _Stage(lay=lay, codes={_REGION: _EVALUATE_FAILURE}))

        assert set(run.qc_values().values()) == {0}
        assert run.result == _EVALUATE_FAILURE

    @pytest.mark.parametrize(
        ("reused", "qc_values"),
        [
            ({_REGION: _complete(), _OTHER_REGION: _complete()}, {1}),
            ({_REGION: _LOST_ONE, _OTHER_REGION: _complete()}, {0, 1}),
        ],
        ids=["usable", "mixed"],
    )
    def test_per_region_status_wins_over_the_report(self, tmp_path, reused, qc_values):
        run = _finish(tmp_path, _Stage(reused=reused, per_region_status=_EVALUATE_FAILURE))

        assert set(run.qc_values().values()) == qc_values
        assert run.result == _EVALUATE_FAILURE


def _file_reasons(path, width=_WIDTH):
    """The non-empty results of the shared check on ``path``, for _COHORT."""
    found = fetch_embeddings_csv_reasons(str(path), _REGION, _COHORT, width)
    return {subject: reasons for subject, reasons in found.items() if reasons}


class TestSharedCheck:
    """The owner of the embeddings CSV validity rule, for its named consumer: the stage 5 input QC."""

    def test_valid_file_gives_no_reason(self, tmp_path):
        path = tmp_path / "full_embeddings.csv"
        path.write_bytes(_complete())

        assert _file_reasons(path) == {}

    def test_absent_file_gives_each_expected_subject_its_reason(self, tmp_path):
        reason = [f"missing_embeddings:{_REGION}"]

        assert _file_reasons(tmp_path / "absent.csv") == {_SUBJECT: reason, _OTHER_SUBJECT: reason}

    def test_unknown_width_gives_each_expected_subject_its_reason(self, tmp_path):
        path = tmp_path / "full_embeddings.csv"
        path.write_bytes(_complete())
        reason = [f"unknown_width:{_REGION}"]

        assert _file_reasons(path, None) == {_SUBJECT: reason, _OTHER_SUBJECT: reason}

    def test_report_holds_the_reasons_of_the_shared_check(self, tmp_path):
        table = _table(_line(_SUBJECT, ("nan",) * _WIDTH), _line(_STRAY))

        run = _finish(tmp_path, _Stage(tables={_REGION: table}))

        path = run.output_base / _REGION / "full_embeddings.csv"
        found = fetch_embeddings_csv_reasons(str(path), _REGION, _COHORT, _WIDTH)
        expected = {subject: Counter(reasons) for subject, reasons in found.items() if reasons}
        assert {subject: reasons for subject, reasons in run.reasons().items() if reasons} == expected

#!/usr/bin/env python3
"""
Stage 5 output QC tests of put_together_embeddings.py: output_qc.tsv over the <region>_embeddings.csv copies.

The source of each test is the good stage 4 source of the input QC tests, so the input QC passes it. A fault of a copy
is laid right after the copy step of the run: only the output QC can see it.
"""

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.utils.qc_report import (
    ID_COLUMN,
    QC_COLUMN,
    REASON_COLUMN,
    REASON_SEPARATOR,
    read_qc_tsv,
    write_qc_tsv,
)
from tests.log_assertions import FAIL, OUTPUT, assert_one_subject_error
from tests.test_put_together_embeddings_input_qc import (
    _COHORT,
    _MODULE,
    _OTHER_REGION,
    _OTHER_SUBJECT,
    _REGION,
    _REGIONS,
    _SENTINEL,
    _STRAY,
    _SUBJECT,
    _every_subject_failing,
    _good_table,
    _line,
    _make_empty_source,
    _make_source,
    _messages,
    _one_region_file_absent,
    _Run,
    _run,
    _shared_check,
    _table,
    _wide_table,
    _with_bad_row,
)

pytestmark = pytest.mark.unit  # noqa: V107

_REPORT = "output_qc.tsv"
_OUTPUT_FOLDER = "combined"  # the folder _run gives as --output_path
_HEADER_WIDTH = 3
_WIDE_HEADER_WIDTH = 4
_NOTHING_USABLE_SOURCES = [
    pytest.param(_every_subject_failing, id="every_subject_failing"),
    pytest.param(_one_region_file_absent, id="one_region_file_absent"),
    pytest.param(_make_empty_source, id="no_region_folder"),
]

Fault = Callable[[Path], None]


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage5_input_qc():
    """Run the real stage 5 input QC of the source: overrides the conftest fixture of the same name."""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage5_output_qc():
    """Run the real stage 5 output QC of the copies: overrides the conftest fixture of the same name."""


def _copy_name(region: str) -> str:
    return f"{region}_embeddings.csv"


def _replace_copies(contents: Mapping[str, bytes | str]) -> Fault:
    """A fault writing the content of each region over its copy, text as UTF-8."""

    def fault(output: Path) -> None:
        for region, content in contents.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            (output / _copy_name(region)).write_bytes(data)

    return fault


def _delete_copy(region: str) -> Fault:
    def fault(output: Path) -> None:
        (output / _copy_name(region)).unlink()

    return fault


def _finish(
    tmp_path: Path,
    fault: Fault | None = None,
    tables: Mapping[str, str] | None = None,
    options: tuple[str, ...] = (),
    has_sidecars: bool = True,
) -> _Run:
    """Run the stage over the good source changed by ``tables``, laying ``fault`` right after its copy step."""
    source = _make_source(tmp_path, tables or {}, has_sidecars=has_sidecars)
    copy_embeddings = PutTogetherEmbeddings._copy_embeddings

    def copy_then_fault(script: PutTogetherEmbeddings, *args: str) -> int:
        copied = copy_embeddings(script, *args)
        if fault is not None:
            fault(Path(script.args.output_path))
        return copied

    with patch.object(PutTogetherEmbeddings, "_copy_embeddings", copy_then_fault):
        return _run(tmp_path, source, *options)


def _rows(run: _Run) -> dict[str, tuple[int, str]]:
    """(qc, reason) by participant_id of the output report."""
    report = run.output / _REPORT
    assert report.is_file(), f"{report} was not written"
    return {row[ID_COLUMN]: (row[QC_COLUMN], row[REASON_COLUMN]) for row in read_qc_tsv(report)}


def _assert_each_row_holds(run: _Run, reason: str) -> None:
    """Each cohort row of the output report is qc 0 and holds ``reason`` of _REGION."""
    rows = _rows(run)
    assert set(rows) == set(_COHORT)
    for qc, cell in rows.values():
        assert qc == 0
        assert f"{reason}:{_REGION}" in cell.split(REASON_SEPARATOR)


def _lose_other_subject(tmp_path: Path) -> _Run:
    """A run whose copy of _REGION loses the row of _OTHER_SUBJECT; the report must keep _SUBJECT."""
    return _finish(tmp_path, _replace_copies({_REGION: _table(_line(_SUBJECT))}))


class TestReport:
    def test_report_follows_the_copies(self, tmp_path):
        copies_at_write = {}

        def spy(path, rows, *args):
            copies_at_write[Path(path).name] = len(list(Path(path).parent.glob("*_embeddings.csv")))
            write_qc_tsv(path, rows, *args)

        with patch(f"{_MODULE}.write_qc_tsv", spy):
            _finish(tmp_path)

        assert copies_at_write.get(_REPORT) == len(_REGIONS)


class TestRows:
    @pytest.mark.parametrize(
        ("tables", "expected"),
        [
            pytest.param({}, set(_COHORT), id="every_subject_kept"),
            pytest.param({_REGION: _table(_line(_SUBJECT))}, {_SUBJECT}, id="subject_excluded_by_the_input_qc"),
        ],
    )
    def test_one_row_per_qc_1_subject(self, tmp_path, tables, expected):
        assert set(_rows(_finish(tmp_path, tables=tables))) == expected

    def test_usable_subject_row(self, tmp_path):
        run = _finish(tmp_path)

        assert _rows(run) == {subject: (1, "") for subject in _COHORT}
        assert run.status == 0

    def test_reasons_come_from_the_shared_check(self, tmp_path):
        def fake_check(csv_path, region, expected_subjects, width):
            in_output = Path(csv_path).parent == tmp_path / _OUTPUT_FOLDER
            return {_OTHER_SUBJECT: [_SENTINEL]} if in_output else {}

        with _shared_check(fake_check):
            run = _finish(tmp_path)

        assert _rows(run) == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (0, _SENTINEL)}


class TestRegions:
    @pytest.mark.parametrize("region", [_REGIONS[0], _REGIONS[-1]], ids=["first_region", "last_region"])
    def test_each_default_region_is_checked(self, tmp_path, region):
        rows = _rows(_finish(tmp_path, _delete_copy(region)))

        assert rows == {subject: (0, f"missing_embeddings:{region}") for subject in _COHORT}

    def test_named_regions_are_the_only_ones_checked(self, tmp_path):
        run = _finish(tmp_path, options=("--regions", _REGION, _OTHER_REGION))

        assert {qc for qc, _ in _rows(run).values()} == {1}


class TestWidth:
    @pytest.mark.parametrize(
        ("has_sidecars", "tables", "expected"),
        [
            pytest.param(True, {}, _HEADER_WIDTH, id="sidecar"),
            pytest.param(False, {_REGION: _wide_table()}, _WIDE_HEADER_WIDTH, id="header_of_the_region_file"),
        ],
    )
    def test_copy_gets_the_width_of_the_input_qc(self, tmp_path, has_sidecars, tables, expected):
        widths = {}

        def fake_check(csv_path, region, expected_subjects, width):
            widths[Path(csv_path).parent.name] = width
            return {}

        with _shared_check(fake_check):
            _finish(tmp_path, tables=tables, options=("--regions", _REGION), has_sidecars=has_sidecars)

        assert widths.get(_OUTPUT_FOLDER) == expected

    @pytest.mark.parametrize("has_sidecars", [True, False], ids=["sidecar", "header_of_the_region_file"])
    def test_row_of_another_width_fails(self, tmp_path, has_sidecars):
        run = _finish(tmp_path, _replace_copies({_REGION: _wide_table()}), has_sidecars=has_sidecars)

        _assert_each_row_holds(run, "wrong_width")

    def test_no_second_warning_without_a_sidecar(self, tmp_path, pipeline_caplog):
        run = _finish(tmp_path, has_sidecars=False)

        assert set(_rows(run)) == set(_COHORT)
        warnings = _messages(pipeline_caplog, logging.WARNING)
        assert len([message for message in warnings if _REGION in message]) == 1


class TestWholeFileFault:
    def test_absent_copy(self, tmp_path):
        _assert_each_row_holds(_finish(tmp_path, _delete_copy(_REGION)), "missing_embeddings")

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(b"\xff\xfe\n", id="not_utf8"),
            pytest.param(b"", id="empty_file"),
            pytest.param(b"subject,dim1\nsub-01,0.5\n", id="no_id_column"),
        ],
    )
    def test_unreadable_copy(self, tmp_path, content):
        run = _finish(tmp_path, _replace_copies({_REGION: content}))

        _assert_each_row_holds(run, "unreadable_embeddings")

    def test_empty_id_cell(self, tmp_path):
        run = _finish(tmp_path, _replace_copies({_REGION: _with_bad_row(_line(""))}))

        _assert_each_row_holds(run, "unreadable_embeddings")


class TestSubjectFault:
    def test_missing_subject_row(self, tmp_path):
        assert _rows(_lose_other_subject(tmp_path)) == {
            _SUBJECT: (1, ""),
            _OTHER_SUBJECT: (0, f"missing_row:{_REGION}"),
        }

    def test_unexpected_subject_row(self, tmp_path):
        stray_copy = _table(*(_line(subject) for subject in (*_COHORT, _STRAY)))

        run = _finish(tmp_path, _replace_copies({_REGION: stray_copy}))

        assert _rows(run) == {
            _SUBJECT: (1, ""),
            _OTHER_SUBJECT: (1, ""),
            _STRAY: (0, f"unexpected_row:{_REGION}"),
        }

    def test_excluded_subject_left_in_a_copy(self, tmp_path):
        run = _finish(
            tmp_path,
            _replace_copies({_OTHER_REGION: _good_table()}),
            tables={_REGION: _table(_line(_SUBJECT))},
        )

        assert _rows(run) == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (0, f"unexpected_row:{_OTHER_REGION}")}

    @pytest.mark.parametrize(
        ("bad_lines", "reason"),
        [
            pytest.param((_line(_OTHER_SUBJECT, ("0.5", "0.5")),), "wrong_width", id="short_row"),
            pytest.param((_line(_OTHER_SUBJECT, ("abc", "0.5", "0.5")),), "non_numeric", id="text_cell"),
            pytest.param((_line(_OTHER_SUBJECT, ("nan", "0.5", "0.5")),), "non_finite", id="nan_cell"),
            pytest.param((_line(_OTHER_SUBJECT, ("-inf", "0.5", "0.5")),), "non_finite", id="infinite_cell"),
            pytest.param((_line(_OTHER_SUBJECT, ("", "0.5", "0.5")),), "non_finite", id="empty_cell"),
            pytest.param((_line(_OTHER_SUBJECT),) * 2, "duplicate_row", id="repeated_row"),
        ],
    )
    def test_failing_subject_row(self, tmp_path, bad_lines, reason):
        run = _finish(tmp_path, _replace_copies({_REGION: _with_bad_row(*bad_lines)}))

        assert _rows(run) == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (0, f"{reason}:{_REGION}")}

    def test_several_reasons_joined_once(self, tmp_path):
        copies = {
            _REGION: _with_bad_row(_line(_OTHER_SUBJECT, ("abc", "abc", "0.5"))),
            _OTHER_REGION: _table(_line(_SUBJECT)),
        }

        _, cell = _rows(_finish(tmp_path, _replace_copies(copies)))[_OTHER_SUBJECT]

        held = Counter(cell.split(REASON_SEPARATOR))
        assert held == {f"non_numeric:{_REGION}": 1, f"missing_row:{_OTHER_REGION}": 1}


class TestRecords:
    def test_one_error_per_failing_subject(self, tmp_path, pipeline_caplog):
        _lose_other_subject(tmp_path)

        errors = _messages(pipeline_caplog, logging.ERROR)
        assert_one_subject_error(errors, _OTHER_SUBJECT, f"missing_row:{_REGION}")

    def test_usable_subject_gets_no_error(self, tmp_path, pipeline_caplog):
        run = _finish(tmp_path)

        assert set(_rows(run)) == set(_COHORT)
        assert _messages(pipeline_caplog, logging.ERROR) == []

    @pytest.mark.parametrize(
        "fault",
        [
            pytest.param(_delete_copy(_REGION), id="absent_copy"),
            pytest.param(_replace_copies({_REGION: _table()}), id="each_subject_row_lost"),
        ],
    )
    def test_one_fail_record_names_the_report(self, tmp_path, fault, pipeline_caplog):
        run = _finish(tmp_path, fault)

        failures = _messages(pipeline_caplog, FAIL)
        assert len(failures) == 1, failures
        assert str(run.output / _REPORT) in failures[0]

    def test_usable_subject_gives_no_fail_record(self, tmp_path, pipeline_caplog):
        run = _lose_other_subject(tmp_path)

        assert _rows(run)[_SUBJECT] == (1, "")
        assert _messages(pipeline_caplog, FAIL) == []


_STRAY_IN_THE_REGION_FILE = {_REGION: _table(*(_line(subject) for subject in (*_COHORT, _STRAY)))}
_LOSE_OTHER_SUBJECT = _replace_copies({_REGION: _table(_line(_SUBJECT))})


class TestExitStatus:
    @pytest.mark.parametrize(
        "tables",
        [
            pytest.param({}, id="every_subject_kept_by_the_input_qc"),
            pytest.param(_STRAY_IN_THE_REGION_FILE, id="subject_excluded_by_the_input_qc"),
        ],
    )
    def test_nothing_usable_returns_1(self, tmp_path, tables):
        run = _finish(tmp_path, _delete_copy(_REGION), tables=tables)

        assert {qc for qc, _ in _rows(run).values()} == {0}
        assert run.status == 1

    @pytest.mark.parametrize(
        ("tables", "fault", "usable"),
        [
            pytest.param({}, _LOSE_OTHER_SUBJECT, {_SUBJECT}, id="lost_by_the_output_qc"),
            pytest.param(_STRAY_IN_THE_REGION_FILE, None, set(_COHORT), id="lost_by_the_input_qc"),
            pytest.param(_STRAY_IN_THE_REGION_FILE, _LOSE_OTHER_SUBJECT, {_SUBJECT}, id="lost_by_both"),
        ],
    )
    def test_subject_lost_returns_3(self, tmp_path, tables, fault, usable):
        run = _finish(tmp_path, fault, tables=tables)

        assert {subject for subject, (qc, _) in _rows(run).items() if qc == 1} == usable
        assert run.status == 3


class TestNothingCopied:
    @pytest.mark.parametrize("make_source", _NOTHING_USABLE_SOURCES)
    def test_no_output_report(self, tmp_path, make_source):
        run = _run(tmp_path, make_source(tmp_path))

        assert run.report.is_file()
        assert not (run.output / _REPORT).exists()

    @pytest.mark.parametrize("make_source", _NOTHING_USABLE_SOURCES)
    def test_no_warning_names_the_source(self, tmp_path, make_source, pipeline_caplog):
        run = _run(tmp_path, make_source(tmp_path))

        assert [text for text in _messages(pipeline_caplog, logging.WARNING) if str(run.source) in text] == []

    @pytest.mark.parametrize("make_source", _NOTHING_USABLE_SOURCES)
    def test_no_copied_count(self, tmp_path, make_source, pipeline_caplog):
        run = _run(tmp_path, make_source(tmp_path))

        assert [text for text in _messages(pipeline_caplog, OUTPUT) if str(run.output) in text] == []

#!/usr/bin/env python3
"""
Stage 5 input QC tests of put_together_embeddings.py: input_qc.tsv over the full_embeddings.csv of the regions.

The source of each test is what stage 4 leaves when all goes well: every default region in both hemispheres, the same
subjects in each file, the embedding_width.json of each region, and an output_qc.tsv keeping them all.
"""

import json
import logging
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import (
    HEMISPHERES,
    compute_combined_embeddings_dir,
    compute_region_embeddings_dir,
    compute_region_model_name,
)
from champollion_pipeline.generate_masks import _REGIONS_DEFAULT
from champollion_pipeline.models import embeddings_qc
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.streaming import combine
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.utils.qc_report import (
    ID_COLUMN,
    QC_COLUMN,
    REASON_COLUMN,
    REASON_SEPARATOR,
    read_qc_tsv,
    write_qc_tsv,
)
from tests.log_assertions import FAIL, assert_one_subject_error
from tests.main_stage_helpers import capture_stage_argv, load_main_module, stage_paths_config

pytestmark = pytest.mark.unit  # noqa: V107

_LOGGER_NAME = "champollion_pipeline.put_together_embeddings"
_MODULE = "champollion_pipeline.put_together_embeddings"
_EMBEDDINGS = "full_embeddings.csv"
_REPORT = "input_qc.tsv"
_SIDECAR = "embedding_width.json"
_REGIONS = tuple(compute_region_model_name(region, side) for region in _REGIONS_DEFAULT for side in HEMISPHERES)
_REGION = _REGIONS[0]
_OTHER_REGION = _REGIONS[1]
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_ABSENT_SUBJECT = "sub-03"
_EXCLUDED_SUBJECT = "sub-04"
_STRAY = "sub-99"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)
_KEPT = {subject: 1 for subject in _COHORT}
_GOOD_CELLS = ("0.5", "0.5", "0.5")
_WIDTH = len(_GOOD_CELLS)
_HEADER = "ID,dim1,dim2,dim3\n"
_WIDE_CELLS = ("0.5", "0.5", "0.5", "0.5")
_WIDE_HEADER = "ID,dim1,dim2,dim3,dim4\n"
_SENTINEL = "sentinel"
_MASKS = "test_masks"
_DATASET_REGIONS = _REGIONS_DEFAULT[:2]
_DATASET_REGION_NAMES = [compute_region_model_name(region, side) for region in _DATASET_REGIONS for side in HEMISPHERES]


def _line(subject: str, cells: tuple[str, ...] = _GOOD_CELLS) -> str:
    return ",".join((subject, *cells)) + "\n"


def _table(*lines: str) -> str:
    return _HEADER + "".join(lines)


def _good_table() -> str:
    return _table(*(_line(subject) for subject in _COHORT))


def _with_bad_row(*bad_lines: str) -> str:
    """The table of _COHORT with _OTHER_SUBJECT's row replaced by ``bad_lines``."""
    return _table(_line(_SUBJECT), *bad_lines)


def _wide_table() -> str:
    """A table of _COHORT one cell wider than the sidecars of _make_source."""
    return _WIDE_HEADER + "".join(_line(subject, _WIDE_CELLS) for subject in _COHORT)


def _write_sidecar(source: Path, region: str, content: bytes | None) -> None:
    """Write the embedding_width.json of ``region``; None removes it."""
    path = source / region / _SIDECAR
    if content is None:
        path.unlink()
    else:
        path.write_bytes(content)


def _make_source(
    tmp_path: Path,
    tables: Mapping[str, str],
    stage4_report: Mapping[str, int] | None = _KEPT,
    has_sidecars: bool = True,
) -> Path:
    """The region embeddings folder: each region of _REGIONS, with its text of ``tables`` or a good table.

    ``stage4_report`` gives the qc of each subject of output_qc.tsv; None leaves that file out.
    ``has_sidecars`` False leaves out each embedding_width.json, as a streaming run does.
    """
    source = tmp_path / "region_embeddings"
    for region in _REGIONS:
        (source / region).mkdir(parents=True)
        (source / region / _EMBEDDINGS).write_text(tables.get(region, _good_table()), encoding="utf-8")
        if has_sidecars:
            _write_sidecar(source, region, json.dumps({"width": _WIDTH}).encode("utf-8"))
    if stage4_report is not None:
        rows = [{ID_COLUMN: subject, QC_COLUMN: qc, REASON_COLUMN: ""} for subject, qc in stage4_report.items()]
        write_qc_tsv(source / "output_qc.tsv", rows)
    return source


def _make_empty_source(tmp_path: Path) -> Path:
    source = tmp_path / "region_embeddings"
    source.mkdir()
    return source


@dataclass(frozen=True)
class _Run:
    status: int
    source: Path
    output: Path

    @property
    def report(self) -> Path:
        return self.output / _REPORT

    def rows(self) -> dict[str, tuple[int, str]]:
        """(qc, reason) by participant_id of the report."""
        records = read_qc_tsv(self.report)
        return {record[ID_COLUMN]: (record[QC_COLUMN], record[REASON_COLUMN]) for record in records}

    def copied(self) -> list[str]:
        return sorted(path.name for path in self.output.glob("*_embeddings.csv"))

    def copy_of(self, region: str) -> Path:
        return self.output / f"{region}_embeddings.csv"


def _run(tmp_path: Path, source: Path, *options: str) -> _Run:
    output = tmp_path / "combined"
    script = PutTogetherEmbeddings()
    script.parse_args([str(source), "--output_path", str(output), *options])
    return _Run(script.run(), source, output)


def _run_tables(tmp_path: Path, tables: Mapping[str, str] | None = None) -> _Run:
    return _run(tmp_path, _make_source(tmp_path, tables or {}))


def _run_subject_lost(tmp_path: Path) -> _Run:
    """A run where _REGION holds no row of _OTHER_SUBJECT; the run must keep _SUBJECT."""
    run = _run_tables(tmp_path, {_REGION: _table(_line(_SUBJECT))})
    assert run.rows()[_OTHER_SUBJECT][0] == 0
    assert run.rows()[_SUBJECT][0] == 1
    return run


def _messages(capture, level: int) -> list[str]:
    return [
        record.getMessage() for record in capture.records if record.levelno == level and record.name == _LOGGER_NAME
    ]


@contextmanager
def _shared_check(fake: Callable) -> Iterator[None]:
    """Replace the shared check, in its own module and in the stage module."""
    with (
        patch.object(embeddings_qc, "fetch_embeddings_csv_reasons", fake),
        patch(f"{_MODULE}.fetch_embeddings_csv_reasons", fake, create=True),
    ):
        yield


def _checked_regions(tmp_path: Path, source: Path, *options: str) -> set[str]:
    """The regions the shared check was called on in a run."""
    regions = []

    def fake_check(csv_path, region, expected_subjects, width):
        regions.append(region)
        return {}

    with _shared_check(fake_check):
        _run(tmp_path, source, *options)
    return set(regions)


def _checked_widths(tmp_path: Path, source: Path) -> dict[str, int | None]:
    """The width the shared check was given for each region in a run."""
    widths = {}

    def fake_check(csv_path, region, expected_subjects, width):
        widths[region] = width
        return {}

    with _shared_check(fake_check):
        _run(tmp_path, source)
    return widths


def _streaming_settings(tmp_path: Path) -> StreamingSettings:
    return StreamingSettings(
        subjects_dir=str(tmp_path / "subjects"), output_dir=str(tmp_path / "out"), log_dir=str(tmp_path / "logs")
    )


def _streaming_dataset(tmp_path: Path, regions: tuple[str, ...] = ()) -> DatasetConfig:
    root = str(tmp_path / "dataset")
    Path(compute_region_embeddings_dir(root, _MASKS)).mkdir(parents=True)
    return DatasetConfig(datasets_root=root, masks_version=_MASKS, regions=list(regions))


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage5_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


class TestReport:
    def test_report_exists_before_the_first_copy(self, tmp_path):
        source = _make_source(tmp_path, {})
        report = tmp_path / "combined" / _REPORT
        seen = []
        real_copyfile = shutil.copyfile

        def spy(src, dst, **kwargs):
            seen.append(report.exists())
            return real_copyfile(src, dst, **kwargs)

        with patch(f"{_MODULE}.shutil.copyfile", side_effect=spy):
            _run(tmp_path, source)

        assert seen, "no file was copied"
        assert all(seen), "a file was copied before input_qc.tsv existed"

    def test_report_goes_through_the_shared_writer(self, tmp_path):
        with patch(f"{_MODULE}.write_qc_tsv", create=True, side_effect=write_qc_tsv) as writer:
            run = _run_tables(tmp_path)

        assert [str(call.args[0]) for call in writer.call_args_list] == [str(run.report)]


class TestRows:
    def test_usable_subject_row(self, tmp_path):
        run = _run_tables(tmp_path)

        assert run.rows() == {subject: (1, "") for subject in _COHORT}

    @pytest.mark.parametrize(
        ("bad_lines", "reason"),
        [
            pytest.param((_line(_OTHER_SUBJECT, ("0.5", "abc", "0.5")),), "non_numeric", id="text_cell"),
            pytest.param((_line(_OTHER_SUBJECT, ("0.5", "nan", "0.5")),), "non_finite", id="nan_cell"),
            pytest.param((_line(_OTHER_SUBJECT, ("0.5", "inf", "0.5")),), "non_finite", id="infinite_cell"),
            pytest.param((_line(_OTHER_SUBJECT, ("0.5", "", "0.5")),), "non_finite", id="empty_cell"),
            pytest.param((_line(_OTHER_SUBJECT), _line(_OTHER_SUBJECT)), "duplicate_row", id="repeated_row"),
        ],
    )
    def test_failing_subject_row(self, tmp_path, bad_lines, reason):
        run = _run_tables(tmp_path, {_REGION: _with_bad_row(*bad_lines)})

        assert run.rows() == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (0, f"{reason}:{_REGION}")}

    def test_reasons_come_from_the_shared_check(self, tmp_path):
        def fake_check(csv_path, region, expected_subjects, width):
            return {_OTHER_SUBJECT: [f"{_SENTINEL}:{region}"]}

        with _shared_check(fake_check):
            run = _run_tables(tmp_path)

        qc, reason = run.rows()[_OTHER_SUBJECT]
        assert qc == 0
        assert set(reason.split(REASON_SEPARATOR)) == {f"{_SENTINEL}:{region}" for region in _REGIONS}

    def test_several_reasons_joined_once(self, tmp_path):
        tables = {
            _REGION: _with_bad_row(_line(_OTHER_SUBJECT, ("abc", "abc", "0.5"))),
            _OTHER_REGION: _with_bad_row(_line(_OTHER_SUBJECT, ("nan", "0.5", "0.5"))),
        }

        run = _run_tables(tmp_path, tables)

        qc, reason = run.rows()[_OTHER_SUBJECT]
        assert qc == 0
        expected = {f"non_numeric:{_REGION}": 1, f"non_finite:{_OTHER_REGION}": 1}
        assert Counter(reason.split(REASON_SEPARATOR)) == expected


class TestLogging:
    def test_one_error_per_excluded_subject(self, tmp_path, pipeline_caplog):
        bad_line = _line(_OTHER_SUBJECT, ("abc", "0.5", "0.5"))

        _run_tables(tmp_path, {_REGION: _with_bad_row(bad_line)})

        errors = _messages(pipeline_caplog, logging.ERROR)
        assert_one_subject_error(errors, _OTHER_SUBJECT, f"non_numeric:{_REGION}")


def _no_folder_of_a_region(source: Path) -> None:
    shutil.rmtree(source / _REGION)


def _folder_of_cka_results(source: Path) -> None:
    (source / "cka_results").mkdir()


def _folder_of_a_stray_model(source: Path) -> None:
    (source / "stray_left").mkdir()
    (source / "stray_left" / _EMBEDDINGS).write_text(_good_table(), encoding="utf-8")


class TestExpectedRegions:
    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(_no_folder_of_a_region, id="no_folder_of_a_region"),
            pytest.param(_folder_of_cka_results, id="cka_results_folder"),
            pytest.param(_folder_of_a_stray_model, id="stray_model_folder"),
        ],
    )
    def test_default_regions_checked(self, tmp_path, change):
        source = _make_source(tmp_path, {})
        change(source)

        assert _checked_regions(tmp_path, source) == set(_REGIONS)

    def test_named_regions_checked(self, tmp_path):
        source = _make_source(tmp_path, {})

        checked = _checked_regions(tmp_path, source, "--regions", _REGION, _OTHER_REGION)

        assert checked == {_REGION, _OTHER_REGION}

    def test_named_regions_copied(self, tmp_path):
        run = _run(tmp_path, _make_source(tmp_path, {}), "--regions", _REGION, _OTHER_REGION)

        assert run.copied() == sorted(f"{region}_embeddings.csv" for region in (_REGION, _OTHER_REGION))

    def test_main_passes_the_regions(self):
        main_module = load_main_module("_champollion_main_stage5_input_qc_under_test")
        config = stage_paths_config(main_module)
        config.dataset.regions = list(_DATASET_REGIONS)  # noqa: V101

        argv = capture_stage_argv(
            main_module.PutTogetherEmbeddingsStage,
            f"{_MODULE}.PutTogetherEmbeddings",
            "PutTogetherEmbeddingsStage",
            config,
            "test_put_together_embeddings_input_qc",
        )

        assert PutTogetherEmbeddings().parse_args(argv).regions == _DATASET_REGION_NAMES

    def test_streaming_passes_the_regions(self, tmp_path, monkeypatch):
        seen = []

        class Recording(PutTogetherEmbeddings):
            def run(self) -> int:
                seen.append(getattr(self.args, "regions", None))
                return 0

        monkeypatch.setattr(combine, "PutTogetherEmbeddings", Recording)

        combine.run_combine(_streaming_settings(tmp_path), _streaming_dataset(tmp_path, _DATASET_REGIONS))

        assert seen == [_DATASET_REGION_NAMES]


class TestExpectedSubjects:
    def test_rows_follow_the_stage_4_report(self, tmp_path):
        stage4_report = {**_KEPT, _ABSENT_SUBJECT: 1, _EXCLUDED_SUBJECT: 0}

        run = _run(tmp_path, _make_source(tmp_path, {}, stage4_report))

        assert set(run.rows()) == {*_COHORT, _ABSENT_SUBJECT}

    def test_rows_follow_the_region_files_without_a_report(self, tmp_path):
        tables = {_REGION: _table(_line(_SUBJECT), _line(_ABSENT_SUBJECT))}

        run = _run(tmp_path, _make_source(tmp_path, tables, stage4_report=None))

        assert set(run.rows()) == {*_COHORT, _ABSENT_SUBJECT}

    @pytest.mark.parametrize(
        "stage4_report",
        [
            pytest.param(_KEPT, id="id_outside_the_report"),
            pytest.param({**_KEPT, _STRAY: 0}, id="id_excluded_by_stage_4"),
        ],
    )
    def test_unexpected_subject_row(self, tmp_path, stage4_report):
        tables = {_REGION: _table(*(_line(subject) for subject in (*_COHORT, _STRAY)))}

        run = _run(tmp_path, _make_source(tmp_path, tables, stage4_report))

        assert run.rows()[_STRAY] == (0, f"unexpected_row:{_REGION}")


class TestSubjectLost:
    def test_missing_subject_row(self, tmp_path):
        run = _run_tables(tmp_path, {_REGION: _table(_line(_SUBJECT))})

        assert run.rows() == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (0, f"missing_row:{_REGION}")}

    def test_returns_3(self, tmp_path):
        assert _run_subject_lost(tmp_path).status == 3

    def test_copies_each_region(self, tmp_path):
        run = _run_subject_lost(tmp_path)

        assert run.copied() == sorted(f"{region}_embeddings.csv" for region in _REGIONS)

    def test_copies_hold_no_excluded_subject(self, tmp_path):
        run = _run_subject_lost(tmp_path)

        holding = [region for region in _REGIONS if _OTHER_SUBJECT in run.copy_of(region).read_text(encoding="utf-8")]
        assert holding == []

    def test_copies_keep_the_other_rows(self, tmp_path):
        run = _run_subject_lost(tmp_path)

        expected = _table(_line(_SUBJECT)).encode("utf-8")
        assert {run.copy_of(region).read_bytes() for region in _REGIONS} == {expected}

    def test_region_files_stay_untouched(self, tmp_path):
        run = _run_subject_lost(tmp_path)

        assert (run.source / _REGION / _EMBEDDINGS).read_text(encoding="utf-8") == _table(_line(_SUBJECT))
        assert (run.source / _OTHER_REGION / _EMBEDDINGS).read_text(encoding="utf-8") == _good_table()

    def test_streaming_accepts_a_partial_run(self, tmp_path, monkeypatch):
        class Partial(PutTogetherEmbeddings):
            def run(self) -> int:
                return 3

        monkeypatch.setattr(combine, "PutTogetherEmbeddings", Partial)
        dataset = _streaming_dataset(tmp_path)

        combined = combine.run_combine(_streaming_settings(tmp_path), dataset)

        assert combined == compute_combined_embeddings_dir(dataset.datasets_root, _MASKS)


_NO_USABLE_SIDECAR = pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="no_file"),
        pytest.param(b"{", id="not_json"),
        pytest.param(b"[3]", id="not_an_object"),
        pytest.param(b"{}", id="no_key"),
        pytest.param(b'{"width": 0}', id="zero"),
        pytest.param(b'{"width": -3}', id="negative"),
        pytest.param(b'{"width": true}', id="boolean"),
        pytest.param(b'{"width": 3.5}', id="float"),
        pytest.param(b'{"width": "3"}', id="text"),
    ],
)


class TestWidth:
    def test_sidecar_gives_the_width(self, tmp_path):
        source = _make_source(tmp_path, {})
        _write_sidecar(source, _REGION, b'{"width": 7}')

        assert _checked_widths(tmp_path, source)[_REGION] == 7

    def test_sidecar_wins_over_the_header(self, tmp_path):
        run = _run_tables(tmp_path, {_REGION: _wide_table()})

        assert run.rows() == {subject: (0, f"wrong_width:{_REGION}") for subject in _COHORT}

    @_NO_USABLE_SIDECAR
    def test_header_gives_the_width_without_a_usable_sidecar(self, tmp_path, content):
        source = _make_source(tmp_path, {_REGION: _wide_table()})
        _write_sidecar(source, _REGION, content)

        assert _checked_widths(tmp_path, source)[_REGION] == len(_WIDE_CELLS)

    def test_source_without_sidecars_keeps_every_subject(self, tmp_path):
        run = _run(tmp_path, _make_source(tmp_path, {}, has_sidecars=False))

        assert run.rows() == {subject: (1, "") for subject in _COHORT}
        assert run.status == 0

    @_NO_USABLE_SIDECAR
    def test_one_warning_names_the_region(self, tmp_path, content, pipeline_caplog):
        source = _make_source(tmp_path, {})
        _write_sidecar(source, _REGION, content)

        _run(tmp_path, source)

        warnings = _messages(pipeline_caplog, logging.WARNING)
        assert len([message for message in warnings if _REGION in message]) == 1
        assert [message for message in warnings if _OTHER_REGION in message] == []

    @pytest.mark.parametrize("has_sidecars", [True, False], ids=["sidecar_width", "header_width"])
    def test_row_of_another_width(self, tmp_path, has_sidecars):
        tables = {_REGION: _with_bad_row(_line(_OTHER_SUBJECT, _WIDE_CELLS))}

        run = _run(tmp_path, _make_source(tmp_path, tables, has_sidecars=has_sidecars))

        assert run.rows() == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (0, f"wrong_width:{_REGION}")}


def _remove_file(path: Path) -> None:
    path.unlink()


def _remove_folder(path: Path) -> None:
    shutil.rmtree(path.parent)


def _write_bytes(data: bytes) -> Callable[[Path], None]:
    return lambda path: path.write_bytes(data)


class TestWholeFileFault:
    @pytest.mark.parametrize(
        ("fault", "reason"),
        [
            pytest.param(_remove_file, "missing_embeddings", id="absent_file"),
            pytest.param(_remove_folder, "missing_embeddings", id="absent_folder"),
            pytest.param(_write_bytes(b"\xff\xfe\x00"), "unreadable_embeddings", id="not_utf8"),
            pytest.param(_write_bytes(b""), "unreadable_embeddings", id="empty_file"),
            pytest.param(_write_bytes(b"Subject,dim1\nsub-01,0.5\n"), "unreadable_embeddings", id="no_id_column"),
            pytest.param(
                _write_bytes(_good_table().encode("utf-8") + _line("").encode("utf-8")),
                "unreadable_embeddings",
                id="empty_id_cell",
            ),
        ],
    )
    def test_fault_reaches_each_row(self, tmp_path, fault, reason):
        tables = {_OTHER_REGION: _table(*(_line(subject) for subject in (*_COHORT, _STRAY)))}
        source = _make_source(tmp_path, tables)
        fault(source / _REGION / _EMBEDDINGS)

        rows = _run(tmp_path, source).rows()

        assert set(rows) == {*_COHORT, _STRAY}
        for qc, cell in rows.values():
            assert qc == 0
            assert f"{reason}:{_REGION}" in cell.split(REASON_SEPARATOR)


def _every_subject_failing(tmp_path: Path) -> Path:
    bad_cells = ("abc", "0.5", "0.5")
    return _make_source(tmp_path, {_REGION: _table(*(_line(subject, bad_cells) for subject in _COHORT))})


def _one_region_file_absent(tmp_path: Path) -> Path:
    source = _make_source(tmp_path, {})
    (source / _REGION / _EMBEDDINGS).unlink()
    return source


@pytest.fixture(  # noqa: V103
    params=[
        pytest.param(_every_subject_failing, id="every_subject_failing"),
        pytest.param(_one_region_file_absent, id="one_region_file_absent"),
        pytest.param(_make_empty_source, id="no_region_folder"),
    ]
)
def unusable_source(request, tmp_path) -> Path:
    """A source whose run leaves no qc 1 row. The run stays in the test: caplog keeps the records of a phase alone."""
    return request.param(tmp_path)


class TestNothingUsable:
    def test_returns_1(self, tmp_path, unusable_source):
        assert _run(tmp_path, unusable_source).status == 1

    def test_one_fail_record_names_the_report(self, tmp_path, unusable_source, pipeline_caplog):
        report = _run(tmp_path, unusable_source).report

        [failure] = _messages(pipeline_caplog, FAIL)
        assert str(report) in failure

    def test_copies_nothing(self, tmp_path, unusable_source):
        run = _run(tmp_path, unusable_source)

        assert run.report.exists()
        assert run.copied() == []


class TestEverySubjectKept:
    def test_returns_0(self, tmp_path):
        run = _run_tables(tmp_path)

        assert {qc for qc, _ in run.rows().values()} == {1}
        assert run.status == 0

    def test_copies_each_file_unchanged(self, tmp_path):
        run = _run_tables(tmp_path)

        assert {qc for qc, _ in run.rows().values()} == {1}
        assert run.copied() == sorted(f"{region}_embeddings.csv" for region in _REGIONS)
        for region in _REGIONS:
            expected = (run.source / region / _EMBEDDINGS).read_bytes()
            assert run.copy_of(region).read_bytes() == expected

#!/usr/bin/env python3
"""
generate_embeddings: per-region check of the crops' Mask version against each region model's Mask version for a local
model set (MODELVER epic, TASK-MODELVER-BDRABCZUK-86405DAD28CE).

The per-region Mask version resolver (models.version_options.resolve_region_mask_versions) reads, per region model
name, model_set.json masks, else the PROVENANCE.csv rows of the name, else the models folder name. Stage 4 compares
what it finds with the crops it will read (--masks, or the --region-masks map):

- REQ-MODELVER-BDRABCZUK-440EB622BC61: map <V> for <R> vs resolved <K>: exit 2 naming <R>, <V>, <K>.
- REQ-MODELVER-BDRABCZUK-FCE290A16A59: no map, no model_set.json, resolved <K> for <R> vs --masks <M>: exit 2.
- REQ-MODELVER-BDRABCZUK-A670F4A83880, REQ-MODELVER-BDRABCZUK-AD2E14B6276C: the same with --allow-version-mismatch,
  one WARNING record per region model name (user decision 2026-11-23 ~21:33 (3)).
- REQ-MODELVER-BDRABCZUK-E507E2E0EB9B (supersedes REQ-MODELVER-BDRABCZUK-8FA8D4F8599A): model_set.json masks vs
  --masks accepted: exactly one WARNING record naming both, no UserWarning.
- REQ-MODELVER-BDRABCZUK-82FF84FAD95C: unresolved region model names with a map: the run goes on after one WARNING
  record (user decision (1)).
- REQ-MODELVER-BDRABCZUK-D99E7930528B, REQ-MODELVER-BDRABCZUK-1C90B4E059D7: a region model name the resolver reports
  as conflicting is a mismatch: exit 2, or one WARNING record per region model name with --allow-version-mismatch
  (user decision (2)).

User answers 2026-11-23 ~22:15 to the design questions:

- REQ-MODELVER-BDRABCZUK-56046AECE972 (Q2): an unmarked set whose region model names agree with their crops: no
  UserWarning, no WARNING record.
- REQ-MODELVER-BDRABCZUK-4B945E3FF5C1 (Q3): map run, model_set.json refused by read_model_set: exit 2 naming the file,
  with or without --allow-version-mismatch.
- REQ-MODELVER-BDRABCZUK-1DEF28F26ED5 (Q3): map run, model_set.json masks not a known Mask version: one WARNING record.
- REQ-MODELVER-BDRABCZUK-3AE63A554F2A (Q4): a region folder named none of the 56 region model names is unresolved.
- REQ-MODELVER-BDRABCZUK-DBFA0EBE52B2, REQ-MODELVER-BDRABCZUK-524734E7EFE7 (Q5): a mismatch exit logs no unresolved
  WARNING record; several mismatches exit 2 with one error message naming each.

User answers 2026-11-23 ~22:40:

- REQ-MODELVER-BDRABCZUK-BA8C954A2941 (1): no map, model_set.json masks <M> equal to --masks but not a known Mask
  version: the run goes on after exactly one WARNING record naming models_path and <M>.
- REQ-MODELVER-BDRABCZUK-3AE63A554F2A (2): the foreign region folder counts as unresolved under a map too.
- REQ-MODELVER-BDRABCZUK-48ED0E8AD3FC (3, supersedes REQ-MODELVER-BDRABCZUK-1865332778D2 and
  REQ-MODELVER-BDRABCZUK-5A0D8B461FBD): no map, unresolved names, --masks typed or defaulted (from --model-version or
  the folder name): the run goes on after exactly one WARNING record naming models_path, <U> and the resolved --masks;
  no UserWarning.

No model weights, no evaluate.py, no network, no real dataset: only the real GenerateEmbeddings parser on tmp_path
model sets whose region folders hold an empty logs/ directory.
"""

import json
import logging
import warnings

import pytest

from champollion_pipeline.generate_embeddings import MODEL_SET_FILE_NAME, GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

C25 = "canonical_25"
C26 = "canonical_corrected_26_1"
R1 = "SC-sylv_left"
R2 = "SC-sylv_right"
PROVENANCE_HEADER = "region,mask_version_trained_on\n"


def _model_set(tmp_path, regions, provenance=None, marker_masks=None, folder="models"):
    """A local model set tmp_path/<folder> with one <region>/logs/ folder per region.

    provenance: (region model name, Mask version) rows written as PROVENANCE.csv, or its raw text (a str);
    marker_masks: write model_set.json naming that masks value.
    """
    models = tmp_path / folder
    models.mkdir(parents=True)
    for region in regions:
        (models / region / "logs").mkdir(parents=True)
    if isinstance(provenance, str):
        (models / "PROVENANCE.csv").write_text(provenance)
    elif provenance is not None:
        rows = "".join(f"{region},{version}\n" for region, version in provenance)
        (models / "PROVENANCE.csv").write_text(PROVENANCE_HEADER + rows)
    if marker_masks is not None:
        marker = {"model_version": marker_masks, "masks": marker_masks}
        (models / MODEL_SET_FILE_NAME).write_text(json.dumps(marker))
    return models


def _datasets_root(tmp_path) -> str:
    return str(tmp_path / "TEST01")


def _parse(argv: list[str]) -> tuple[int | str | None, list[str]]:
    """Parse argv with the real parser, recording every Python warning; (exit code or None, UserWarning messages)."""
    code = None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            GenerateEmbeddings().parse_args(argv)
        except SystemExit as exc:
            code = exc.code
    return code, [str(w.message) for w in record if issubclass(w.category, UserWarning)]


def _parse_ok(argv: list[str]) -> list[str]:
    """Parse argv; fail on SystemExit; return the UserWarning messages."""
    code, user_warnings = _parse(argv)
    if code is not None:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {code})")
    return user_warnings


def _warnings(pipeline_caplog) -> list[str]:
    return [r.getMessage() for r in pipeline_caplog.records if r.levelno == logging.WARNING]


def _naming(messages: list[str], *words: str) -> list[str]:
    return [message for message in messages if all(word in message for word in words)]


def _assert_one_warning_naming(pipeline_caplog, *words: str) -> None:
    """Exactly one WARNING record was logged, and it names each of words."""
    messages = _warnings(pipeline_caplog)
    assert len(messages) == 1, messages
    assert _naming(messages, *words), messages


def _expect_exit_2(argv: list[str], capsys) -> str:
    code, _ = _parse(argv)
    stderr = capsys.readouterr().err
    assert code == 2, f"exit {code!r}, stderr {stderr!r}"
    return stderr


class TestMapMismatchExits2:
    """REQ-MODELVER-BDRABCZUK-440EB622BC61: the map names <V> for <R>, the resolver gives <K>: exit 2 naming all."""

    def test_exits_2_naming_region_and_both_versions(self, tmp_path, capsys):
        models = _model_set(tmp_path, [R1], provenance=[(R1, C25)])
        argv = [str(models), _datasets_root(tmp_path), "--region-masks", f"{R1}={C26}", "--model-version", C26]
        stderr = _expect_exit_2(argv, capsys)
        assert R1 in stderr and C25 in stderr and C26 in stderr, stderr


class TestUnmarkedMismatchExits2:
    """REQ-MODELVER-BDRABCZUK-FCE290A16A59: no model_set.json, the resolver gives <K> for <R>, --masks <M>: exit 2."""

    def test_exits_2_naming_region_and_both_versions(self, tmp_path, capsys):
        models = _model_set(tmp_path, [R1], provenance=[(R1, C25)])
        stderr = _expect_exit_2([str(models), _datasets_root(tmp_path), "--masks", C26], capsys)
        assert R1 in stderr and C25 in stderr and C26 in stderr, stderr


class _PerRegionWarningRecords:
    """Shared tests of an accepted per-region mismatch or conflict on R1 and R2 between canonical_25 and
    canonical_corrected_26_1: one WARNING record per region model name. A subclass gives _argv(tmp_path)."""

    @staticmethod
    def _argv(tmp_path) -> list[str]:
        raise NotImplementedError

    def test_one_warning_record_per_region_naming_both_versions(self, tmp_path, pipeline_caplog):
        _parse_ok(self._argv(tmp_path))
        messages = _warnings(pipeline_caplog)
        assert len(_naming(messages, R1, C25, C26)) == 1, messages
        assert len(_naming(messages, R2, C25, C26)) == 1, messages

    def test_each_record_names_one_region_model_name(self, tmp_path, pipeline_caplog):
        _parse_ok(self._argv(tmp_path))
        messages = _warnings(pipeline_caplog)
        assert _naming(messages, R1, C25, C26) and not _naming(messages, R1, R2), messages


class TestMapMismatchAllowed(_PerRegionWarningRecords):
    """REQ-MODELVER-BDRABCZUK-A670F4A83880: accepted map mismatches give one WARNING record per region model name."""

    @staticmethod
    def _argv(tmp_path) -> list[str]:
        models = _model_set(tmp_path, [R1, R2], provenance=[(R1, C25), (R2, C25)])
        return [
            str(models),
            _datasets_root(tmp_path),
            "--region-masks",
            f"{R1}={C26}",
            f"{R2}={C26}",
            "--model-version",
            C26,
            "--allow-version-mismatch",
        ]


class TestUnmarkedMismatchAllowed(_PerRegionWarningRecords):
    """REQ-MODELVER-BDRABCZUK-AD2E14B6276C: accepted mismatches of an unmarked local model set against --masks give
    one WARNING record per region model name."""

    @staticmethod
    def _argv(tmp_path) -> list[str]:
        models = _model_set(tmp_path, [R1, R2], provenance=[(R1, C25), (R2, C25)])
        return [str(models), _datasets_root(tmp_path), "--masks", C26, "--allow-version-mismatch"]

    def test_emits_no_user_warning(self, tmp_path):
        assert _parse_ok(self._argv(tmp_path)) == []


class TestMarkerMismatchAllowedRecord:
    """REQ-MODELVER-BDRABCZUK-E507E2E0EB9B: model_set.json masks <K> vs --masks <M> accepted: exactly one WARNING record
    naming both, not a UserWarning."""

    @staticmethod
    def _argv(tmp_path) -> list[str]:
        models = _model_set(tmp_path, [R1, R2], marker_masks=C26)
        return [str(models), _datasets_root(tmp_path), "--masks", C25, "--allow-version-mismatch"]

    def test_logs_exactly_one_warning_record_naming_both(self, tmp_path, pipeline_caplog):
        _parse_ok(self._argv(tmp_path))
        messages = _warnings(pipeline_caplog)
        assert len(_naming(messages, C25, C26)) == 1, messages

    def test_emits_no_user_warning(self, tmp_path):
        assert _parse_ok(self._argv(tmp_path)) == []


class TestUnresolvedWithMapWarns:
    """REQ-MODELVER-BDRABCZUK-82FF84FAD95C: map given, the resolver leaves <U> unresolved: the run goes on after
    exactly one WARNING record naming models_path, <U> and the mapped Mask versions of <U>."""

    def test_logs_exactly_one_warning_record_and_runs_on(self, tmp_path, pipeline_caplog):
        models = _model_set(tmp_path, [R1])
        argv = [str(models), _datasets_root(tmp_path), "--region-masks", f"{R1}={C25}", "--model-version", C25]
        _parse_ok(argv)
        _assert_one_warning_naming(pipeline_caplog, str(models), R1, C25)


class TestConflictExits2:
    """REQ-MODELVER-BDRABCZUK-D99E7930528B: the resolver reports <R> as conflicting between <A> and <B>: exit 2 naming
    <R>, <A> and <B>."""

    def test_disagreeing_provenance_rows_exit_2(self, tmp_path, capsys):
        models = _model_set(tmp_path, [R1], provenance=[(R1, C25), (R1, C26)])
        stderr = _expect_exit_2([str(models), _datasets_root(tmp_path), "--masks", C25], capsys)
        assert R1 in stderr and C25 in stderr and C26 in stderr, stderr

    def test_marker_against_folder_name_exits_2(self, tmp_path, capsys):
        """model_set.json names canonical_corrected_26_1, the folder is canonical_25, --masks equals the marker."""
        models = _model_set(tmp_path, [R1], marker_masks=C26, folder=C25)
        stderr = _expect_exit_2([str(models), _datasets_root(tmp_path), "--masks", C26], capsys)
        assert R1 in stderr and C26 in stderr, stderr


class TestConflictAllowed(_PerRegionWarningRecords):
    """REQ-MODELVER-BDRABCZUK-1C90B4E059D7: with --allow-version-mismatch a conflicting <R> gives one WARNING record
    naming <R>, <A> and <B>, per region model name."""

    @staticmethod
    def _argv(tmp_path) -> list[str]:
        rows = [(R1, C25), (R1, C26), (R2, C25), (R2, C26)]
        models = _model_set(tmp_path, [R1, R2], provenance=rows)
        return [str(models), _datasets_root(tmp_path), "--masks", C25, "--allow-version-mismatch"]


UNRESOLVED_WITHOUT_MAP_SETUPS = {
    "--masks typed, no PROVENANCE.csv": {"folder": "models", "provenance": None, "extra": ["--masks", C25]},
    "--masks typed, PROVENANCE.csv lacking a column": {
        "folder": "models",
        "provenance": "region\n" + f"{R1}\n",
        "extra": ["--masks", C25],
    },
    "--masks typed, unknown Mask version row": {
        "folder": "models",
        "provenance": [(R1, "canonical_99")],
        "extra": ["--masks", C25],
    },
    "--masks from --model-version": {"folder": "models", "provenance": None, "extra": ["--model-version", C25]},
    "--masks from the folder name": {"folder": C25, "provenance": [(R1, "canonical_99")], "extra": []},
}


class TestUnresolvedWithoutMapWarns:
    """REQ-MODELVER-BDRABCZUK-48ED0E8AD3FC: no map, no model_set.json, the resolver leaves <U> unresolved, --masks typed
    or defaulted: the run goes on after exactly one WARNING record naming models_path, <U> and the resolved --masks;
    no UserWarning."""

    @staticmethod
    def _argv(tmp_path, setup) -> tuple[list[str], str]:
        case = UNRESOLVED_WITHOUT_MAP_SETUPS[setup]
        models = _model_set(tmp_path, [R1], provenance=case["provenance"], folder=case["folder"])
        return [str(models), _datasets_root(tmp_path), *case["extra"]], str(models)

    @pytest.mark.parametrize("setup", sorted(UNRESOLVED_WITHOUT_MAP_SETUPS))
    def test_logs_exactly_one_warning_record_and_runs_on(self, tmp_path, pipeline_caplog, setup):
        argv, models_path = self._argv(tmp_path, setup)
        _parse_ok(argv)
        _assert_one_warning_naming(pipeline_caplog, models_path, R1, C25)

    @pytest.mark.parametrize("setup", sorted(UNRESOLVED_WITHOUT_MAP_SETUPS))
    def test_emits_no_user_warning(self, tmp_path, setup):
        argv, _ = self._argv(tmp_path, setup)
        assert _parse_ok(argv) == []


AGREEING_SETUPS = {
    "folder name": {"folder": C25, "provenance": None},
    "PROVENANCE.csv rows": {"folder": "models", "provenance": [(R1, C25), (R2, C25)]},
}


class TestAgreeingUnmarkedSetIsSilent:
    """REQ-MODELVER-BDRABCZUK-56046AECE972: no model_set.json, each region model name resolves to its crops' Mask
    version: no UserWarning (the old "model_set.json not found" warning is gone) and no WARNING record."""

    @pytest.mark.parametrize("setup", sorted(AGREEING_SETUPS))
    def test_no_user_warning_and_no_warning_record(self, tmp_path, pipeline_caplog, setup):
        models = _model_set(tmp_path, [R1, R2], **AGREEING_SETUPS[setup])
        user_warnings = _parse_ok([str(models), _datasets_root(tmp_path), "--masks", C25])
        assert user_warnings == [], user_warnings
        assert _warnings(pipeline_caplog) == []


REFUSED_MARKERS = {
    "not JSON": "{not json",
    "no model_version": json.dumps({"masks": C25}),
}


class TestMapUnreadableMarkerExits2:
    """REQ-MODELVER-BDRABCZUK-4B945E3FF5C1: map run whose model_set.json read_model_set refuses: exit 2 naming the
    model_set.json path, with or without --allow-version-mismatch."""

    @pytest.mark.parametrize("allow", [False, True], ids=["strict", "allow-version-mismatch"])
    @pytest.mark.parametrize("marker", sorted(REFUSED_MARKERS))
    def test_exits_2_naming_the_marker(self, tmp_path, capsys, marker, allow):
        models = _model_set(tmp_path, [R1])
        (models / MODEL_SET_FILE_NAME).write_text(REFUSED_MARKERS[marker])
        argv = [str(models), _datasets_root(tmp_path), "--region-masks", f"{R1}={C25}", "--model-version", C25]
        stderr = _expect_exit_2(argv + (["--allow-version-mismatch"] if allow else []), capsys)
        assert str(models / MODEL_SET_FILE_NAME) in stderr, stderr


class TestMapUnknownMarkerMasksWarns:
    """REQ-MODELVER-BDRABCZUK-1DEF28F26ED5: map run whose model_set.json masks is not a known Mask version: the run
    goes on after exactly one WARNING record naming models_path, <U> and the mapped Mask versions of <U>."""

    def test_logs_exactly_one_warning_record_and_runs_on(self, tmp_path, pipeline_caplog):
        models = _model_set(tmp_path, [R1], marker_masks="canonical_99")
        argv = [str(models), _datasets_root(tmp_path), "--region-masks", f"{R1}={C25}", "--model-version", C25]
        _parse_ok(argv)
        _assert_one_warning_naming(pipeline_caplog, str(models), R1, C25)


UNKNOWN_MASKS = "canonical_99"
UNKNOWN_MARKER_MASKS_ARGV = {
    "--masks typed": ["--masks", UNKNOWN_MASKS],
    "--masks from --model-version": ["--model-version", UNKNOWN_MASKS],
}


class TestUnknownMarkerMasksEqualToMasksWarns:
    """REQ-MODELVER-BDRABCZUK-BA8C954A2941: no map, model_set.json masks <M> equal to the resolved --masks but not a
    known Mask version: the run goes on after exactly one WARNING record naming models_path and <M>."""

    @pytest.mark.parametrize("masks", sorted(UNKNOWN_MARKER_MASKS_ARGV))
    def test_logs_exactly_one_warning_record_and_runs_on(self, tmp_path, pipeline_caplog, masks):
        models = _model_set(tmp_path, [R1, R2], marker_masks=UNKNOWN_MASKS)
        _parse_ok([str(models), _datasets_root(tmp_path), *UNKNOWN_MARKER_MASKS_ARGV[masks]])
        _assert_one_warning_naming(pipeline_caplog, str(models), UNKNOWN_MASKS)


FOREIGN_REGION = "NOTAREGION_left"


FOREIGN_REGION_CROPS_ARGV = {
    "--masks": ["--masks", C25],
    "map": ["--region-masks", f"{R1}={C25}", "--model-version", C25],
}


class TestForeignRegionFolderUnresolved:
    """REQ-MODELVER-BDRABCZUK-3AE63A554F2A: a region folder named none of the 56 region model names is reported as an
    unresolved region model name in the one WARNING record, with --masks or with a map (user answer 2026-11-23 ~22:40
    (2): the map cannot name the folder, as the map takes only the 56 region model names)."""

    @pytest.mark.parametrize("crops", sorted(FOREIGN_REGION_CROPS_ARGV))
    def test_foreign_folder_named_in_the_unresolved_record(self, tmp_path, pipeline_caplog, crops):
        models = _model_set(tmp_path, [R1, FOREIGN_REGION], folder=C25)
        _parse_ok([str(models), _datasets_root(tmp_path), *FOREIGN_REGION_CROPS_ARGV[crops]])
        _assert_one_warning_naming(pipeline_caplog, str(models), FOREIGN_REGION)


class TestMismatchWithUnresolvedExits2:
    """REQ-MODELVER-BDRABCZUK-DBFA0EBE52B2: R1 mismatches --masks while R2 stays unresolved: exit 2, and no WARNING
    record naming R2 is logged first."""

    def test_exits_2_without_unresolved_record(self, tmp_path, capsys, pipeline_caplog):
        models = _model_set(tmp_path, [R1, R2], provenance=[(R1, C25)])
        _expect_exit_2([str(models), _datasets_root(tmp_path), "--masks", C26], capsys)
        messages = _warnings(pipeline_caplog)
        assert not _naming(messages, R2), messages


SEVERAL_MISMATCH_ARGV = {
    "--masks": ["--masks", C26],
    "map": ["--region-masks", f"{R1}={C26}", f"{R2}={C26}", "--model-version", C26],
}


class TestSeveralMismatchesOneMessage:
    """REQ-MODELVER-BDRABCZUK-524734E7EFE7: R1 and R2 both mismatch their crops: exit 2 with one error message naming
    each region model name with both Mask versions."""

    @pytest.mark.parametrize("crops", sorted(SEVERAL_MISMATCH_ARGV))
    def test_one_error_message_naming_each(self, tmp_path, capsys, crops):
        models = _model_set(tmp_path, [R1, R2], provenance=[(R1, C25), (R2, C25)])
        stderr = _expect_exit_2([str(models), _datasets_root(tmp_path), *SEVERAL_MISMATCH_ARGV[crops]], capsys)
        assert stderr.count("error:") == 1, stderr
        assert all(word in stderr for word in (R1, R2, C25, C26)), stderr

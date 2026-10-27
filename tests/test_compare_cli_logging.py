"""LOGGING convention in compare_sulcal.cli, the compare command front end (TASK-LOGGING-BDRABCZUK-E09CA1C56517).

The compare report, progress, warnings and errors go through the ``compare_sulcal.cli`` module
logger instead of print(); tests assert the records with caplog, not capsys. The CLI's runtime
routing (``Compare.main()`` -> ``init_process()`` -> ``setup_logging()``) sends LOG and OUTPUT
records to stdout and WARNING and above to stderr.

Errors (ERROR):
REQ-LOGGING-BDRABCZUK-28D7D679C6EC: soma.aims missing for an aims mode -> record containing soma.aims.
REQ-LOGGING-BDRABCZUK-308824D7FD8E: unknown mode -> record containing unknown mode and the mode.
REQ-LOGGING-BDRABCZUK-3FE5A8581207: crops/embeddings --set_a/--set_b not a directory -> option and value.
REQ-LOGGING-BDRABCZUK-51E9FF645858: one option of an embeddings option pair -> both option names.
REQ-LOGGING-BDRABCZUK-1B75B0890603: embeddings --k below 1 -> record containing --k.
REQ-LOGGING-BDRABCZUK-39408F1DBED5: unusable --covariate CSV -> the covariate loader's reason.
Warnings (WARNING):
REQ-LOGGING-BDRABCZUK-5F79D70A27D4: skipped region/side pair -> pair key and skip reason.
REQ-LOGGING-BDRABCZUK-ABC24F5D0D93: skipped UMAP projection -> pair key and UMAP skip reason.
REQ-LOGGING-BDRABCZUK-09536F7D87E1: subject without graph -> side and subject name.
REQ-LOGGING-BDRABCZUK-BFA363086F48: mask shape mismatch -> mask name and both shapes.
Progress (LOG):
REQ-LOGGING-BDRABCZUK-8E0EEB749F11: side load start -> side, subjects found, worker count.
REQ-LOGGING-BDRABCZUK-C0E7F5A2B021: side load end -> side, subjects loaded, subjects found.
REQ-LOGGING-BDRABCZUK-4853DF31C9F7: campaign load start -> campaign label and graph sub-path.
REQ-LOGGING-BDRABCZUK-851ED5321283: mask statistics load -> campaign label and mask directory.
Report (OUTPUT):
REQ-LOGGING-BDRABCZUK-A30AC0BB41ED: mask set counts.
REQ-LOGGING-BDRABCZUK-711B0AB759FB: written JSON report path.
REQ-LOGGING-BDRABCZUK-BAB5A93963DD: --xor_dir value.
REQ-LOGGING-BDRABCZUK-C36C3A5A0230: bucket table heading with the --bucket_step value.
REQ-LOGGING-BDRABCZUK-CD7AC89F7B1E: one record per bucket with label and mask count.
REQ-LOGGING-BDRABCZUK-DA74C3915C42: Unchanged masks: <unchanged>/<diffed>.
REQ-LOGGING-BDRABCZUK-966A96D86094: crops pair summary.
REQ-LOGGING-BDRABCZUK-C220E696F1F5: embeddings pair summary.
REQ-LOGGING-BDRABCZUK-953710019AEC: written per_subject.csv and summary.json paths.
REQ-LOGGING-BDRABCZUK-4F52E575CF0D: campaign subject overlap counts.
REQ-LOGGING-BDRABCZUK-D86FB7EA45B5: ratio table heading with both labels.
REQ-LOGGING-BDRABCZUK-490311FC3524: one record per ratio table row (sulcus, ratio).
REQ-LOGGING-BDRABCZUK-FE2D9A39729A: No sulcus rows found with both labels.
REQ-LOGGING-BDRABCZUK-9557B970AE39: CSV written to: <absolute --output>.
Static:
REQ-LOGGING-BDRABCZUK-05321B7586C6: no static logging text of cli.py matches the MCP fold regex.
REQ-LOGGING-BDRABCZUK-8383D55DEF0B (src/compare_masks.py keeps the report on stdout) is verified by
tests/test_compare_masks.py::TestMain::test_both_metrics_and_console_summary;
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 / -22ABEF2F2CBD (no print call, module logger) by
tests/test_logging_convention.py through LOG_CONVERTED_MODULES.
"""

import ast
import json
import logging
import re
from argparse import Namespace
from os.path import abspath
from pathlib import Path

import numpy as np
import pytest

import compare_sulcal.brainvisa
from compare_sulcal import crops, databases, embeddings, masks
from compare_sulcal.cli import Compare, _score_common_masks
from tests.log_assertions import COMPARE_LOGGER_NAME, LOG, OUTPUT, logged_messages

CLI_LOGGER = "compare_sulcal.cli"
CLI_PY = Path(__file__).resolve().parents[1] / "src" / "compare_sulcal" / "cli.py"
# Copy of champollion_sulcal_mcp runner.FOLD_RE (it reads the stage output line by line).
MCP_FOLD_RE = re.compile(r"[Ff]old\s+\d+\s*(?:/|of)\s*\d+")

LABEL_A = "campaignAlpha"
LABEL_B = "campaignBeta"
GRAPH_A = "graph_a"
GRAPH_B = "graph_b"


@pytest.fixture
def compare_caplog(caplog):
    """caplog capturing every level of the compare_sulcal loggers."""
    caplog.set_level(1, logger=COMPARE_LOGGER_NAME)
    return caplog


def _matching(caplog, level: int, *texts: str) -> list[str]:
    """compare_sulcal.cli messages at ``level`` containing every one of ``texts``."""
    return [message for message in logged_messages(caplog, CLI_LOGGER, level) if all(t in message for t in texts)]


def _assert_one(caplog, level: int, *texts: str) -> str:
    """Return the single compare_sulcal.cli message at ``level`` containing ``texts``; fail listing the records."""
    found = _matching(caplog, level, *texts)
    captured = [(record.name, record.levelname, record.getMessage()) for record in caplog.records]
    assert len(found) == 1, f"expected one {logging.getLevelName(level)} record with {texts}; captured: {captured}"
    return found[0]


def _script(argv: list) -> Compare:
    script = Compare()
    script.parse_args(argv)
    return script


def _names_side(message: str, side: str) -> bool:
    """True when ``side`` (L or R) stands as a word of ``message``."""
    return re.search(rf"\b{side}\b", message) is not None


def _set_dirs(tmp_path: Path) -> tuple[Path, Path]:
    set_a, set_b = tmp_path / "set_a", tmp_path / "set_b"
    set_a.mkdir()
    set_b.mkdir()
    return set_a, set_b


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_aims_logged_at_error(self, monkeypatch, compare_caplog, tmp_path):
        monkeypatch.setattr(compare_sulcal.brainvisa, "aims", None)
        script = _script(["crops", "--set_a", str(tmp_path), "--set_b", str(tmp_path)])

        assert script.run() == 1

        _assert_one(compare_caplog, logging.ERROR, "soma.aims")

    def test_unknown_mode_logged_at_error(self, compare_caplog):
        script = Compare()
        script.args = Namespace(mode="bogus")

        assert script.run() == 1

        _assert_one(compare_caplog, logging.ERROR, "unknown mode", "bogus")

    @pytest.mark.parametrize("mode", ["crops", "embeddings"])
    @pytest.mark.parametrize("flag", ["--set_a", "--set_b"])
    def test_set_not_a_directory_logged_at_error(self, compare_caplog, tmp_path, mode, flag):
        set_a, set_b = _set_dirs(tmp_path)
        missing = tmp_path / "missing"
        paths = {"--set_a": set_a, "--set_b": set_b, flag: missing}
        script = _script([mode, "--set_a", str(paths["--set_a"]), "--set_b", str(paths["--set_b"])])

        assert script.run() == 1

        _assert_one(compare_caplog, logging.ERROR, flag, str(missing))

    @pytest.mark.parametrize(
        ("given", "pair"),
        [
            (["--umap_a", "u"], ("--umap_a", "--umap_b")),
            (["--umap_b", "u"], ("--umap_a", "--umap_b")),
            (["--covariate", "c.csv"], ("--covariate", "--covariate_column")),
            (["--covariate_column", "age"], ("--covariate", "--covariate_column")),
        ],
    )
    def test_half_option_pair_logged_at_error(self, compare_caplog, tmp_path, given, pair):
        set_a, set_b = _set_dirs(tmp_path)
        script = _script(["embeddings", "--set_a", str(set_a), "--set_b", str(set_b), *given])

        assert script.run() == 1

        _assert_one(compare_caplog, logging.ERROR, *pair)

    def test_k_below_one_logged_at_error(self, compare_caplog, tmp_path):
        set_a, set_b = _set_dirs(tmp_path)
        script = _script(["embeddings", "--set_a", str(set_a), "--set_b", str(set_b), "--k", "0"])

        assert script.run() == 1

        _assert_one(compare_caplog, logging.ERROR, "--k")

    def test_covariate_load_failure_logged_at_error(self, compare_caplog, tmp_path):
        set_a, set_b = _set_dirs(tmp_path)
        covariate = tmp_path / "covariate.csv"
        covariate.write_text("ID,age\nsub-01,30\n")
        reason = embeddings._load_covariate_csv(str(covariate), "age")
        assert isinstance(reason, str), "fixture CSV must make the covariate loader fail"
        script = _script(
            [
                "embeddings",
                "--set_a",
                str(set_a),
                "--set_b",
                str(set_b),
                "--covariate",
                str(covariate),
                "--covariate_column",
                "age",
                "--output",
                str(tmp_path / "out"),
            ]
        )

        assert script.run() == 1

        _assert_one(compare_caplog, logging.ERROR, reason)


# --------------------------------------------------------------------------- #
# Fake per-pair results for the crops and embeddings subcommands
# --------------------------------------------------------------------------- #

REGION = "S.C.-sylv."
SIDE = "L"
KEY = f"{REGION}_{SIDE}"


def _crops_result(skipped=None, summary=None) -> dict:
    return {"key": KEY, "region": REGION, "side": SIDE, "skipped": skipped, "summary": summary, "rows": []}


def _embeddings_result(skipped=None, summary=None, umap_skipped=()) -> dict:
    return {
        "key": KEY,
        "region": REGION,
        "side": SIDE,
        "skipped": skipped,
        "umap_skipped": list(umap_skipped),
        "summary": summary,
        "rows": [],
    }


def _run_crops(monkeypatch, tmp_path, result: dict, *extra: str) -> int:
    set_a, set_b = _set_dirs(tmp_path)
    monkeypatch.setattr(crops, "_list_crop_regions", lambda _a, _b, _regions: [REGION])
    monkeypatch.setattr(crops, "_compare_region_side", lambda *_args, **_kwargs: result)
    script = _script(
        [
            "crops",
            "--set_a",
            str(set_a),
            "--set_b",
            str(set_b),
            "--side",
            SIDE,
            "--output",
            str(tmp_path / "out"),
            *extra,
        ]
    )
    return script.run()


def _run_embeddings(monkeypatch, tmp_path, result: dict) -> int:
    set_a, set_b = _set_dirs(tmp_path)
    monkeypatch.setattr(embeddings, "_list_embedding_pairs", lambda _a, _b, _regions, _sides: [(REGION, SIDE)])
    monkeypatch.setattr(embeddings, "_compare_embeddings_region_side", lambda *_args: result)
    script = _script(
        ["embeddings", "--set_a", str(set_a), "--set_b", str(set_b), "--side", SIDE, "--output", str(tmp_path / "out")]
    )
    return script.run()


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


class TestWarnings:
    @pytest.mark.parametrize("mode", ["crops", "embeddings"])
    def test_skipped_pair_logged_at_warning(self, monkeypatch, compare_caplog, tmp_path, mode):
        skipped = {"region": REGION, "side": SIDE, "reason": "missing_in_b"}
        if mode == "crops":
            assert _run_crops(monkeypatch, tmp_path, _crops_result(skipped=skipped)) == 0
        else:
            assert _run_embeddings(monkeypatch, tmp_path, _embeddings_result(skipped=skipped)) == 0

        _assert_one(compare_caplog, logging.WARNING, KEY, "missing_in_b")

    def test_umap_skip_logged_at_warning(self, monkeypatch, compare_caplog, tmp_path):
        umap_skipped = [{"region": REGION, "side": SIDE, "reason": "umap_pkl_missing_a"}]
        result = _embeddings_result(umap_skipped=umap_skipped)

        assert _run_embeddings(monkeypatch, tmp_path, result) == 0

        _assert_one(compare_caplog, logging.WARNING, KEY, "umap_pkl_missing_a")

    def test_subject_without_graph_logged_at_warning(self, monkeypatch, compare_caplog, tmp_path):
        _stub_subjects(monkeypatch, ["sub-98", "sub-99"], without_graph={"sub-99"})
        script = _script(_databases_argv(tmp_path))

        script._load_database(GRAPH_A, ["R"], str(tmp_path), 1, str(tmp_path))

        message = _assert_one(compare_caplog, logging.WARNING, "sub-99")
        assert _names_side(message, "R"), message

    def test_shape_mismatch_logged_at_warning(self, monkeypatch, compare_caplog):
        volumes = {"a/bad": np.zeros((4, 4, 4)), "b/bad": np.zeros((2, 2, 2))}
        monkeypatch.setattr(masks, "load_mask_vol", lambda path: volumes[path])
        _score_common_masks(
            ["L/bad.nii.gz"], {"L/bad.nii.gz": "a/bad"}, {"L/bad.nii.gz": "b/bad"}, True, True, 1.0, None
        )

        _assert_one(compare_caplog, logging.WARNING, "L/bad.nii.gz", "(4, 4, 4)", "(2, 2, 2)")


# --------------------------------------------------------------------------- #
# Databases helpers
# --------------------------------------------------------------------------- #


def _databases_argv(tmp_path: Path, output=None, *extra: str) -> list:
    return [
        "databases",
        "--labeled_subjects_dir",
        str(tmp_path),
        "--path_to_graph_a",
        GRAPH_A,
        "--path_to_graph_b",
        GRAPH_B,
        "--label_a",
        LABEL_A,
        "--label_b",
        LABEL_B,
        "--side",
        "L",
        "--njobs",
        "1",
        "--output",
        str(output or tmp_path / "db.csv"),
        *extra,
    ]


def _stub_subjects(monkeypatch, subjects: list, without_graph=frozenset()) -> None:
    """Serve ``subjects`` from the subject lister; those in ``without_graph`` have no graph."""
    monkeypatch.setattr(
        compare_sulcal.brainvisa,
        "get_all_subjects_as_dictionary",
        lambda _dirs, _patterns, _side: [{"subject": s, "dir": "d", "graph_file": "g"} for s in subjects],
    )
    monkeypatch.setattr(
        databases,
        "_get_subject_voxel_counts",
        lambda sub, _bv: (sub["subject"], None if sub["subject"] in without_graph else {"S.C.": 7}),
    )


def _fake_campaigns(monkeypatch, data_a: dict, data_b: dict) -> None:
    by_graph = {GRAPH_A: data_a, GRAPH_B: data_b}
    monkeypatch.setattr(
        Compare,
        "_load_database",
        lambda _self, path_to_graph, *_args: {s: dict(c) for s, c in by_graph[path_to_graph].items()},
    )


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


class TestProgress:
    def test_side_load_start_logged_at_log(self, monkeypatch, compare_caplog, tmp_path):
        _stub_subjects(monkeypatch, ["sub-01", "sub-02", "sub-03"])
        script = _script(_databases_argv(tmp_path))

        script._load_database(GRAPH_A, ["R"], str(tmp_path), 1, str(tmp_path))

        found = [m for m in _matching(compare_caplog, LOG, "3", "1", "subjects found") if _names_side(m, "R")]
        assert len(found) == 1, logged_messages(compare_caplog, CLI_LOGGER, LOG)

    def test_side_load_end_logged_at_log(self, monkeypatch, compare_caplog, tmp_path):
        _stub_subjects(monkeypatch, ["sub-01", "sub-02", "sub-03"], without_graph={"sub-03"})
        script = _script(_databases_argv(tmp_path))

        script._load_database(GRAPH_A, ["R"], str(tmp_path), 1, str(tmp_path))

        message = _assert_one(compare_caplog, LOG, "2/3")
        assert _names_side(message, "R"), message

    def test_campaign_load_start_logged_at_log(self, monkeypatch, compare_caplog, tmp_path):
        _fake_campaigns(monkeypatch, {"sub-01": {"S.C.": 1}}, {"sub-01": {"S.C.": 2}})

        assert _script(_databases_argv(tmp_path)).run() == 0

        _assert_one(compare_caplog, LOG, LABEL_A, GRAPH_A)
        _assert_one(compare_caplog, LOG, LABEL_B, GRAPH_B)

    def test_mask_stats_load_logged_at_log(self, monkeypatch, compare_caplog, tmp_path):
        _fake_campaigns(monkeypatch, {"sub-01": {"S.C.": 1}}, {"sub-01": {"S.C.": 2}})
        monkeypatch.setattr(databases, "_mask_stats", lambda _path, _bv: {})
        masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
        masks_a.mkdir()
        masks_b.mkdir()
        argv = _databases_argv(tmp_path, None, "--masks_a", str(masks_a), "--masks_b", str(masks_b))

        assert _script(argv).run() == 0

        _assert_one(compare_caplog, LOG, LABEL_A, str(masks_a))
        _assert_one(compare_caplog, LOG, LABEL_B, str(masks_b))


# --------------------------------------------------------------------------- #
# Masks / cortical_tiles report
# --------------------------------------------------------------------------- #


def _one_voxel(index: int) -> np.ndarray:
    volume = np.zeros((4, 4, 4))
    volume[index, 0, 0] = 1
    return volume


@pytest.fixture
def mask_run(monkeypatch, tmp_path):
    """Run the masks subcommand on in-memory sets: 3 masks each, 2 common, 1 of them with a shape mismatch."""
    set_a, set_b = _set_dirs(tmp_path)
    found = {
        str(set_a): {"L/shared.nii.gz": "a/shared", "L/bad.nii.gz": "a/bad", "L/only_a.nii.gz": "a/only"},
        str(set_b): {"L/shared.nii.gz": "b/shared", "L/bad.nii.gz": "b/bad", "L/only_b.nii.gz": "b/only"},
    }
    volumes = {
        "a/shared": _one_voxel(0),
        "b/shared": _one_voxel(2),
        "a/bad": np.zeros((4, 4, 4)),
        "b/bad": np.zeros((2, 2, 2)),
    }
    monkeypatch.setattr(masks, "find_masks", lambda directory, _pattern: found[str(directory)])
    monkeypatch.setattr(masks, "load_mask_vol", lambda path: volumes[path])
    monkeypatch.setattr(masks, "save_xor_vol", lambda *_args: None)
    output = tmp_path / "report.json"

    def _run(*extra: str) -> int:
        argv = ["masks", "--set_a", str(set_a), "--set_b", str(set_b), "--output", str(output), "--bucket_step", "1"]
        return _script([*argv, *extra]).run()

    return Namespace(run=_run, output=output, tmp_path=tmp_path)


class TestMaskReport:
    def test_mask_set_counts_logged_at_output(self, mask_run, compare_caplog):
        assert mask_run.run("--metric", "diff") == 0

        expected = {
            "Masks in set_a": 3,
            "Masks in set_b": 3,
            "Common": 2,
            "Only in set_a": 1,
            "Only in set_b": 1,
            "Compared": 1,
            "Skipped (shape mismatch)": 1,
        }
        messages = logged_messages(compare_caplog, CLI_LOGGER, OUTPUT)
        for label, count in expected.items():
            found = [m for m in messages if m.startswith(label) and m.rstrip().endswith(f" {count}")]
            assert len(found) == 1, f"{label}: {count} not in OUTPUT records {messages}"

    def test_report_path_logged_at_output(self, mask_run, compare_caplog):
        assert mask_run.run() == 0

        _assert_one(compare_caplog, OUTPUT, str(mask_run.output))

    @pytest.mark.parametrize("mode", ["masks", "crops"])
    def test_xor_dir_logged_at_output(self, mask_run, monkeypatch, compare_caplog, mode):
        xor_dir = mask_run.tmp_path / "xor"
        if mode == "masks":
            assert mask_run.run("--xor_dir", str(xor_dir)) == 0
        else:
            crops_tmp = mask_run.tmp_path / "crops"
            crops_tmp.mkdir()
            assert _run_crops(monkeypatch, crops_tmp, _crops_result(), "--xor_dir", str(xor_dir)) == 0

        _assert_one(compare_caplog, OUTPUT, str(xor_dir))

    def test_bucket_headings_logged_at_output(self, mask_run, compare_caplog):
        assert mask_run.run("--metric", "both") == 0

        _assert_one(compare_caplog, OUTPUT, "Wasserstein buckets (step=1.0)")
        _assert_one(compare_caplog, OUTPUT, "Diff buckets (step=1.0)")

    def test_bucket_counts_logged_at_output(self, mask_run, compare_caplog):
        assert mask_run.run("--metric", "diff") == 0

        changed = json.loads(mask_run.output.read_text())["diffs_per_mask"]["L/shared.nii.gz"]["changed"]
        label = masks.bucket_label(changed, 1.0)
        message = _assert_one(compare_caplog, OUTPUT, label, "mask(s)")
        assert re.search(r"(?<!\d)1 mask\(s\)", message), message

    def test_unchanged_count_logged_at_output(self, mask_run, compare_caplog):
        assert mask_run.run("--metric", "diff") == 0

        _assert_one(compare_caplog, OUTPUT, "Unchanged masks: 0/1")


# --------------------------------------------------------------------------- #
# Crops / embeddings pair report
# --------------------------------------------------------------------------- #


class TestPairReport:
    def test_crops_pair_summary_logged_at_output(self, monkeypatch, compare_caplog, tmp_path):
        summary = {"n_subjects_compared": 4, "subjects_changed": 1, "pct_lost_mean": 0.25, "dice_min": 0.5}

        assert _run_crops(monkeypatch, tmp_path, _crops_result(summary=summary)) == 0

        _assert_one(compare_caplog, OUTPUT, KEY, "n=4", "changed=1", "0.2500", "0.5000")

    def test_embeddings_pair_summary_logged_at_output(self, monkeypatch, compare_caplog, tmp_path):
        summary = {"n_subjects": 5, "cka": 0.9, "knn_overlap_mean": 0.7}

        assert _run_embeddings(monkeypatch, tmp_path, _embeddings_result(summary=summary)) == 0

        _assert_one(compare_caplog, OUTPUT, KEY, "n=5", "0.9000", "0.7000")

    @pytest.mark.parametrize("mode", ["crops", "embeddings"])
    def test_written_files_logged_at_output(self, monkeypatch, compare_caplog, tmp_path, mode):
        if mode == "crops":
            assert _run_crops(monkeypatch, tmp_path, _crops_result()) == 0
        else:
            assert _run_embeddings(monkeypatch, tmp_path, _embeddings_result()) == 0

        out = tmp_path / "out"
        _assert_one(compare_caplog, OUTPUT, str(out / "per_subject.csv"), str(out / "summary.json"))


# --------------------------------------------------------------------------- #
# Databases report
# --------------------------------------------------------------------------- #


class TestDatabasesReport:
    def test_subject_overlap_logged_at_output(self, monkeypatch, compare_caplog, tmp_path):
        data_a = {"sub-01": {"S.C.": 100}, "sub-02": {"S.C.": 200}}
        data_b = {"sub-02": {"S.C.": 400}, "sub-03": {"S.C.": 300}, "sub-04": {"S.C.": 300}}
        _fake_campaigns(monkeypatch, data_a, data_b)

        assert _script(_databases_argv(tmp_path)).run() == 0

        messages = logged_messages(compare_caplog, CLI_LOGGER, OUTPUT)
        for marker, count in ((f"{LABEL_A} only", 1), (f"{LABEL_B} only", 2), ("in both", 1)):
            found = [m for m in messages if marker in m and m.rstrip().endswith(f" {count}")]
            assert len(found) == 1, f"{marker}: {count} not in OUTPUT records {messages}"

    def test_ratio_table_heading_logged_at_output(self, monkeypatch, compare_caplog, tmp_path):
        _fake_campaigns(monkeypatch, {"sub-01": {"S.C.": 100}}, {"sub-01": {"S.C.": 120}})

        assert _script(_databases_argv(tmp_path)).run() == 0

        _assert_one(compare_caplog, OUTPUT, "Sulci with biggest voxel-density difference", LABEL_A, LABEL_B)

    def test_ratio_table_rows_logged_at_output(self, monkeypatch, compare_caplog, tmp_path):
        data_a = {"sub-01": {"S.C.": 100, "F.C.M.": 40}}
        data_b = {"sub-01": {"S.C.": 120, "F.C.M.": 10}}
        _fake_campaigns(monkeypatch, data_a, data_b)

        assert _script(_databases_argv(tmp_path)).run() == 0

        for sulcus, ratio in (("S.C.", 1.2), ("F.C.M.", 0.25)):
            message = _assert_one(compare_caplog, OUTPUT, f" {sulcus} ", f"{ratio:.3f}")
            assert message.rstrip().endswith(f"{ratio:.3f}"), message

    def test_no_rows_logged_at_output(self, monkeypatch, compare_caplog, tmp_path):
        _fake_campaigns(monkeypatch, {}, {})

        assert _script(_databases_argv(tmp_path)).run() == 0

        _assert_one(compare_caplog, OUTPUT, "No sulcus rows found", LABEL_A, LABEL_B)

    def test_csv_path_logged_at_output(self, monkeypatch, compare_caplog, tmp_path):
        _fake_campaigns(monkeypatch, {"sub-01": {"S.C.": 100}}, {"sub-01": {"S.C.": 120}})
        monkeypatch.chdir(tmp_path)

        assert _script(_databases_argv(tmp_path, "out/db.csv")).run() == 0

        message = _assert_one(compare_caplog, OUTPUT, "CSV written to: ")
        assert message.endswith(f"CSV written to: {abspath('out/db.csv')}"), message


# --------------------------------------------------------------------------- #
# Static: no fold-progress text
# --------------------------------------------------------------------------- #


def _logging_call_texts(tree: ast.Module) -> list[str]:
    """Static string parts of the ``logger.<method>(...)`` calls in ``tree``."""
    texts = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            for arg in node.args:
                texts.extend(
                    part.value
                    for part in ast.walk(arg)
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
    return texts


def test_no_static_message_matches_fold_pattern():
    texts = _logging_call_texts(ast.parse(CLI_PY.read_text(encoding="utf-8")))

    assert texts, "src/compare_sulcal/cli.py has no logger call"
    assert [text for text in texts if MCP_FOLD_RE.search(text)] == []

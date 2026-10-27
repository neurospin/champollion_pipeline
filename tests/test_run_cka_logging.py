"""LOGGING convention in src/run_cka.py (TASK-LOGGING-BDRABCZUK-B8825FF10EEB).

The CKA comparison script reports through the ``run_cka`` module logger instead of print();
tests assert the records with caplog, not capsys. ``main()`` calls ``init_pipeline_process()``
and ``ScriptBuilder.main()``, whose root handlers send LOG and OUTPUT records to stdout and
WARNING and above to stderr.

Progress (LOG):
REQ-LOGGING-BDRABCZUK-52F44F0EBC47: start of a region pair -> the pair label.
Warnings (WARNING):
REQ-LOGGING-BDRABCZUK-0D249EB46387: CKA command with a non-zero exit code -> pair label and exit code.
REQ-LOGGING-BDRABCZUK-64FBF9C84148: nested-mode region without embeddings file -> region and file path.
REQ-LOGGING-BDRABCZUK-852F8E7D2E2F: flat-mode CSV files without match in --path_b -> their count.
Results (OUTPUT):
REQ-LOGGING-BDRABCZUK-7956F44CCAE1: directory-mode completion -> processed and total counts, --output_dir.
Static:
REQ-LOGGING-BDRABCZUK-DA4337AC550F: no static logging text matches the MCP fold regex.
(REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)

The CKA subprocess is never launched: ``execute_command`` is replaced by a recorder.
"""

import ast
import logging
import re
from pathlib import Path

import pytest

from run_cka import RunCKA
from tests.log_assertions import LOG, OUTPUT

LOGGER = "run_cka"
SCRIPT_PY = Path(__file__).resolve().parents[1] / "src" / "run_cka.py"
# Copy of champollion_sulcal_mcp runner.FOLD_RE (it reads the stage output line by line).
MCP_FOLD_RE = re.compile(r"[Ff]old\s+(\d+)\s*(?:/|of)\s*(\d+)")


@pytest.fixture
def cka_caplog(caplog):
    """caplog capturing every level of the run_cka logger."""
    caplog.set_level(1, logger=LOGGER)
    return caplog


def _messages(caplog, level: int) -> list[str]:
    """run_cka messages captured at ``level``, in emission order."""
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == level]


def _assert_one(caplog, level: int, *texts: str) -> str:
    """Return the single message at ``level`` containing every one of ``texts``; fail listing the records."""
    found = [message for message in _messages(caplog, level) if all(text in message for text in texts)]
    captured = [(record.name, record.levelname, record.getMessage()) for record in caplog.records]
    assert len(found) == 1, f"expected one {logging.getLevelName(level)} record with {texts}; captured: {captured}"
    return found[0]


def _make_script(argv: list, exit_code: int = 0) -> RunCKA:
    """Build a RunCKA with parsed args and an execute_command that returns ``exit_code`` without running."""
    script = RunCKA()
    script.args = script.parse_args(argv)
    script.executed = []

    def _fake_execute(cmd, shell=False):
        script.executed.append(cmd)
        return exit_code

    script.execute_command = _fake_execute
    return script


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ID,dim0\nsub-01,1.0\n")
    return path


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


class TestProgress:
    def test_pair_start_logged_at_log(self, cka_caplog, tmp_path):
        script = _make_script(["--path_a", "a", "--path_b", "b", "--output_dir", str(tmp_path)])

        assert script._run_pairs([("SC_left", "a.csv", "b.csv")]) == 1

        _assert_one(cka_caplog, LOG, "SC_left")


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


class TestWarnings:
    def test_failed_cka_command_logged_at_warning(self, cka_caplog, tmp_path):
        script = _make_script(["--path_a", "a", "--path_b", "b", "--output_dir", str(tmp_path)], exit_code=7)

        assert script._run_pairs([("SC_left", "a.csv", "b.csv")]) == 1

        message = _assert_one(cka_caplog, logging.WARNING, "SC_left")
        assert re.search(r"\b7\b", message), message

    @pytest.mark.parametrize("missing_side", ["a", "b"])
    def test_region_missing_file_logged_at_warning(self, cka_caplog, tmp_path, missing_side):
        for side in ("a", "b"):
            (tmp_path / side / "SC_left").mkdir(parents=True)
            _touch(tmp_path / side / "SC_right" / "full_embeddings.csv")
            if side != missing_side:
                _touch(tmp_path / side / "SC_left" / "full_embeddings.csv")
        missing = tmp_path / missing_side / "SC_left" / "full_embeddings.csv"
        script = _make_script(
            ["--path_a", str(tmp_path / "a"), "--path_b", str(tmp_path / "b"), "--output_dir", str(tmp_path / "o")]
        )

        assert script.run() == 0

        _assert_one(cka_caplog, logging.WARNING, "SC_left", str(missing))

    @pytest.mark.parametrize(
        ("names_a", "names_b"),
        [
            (
                ["SC_left_embeddings.csv", "extra1_embeddings.csv", "extra2_embeddings.csv"],
                ["SC_left_embeddings.csv"],
            ),
            (
                [
                    "SC_left_name07-58-00--111_embeddings.csv",
                    "SC_right_name08-00-00--222_embeddings.csv",
                    "FIP_left_name09-00-00--333_embeddings.csv",
                ],
                ["other_name07-58-00--111_embeddings.csv"],
            ),
        ],
        ids=["basename", "model_id"],
    )
    def test_unmatched_flat_files_logged_at_warning(self, cka_caplog, tmp_path, names_a, names_b):
        for name in names_a:
            _touch(tmp_path / "a" / name)
        for name in names_b:
            _touch(tmp_path / "b" / name)
        script = _make_script(
            ["--path_a", str(tmp_path / "a"), "--path_b", str(tmp_path / "b"), "--output_dir", str(tmp_path / "o")]
        )

        assert script.run() == 0

        unmatched = [m for m in _messages(cka_caplog, logging.WARNING) if re.search(r"\b2\b", m)]
        captured = [(record.levelname, record.getMessage()) for record in cka_caplog.records]
        assert len(unmatched) == 1, f"expected one WARNING record with the unmatched count 2; captured: {captured}"


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class TestResults:
    def test_completion_logged_at_output(self, cka_caplog, tmp_path):
        # Three regions under path_a, one without its embeddings file: 2 pairs run of 3 regions.
        for side in ("a", "b"):
            for region in ("SC_left", "SC_right", "FIP_left"):
                (tmp_path / side / region).mkdir(parents=True)
                if not (side == "a" and region == "FIP_left"):
                    _touch(tmp_path / side / region / "full_embeddings.csv")
        output_dir = str(tmp_path / "cka_out")
        script = _make_script(
            ["--path_a", str(tmp_path / "a"), "--path_b", str(tmp_path / "b"), "--output_dir", output_dir]
        )

        assert script.run() == 0

        message = _assert_one(cka_caplog, OUTPUT, output_dir)
        counts = message.replace(output_dir, "")
        assert re.search(r"\b2\b", counts), message
        assert re.search(r"\b3\b", counts), message


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
    texts = _logging_call_texts(ast.parse(SCRIPT_PY.read_text(encoding="utf-8")))

    assert texts, "src/run_cka.py has no logger call"
    assert [text for text in texts if MCP_FOLD_RE.search(text)] == []

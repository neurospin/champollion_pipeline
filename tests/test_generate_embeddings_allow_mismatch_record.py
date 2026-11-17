#!/usr/bin/env python3
"""
generate_embeddings: an accepted --masks / --model-version mismatch is one WARNING log record (MODELVER epic).

User decision 2026-11-17 09:49 (3), DEF-MODELVER-BDRABCZUK-291232AA949D: --allow-version-mismatch stays an
explicit opt-in; with it, the mismatch is reported as exactly one log record at level WARNING naming both
versions, through the champollion_pipeline logger, and never as a Python UserWarning.

- REQ-MODELVER-BDRABCZUK-C8112359118C (system): an accepted version mismatch is one WARNING record, not a
  UserWarning.
- REQ-MODELVER-BDRABCZUK-790B770EB0B1 (component, supersedes REQ-MODELVER-BDRABCZUK-55C6CBEED962): exactly one
  WARNING record naming <M> and <V> through the champollion_pipeline logger.
- REQ-MODELVER-BDRABCZUK-407A2E62D753 (component): no Python UserWarning.

No model, no evaluate.py, no network: only the real GenerateEmbeddings parser, on a Hugging Face repository id
(so no local model set check runs).
"""

import logging
import warnings

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

HF_REPO = "neurospin/Champollion_V1"
MODEL_VERSION = "canonical_corrected_26_1"
MASKS = "canonical_25"


def _argv(tmp_path) -> list[str]:
    return [
        HF_REPO,
        str(tmp_path / "TEST01"),
        "--masks",
        MASKS,
        "--model-version",
        MODEL_VERSION,
        "--allow-version-mismatch",
    ]


def _parse_recording_warnings(argv: list[str]) -> list[warnings.WarningMessage]:
    """Parse argv with the real parser, recording every Python warning; fail on rejection."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            GenerateEmbeddings().parse_args(argv)
        except SystemExit as exc:
            pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
    return list(record)


class TestAllowMismatchWarningRecord:
    """REQ-MODELVER-BDRABCZUK-C8112359118C, REQ-MODELVER-BDRABCZUK-790B770EB0B1, REQ-MODELVER-BDRABCZUK-407A2E62D753."""

    def test_logs_exactly_one_warning_record_naming_both(self, tmp_path, pipeline_caplog):
        _parse_recording_warnings(_argv(tmp_path))
        warning_messages = [r.getMessage() for r in pipeline_caplog.records if r.levelno == logging.WARNING]
        naming_both = [message for message in warning_messages if MASKS in message and MODEL_VERSION in message]
        assert len(naming_both) == 1, [(r.name, r.levelname, r.getMessage()) for r in pipeline_caplog.records]

    def test_emits_no_user_warning(self, tmp_path):
        record = _parse_recording_warnings(_argv(tmp_path))
        user_warnings = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
        assert user_warnings == []

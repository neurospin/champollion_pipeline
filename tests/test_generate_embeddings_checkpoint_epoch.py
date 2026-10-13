#!/usr/bin/env python3
"""
generate_embeddings parser: the --checkpoint-epoch option (EPOCH80 parser task).
Removal of --use_last_checkpoint is pinned separately, in
tests/test_generate_embeddings_no_last_checkpoint.py (EPOCH80 dead-code task).

Epoch-80 harmonisation (user decision 2026-10-13): each region is evaluated with
the checkpoint of a configurable epoch, default 80. These tests exercise only the
real GenerateEmbeddings (ScriptBuilder) parser: no model, no evaluate.py, no network.

Lightning numbers epochs from 0 (``epoch=0-step=*.ckpt``), so 0 is a valid epoch
and only negative or non-integer values are rejected.
"""

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

OPTION = "--checkpoint-epoch"
POSITIONALS = ["/models", "/data/MYDATASET"]
_ABSENT = "<no checkpoint_epoch attribute>"


def parse_embeddings_args(argv: list[str]):
    """Parse argv with the real GenerateEmbeddings parser; fail (not error) on rejection."""
    try:
        return GenerateEmbeddings().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")


def parse_expecting_exit(argv: list[str]) -> SystemExit:
    """Parse argv with the real parser and return the SystemExit it raises."""
    with pytest.raises(SystemExit) as excinfo:
        GenerateEmbeddings().parse_args(argv)
    return excinfo.value


class TestCheckpointEpochOption:
    """REQ-EPOCH80-BDRABCZUK-022CBA4B41E5: value parses as int; omitted -> 80."""

    def test_given_value_parses_as_int(self):
        namespace = parse_embeddings_args([*POSITIONALS, OPTION, "120"])
        value = vars(namespace).get("checkpoint_epoch", _ABSENT)
        assert type(value) is int and value == 120, value

    def test_zero_parses_as_int(self):
        namespace = parse_embeddings_args([*POSITIONALS, OPTION, "0"])
        value = vars(namespace).get("checkpoint_epoch", _ABSENT)
        assert type(value) is int and value == 0, value

    def test_omitted_defaults_to_80(self):
        namespace = parse_embeddings_args(POSITIONALS)
        value = vars(namespace).get("checkpoint_epoch", _ABSENT)
        assert type(value) is int and value == 80, value


class TestCheckpointEpochInvalidValueRejected:
    """REQ-EPOCH80-BDRABCZUK-A01743BD1CE1: negative or non-integer value exits with status 2."""

    @pytest.mark.parametrize("value", ["-1", "-80", "abc", "1.5", "80.0"])
    def test_invalid_value_exits_with_status_2(self, value, capsys):
        exc = parse_expecting_exit([*POSITIONALS, f"{OPTION}={value}"])
        err = capsys.readouterr().err
        assert exc.code == 2, err
        # The rejection must come from the option's own validation, not from
        # argparse not knowing the option at all.
        assert "unrecognized arguments" not in err, err
        assert OPTION in err, err

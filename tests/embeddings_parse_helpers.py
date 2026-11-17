"""Shared helpers of the generate_embeddings parser tests (MODELVER epic).

Only the real GenerateEmbeddings parser: no model, no evaluate.py, no network.
"""

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

# A tmp models dir with an arbitrary name is not a Mask version: the parser needs --masks
# (REQ-MODELVER-BDRABCZUK-1B06D337ECC0); canonical_25 keeps the masks these tests always ran with.
MASKS_ARGS = ("--masks", "canonical_25")

_ABSENT = "<no masks attribute>"


def parse_or_fail(argv: list[str]) -> GenerateEmbeddings:
    """Return a GenerateEmbeddings whose args are argv parsed by the real parser; fail on rejection."""
    script = GenerateEmbeddings()
    try:
        script.parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
    return script


def masks_of(script: GenerateEmbeddings) -> str:
    """The parsed --masks value, or a marker string when the namespace has no masks attribute."""
    return vars(script.args).get("masks", _ABSENT)

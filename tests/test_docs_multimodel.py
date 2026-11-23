#!/usr/bin/env python3
"""Documentation guards: the docs describe several models per region (TASK-MULTIMODEL docs half).

Run by hand (``verify_ignore`` skips ``tests/test_docs_*.py``): ``pixi run pytest tests/test_docs_multimodel.py``.
The parsers are only built, never parsed: no filesystem writes, no network.

The README.md stage 4-6 text (README sections 5-7) moved to the Stage sections of docs/usage.md
(TASK-DOCS-BDRABCZUK-84BE8AB6FC73); the first ten Requirements below name that place since.

Requirements pinned here (one test each):

* REQ-DOCS-BDRABCZUK-66C5D5EEBB15 - usage.md stage 4 ``--per-model-folders`` row.
* REQ-DOCS-BDRABCZUK-1385E20E250F - usage.md stage 4 ``--checkpoint-epoch`` row: repeatable, ``<id>=N``.
* REQ-DOCS-BDRABCZUK-56C754B68BEB - subsection "Models of one region" defines the model id.
* REQ-DOCS-BDRABCZUK-B066B64747EC - both models_path layouts.
* REQ-DOCS-BDRABCZUK-AB302E0CC1E4 - a same-id pair is refused.
* REQ-DOCS-BDRABCZUK-152173387857 - per-model output file and when it applies.
* REQ-DOCS-BDRABCZUK-B05E72E488FD - usage.md stage 5 ``--model``.
* REQ-DOCS-BDRABCZUK-024B6C36A4E6 - usage.md stage 5 combined folder per model id.
* REQ-DOCS-BDRABCZUK-11657C43203B - usage.md stage 6 ``--model``.
* REQ-DOCS-BDRABCZUK-BC9B388B4115 - usage.md ``provenance.json`` ``models`` list.
* REQ-DOCS-BDRABCZUK-16F6F559236E - usage.md stage 4 multi-model example.
* REQ-DOCS-BDRABCZUK-9C40A807F2EA - usage.md stage 5 ``--model`` example.
* REQ-DOCS-BDRABCZUK-D31F7FA24517 - usage.md stage 6 ``--model`` example.
* REQ-DOCS-BDRABCZUK-7FC72BE09F92 - usage.md streaming over models.
* REQ-DOCS-BDRABCZUK-526A7F6DD3DF - usage.md Config keys.
* REQ-DOCS-BDRABCZUK-7C7EADE454A9 - workflow.md inference per model.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from tests.docs_test_helpers import md_section, option_strings, read_text, required_examples

ROOT = Path(__file__).resolve().parents[1]
USAGE = read_text(ROOT / "docs" / "usage.md")
WORKFLOW = read_text(ROOT / "docs" / "workflow.md")
MODEL_SUBSECTION = "Models of one region"
PARSERS = {
    "champollion-embeddings": GenerateEmbeddings,
    "champollion-combine": PutTogetherEmbeddings,
    "champollion-snapshots": GenerateSnapshots,
}


def _usage_stage(number: int) -> str:
    """Body of the ``## Stage <number> ...`` section of docs/usage.md (README sections 5-7 moved there)."""
    return md_section(USAGE, rf"Stage {number}\b[^\n]*")


def _subsection(text: str, heading: str) -> str:
    m = re.search(rf"^###\s+{re.escape(heading)}\s*$(.*?)(?=^###?\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '### {heading}' subsection found"
    return m.group(1)


def _option_row(flag: str) -> str:
    """The row of the Stage 4 options table of docs/usage.md whose first cell names ``flag``."""
    for line in _usage_stage(4).splitlines():
        if line.startswith("|") and flag in line.strip("|").split("|", 1)[0]:
            return line
    raise AssertionError(f"docs/usage.md Stage 4 has no options table row for {flag}")


def _models_subsection() -> str:
    return _subsection(_usage_stage(4), MODEL_SUBSECTION)


def _stage_section(number: int) -> str:
    return md_section(USAGE, rf"Stage {number}\s+[^\n]*")


def _options_of(command: str) -> set[str]:
    return option_strings(PARSERS[command]().parser)


def _example_with(section: str, command: str, flag: str) -> list[str]:
    """The tokens of the ``pixi run <command>`` example of ``section`` that uses ``flag``."""
    for tokens in required_examples(section, command, f"docs/usage.md {command}"):
        if any(tok.split("=", 1)[0] == flag for tok in tokens):
            return tokens
    raise AssertionError(f"docs/usage.md has no {command} example using {flag}")


def _assert_declared(tokens: list[str], command: str) -> None:
    bogus = sorted({t.split("=", 1)[0] for t in tokens if t.startswith("-")} - _options_of(command))
    assert not bogus, f"{command} example uses options its parser rejects: {bogus}"


@pytest.mark.smoke
class TestReadmeStage4:
    def test_per_model_folders_row(self):  # REQ-DOCS-BDRABCZUK-66C5D5EEBB15
        assert "--per-model-folders" in _options_of("champollion-embeddings")
        assert "model id" in _option_row("--per-model-folders")

    def test_checkpoint_epoch_row(self):  # REQ-DOCS-BDRABCZUK-1385E20E250F
        row = _option_row("--checkpoint-epoch")
        assert "repeatable" in row.lower() and "<id>=N" in row

    def test_model_id_defined(self):  # REQ-DOCS-BDRABCZUK-56C754B68BEB
        assert "folder path between the region folder and the model folder" in _models_subsection()

    def test_both_layouts(self):  # REQ-DOCS-BDRABCZUK-B066B64747EC
        body = _models_subsection()
        assert "<models_path>/<Region>_<side>/<name>/logs" in body
        assert "one models_path per variant" in body

    def test_same_id_pair_refused(self):  # REQ-DOCS-BDRABCZUK-AB302E0CC1E4
        body = _models_subsection()
        assert "same model id" in body and "refused" in body

    def test_output_file(self):  # REQ-DOCS-BDRABCZUK-152173387857
        body = _models_subsection()
        assert "<region>/<model id>/full_embeddings.csv" in body
        assert "two or more models" in body and "--per-model-folders" in body


@pytest.mark.smoke
class TestReadmeStages5And6:
    def test_combine_model_option(self):  # REQ-DOCS-BDRABCZUK-B05E72E488FD
        assert "--model" in _options_of("champollion-combine")
        body = _usage_stage(5)
        assert "--model" in body and "only the named model id" in body

    def test_combine_folder_per_model_id(self):  # REQ-DOCS-BDRABCZUK-024B6C36A4E6
        assert "<output_path>/<model id>/<region>_embeddings.csv" in _usage_stage(5)

    def test_snapshots_model_option(self):  # REQ-DOCS-BDRABCZUK-11657C43203B
        assert "--model" in _options_of("champollion-snapshots")
        body = _usage_stage(6)
        assert "--model" in body and "two or more model id folders" in body and "refused" in body

    def test_provenance_models_list(self):  # REQ-DOCS-BDRABCZUK-BC9B388B4115
        body = _subsection(_usage_stage(4), "Run record: `provenance.json`")
        for key in ("models", "region", "model_id", "checkpoint_epoch", "weights_source"):
            assert re.search(rf"`{key}`", body), f"provenance subsection does not name `{key}`"


@pytest.mark.smoke
class TestUsage:
    def test_stage4_example(self):  # REQ-DOCS-BDRABCZUK-16F6F559236E
        tokens = _example_with(_stage_section(4), "champollion-embeddings", "--per-model-folders")
        assert any(re.fullmatch(r"[\w./-]+=\d+", t) for t in tokens), f"no <id>=N value in {tokens}"
        assert "--checkpoint-epoch" in tokens
        _assert_declared(tokens, "champollion-embeddings")

    @pytest.mark.parametrize(("number", "command"), [(5, "champollion-combine"), (6, "champollion-snapshots")])
    def test_model_example(self, number, command):  # REQ-DOCS-BDRABCZUK-9C40A807F2EA, ...-D31F7FA24517
        _assert_declared(_example_with(_stage_section(number), command, "--model"), command)

    def test_streaming_over_models(self):  # REQ-DOCS-BDRABCZUK-7FC72BE09F92
        body = md_section(USAGE, "Streaming mode")
        assert "each model of the models path" in body and "one output per model id" in body

    def test_config_keys(self):  # REQ-DOCS-BDRABCZUK-526A7F6DD3DF
        for key in ("dataset.per_model_folders", "dataset.checkpoint_epochs", "dataset.model"):
            assert re.search(rf"`{re.escape(key)}`", USAGE), f"docs/usage.md does not name `{key}`"


@pytest.mark.smoke
class TestWorkflow:
    def test_inference_per_model(self):  # REQ-DOCS-BDRABCZUK-7C7EADE454A9
        m = re.search(r"^###\s+Inference[^\n]*$(.*?)(?=^---|^##\s)", WORKFLOW, re.MULTILINE | re.DOTALL)
        assert m, "docs/workflow.md has no Inference section"
        assert "one inference pass per trained model directory" in m.group(1) and "model id" in m.group(1)

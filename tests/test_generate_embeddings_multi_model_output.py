#!/usr/bin/env python3
"""
Stage 4 batch over several models per region (TASK-MULTIMODEL-BDRABCZUK-98B60783E569).

Every test drives the stage through its script interface (GenerateEmbeddings.parse_args, then run()) on a fixture
models tree; fetch_models is patched to return that tree and execute_command is replaced by a fake evaluate.py.
The model id of a model directory is the path between the region folder (the nearest *_left / *_right folder) and
the model directory; its model label is <region>/<model id> for a non-empty model id, else the region name
(T1 TASK-MULTIMODEL-BDRABCZUK-762D497260FE). Each class names its Requirement by title (``alm get <id>``).

The input and output QC are stubbed by tests/conftest.py here; tests/test_generate_embeddings_multi_model_qc.py
runs them for real.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.log_assertions import FAIL, LOG

pytestmark = pytest.mark.unit  # noqa: V107

# The models folder is named after a Mask version, so the parser takes --masks from it.
LOCAL_MODELS = "canonical_25"
DEFAULT_EPOCH = 80
EMBEDDINGS = "full_embeddings.csv"
FLAT = "SC-sylv_left"
MULTI = "STs_right"
HF_HASHES = {FLAT: "name08-32-57_200", MULTI: "name09-11-02_31"}
COHORT = ("sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06")
WIDTH = 3
LOGGER_NAME = "champollion_pipeline.generate_embeddings"
EVALUATE_FAILED = 1


def make_model(models: Path, *parts: str, epochs=(DEFAULT_EPOCH,), width: int | None = WIDTH) -> Path:
    """Create the model directory models/<parts>/ with logs/ and one epoch=<e>-step=1.ckpt per epoch of ``epochs``.

    ``width``: the backbone_output_size of its .hydra/config.yaml, None for no config."""
    model = models.joinpath(*parts)
    ckpt_dir = model / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for epoch in epochs:
        (ckpt_dir / f"epoch={epoch}-step=1.ckpt").write_text("placeholder")
    if width is not None:
        (model / ".hydra").mkdir()
        (model / ".hydra" / "config.yaml").write_text(f"backbone_output_size: {width}\n")
    return model


def models_root(root: Path) -> Path:
    return Path(root) / LOCAL_MODELS


def embeddings_table(subjects=COHORT, width=WIDTH, seed=0) -> bytes:
    """An embeddings table as evaluate.py writes it: an ID column, then ``width`` finite values per subject."""
    values = np.random.default_rng(seed).normal(size=(len(subjects), width))
    header = ",".join(["ID", *(f"dim{index}" for index in range(1, width + 1))])
    lines = [
        ",".join([subject, *(f"{value:.6f}" for value in row)]) for subject, row in zip(subjects, values, strict=True)
    ]
    return "\n".join([header, *lines, ""]).encode()


def model_of_command(command) -> str:
    """The real path of the model directory whose checkpoint the -m directory of an evaluate.py command holds."""
    ckpts = sorted(Path(command[command.index("-m") + 1]).glob("logs/lightning_logs/version_0/checkpoints/*.ckpt"))
    if not ckpts:
        return ""
    # <model>/logs/lightning_logs/version_<n>/checkpoints/<ckpt>
    return os.path.realpath(Path(os.path.realpath(ckpts[0])).parents[4])


@dataclass
class EvaluateCall:
    model: str  # real path of the model directory
    saving: Path  # the -s path
    ckpts: list[str]  # names of the .ckpt files evaluate.py would load
    content: bytes  # what it wrote, empty when it failed


@dataclass
class FakeEvaluate:
    """Stand-in for execute_command: writes an embeddings table to the -s path, as evaluate.py would.

    ``failing``: model directories whose command returns a non-zero status and writes nothing.
    ``tables``: bytes to write per model directory; by default a valid table distinct per call."""

    subjects: tuple[str, ...] = COHORT
    failing: tuple[Path, ...] = ()
    tables: dict = field(default_factory=dict)
    calls: list[EvaluateCall] = field(default_factory=list)

    def __call__(self, command, **_kwargs):
        model = model_of_command(command)
        saving = Path(command[command.index("-s") + 1])
        ckpts = [
            path.name
            for path in sorted(Path(command[command.index("-m") + 1]).glob("logs/lightning_logs/*/checkpoints/*.ckpt"))
        ]
        if model in {os.path.realpath(path) for path in self.failing}:
            self.calls.append(EvaluateCall(model, saving, ckpts, b""))
            return EVALUATE_FAILED
        tables = {os.path.realpath(path): content for path, content in self.tables.items()}
        content = tables.get(model, embeddings_table(self.subjects, seed=len(self.calls) + 1))
        saving.parent.mkdir(parents=True, exist_ok=True)
        saving.write_bytes(content)
        self.calls.append(EvaluateCall(model, saving, ckpts, content))
        return 0

    def call_of(self, model: Path) -> EvaluateCall:
        """The one call made for ``model``."""
        calls = [call for call in self.calls if call.model == os.path.realpath(model)]
        assert len(calls) == 1, f"expected one evaluate.py command for {model}, got {len(calls)}: {self.calls}"
        return calls[0]

    def saved(self) -> set[Path]:
        return {call.saving for call in self.calls}


@dataclass
class Run:
    """What a stage run left: its status (an int, a SystemExit code, or the exception raised) and its files."""

    output: Path
    status: object
    evaluate: FakeEvaluate

    def embeddings_files(self) -> set[str]:
        """The full_embeddings.csv files under the output base directory, as POSIX paths relative to it."""
        return {path.relative_to(self.output).as_posix() for path in self.output.rglob(EMBEDDINGS)}


def parse(root: Path, *extra: str, dataset: Path | None = None) -> GenerateEmbeddings:
    """A GenerateEmbeddings parsed on the models tree under ``root``, its output base directory being root/out."""
    script = GenerateEmbeddings()
    dataset = dataset or Path(root) / "dataset"
    script.parse_args([str(models_root(root)), str(dataset), "--output", str(Path(root) / "out"), *extra])
    return script


def run_parsed(root: Path, script: GenerateEmbeddings, evaluate: FakeEvaluate | None = None) -> Run:
    """Run ``script`` with fetch_models returning the models tree and ``evaluate`` as evaluate.py."""
    evaluate = evaluate or FakeEvaluate()
    script.execute_command = evaluate
    with patch.object(GenerateEmbeddings, "fetch_models", return_value=str(models_root(root))):
        try:
            status = script.run()
        except SystemExit as stop:
            status = stop.code
        except Exception as error:  # a refusal by exception is a status the tests read
            status = error
    return Run(Path(root) / "out", status, evaluate)


def run_stage(root: Path, *extra: str, evaluate: FakeEvaluate | None = None, dataset: Path | None = None) -> Run:
    return run_parsed(root, parse(root, *extra, dataset=dataset), evaluate)


def lay_two_models_in_one_region(root: Path, *, flat: bool = True) -> tuple[Path, Path]:
    """MULTI holds run_1 and run_2; FLAT, when ``flat``, holds one model with an empty model id."""
    models = models_root(root)
    if flat:
        make_model(models, FLAT)
    return make_model(models, MULTI, "run_1"), make_model(models, MULTI, "run_2")


def lay_single_models(root: Path, layout: str) -> dict[str, Path]:
    """One model per region: flat (empty model id) or the HF cache layout <region>/<hash> (non-empty model id)."""
    models = models_root(root)
    if layout == "flat":
        return {region: make_model(models, region) for region in (FLAT, MULTI)}
    return {region: make_model(models, region, HF_HASHES[region]) for region in (FLAT, MULTI)}


def stage_messages(pipeline_caplog, level: int) -> list[str]:
    return [item.getMessage() for item in pipeline_caplog.records if (item.name, item.levelno) == (LOGGER_NAME, level)]


class TestSingleModelEmbeddingsFilesUnchanged:
    """Single-model embeddings files unchanged (stakeholder): path and bytes as pipeline commit c29c56e."""

    @pytest.mark.parametrize("layout", ["flat", "hf_cache"])
    def test_each_region_file_at_the_region_folder_with_the_evaluate_bytes(self, tmp_path, layout):
        models = lay_single_models(tmp_path, layout)

        run = run_stage(tmp_path)

        assert run.status == 0, run.status
        assert run.embeddings_files() == {f"{FLAT}/{EMBEDDINGS}", f"{MULTI}/{EMBEDDINGS}"}
        for region, model in models.items():
            written = (run.output / region / EMBEDDINGS).read_bytes()
            assert written == run.evaluate.call_of(model).content


class TestSeveralModelsOneFileEach:
    """Several models of one region embedded in one run, one output each (stakeholder S1)."""

    def test_two_models_of_one_region_give_two_embeddings_files_in_one_run(self, tmp_path):
        first, second = lay_two_models_in_one_region(tmp_path, flat=False)

        run = run_stage(tmp_path)

        assert len(run.embeddings_files()) == 2, run.embeddings_files()
        for model in (first, second):
            call = run.evaluate.call_of(model)
            assert call.saving.read_bytes() == call.content


class TestMultiModelRegionPath:
    """Two or more models of a region: one subfolder per model id."""

    @pytest.mark.parametrize("ids", [("run_1", "run_2"), ("exp/v2", "run_3")], ids=["flat_ids", "nested_id"])
    def test_each_model_of_the_region_writes_under_its_model_id(self, tmp_path, ids):
        models = models_root(tmp_path)
        make_model(models, FLAT)
        for model_id in ids:
            make_model(models, MULTI, *model_id.split("/"))

        run = run_stage(tmp_path)

        assert run.embeddings_files() == {
            f"{FLAT}/{EMBEDDINGS}",
            *(f"{MULTI}/{model_id}/{EMBEDDINGS}" for model_id in ids),
        }

    def test_each_model_file_holds_the_bytes_of_its_own_evaluate_command(self, tmp_path):
        first, second = lay_two_models_in_one_region(tmp_path)

        run = run_stage(tmp_path)

        for model, model_id in ((first, "run_1"), (second, "run_2")):
            path = run.output / MULTI / model_id / EMBEDDINGS
            assert path.is_file(), run.embeddings_files()
            assert path.read_bytes() == run.evaluate.call_of(model).content


class TestPerModelFoldersPath:
    """--per-model-folders: subfolder for a non-empty model id."""

    def test_single_hf_cache_model_writes_under_its_model_id(self, tmp_path):
        lay_single_models(tmp_path, "hf_cache")

        run = run_stage(tmp_path, "--per-model-folders")

        assert run.embeddings_files() == {f"{region}/{HF_HASHES[region]}/{EMBEDDINGS}" for region in (FLAT, MULTI)}


class TestEmptyModelIdPath:
    """Empty model id: no model subfolder."""

    @pytest.mark.parametrize("extra", [(), ("--per-model-folders",)], ids=["no_flag", "per_model_folders"])
    def test_flat_models_write_in_the_region_folder(self, tmp_path, extra):
        lay_single_models(tmp_path, "flat")

        run = run_stage(tmp_path, *extra)

        assert run.embeddings_files() == {f"{FLAT}/{EMBEDDINGS}", f"{MULTI}/{EMBEDDINGS}"}

    def test_empty_model_id_beside_a_named_model_of_its_region(self, tmp_path):
        """variant_a/STs_right (empty model id) and variant_b/STs_right/run_2: two models of one region."""
        models = models_root(tmp_path)
        empty = make_model(models, "variant_a", MULTI)
        named = make_model(models, "variant_b", MULTI, "run_2")

        run = run_stage(tmp_path)

        assert run.embeddings_files() == {f"{MULTI}/{EMBEDDINGS}", f"{MULTI}/run_2/{EMBEDDINGS}"}
        assert (run.output / MULTI / EMBEDDINGS).read_bytes() == run.evaluate.call_of(empty).content
        assert (run.output / MULTI / "run_2" / EMBEDDINGS).read_bytes() == run.evaluate.call_of(named).content


class TestSingleModelPath:
    """One model per region without --per-model-folders: region folder."""

    @pytest.mark.parametrize("layout", ["flat", "hf_cache"])
    def test_one_model_per_region_writes_in_the_region_folder(self, tmp_path, layout):
        lay_single_models(tmp_path, layout)

        run = run_stage(tmp_path)

        assert run.embeddings_files() == {f"{FLAT}/{EMBEDDINGS}", f"{MULTI}/{EMBEDDINGS}"}


class TestSkipPerModel:
    """Without --overwrite: one evaluate.py command per model lacking an embeddings file."""

    def test_only_the_model_without_an_embeddings_file_is_evaluated(self, tmp_path):
        _first, second = lay_two_models_in_one_region(tmp_path, flat=False)
        existing = tmp_path / "out" / MULTI / "run_1" / EMBEDDINGS
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"kept\n")

        run = run_stage(tmp_path)

        assert [call.model for call in run.evaluate.calls] == [os.path.realpath(second)]
        assert run.evaluate.saved() == {run.output / MULTI / "run_2" / EMBEDDINGS}
        assert existing.read_bytes() == b"kept\n"

    def test_each_model_without_a_file_is_evaluated_once(self, tmp_path):
        first, second = lay_two_models_in_one_region(tmp_path)

        run = run_stage(tmp_path)

        assert len(run.evaluate.calls) == 3
        assert run.evaluate.call_of(first).saving != run.evaluate.call_of(second).saving


class TestOverwritePerModel:
    """With --overwrite: one evaluate.py command per model."""

    def test_every_existing_model_file_is_recomputed(self, tmp_path):
        first, second = lay_two_models_in_one_region(tmp_path, flat=False)
        for model_id in ("run_1", "run_2"):
            existing = tmp_path / "out" / MULTI / model_id / EMBEDDINGS
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"stale\n")

        run = run_stage(tmp_path, "--overwrite")

        assert run.evaluate.saved() == {run.output / MULTI / model_id / EMBEDDINGS for model_id in ("run_1", "run_2")}
        for model in (first, second):
            call = run.evaluate.call_of(model)
            assert call.saving.read_bytes() == call.content


class TestFailedModelDoesNotStopOthers:
    """A failed model does not stop the other models."""

    def test_second_model_of_the_region_runs_after_the_first_fails(self, tmp_path):
        first, second = lay_two_models_in_one_region(tmp_path)

        run = run_stage(tmp_path, evaluate=FakeEvaluate(failing=(first,)))

        call = run.evaluate.call_of(second)
        assert call.saving == run.output / MULTI / "run_2" / EMBEDDINGS
        assert call.saving.read_bytes() == call.content
        assert not (run.output / MULTI / "run_1" / EMBEDDINGS).exists()


class TestFailedModelNamedByLabel:
    """Failure summary names the failed model by label."""

    def test_fail_record_names_the_model_label(self, tmp_path, pipeline_caplog):
        first, _second = lay_two_models_in_one_region(tmp_path)

        run_stage(tmp_path, evaluate=FakeEvaluate(failing=(first,)))

        failures = stage_messages(pipeline_caplog, FAIL)
        assert any(f"{MULTI}/run_1" in message for message in failures), failures
        assert not any(f"{MULTI}/run_2" in message for message in failures), failures


class TestPerModelCheckpointEpoch:
    """Per-id checkpoint epoch reaches evaluate.py."""

    def test_model_named_by_an_id_value_is_evaluated_with_its_epoch(self, tmp_path):
        models = models_root(tmp_path)
        first = make_model(models, MULTI, "run_1", epochs=(DEFAULT_EPOCH,))
        second = make_model(models, MULTI, "run_2", epochs=(70,))

        run = run_stage(tmp_path, "--checkpoint-epoch", "run_2=70")

        assert run.evaluate.call_of(second).ckpts == ["epoch=70-step=1.ckpt"]
        assert run.evaluate.call_of(first).ckpts == [f"epoch={DEFAULT_EPOCH}-step=1.ckpt"]


class TestUnknownEpochModelIdRefused:
    """Unknown model id in --checkpoint-epoch refused."""

    def test_unknown_id_ends_with_status_2_before_evaluate(self, tmp_path):
        lay_two_models_in_one_region(tmp_path)
        script = parse(tmp_path, "--checkpoint-epoch", "nope=70")

        run = run_parsed(tmp_path, script)

        assert run.status == 2, run.status
        assert run.evaluate.calls == []


class TestCheckpointEpochRepeatable:
    """--checkpoint-epoch repeatable."""

    def test_bare_default_and_id_value_both_apply(self, tmp_path):
        models = models_root(tmp_path)
        first = make_model(models, MULTI, "run_1", epochs=(60,))
        second = make_model(models, MULTI, "run_2", epochs=(70,))

        run = run_stage(tmp_path, "--checkpoint-epoch", "60", "--checkpoint-epoch", "run_2=70")

        assert run.evaluate.call_of(first).ckpts == ["epoch=60-step=1.ckpt"]
        assert run.evaluate.call_of(second).ckpts == ["epoch=70-step=1.ckpt"]


class TestCheckpointEpochIdValueAccepted:
    """--checkpoint-epoch accepts <model id>=N."""

    @pytest.mark.parametrize("value", ["run_2=70", "exp/v2=0", "name08-32-57_200=80"])
    def test_id_value_parses(self, tmp_path, value):
        script = parse(tmp_path, f"--checkpoint-epoch={value}")

        assert script.args is not None


class TestCheckpointEpochRefusedValues:
    """--checkpoint-epoch values refused by parse_epoch_specs exit 2 (REQ-EPOCH80-BDRABCZUK-A01743BD1CE1 revised)."""

    @pytest.mark.parametrize(
        "values",
        [("=70",), ("run_2=x",), ("run_2=-1",), ("60", "70"), ("run_2=70", "run_2=71")],
        ids=["empty_id", "id_value_not_integer", "id_value_negative", "two_bare_values", "id_given_twice"],
    )
    def test_refused_values_exit_with_status_2(self, tmp_path, values, capsys):
        options = [f"--checkpoint-epoch={value}" for value in values]

        with pytest.raises(SystemExit) as stop:
            parse(tmp_path, *options)

        err = capsys.readouterr().err
        assert stop.value.code == 2, err
        assert "--checkpoint-epoch" in err, err


class TestSameModelIdRefused:
    """Two models of one region with one model id refused by the stage."""

    def test_refused_before_any_output_file(self, tmp_path):
        models = models_root(tmp_path)
        make_model(models, "variant_a", MULTI, "run_1")
        make_model(models, "variant_b", MULTI, "run_1")

        run = run_stage(tmp_path)

        assert run.status not in (0, 3), run.status
        written = sorted(path for path in run.output.rglob("*") if path.is_file()) if run.output.exists() else []
        assert written == []
        assert run.evaluate.calls == []


class TestFoldLineNamesModelLabel:
    """Fold line names the model label, one per model."""

    def test_one_fold_line_per_model_with_its_label(self, tmp_path, pipeline_caplog):
        lay_two_models_in_one_region(tmp_path)

        run_stage(tmp_path)

        folds = [message for message in stage_messages(pipeline_caplog, LOG) if "Fold " in message]
        expected = [f"Fold 1/3: {FLAT}", f"Fold 2/3: {MULTI}/run_1", f"Fold 3/3: {MULTI}/run_2"]
        for text in expected:
            assert any(text in message for message in folds), (text, folds)
        assert len(folds) == 3, folds


class TestMissingWeightsNamedByLabel:
    """Missing weights named by model label."""

    @pytest.mark.parametrize("layout", ["multi", "hf_cache"])
    def test_error_names_the_model_label(self, tmp_path, pipeline_caplog, layout):
        models = models_root(tmp_path)
        if layout == "multi":
            make_model(models, MULTI, "run_1")
            make_model(models, MULTI, "run_2", epochs=())
            label = f"{MULTI}/run_2"
        else:
            make_model(models, FLAT, HF_HASHES[FLAT], epochs=())
            label = f"{FLAT}/{HF_HASHES[FLAT]}"

        run = run_stage(tmp_path)

        reported = [str(run.status), *(item.getMessage() for item in pipeline_caplog.records)]
        assert any(label in text for text in reported), reported
        assert run.evaluate.calls == []

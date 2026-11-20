#!/usr/bin/env python3
"""
Stage 4 quality records over several models per region (TASK-MULTIMODEL-BDRABCZUK-98B60783E569, folding in the
scope of TASK-MULTIMODEL-BDRABCZUK-B4A7DB6A4E50): output QC reasons, width sidecar, removal of excluded subjects,
provenance models list and CKA names, per model.

The input and output QC run for real here (the conftest stubs are overridden); evaluate.py is faked and the models
tree is a fixture (helpers in tests/test_generate_embeddings_multi_model_output.py). Each class names its
Requirement by title (``alm get <id>``).
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.stage2_layout import add_crops
from tests.test_generate_embeddings_multi_model_output import (
    COHORT,
    DEFAULT_EPOCH,
    EMBEDDINGS,
    FLAT,
    HF_HASHES,
    LOCAL_MODELS,
    MULTI,
    FakeEvaluate,
    embeddings_table,
    lay_single_models,
    lay_two_models_in_one_region,
    make_model,
    models_root,
    run_stage,
)

pytestmark = pytest.mark.unit  # noqa: V107

CROPS = {FLAT: "S.C.-sylv.", MULTI: "S.T.s."}
EXCLUDED = "sub-02"
PROVENANCE_KEYS = ["masks", "model_version", "checkpoint_epoch", "models_source", "pipeline_commit"]


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage4_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage4_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


def lay_dataset(root: Path, excluded_from_multi: tuple[str, ...] = ()) -> Path:
    """The dataset of the run: the skeleton crops of FLAT and MULTI over COHORT, less ``excluded_from_multi`` in MULTI.

    A subject missing from one crop is qc 0 in input_qc.tsv."""
    dataset = Path(root) / "dataset"
    crops_dir = dataset / "derivatives" / f"cortical_tiles-{CORTICAL_TILES_VERSION}" / "crops" / LOCAL_MODELS / "2mm"
    add_crops(crops_dir, COHORT, (CROPS[FLAT],))
    add_crops(crops_dir, tuple(s for s in COHORT if s not in excluded_from_multi), (CROPS[MULTI],))
    return dataset


def output_qc_reasons(output: Path) -> dict[str, Counter]:
    """The reasons of each participant_id of output_qc.tsv."""
    report = output / "output_qc.tsv"
    assert report.is_file(), f"{report} was not written"
    return {row["participant_id"]: Counter(filter(None, row["reason"].split(";"))) for row in read_qc_tsv(report)}


def read_provenance(output: Path) -> dict:
    return json.loads((output / "provenance.json").read_text())


def epoch_ckpt(model: Path, epoch: int = DEFAULT_EPOCH) -> Path:
    return model / "logs" / "lightning_logs" / "version_0" / "checkpoints" / f"epoch={epoch}-step=1.ckpt"


def table_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row[0] for row in csv.reader(handle) if row][1:]


class TestOutputQcReasonsNameModelLabel:
    """Output QC reasons name the model label."""

    def test_missing_row_of_a_second_model_names_its_label(self, tmp_path):
        dataset = lay_dataset(tmp_path)
        _first, second = lay_two_models_in_one_region(tmp_path)
        short = embeddings_table(tuple(s for s in COHORT if s != EXCLUDED))

        run = run_stage(tmp_path, dataset=dataset, evaluate=FakeEvaluate(tables={second: short}))

        reasons = output_qc_reasons(run.output)
        assert reasons[EXCLUDED] == Counter({f"missing_row:{MULTI}/run_2": 1}), reasons[EXCLUDED]

    def test_single_hf_cache_model_reason_names_its_label(self, tmp_path):
        dataset = lay_dataset(tmp_path)
        models = lay_single_models(tmp_path, "hf_cache")
        short = embeddings_table(tuple(s for s in COHORT if s != EXCLUDED))

        run = run_stage(tmp_path, dataset=dataset, evaluate=FakeEvaluate(tables={models[FLAT]: short}))

        reasons = output_qc_reasons(run.output)
        assert reasons[EXCLUDED] == Counter({f"missing_row:{FLAT}/{HF_HASHES[FLAT]}": 1}), reasons[EXCLUDED]

    def test_empty_model_id_reason_keeps_the_region_name(self, tmp_path):
        """REQ-NTUI-BDRABCZUK-D956BBAA33E8 family: a flat model keeps <region> (guard)."""
        dataset = lay_dataset(tmp_path)
        models = lay_single_models(tmp_path, "flat")
        short = embeddings_table(tuple(s for s in COHORT if s != EXCLUDED))

        run = run_stage(tmp_path, dataset=dataset, evaluate=FakeEvaluate(tables={models[FLAT]: short}))

        assert output_qc_reasons(run.output)[EXCLUDED] == Counter({f"missing_row:{FLAT}": 1})


class TestWidthSidecarBesideModelFile:
    """Width sidecar beside each model embeddings file."""

    def test_each_model_folder_holds_the_width_of_its_model(self, tmp_path):
        dataset = lay_dataset(tmp_path)
        models = models_root(tmp_path)
        make_model(models, FLAT)
        first = make_model(models, MULTI, "run_1", width=3)
        second = make_model(models, MULTI, "run_2", width=4)
        tables = {first: embeddings_table(width=3), second: embeddings_table(width=4)}

        run = run_stage(tmp_path, dataset=dataset, evaluate=FakeEvaluate(tables=tables))

        for model_id, width in (("run_1", 3), ("run_2", 4)):
            sidecar = run.output / MULTI / model_id / "embedding_width.json"
            assert sidecar.is_file(), sorted(p.relative_to(run.output).as_posix() for p in run.output.rglob("*"))
            assert json.loads(sidecar.read_text())["width"] == width
        assert json.loads((run.output / FLAT / "embedding_width.json").read_text())["width"] == 3


class TestExcludedRowsRemovedFromEachModelFile:
    """Excluded subjects removed from each model embeddings file."""

    def test_qc0_subject_leaves_every_model_file_reused_ones_too(self, tmp_path):
        dataset = lay_dataset(tmp_path, excluded_from_multi=(EXCLUDED,))
        lay_two_models_in_one_region(tmp_path)
        reused = tmp_path / "out" / MULTI / "run_1" / EMBEDDINGS
        reused.parent.mkdir(parents=True)
        reused.write_bytes(embeddings_table(seed=99))

        run = run_stage(tmp_path, dataset=dataset)

        expected = [run.output / FLAT / EMBEDDINGS, reused, run.output / MULTI / "run_2" / EMBEDDINGS]
        for path in expected:
            assert path.is_file(), run.embeddings_files()
            ids = table_ids(path)
            assert EXCLUDED not in ids, (path, ids)
            assert set(ids) == set(COHORT) - {EXCLUDED}, (path, ids)


class TestProvenanceModelsList:
    """provenance.json models list with several models or --per-model-folders."""

    def test_several_models_of_a_region_list_each_model_last(self, tmp_path):
        dataset = lay_dataset(tmp_path)
        models = models_root(tmp_path)
        flat = make_model(models, FLAT)
        first, second = make_model(models, MULTI, "run_1"), make_model(models, MULTI, "run_2", epochs=(70,))

        run = run_stage(tmp_path, "--checkpoint-epoch", "run_2=70", dataset=dataset)

        provenance = read_provenance(run.output)
        assert list(provenance)[-1] == "models", list(provenance)
        assert list(provenance)[:-1] == PROVENANCE_KEYS
        listed = sorted(provenance["models"], key=lambda entry: (entry["region"], entry["model_id"]))
        expected = [
            (FLAT, "", DEFAULT_EPOCH, epoch_ckpt(flat)),
            (MULTI, "run_1", DEFAULT_EPOCH, epoch_ckpt(first)),
            (MULTI, "run_2", 70, epoch_ckpt(second, 70)),
        ]
        assert [sorted(entry) for entry in listed] == [["checkpoint_epoch", "model_id", "region", "weights_source"]] * 3
        assert [
            (entry["region"], entry["model_id"], entry["checkpoint_epoch"], Path(entry["weights_source"]).resolve())
            for entry in listed
        ] == [(region, model_id, epoch, path.resolve()) for region, model_id, epoch, path in expected]

    def test_per_model_folders_lists_single_models(self, tmp_path):
        dataset = lay_dataset(tmp_path)
        models = lay_single_models(tmp_path, "hf_cache")

        run = run_stage(tmp_path, "--per-model-folders", dataset=dataset)

        provenance = read_provenance(run.output)
        assert list(provenance)[-1] == "models", list(provenance)
        listed = sorted((entry["region"], entry["model_id"]) for entry in provenance["models"])
        assert listed == [(region, HF_HASHES[region]) for region in sorted(models)]


class TestProvenanceKeysUnchangedForSingleModels:
    """provenance.json keys unchanged for single models (guard)."""

    @pytest.mark.parametrize("layout", ["flat", "hf_cache"])
    def test_keys_are_the_pre_multi_model_keys_in_order(self, tmp_path, layout):
        dataset = lay_dataset(tmp_path)
        lay_single_models(tmp_path, layout)

        run = run_stage(tmp_path, dataset=dataset)

        assert list(read_provenance(run.output)) == PROVENANCE_KEYS


class TestCkaNamesModelsByLabel:
    """CKA results name each model by label."""

    def test_cka_matrix_rows_are_the_model_labels(self, tmp_path):
        dataset = lay_dataset(tmp_path)
        models = models_root(tmp_path)
        labels = []
        for region in (FLAT, MULTI):
            for model_id in ("run_1", "run_2"):
                make_model(models, region, model_id)
                labels.append(f"{region}/{model_id}")

        run = run_stage(tmp_path, "--run-cka", dataset=dataset)

        matrix = run.output / "cka_results" / "cka_matrix.csv"
        assert matrix.is_file(), sorted(p.relative_to(run.output).as_posix() for p in run.output.rglob("*"))
        assert sorted(table_ids(matrix)) == sorted(labels)

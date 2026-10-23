#!/usr/bin/env python3
"""
provenance.json written by generate_embeddings next to the region embeddings (MODELVER epic,
TASK-MODELVER-BDRABCZUK-10E2986F4C32).

After the per-region evaluation loop, <region embeddings output dir>/provenance.json is a JSON
object with the keys:

- masks: the --masks value in effect;
- model_version: model_version of the fetched model set's model_set.json, else --model-version,
  else null;
- checkpoint_epoch: the --checkpoint-epoch value that selected the region checkpoints;
- models_source: {"type": "local", "path": <absolute models_path>} or
  {"type": "huggingface", "repo_id": <repository id>, "subfolder": <--model-version or null>};
- pipeline_commit: the 40-hex git commit of the champollion_pipeline checkout, else null.

No network, no real evaluate.py, no real git: huggingface_hub.snapshot_download is a fake that lays
out region directories, execute_command is a fake that writes the -s output, and the pipeline
commit comes from generate_embeddings.read_pipeline_commit, patched in the stage tests; its own
tests replace subprocess.run.
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline import generate_embeddings as ge
from champollion_pipeline.derivatives_layout import compute_region_embeddings_dir
from champollion_pipeline.generate_embeddings import GenerateEmbeddings, ModelSet, write_model_set

pytestmark = pytest.mark.unit  # noqa: V107

PROVENANCE = "provenance.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPO_ID = "neurospin/Champollion_V1"
VERSION = "canonical_corrected_26_1"
REGIONS = ["Aaa_left", "Bbb_right"]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def lay_out_regions(models: Path, regions: list[str] = REGIONS, epoch: int = 80) -> Path:
    """Create <models>/<region>/logs/lightning_logs/version_0/checkpoints/epoch=<epoch>-step=1.ckpt per region."""
    for region in regions:
        ckpt_dir = models / region / "logs" / "lightning_logs" / "version_0" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / f"epoch={epoch}-step=1.ckpt").write_text("placeholder")
    return models


def fake_evaluate(cmd, **_kwargs) -> int:
    """execute_command stand-in: write the -s output the way evaluate.py would, return 0."""
    out = Path(cmd[cmd.index("-s") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("ID,dim1\n")
    return 0


def fake_snapshot_download(repo_id, local_dir, allow_patterns=None, **_options):
    """snapshot_download stand-in: lay out region model directories under the requested subfolder."""
    root = Path(local_dir)
    for pattern in allow_patterns or ["*"]:
        lay_out_regions(root / pattern.rstrip("*").rstrip("/"))
    return local_dir


def run_stage(argv: list[str], commit: str | None = COMMIT) -> None:
    """Parse argv with the real parser and run the stage with the fake evaluate and a fixed pipeline commit."""
    script = GenerateEmbeddings()
    script.parse_args(argv)
    script.execute_command = fake_evaluate
    with (
        patch.object(ge, "read_pipeline_commit", return_value=commit),
        patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download),
    ):
        script.run()


def read_provenance(output_dir: Path) -> dict:
    path = output_dir / PROVENANCE
    assert path.is_file(), f"no {PROVENANCE} in {output_dir}: {sorted(os.listdir(output_dir))}"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert type(data) is dict, f"{path} is not a JSON object: {data!r}"
    return data


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """cwd = tmp_path, so a Hugging Face repo id is never a local directory; keep SSL_CERT_FILE untouched."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "no-ca.crt"))
    return tmp_path


def local_argv(workdir: Path, *extra: str) -> list[str]:
    return [
        str(workdir / "models"),
        str(workdir / "dataset"),
        "--models-cache",
        str(workdir / "cache"),
        "--output",
        str(workdir / "out"),
        *extra,
    ]


# --------------------------------------------------------------------------- #
# REQ-MODELVER-BDRABCZUK-E821EF840132: file, place, masks
# --------------------------------------------------------------------------- #


class TestProvenanceWrittenWithMasks:
    """REQ-MODELVER-BDRABCZUK-E821EF840132: provenance.json in the output directory; masks = --masks in effect."""

    def test_written_in_output_dir_with_default_masks(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(local_argv(workdir))
        assert read_provenance(workdir / "out")["masks"] == "canonical_25"

    def test_written_in_default_region_embeddings_dir_with_given_masks(self, workdir):
        lay_out_regions(workdir / "models")
        dataset = workdir / "dataset"
        run_stage([str(workdir / "models"), str(dataset), "--models-cache", str(workdir / "cache"), "--masks", "m9"])
        output_dir = Path(compute_region_embeddings_dir(str(dataset), "m9"))
        assert read_provenance(output_dir)["masks"] == "m9"

    def test_written_when_every_region_is_skipped(self, workdir):
        lay_out_regions(workdir / "models")
        for region in REGIONS:
            existing = workdir / "out" / region / "full_embeddings.csv"
            existing.parent.mkdir(parents=True)
            existing.write_text("ID,dim1\n")
        run_stage(local_argv(workdir))
        assert read_provenance(workdir / "out")["masks"] == "canonical_25"


# --------------------------------------------------------------------------- #
# REQ-MODELVER-BDRABCZUK-F981AB1172B4: model_version
# --------------------------------------------------------------------------- #


class TestProvenanceModelVersion:
    """REQ-MODELVER-BDRABCZUK-F981AB1172B4: model_version from model_set.json, else --model-version, else null."""

    def test_from_local_model_set_marker(self, workdir):
        models = lay_out_regions(workdir / "models")
        write_model_set(str(models), ModelSet("marker_version", "canonical_25"))
        run_stage(local_argv(workdir))
        assert read_provenance(workdir / "out")["model_version"] == "marker_version"

    def test_from_marker_when_regions_are_restricted(self, workdir):
        models = lay_out_regions(workdir / "models")
        write_model_set(str(models), ModelSet("marker_version", "canonical_25"))
        run_stage(local_argv(workdir, "--regions", REGIONS[0]))
        assert read_provenance(workdir / "out")["model_version"] == "marker_version"

    def test_from_model_version_option_without_marker(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(local_argv(workdir, "--model-version", "v_option"))
        assert read_provenance(workdir / "out")["model_version"] == "v_option"

    def test_null_without_marker_or_option(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(local_argv(workdir))
        provenance = read_provenance(workdir / "out")
        assert "model_version" in provenance and provenance["model_version"] is None

    def test_from_marker_written_by_hugging_face_download(self, workdir):
        run_stage(
            [
                REPO_ID,
                str(workdir / "dataset"),
                "--model-version",
                VERSION,
                "--models-cache",
                str(workdir / "cache"),
                "--output",
                str(workdir / "out"),
            ]
        )
        assert read_provenance(workdir / "out")["model_version"] == VERSION


# --------------------------------------------------------------------------- #
# REQ-MODELVER-BDRABCZUK-4BD0061E6FC2: checkpoint_epoch
# --------------------------------------------------------------------------- #


class TestProvenanceCheckpointEpoch:
    """REQ-MODELVER-BDRABCZUK-4BD0061E6FC2: checkpoint_epoch = the --checkpoint-epoch value."""

    def test_default_epoch_80(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(local_argv(workdir))
        epoch = read_provenance(workdir / "out")["checkpoint_epoch"]
        assert type(epoch) is int and epoch == 80

    def test_given_epoch(self, workdir):
        lay_out_regions(workdir / "models", epoch=12)
        run_stage(local_argv(workdir, "--checkpoint-epoch", "12"))
        epoch = read_provenance(workdir / "out")["checkpoint_epoch"]
        assert type(epoch) is int and epoch == 12

    def test_marker_epoch_does_not_replace_option(self, workdir):
        models = lay_out_regions(workdir / "models")
        write_model_set(str(models), ModelSet("marker_version", "canonical_25", 12))
        run_stage(local_argv(workdir))
        assert read_provenance(workdir / "out")["checkpoint_epoch"] == 80


# --------------------------------------------------------------------------- #
# REQ-MODELVER-BDRABCZUK-E46B4CCA8743: models_source
# --------------------------------------------------------------------------- #


class TestProvenanceModelsSource:
    """REQ-MODELVER-BDRABCZUK-E46B4CCA8743: models_source = local path, or HF repo id and subfolder."""

    def test_local_absolute_path(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(local_argv(workdir))
        expected = {"type": "local", "path": os.path.abspath(workdir / "models")}
        assert read_provenance(workdir / "out")["models_source"] == expected

    def test_local_relative_path_made_absolute(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(["models", "dataset", "--models-cache", "cache", "--output", str(workdir / "out")])
        expected = {"type": "local", "path": os.path.abspath("models")}
        assert read_provenance(workdir / "out")["models_source"] == expected

    def test_hugging_face_repo_id_and_subfolder(self, workdir):
        run_stage(
            [
                REPO_ID,
                str(workdir / "dataset"),
                "--model-version",
                VERSION,
                "--models-cache",
                str(workdir / "cache"),
                "--output",
                str(workdir / "out"),
            ]
        )
        expected = {"type": "huggingface", "repo_id": REPO_ID, "subfolder": VERSION}
        assert read_provenance(workdir / "out")["models_source"] == expected

    def test_hugging_face_url_gives_repo_id(self, workdir):
        run_stage(
            [
                f"https://huggingface.co/{REPO_ID}",
                str(workdir / "dataset"),
                "--model-version",
                VERSION,
                "--models-cache",
                str(workdir / "cache"),
                "--output",
                str(workdir / "out"),
            ]
        )
        expected = {"type": "huggingface", "repo_id": REPO_ID, "subfolder": VERSION}
        assert read_provenance(workdir / "out")["models_source"] == expected

    def test_hugging_face_without_model_version_has_null_subfolder(self, workdir):
        run_stage(
            [
                REPO_ID,
                str(workdir / "dataset"),
                "--models-cache",
                str(workdir / "cache"),
                "--output",
                str(workdir / "out"),
            ]
        )
        expected = {"type": "huggingface", "repo_id": REPO_ID, "subfolder": None}
        assert read_provenance(workdir / "out")["models_source"] == expected


# --------------------------------------------------------------------------- #
# REQ-MODELVER-BDRABCZUK-EA3EDB342A85: pipeline_commit
# --------------------------------------------------------------------------- #


def fake_git_run(stdout: str = "", returncode: int = 0, raises: BaseException | None = None):
    """subprocess.run stand-in answering any command with stdout/returncode (str or bytes as asked), or raising."""

    def fake(args, *_args, **kwargs):
        if raises is not None:
            raise raises
        textual = kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding")
        out = stdout if textual else stdout.encode()
        if kwargs.get("check") and returncode != 0:
            raise subprocess.CalledProcessError(returncode, args, output=out)
        return subprocess.CompletedProcess(args, returncode, stdout=out, stderr="" if textual else b"")

    return fake


class TestProvenancePipelineCommit:
    """REQ-MODELVER-BDRABCZUK-EA3EDB342A85: pipeline_commit = 40-hex pipeline commit, else null."""

    def test_stage_records_the_pipeline_commit(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(local_argv(workdir), commit=COMMIT)
        assert read_provenance(workdir / "out")["pipeline_commit"] == COMMIT

    def test_stage_records_null_when_commit_unknown(self, workdir):
        lay_out_regions(workdir / "models")
        run_stage(local_argv(workdir), commit=None)
        provenance = read_provenance(workdir / "out")
        assert "pipeline_commit" in provenance and provenance["pipeline_commit"] is None

    def test_reader_returns_git_head_commit(self):
        with patch("subprocess.run", side_effect=fake_git_run(stdout=COMMIT + "\n")):
            assert ge.read_pipeline_commit() == COMMIT

    def test_reader_null_when_git_missing(self):
        with patch("subprocess.run", side_effect=fake_git_run(raises=FileNotFoundError("git"))):
            assert ge.read_pipeline_commit() is None

    def test_reader_null_when_not_a_git_checkout(self):
        with patch("subprocess.run", side_effect=fake_git_run(stdout="", returncode=128)):
            assert ge.read_pipeline_commit() is None

    def test_reader_null_when_output_is_not_a_commit(self):
        with patch("subprocess.run", side_effect=fake_git_run(stdout="HEAD\n")):
            assert ge.read_pipeline_commit() is None

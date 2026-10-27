#!/usr/bin/env python3
"""
Embeddings evaluate a region that holds an epoch-N checkpoint with that
checkpoint only (TASK-EPOCH80-BDRABCZUK-EABDCCF7F67F, user decisions 2026-10-13).

An epoch-N checkpoint (TERM-EPOCH80-BDRABCZUK-F4174305903D) is a Lightning file
``epoch=<N>-step=<S>.ckpt`` in any ``logs/lightning_logs/version_*/checkpoints/``
of a region model directory, N being ``args.checkpoint_epoch`` (default 80, set
by the parser task TASK-EPOCH80-BDRABCZUK-7DD4EABB61C5). These tests set
``args.checkpoint_epoch`` directly on the parsed namespace.

evaluate.py (champollion_V1, frozen) loads the first .ckpt globbed in
``<-m>/logs/lightning_logs/version_0/checkpoints/`` and ``<-m>/.hydra/config.yaml``.
_run_per_region runs with execute_command replaced by a fake that inspects the
``-m`` directory at call time, the way evaluate.py would. No real evaluate.py,
no real model, no network: small torch files in tmp_path.
"""

from pathlib import Path

import pytest
import torch

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.test_generate_embeddings_best_weights import (
    BEST_STATE,
    HYDRA_CONFIG,
    REGION,
    ckpt_dir_of,
    lines_naming,
    make_model_dir,
    record_text,
    snapshot_tree,
    write_converted_ckpt,
    write_weights,
)

pytestmark = pytest.mark.unit  # noqa: V107

DEFAULT_EPOCH = 80


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def write_epoch_ckpt(model_dir: Path, epoch: int, step: int, version: int = 0) -> Path:
    """Write a Lightning-style checkpoint whose bytes are unique to (epoch, step, version)."""
    ckpt_dir = ckpt_dir_of(model_dir, version)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch={epoch}-step={step}.ckpt"
    state = {"backbones.0.encoder.w": torch.full((4,), float(epoch * 1000 + version)), "step": torch.tensor(step)}
    torch.save({"state_dict": state, "epoch": epoch, "global_step": step, "optimizer_states": []}, str(path))
    return path


class BytesEvaluate:
    """Stand-in for execute_command recording what evaluate.py would see at call time.

    Per call: the -m path, the name and bytes (symlinks followed) of every .ckpt
    in <-m>/logs/lightning_logs/version_0/checkpoints/, and the bytes of
    <-m>/.hydra/config.yaml. Writes the -s output and returns 0.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, cmd, **_kwargs):
        model_arg = cmd[cmd.index("-m") + 1]
        ckpts = sorted(Path(model_arg).glob("logs/lightning_logs/version_0/checkpoints/*.ckpt"))
        config = Path(model_arg) / ".hydra" / "config.yaml"
        self.calls.append(
            {
                "model_arg": model_arg,
                "ckpt_names": [p.name for p in ckpts],
                "ckpt_bytes": [p.read_bytes() for p in ckpts],
                "config_bytes": config.read_bytes() if config.is_file() else None,
            }
        )
        out = Path(cmd[cmd.index("-s") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ID,dim1\n")
        return 0


def run_region(tmp_path: Path, checkpoint_epoch: int = DEFAULT_EPOCH) -> BytesEvaluate:
    """Run _run_per_region over tmp_path/models with args.checkpoint_epoch and the fake evaluate."""
    script = GenerateEmbeddings()
    script.parse_args([str(tmp_path / "models"), str(tmp_path / "dataset")])
    script.args.checkpoint_epoch = checkpoint_epoch
    fake = BytesEvaluate()
    script.execute_command = fake
    code = script._run_per_region("evaluate.py", str(tmp_path / "crops"), str(tmp_path / "out"))
    assert code == 0
    assert len(fake.calls) == 1
    return fake


def assert_evaluate_sees_only(call: dict, expected: Path) -> None:
    assert len(call["ckpt_names"]) == 1, (
        f"evaluate.py would glob {call['ckpt_names']} in {call['model_arg']}; expected only {expected.name}"
    )
    assert call["ckpt_bytes"][0] == expected.read_bytes(), (
        f"evaluate.py would load {call['ckpt_names'][0]} from {call['model_arg']}, whose bytes differ from {expected}"
    )


# --------------------------------------------------------------------------- #
# REQ-EPOCHSEL-BDRABCZUK-8D8F3519F164
# --------------------------------------------------------------------------- #


class TestEvaluateSeesOnlyEpochNCkpt:
    """REQ-EPOCHSEL-...-8D8F3519F164: version_0 of -m holds exactly one .ckpt, byte-identical
    to the highest-step epoch-N checkpoint."""

    def test_epoch_n_in_version_0_beside_best_weights(self, tmp_path):
        """HF canonical_25 layout: best weights never win over the epoch-80 checkpoint."""
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        epoch_n = write_epoch_ckpt(model_dir, 80, 106272)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], epoch_n)

    @pytest.mark.parametrize("with_best_weights", [True, False], ids=["with_best_weights", "without_best_weights"])
    def test_epoch_n_in_version_1_beside_aborted_version_0(self, tmp_path, with_best_weights):
        """Jean-Zay cc26_1 layout: aborted run in version_0, epoch 80 in version_1."""
        model_dir = make_model_dir(tmp_path)
        if with_best_weights:
            write_weights(model_dir, BEST_STATE)
        write_epoch_ckpt(model_dir, 2, 3936, version=0)
        epoch_n = write_epoch_ckpt(model_dir, 80, 106272, version=1)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], epoch_n)

    def test_stale_converted_best_model_ckpt_beside_epoch_n(self, tmp_path):
        """A best_model.ckpt left in version_0 by an earlier conversion is ignored."""
        model_dir = make_model_dir(tmp_path)
        epoch_n = write_epoch_ckpt(model_dir, 80, 106272)
        write_converted_ckpt(model_dir, BEST_STATE)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], epoch_n)

    def test_highest_step_wins_numerically(self, tmp_path):
        """step=106272 beats step=99999 although it sorts first as text."""
        model_dir = make_model_dir(tmp_path)
        write_epoch_ckpt(model_dir, 80, 99999, version=0)
        highest = write_epoch_ckpt(model_dir, 80, 106272, version=1)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], highest)

    def test_configured_epoch_selected(self, tmp_path):
        """args.checkpoint_epoch=50 selects the epoch-50 checkpoint over epoch 80."""
        model_dir = make_model_dir(tmp_path)
        epoch_50 = write_epoch_ckpt(model_dir, 50, 66000)
        write_epoch_ckpt(model_dir, 80, 106272)

        fake = run_region(tmp_path, checkpoint_epoch=50)

        assert_evaluate_sees_only(fake.calls[0], epoch_50)

    def test_neighbour_epochs_not_selected(self, tmp_path):
        """epoch=8, epoch=79 and epoch=800 are not epoch-80 checkpoints."""
        model_dir = make_model_dir(tmp_path)
        write_epoch_ckpt(model_dir, 8, 10496)
        write_epoch_ckpt(model_dir, 79, 104960)
        write_epoch_ckpt(model_dir, 800, 1049600)
        epoch_n = write_epoch_ckpt(model_dir, 80, 106272, version=1)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], epoch_n)


# --------------------------------------------------------------------------- #
# REQ-EPOCHSEL-BDRABCZUK-6CFCE816BD78 (guard)
# --------------------------------------------------------------------------- #


class TestEvaluateSeesRegionHydraConfig:
    """REQ-EPOCHSEL-...-6CFCE816BD78: -m carries the region's .hydra/config.yaml, byte for byte."""

    def test_config_identical_epoch_n_in_version_1(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_epoch_ckpt(model_dir, 2, 3936, version=0)
        write_epoch_ckpt(model_dir, 80, 106272, version=1)

        fake = run_region(tmp_path)

        assert fake.calls[0]["config_bytes"] == HYDRA_CONFIG.encode()


# --------------------------------------------------------------------------- #
# REQ-EPOCHSEL-BDRABCZUK-13E49EEC035F (guard)
# --------------------------------------------------------------------------- #


class TestRegionModelDirUnchanged:
    """REQ-EPOCHSEL-...-13E49EEC035F: no file under the region model dir is added, removed or rewritten."""

    def test_dir_unchanged_with_stale_converted_ckpt_and_best_weights(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_converted_ckpt(model_dir, BEST_STATE)
        write_epoch_ckpt(model_dir, 2, 3936, version=0)
        write_epoch_ckpt(model_dir, 80, 106272, version=1)
        before = snapshot_tree(model_dir)

        run_region(tmp_path)

        assert snapshot_tree(model_dir) == before


# --------------------------------------------------------------------------- #
# REQ-EPOCHSEL-BDRABCZUK-D80BDF703B49
# --------------------------------------------------------------------------- #


class TestModelArgIsRemovedOutsideDir:
    """REQ-EPOCHSEL-...-D80BDF703B49: -m is outside the region model dir and gone once the stage returns."""

    def test_model_arg_outside_region_dir_and_removed(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_epoch_ckpt(model_dir, 80, 106272)

        fake = run_region(tmp_path)

        model_arg = Path(fake.calls[0]["model_arg"])
        region_dir = model_dir.resolve()
        assert model_arg.resolve() != region_dir and region_dir not in model_arg.resolve().parents, (
            f"-m {model_arg} is the region model directory {model_dir} or lies inside it"
        )
        assert not model_arg.exists(), f"-m {model_arg} still exists after the embeddings stage returned"


# --------------------------------------------------------------------------- #
# REQ-EPOCHSEL-BDRABCZUK-00A1091ADECB
# --------------------------------------------------------------------------- #


class TestEpochNCkptLogged:
    """REQ-EPOCHSEL-...-00A1091ADECB: one console line names the region and the epoch-N
    checkpoint's path under the region model dir."""

    def test_one_line_names_region_and_epoch_n_ckpt(self, tmp_path, pipeline_caplog):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_epoch_ckpt(model_dir, 2, 3936, version=0)
        epoch_n = write_epoch_ckpt(model_dir, 80, 106272, version=1)

        run_region(tmp_path)

        out = record_text(pipeline_caplog.records)
        assert len(lines_naming(out, REGION, str(epoch_n))) == 1, out

#!/usr/bin/env python3
"""
Interim stand-in: a region model directory with logs/best_model_weights.pt and no
epoch-N checkpoint (the cc26_1 HF models) is evaluated with best_model_weights.pt
in place of epoch N, through a temporary mirror, the model directory never being
written, with one warning per such region. A region holding neither stops the
stage before any region is evaluated (TASK-EPOCH80-BDRABCZUK-167CF558ABF3, user
decision 2026-10-13).

An epoch-N checkpoint (TERM-EPOCH80-BDRABCZUK-F4174305903D) is a Lightning file
``epoch=<N>-step=<S>.ckpt`` in any ``logs/lightning_logs/version_*/checkpoints/``,
N being ``args.checkpoint_epoch``. As in the sibling tests, the stage is driven
through ``_run_per_region`` with execute_command replaced by a fake evaluate.py
that records the ``-m`` directory at call time. No real evaluate.py, model or
network: small torch files in tmp_path.
"""

import re
from pathlib import Path

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.test_generate_embeddings_best_weights import (
    BEST_STATE,
    HYDRA_CONFIG,
    NATIVE_STATE,
    OLD_STATE,
    snapshot_tree,
    write_converted_ckpt,
    write_native_ckpt,
    write_weights,
)
from tests.test_generate_embeddings_epoch_checkpoint import BytesEvaluate, write_epoch_ckpt

pytestmark = pytest.mark.unit  # noqa: V107

DEFAULT_EPOCH = 80
STAND_IN = "Aaa_left"


# --------------------------------------------------------------------------- #
# Builders and runner
# --------------------------------------------------------------------------- #


def make_region_dir(tmp_path: Path, region: str) -> Path:
    """<tmp>/models/<region> with an empty logs/ and the region's .hydra/config.yaml."""
    model_dir = tmp_path / "models" / region
    (model_dir / "logs").mkdir(parents=True)
    (model_dir / ".hydra").mkdir()
    (model_dir / ".hydra" / "config.yaml").write_text(HYDRA_CONFIG)
    return model_dir


def stand_in_best_weights_only(model_dir: Path) -> None:
    """Clean cc26_1 HF layout: best_model_weights.pt, no checkpoint at all."""
    write_weights(model_dir, BEST_STATE)


def stand_in_beside_stale_converted(model_dir: Path) -> None:
    """HF layout after an earlier in-place conversion of older weights."""
    write_weights(model_dir, BEST_STATE)
    write_converted_ckpt(model_dir, OLD_STATE)


def stand_in_beside_other_epoch(model_dir: Path) -> None:
    """Best weights beside a native checkpoint of an epoch other than N."""
    write_weights(model_dir, BEST_STATE)
    write_native_ckpt(model_dir, "epoch=79-step=104960.ckpt", NATIVE_STATE)


STAND_IN_LAYOUTS = {
    "best_weights_only": stand_in_best_weights_only,
    "beside_stale_converted": stand_in_beside_stale_converted,
    "beside_other_epoch": stand_in_beside_other_epoch,
}


def no_weights_empty_logs(_model_dir: Path) -> None:
    """Nothing under logs/."""


def no_weights_other_epoch_only(model_dir: Path) -> None:
    """A native checkpoint of an epoch other than N, no best_model_weights.pt."""
    write_native_ckpt(model_dir, "epoch=79-step=104960.ckpt", NATIVE_STATE)


def no_weights_converted_only(model_dir: Path) -> None:
    """A converted best_model.ckpt left by an earlier run, no best_model_weights.pt."""
    write_converted_ckpt(model_dir, BEST_STATE)


NO_WEIGHTS_LAYOUTS = {
    "empty_logs": no_weights_empty_logs,
    "other_epoch_only": no_weights_other_epoch_only,
    "converted_only": no_weights_converted_only,
}


class StageResult:
    """What one _run_per_region call produced: the fake evaluate, the status, the error text."""

    def __init__(self, fake: BytesEvaluate, status: int | str | None, error: str):
        self.fake = fake
        self.status = status
        self.error = error


def run_stage(tmp_path: Path, checkpoint_epoch: int = DEFAULT_EPOCH) -> StageResult:
    """Run _run_per_region over tmp_path/models; an exception counts as a non-zero status."""
    script = GenerateEmbeddings()
    script.parse_args([str(tmp_path / "models"), str(tmp_path / "dataset")])
    script.args.checkpoint_epoch = checkpoint_epoch
    fake = BytesEvaluate()
    script.execute_command = fake
    try:
        status = script._run_per_region("evaluate.py", str(tmp_path / "crops"), str(tmp_path / "out"))
        error = ""
    except SystemExit as err:
        status, error = err.code, str(err.code)
    except Exception as err:
        status, error = 1, str(err)
    return StageResult(fake, status, error)


def is_non_zero(status: int | str | None) -> bool:
    return status not in (0, None)


def plain_words(line: str) -> list[str]:
    """Words of a line that are not paths (region names also appear inside model paths)."""
    return [word for word in line.split() if "/" not in word]


def warning_lines(output: str, region: str, epoch: int) -> list[str]:
    """Lines holding WARNING, the region name outside paths, best_model_weights.pt and 'epoch <N>'."""
    epoch_text = re.compile(rf"\bepoch {epoch}\b")
    return [
        line
        for line in output.splitlines()
        if "WARNING" in line
        and "best_model_weights.pt" in line
        and epoch_text.search(line)
        and any(region in word for word in plain_words(line))
    ]


# --------------------------------------------------------------------------- #
# REQ-EPOCH80-BDRABCZUK-6B3BAAFB766D
# --------------------------------------------------------------------------- #


class TestStandInModelArgOutsideAndRemoved:
    """REQ-EPOCH80-...-6B3BAAFB766D: -m is a directory outside the region model directory,
    gone once the stage returns."""

    @pytest.mark.parametrize("layout", STAND_IN_LAYOUTS)
    def test_model_arg_outside_region_dir_and_removed(self, tmp_path, layout):
        model_dir = make_region_dir(tmp_path, STAND_IN)
        STAND_IN_LAYOUTS[layout](model_dir)

        result = run_stage(tmp_path)

        assert result.status == 0, result.error
        assert len(result.fake.calls) == 1
        model_arg = Path(result.fake.calls[0]["model_arg"])
        assert not model_arg.resolve().is_relative_to(model_dir.resolve()), (
            f"-m {model_arg} lies inside the region model directory {model_dir}"
        )
        assert not model_arg.exists(), f"-m {model_arg} still exists after the stage returned"


# --------------------------------------------------------------------------- #
# REQ-EPOCH80-BDRABCZUK-1F4C58983462
# --------------------------------------------------------------------------- #


class TestStandInRegionDirUnchanged:
    """REQ-EPOCH80-...-1F4C58983462: file names and bytes under the region model directory unchanged."""

    @pytest.mark.parametrize("layout", STAND_IN_LAYOUTS)
    def test_region_dir_unchanged(self, tmp_path, layout):
        model_dir = make_region_dir(tmp_path, STAND_IN)
        STAND_IN_LAYOUTS[layout](model_dir)
        before = snapshot_tree(model_dir)

        result = run_stage(tmp_path)

        assert result.status == 0, result.error
        after = snapshot_tree(model_dir)
        assert sorted(after) == sorted(before), "files added to or removed from the region model directory"
        changed = [name for name in before if after[name] != before[name]]
        assert not changed, f"files rewritten in the region model directory: {changed}"


# --------------------------------------------------------------------------- #
# REQ-EPOCH80-BDRABCZUK-C14DD05017E6
# --------------------------------------------------------------------------- #


class TestStandInWarning:
    """REQ-EPOCH80-...-C14DD05017E6: exactly one line with WARNING, the region name,
    best_model_weights.pt and 'epoch <N>'."""

    @pytest.mark.parametrize("epoch", [DEFAULT_EPOCH, 50])
    @pytest.mark.parametrize("layout", STAND_IN_LAYOUTS)
    def test_one_warning_line_per_stand_in_region(self, tmp_path, capsys, layout, epoch):
        STAND_IN_LAYOUTS[layout](make_region_dir(tmp_path, STAND_IN))

        result = run_stage(tmp_path, checkpoint_epoch=epoch)

        assert result.status == 0, result.error
        lines = warning_lines(capsys.readouterr().out, STAND_IN, epoch)
        assert len(lines) == 1, f"expected one stand-in warning for {STAND_IN}, got {lines}"

    def test_warning_only_names_its_own_region(self, tmp_path, capsys):
        """Beside a region holding the epoch-N checkpoint, the stand-in region still gets one warning."""
        stand_in_best_weights_only(make_region_dir(tmp_path, STAND_IN))
        write_epoch_ckpt(make_region_dir(tmp_path, "Bbb_left"), DEFAULT_EPOCH, 106272)
        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path / "models"), str(tmp_path / "dataset")])
        script.args.checkpoint_epoch = DEFAULT_EPOCH
        script.execute_command = BytesEvaluate()

        assert script._run_per_region("evaluate.py", str(tmp_path / "crops"), str(tmp_path / "out")) == 0

        assert len(warning_lines(capsys.readouterr().out, STAND_IN, DEFAULT_EPOCH)) == 1


# --------------------------------------------------------------------------- #
# REQ-EPOCH80-BDRABCZUK-9F8762DC8C69
# --------------------------------------------------------------------------- #


class TestNoWeightsFailsFast:
    """REQ-EPOCH80-...-9F8762DC8C69: a region with neither ends the stage non-zero, evaluate.py never run."""

    @pytest.mark.parametrize("layout", NO_WEIGHTS_LAYOUTS)
    def test_no_evaluate_and_non_zero_status(self, tmp_path, layout):
        """The region without weights sorts after a valid stand-in region, which is not evaluated either."""
        stand_in_best_weights_only(make_region_dir(tmp_path, STAND_IN))
        NO_WEIGHTS_LAYOUTS[layout](make_region_dir(tmp_path, "Bbb_left"))

        result = run_stage(tmp_path)

        assert result.fake.calls == [], f"evaluate.py ran for {[c['model_arg'] for c in result.fake.calls]}"
        assert is_non_zero(result.status), f"stage ended with status {result.status!r}"

    @pytest.mark.parametrize("layout", NO_WEIGHTS_LAYOUTS)
    def test_single_region_without_weights(self, tmp_path, layout):
        NO_WEIGHTS_LAYOUTS[layout](make_region_dir(tmp_path, "Bbb_left"))

        result = run_stage(tmp_path)

        assert result.fake.calls == [], "evaluate.py ran for a region without weights"
        assert is_non_zero(result.status), f"stage ended with status {result.status!r}"


# --------------------------------------------------------------------------- #
# REQ-EPOCH80-BDRABCZUK-F10D477D2130
# --------------------------------------------------------------------------- #


class TestNoWeightsRegionsNamed:
    """REQ-EPOCH80-...-F10D477D2130: the error message names every region without weights."""

    def test_error_names_regions_without_weights(self, tmp_path, capsys):
        stand_in_best_weights_only(make_region_dir(tmp_path, STAND_IN))
        no_weights_empty_logs(make_region_dir(tmp_path, "Bbb_left"))
        no_weights_other_epoch_only(make_region_dir(tmp_path, "Ccc_right"))

        result = run_stage(tmp_path)

        captured = capsys.readouterr()
        error_lines = [line for line in (captured.out + captured.err).splitlines() if "error" in line.lower()]
        error_text = "\n".join([result.error, *error_lines])
        for region in ("Bbb_left", "Ccc_right"):
            assert any(region in word for word in plain_words(error_text)), (
                f"no error message names {region}: {error_text!r}"
            )

"""TASK-210: the per-region weights log must be honest when evaluate.py's choice is not.

champollion_V1's evaluate.py (frozen, see TASK-144) loads
glob(<-m>/logs/lightning_logs/version_0/checkpoints/*.ckpt)[0] without sorting.
When the -m directory holds several .ckpt files there, the pipeline cannot know
which one evaluate.py loads, so its console output names every candidate and
warns that the choice depends on filesystem order.

Hermetic: tmp_path model dirs, evaluate.py replaced by FakeEvaluate.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.test_generate_embeddings_best_weights import (
    BEST_STATE,
    NATIVE_STATE,
    OLD_STATE,
    REGION,
    FakeEvaluate,
    make_model_dir,
    write_converted_ckpt,
    write_native_ckpt,
    write_weights,
)

pytestmark = pytest.mark.unit  # noqa: V107

LAST_CKPT_FLAG = "--use_last_checkpoint"
FS_ORDER = "filesystem order"


# --------------------------------------------------------------------------- #
# Layouts: each returns (extra argv, .ckpt files evaluate.py globs in -m)
# --------------------------------------------------------------------------- #

Layout = Callable[[Path], tuple[list[str], list[Path]]]


def last_checkpoint_two_native(model_dir: Path) -> tuple[list[str], list[Path]]:
    """--use_last_checkpoint: -m is the region dir, version_0 holds two native ckpts."""
    write_weights(model_dir, BEST_STATE)
    first = write_native_ckpt(model_dir, "epoch=79-step=104960.ckpt", OLD_STATE)
    second = write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
    return [LAST_CKPT_FLAG], [first, second]


def two_native_no_best_weights(model_dir: Path) -> tuple[list[str], list[Path]]:
    """Locally trained model without best_model_weights.pt: -m is the region dir."""
    first = write_native_ckpt(model_dir, "epoch=79-step=104960.ckpt", OLD_STATE)
    second = write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
    return [], [first, second]


def native_and_leftover_converted(model_dir: Path) -> tuple[list[str], list[Path]]:
    """TASK-144 layout: native epoch ckpt beside a leftover converted best_model.ckpt."""
    native = write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
    converted = write_converted_ckpt(model_dir, BEST_STATE)
    return [], [native, converted]


def best_weights_mirror_over_two_native(model_dir: Path) -> tuple[list[str], list[Path]]:
    """best_model_weights.pt beside two native ckpts: -m is a mirror with one converted ckpt."""
    write_weights(model_dir, BEST_STATE)
    write_native_ckpt(model_dir, "epoch=79-step=104960.ckpt", OLD_STATE)
    write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
    return [], []


def native_only(model_dir: Path) -> tuple[list[str], list[Path]]:
    native = write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
    return [], [native]


def best_weights_only(model_dir: Path) -> tuple[list[str], list[Path]]:
    write_weights(model_dir, BEST_STATE)
    return [], []


def last_checkpoint_one_native(model_dir: Path) -> tuple[list[str], list[Path]]:
    write_weights(model_dir, BEST_STATE)
    native = write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
    return [LAST_CKPT_FLAG], [native]


SEVERAL_CKPT_LAYOUTS = [
    pytest.param(last_checkpoint_two_native, id="last_checkpoint_two_native"),
    pytest.param(two_native_no_best_weights, id="two_native_no_best_weights"),
    pytest.param(native_and_leftover_converted, id="native_and_leftover_converted"),
]

ONE_CKPT_LAYOUTS = [
    pytest.param(best_weights_mirror_over_two_native, id="best_weights_mirror_over_two_native"),
    pytest.param(native_only, id="native_only"),
    pytest.param(best_weights_only, id="best_weights_only"),
    pytest.param(last_checkpoint_one_native, id="last_checkpoint_one_native"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def run_layout(tmp_path: Path, layout: Layout) -> tuple[FakeEvaluate, list[Path]]:
    """Build the layout under tmp_path/models, run _run_per_region with the fake evaluate."""
    model_dir = make_model_dir(tmp_path)
    extra_argv, candidates = layout(model_dir)
    script = GenerateEmbeddings()
    script.parse_args([str(tmp_path / "models"), str(tmp_path / "dataset"), *extra_argv])
    fake = FakeEvaluate()
    script.execute_command = fake
    code = script._run_per_region("evaluate.py", str(tmp_path / "crops"), str(tmp_path / "out"))
    assert code == 0
    assert len(fake.calls) == 1
    return fake, candidates


def names_region(line: str, region: str, paths: list[str]) -> bool:
    """True if region is a word of line outside every path (REQ-CKPTREFRESH-08 convention)."""
    for path in paths:
        line = line.replace(path, " ")
    return any(region in word for word in line.split() if "/" not in word)


def assert_several_ckpts_seen(fake: FakeEvaluate) -> None:
    names = fake.calls[0]["ckpt_names"]
    assert len(names) > 1, f"sanity check: evaluate.py -m should hold several .ckpt, got {names}"


# --------------------------------------------------------------------------- #
# REQ-CKPTLOG-01
# --------------------------------------------------------------------------- #


class TestSeveralCkptsNamedOnOneLine:
    """REQ-CKPTLOG-01: several .ckpt in -m version_0 -> one line names the region and each one."""

    @pytest.mark.parametrize("layout", SEVERAL_CKPT_LAYOUTS)
    def test_each_candidate_ckpt_named_on_one_line(self, tmp_path, capsys, layout):
        fake, candidates = run_layout(tmp_path, layout)
        assert_several_ckpts_seen(fake)
        paths = [str(p) for p in candidates]

        out = capsys.readouterr().out
        lines = [
            line
            for line in out.splitlines()
            if all(path in line for path in paths) and names_region(line, REGION, paths)
        ]

        assert len(lines) == 1, f"expected one line naming {REGION} and each of {paths}; output:\n{out}"


# --------------------------------------------------------------------------- #
# REQ-CKPTLOG-02
# --------------------------------------------------------------------------- #


class TestSeveralCkptsWarnFilesystemOrder:
    """REQ-CKPTLOG-02: several .ckpt in -m version_0 -> one WARNING line on filesystem order."""

    @pytest.mark.parametrize("layout", SEVERAL_CKPT_LAYOUTS)
    def test_filesystem_order_warning_printed(self, tmp_path, capsys, layout):
        fake, candidates = run_layout(tmp_path, layout)
        assert_several_ckpts_seen(fake)
        paths = [str(p) for p in candidates]

        out = capsys.readouterr().out
        lines = [
            line
            for line in out.splitlines()
            if "WARNING" in line and FS_ORDER in line and names_region(line, REGION, paths)
        ]

        assert len(lines) == 1, f"expected one WARNING line naming {REGION} and {FS_ORDER!r}; output:\n{out}"


# --------------------------------------------------------------------------- #
# REQ-CKPTLOG-03
# --------------------------------------------------------------------------- #


class TestSingleCkptNoFilesystemOrderWarning:
    """REQ-CKPTLOG-03: exactly one .ckpt in -m version_0 -> no line mentions filesystem order."""

    @pytest.mark.parametrize("layout", ONE_CKPT_LAYOUTS)
    def test_no_filesystem_order_warning(self, tmp_path, capsys, layout):
        fake, _ = run_layout(tmp_path, layout)
        names = fake.calls[0]["ckpt_names"]
        assert len(names) == 1, f"sanity check: evaluate.py -m should hold one .ckpt, got {names}"

        out = capsys.readouterr().out

        assert FS_ORDER not in out, out

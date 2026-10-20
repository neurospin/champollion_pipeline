"""TASK-210: the per-region weights log must be honest when evaluate.py's choice is not.

champollion_V1's evaluate.py (frozen, see TASK-144) loads
glob(<-m>/logs/lightning_logs/version_0/checkpoints/*.ckpt)[0] without sorting.
When the -m directory holds several .ckpt files there, the pipeline cannot know
which one evaluate.py loads, so its console output names every candidate and
warns that the choice depends on filesystem order.

Several-ckpt layouts hold no epoch-N checkpoint (epochs 78/79): a region holding
one is evaluated through a one-ckpt mirror (REQ-EPOCHSEL-BDRABCZUK-8D8F3519F164).

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
    FakeEvaluate,
    make_model_dir,
    write_native_ckpt,
    write_weights,
)

pytestmark = pytest.mark.unit  # noqa: V107

FS_ORDER = "filesystem order"


# --------------------------------------------------------------------------- #
# Layouts: each returns (extra argv, .ckpt files evaluate.py globs in -m)
# --------------------------------------------------------------------------- #

Layout = Callable[[Path], tuple[list[str], list[Path]]]


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


ONE_CKPT_LAYOUTS = [
    pytest.param(best_weights_mirror_over_two_native, id="best_weights_mirror_over_two_native"),
    pytest.param(native_only, id="native_only"),
    pytest.param(best_weights_only, id="best_weights_only"),
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


# --------------------------------------------------------------------------- #
# REQ-CKPTLOG-01
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# REQ-CKPTLOG-02
# --------------------------------------------------------------------------- #


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

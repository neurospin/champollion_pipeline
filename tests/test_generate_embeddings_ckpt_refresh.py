#!/usr/bin/env python3
"""
Tests for GenerateEmbeddings._ensure_ckpt refreshing a stale converted
best_model.ckpt (REQ-CKPTREFRESH-01..03, TASK-143).

All checkpoints are small real torch files built in tmp_path.
"""

import os
from pathlib import Path

import pytest
import torch

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

OLD_MTIME_NS = 1_000_000_000_000_000_000  # 2001-09-09, well in the past


def make_script():
    script = GenerateEmbeddings()
    script.parse_args(["/m", "/d"])
    return script


def ckpt_dir_of(model_dir: Path) -> Path:
    return model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"


def weights_path_of(model_dir: Path) -> Path:
    return model_dir / "logs" / "best_model_weights.pt"


def write_weights(model_dir: Path, state_dict: dict, wrapped: bool = False) -> Path:
    path = weights_path_of(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict} if wrapped else state_dict, str(path))
    return path


def write_converted_ckpt(model_dir: Path, state_dict: dict) -> Path:
    """Write a best_model.ckpt shaped like a previous _ensure_ckpt conversion."""
    ckpt_dir = ckpt_dir_of(model_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "best_model.ckpt"
    torch.save({"state_dict": state_dict, "epoch": 0, "global_step": 0}, str(path))
    return path


def write_native_ckpt(model_dir: Path) -> Path:
    """Write a Lightning-style checkpoint as produced by local training."""
    ckpt_dir = ckpt_dir_of(model_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "epoch=80-step=106272.ckpt"
    torch.save(
        {
            "state_dict": {"backbones.0.encoder.w": torch.full((3,), 7.0)},
            "epoch": 80,
            "global_step": 106272,
            "optimizer_states": [],
        },
        str(path),
    )
    return path


def set_mtime(path: Path, mtime_ns: int) -> None:
    os.utime(path, ns=(mtime_ns, mtime_ns))


def load_ckpt_state_dict(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"]


def assert_state_dicts_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def snapshot_dir(directory: Path) -> dict:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


OLD_STATE = {"backbones.0.encoder.w": torch.zeros(4)}
NEW_STATE = {"backbones.0.encoder.w": torch.ones(4)}


class TestStaleConvertedCkptRefreshed:
    """REQ-CKPTREFRESH-01: best_model.ckpt ends up matching best_model_weights.pt."""

    def test_stale_ckpt_refreshed_when_weights_newer(self, tmp_path):
        ckpt = write_converted_ckpt(tmp_path, OLD_STATE)
        set_mtime(ckpt, OLD_MTIME_NS)
        write_weights(tmp_path, NEW_STATE)

        make_script()._ensure_ckpt(str(tmp_path))

        assert_state_dicts_equal(load_ckpt_state_dict(ckpt), NEW_STATE)

    def test_stale_ckpt_refreshed_when_weights_mtime_older(self, tmp_path):
        weights = write_weights(tmp_path, NEW_STATE)
        set_mtime(weights, OLD_MTIME_NS)
        ckpt = write_converted_ckpt(tmp_path, OLD_STATE)

        make_script()._ensure_ckpt(str(tmp_path))

        assert_state_dicts_equal(load_ckpt_state_dict(ckpt), NEW_STATE)

    def test_stale_ckpt_refreshed_from_wrapped_weights(self, tmp_path):
        ckpt = write_converted_ckpt(tmp_path, OLD_STATE)
        set_mtime(ckpt, OLD_MTIME_NS)
        write_weights(tmp_path, NEW_STATE, wrapped=True)

        make_script()._ensure_ckpt(str(tmp_path))

        assert_state_dicts_equal(load_ckpt_state_dict(ckpt), NEW_STATE)


class TestUpToDateCkptNotRewritten:
    """REQ-CKPTREFRESH-02: a matching best_model.ckpt is not rewritten."""

    def test_converted_ckpt_not_rewritten_when_weights_touched(self, tmp_path):
        weights = write_weights(tmp_path, NEW_STATE)
        script = make_script()
        script._ensure_ckpt(str(tmp_path))
        ckpt = ckpt_dir_of(tmp_path) / "best_model.ckpt"
        set_mtime(ckpt, OLD_MTIME_NS)
        weights.touch()  # same content, newer mtime (e.g. a re-download)

        script._ensure_ckpt(str(tmp_path))

        assert ckpt.stat().st_mtime_ns == OLD_MTIME_NS

    def test_legacy_matching_ckpt_not_rewritten(self, tmp_path):
        ckpt = write_converted_ckpt(tmp_path, NEW_STATE)
        set_mtime(ckpt, OLD_MTIME_NS)
        write_weights(tmp_path, NEW_STATE)

        make_script()._ensure_ckpt(str(tmp_path))

        assert ckpt.stat().st_mtime_ns == OLD_MTIME_NS


class TestNativeCkptUntouched:
    """REQ-CKPTREFRESH-03: a checkpoints/ dir with a native .ckpt is untouched."""

    def test_native_ckpt_without_weights_untouched(self, tmp_path):
        write_native_ckpt(tmp_path)
        (tmp_path / "logs").mkdir(exist_ok=True)
        before = snapshot_dir(ckpt_dir_of(tmp_path))

        make_script()._ensure_ckpt(str(tmp_path))

        assert snapshot_dir(ckpt_dir_of(tmp_path)) == before

    def test_native_ckpt_with_weights_untouched(self, tmp_path):
        write_native_ckpt(tmp_path)
        write_weights(tmp_path, NEW_STATE)
        before = snapshot_dir(ckpt_dir_of(tmp_path))

        make_script()._ensure_ckpt(str(tmp_path))

        assert snapshot_dir(ckpt_dir_of(tmp_path)) == before

    def test_native_and_converted_ckpt_with_weights_untouched(self, tmp_path):
        write_native_ckpt(tmp_path)
        write_converted_ckpt(tmp_path, OLD_STATE)
        write_weights(tmp_path, NEW_STATE)
        before = snapshot_dir(ckpt_dir_of(tmp_path))

        make_script()._ensure_ckpt(str(tmp_path))

        assert snapshot_dir(ckpt_dir_of(tmp_path)) == before

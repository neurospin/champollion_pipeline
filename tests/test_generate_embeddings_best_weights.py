#!/usr/bin/env python3
"""
Embeddings use logs/best_model_weights.pt when a region model directory holds it,
even next to a native Lightning checkpoint (REQ-CKPTREFRESH-04..08, TASK-176).

evaluate.py (champollion_V1, frozen) loads
glob('<model>/logs/lightning_logs/version_0/checkpoints/*.ckpt')[0] and
'<model>/.hydra/config.yaml'. These tests drive GenerateEmbeddings._run_per_region
with execute_command replaced by a fake that, at call time, inspects the model
directory handed to evaluate.py through ``-m`` exactly the way evaluate.py would.
No real evaluate.py runs. Checkpoints are small real torch files in tmp_path.
"""

from pathlib import Path

import pytest
import torch

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

REGION = "Aaa_left"
BEST_STATE = {"backbones.0.encoder.w": torch.ones(4), "backbones.0.encoder.b": torch.full((2,), 3.0)}
NATIVE_STATE = {"backbones.0.encoder.w": torch.full((4,), 7.0), "backbones.0.encoder.b": torch.zeros(2)}
OLD_STATE = {"backbones.0.encoder.w": torch.zeros(4), "backbones.0.encoder.b": torch.zeros(2)}
HYDRA_CONFIG = "backbone_name: convnet\ndata:\n- input_size: (1, 20, 30, 40)\n"


# --------------------------------------------------------------------------- #
# Model-directory builders
# --------------------------------------------------------------------------- #


def ckpt_dir_of(model_dir: Path, version: int = 0) -> Path:
    return model_dir / "logs" / "lightning_logs" / f"version_{version}" / "checkpoints"


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "models" / REGION
    (model_dir / "logs").mkdir(parents=True)
    (model_dir / ".hydra").mkdir()
    (model_dir / ".hydra" / "config.yaml").write_text(HYDRA_CONFIG)
    return model_dir


def write_weights(model_dir: Path, state_dict: dict, wrapped: bool = False) -> Path:
    path = model_dir / "logs" / "best_model_weights.pt"
    torch.save({"state_dict": state_dict} if wrapped else state_dict, str(path))
    return path


def write_native_ckpt(model_dir: Path, name: str, state_dict: dict, version: int = 0) -> Path:
    """Write a Lightning-style checkpoint as produced by local training."""
    ckpt_dir = ckpt_dir_of(model_dir, version)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    torch.save(
        {"state_dict": state_dict, "epoch": 80, "global_step": 106272, "optimizer_states": []},
        str(path),
    )
    return path


def write_converted_ckpt(model_dir: Path, state_dict: dict) -> Path:
    """Write a best_model.ckpt shaped like a previous _ensure_ckpt conversion."""
    ckpt_dir = ckpt_dir_of(model_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "best_model.ckpt"
    torch.save({"state_dict": state_dict, "epoch": 0, "global_step": 0}, str(path))
    return path


def snapshot_tree(root: Path) -> dict:
    """Relative path -> bytes for every file under root."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def assert_state_dicts_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


# --------------------------------------------------------------------------- #
# Fake evaluate.py: records what evaluate.py would see at call time
# --------------------------------------------------------------------------- #


class FakeEvaluate:
    """Stand-in for execute_command that inspects the -m model directory.

    Records, per call, the -m path, every .ckpt evaluate.py's glob could
    return (with its state_dict) and the .hydra/config.yaml bytes, then writes
    the -s output the way evaluate.py would and returns 0.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, cmd, **_kwargs):
        model_arg = cmd[cmd.index("-m") + 1]
        ckpts = sorted(Path(p) for p in Path(model_arg).glob("logs/lightning_logs/version_0/checkpoints/*.ckpt"))
        config = Path(model_arg) / ".hydra" / "config.yaml"
        self.calls.append(
            {
                "model_arg": model_arg,
                "ckpt_names": [p.name for p in ckpts],
                "ckpt_state_dicts": [
                    torch.load(str(p), map_location="cpu", weights_only=False)["state_dict"] for p in ckpts
                ],
                "config_bytes": config.read_bytes() if config.is_file() else None,
            }
        )
        out = Path(cmd[cmd.index("-s") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ID,dim1\n")
        return 0


def run_region(tmp_path: Path) -> FakeEvaluate:
    """Run _run_per_region over tmp_path/models with the fake evaluate."""
    models = tmp_path / "models"
    script = GenerateEmbeddings()
    script.parse_args([str(models), str(tmp_path / "dataset")])
    fake = FakeEvaluate()
    script.execute_command = fake
    code = script._run_per_region("evaluate.py", str(tmp_path / "crops"), str(tmp_path / "out"))
    assert code == 0
    assert len(fake.calls) == 1
    return fake


def assert_evaluate_sees_only(call: dict, expected: dict) -> None:
    assert len(call["ckpt_names"]) == 1, (
        f"evaluate.py would glob {call['ckpt_names']} in {call['model_arg']}; "
        "expected exactly one .ckpt so glob(...)[0] is deterministic"
    )
    assert_state_dicts_equal(call["ckpt_state_dicts"][0], expected)


# --------------------------------------------------------------------------- #
# REQ-CKPTREFRESH-04
# --------------------------------------------------------------------------- #


class TestEvaluateUsesBestWeights:
    """REQ-CKPTREFRESH-04: evaluate.py sees exactly one .ckpt, equal to best_model_weights.pt."""

    def test_last_epoch_native_ckpt_beside_best_weights(self, tmp_path):
        """HF canonical_25 / Jean-Zay models_canonical_25 layout (epoch=80 last-epoch ckpt)."""
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], BEST_STATE)

    def test_stale_aborted_run_ckpt_in_version_0_and_full_run_in_version_1(self, tmp_path):
        """train_champollion --overwrite after an aborted run (TASK-149 healthcheck layout)."""
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=2-step=3936.ckpt", OLD_STATE, version=0)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE, version=1)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], BEST_STATE)

    def test_native_ckpt_beside_wrapped_best_weights(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE, wrapped=True)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], BEST_STATE)

    def test_native_and_stale_converted_ckpt_beside_best_weights(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
        write_converted_ckpt(model_dir, OLD_STATE)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], BEST_STATE)

    def test_stale_converted_ckpt_only_is_refreshed(self, tmp_path):
        """Stage-level view of REQ-CKPTREFRESH-01: refresh still reaches evaluate.py."""
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_converted_ckpt(model_dir, OLD_STATE)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], BEST_STATE)

    def test_best_weights_only(self, tmp_path):
        """Clean HF layout: no checkpoint yet."""
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)

        fake = run_region(tmp_path)

        assert_evaluate_sees_only(fake.calls[0], BEST_STATE)


# --------------------------------------------------------------------------- #
# REQ-CKPTREFRESH-05
# --------------------------------------------------------------------------- #


class TestEvaluateSeesRegionHydraConfig:
    """REQ-CKPTREFRESH-05: the -m directory carries the region's own .hydra/config.yaml."""

    def test_config_identical_with_native_ckpt(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        fake = run_region(tmp_path)

        assert fake.calls[0]["config_bytes"] == HYDRA_CONFIG.encode()

    def test_config_identical_with_best_weights_only(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)

        fake = run_region(tmp_path)

        assert fake.calls[0]["config_bytes"] == HYDRA_CONFIG.encode()


# --------------------------------------------------------------------------- #
# REQ-CKPTREFRESH-06
# --------------------------------------------------------------------------- #


class TestUserModelDirUnchanged:
    """REQ-CKPTREFRESH-06: no file added, removed or rewritten in the user's model dir."""

    def test_last_epoch_native_ckpt_dir_unchanged(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
        before = snapshot_tree(model_dir)

        run_region(tmp_path)

        assert snapshot_tree(model_dir) == before

    def test_stale_version_0_and_version_1_dir_unchanged(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=2-step=3936.ckpt", OLD_STATE, version=0)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE, version=1)
        before = snapshot_tree(model_dir)

        run_region(tmp_path)

        assert snapshot_tree(model_dir) == before

    def test_native_and_stale_converted_ckpt_dir_unchanged(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)
        write_converted_ckpt(model_dir, OLD_STATE)
        before = snapshot_tree(model_dir)

        run_region(tmp_path)

        assert snapshot_tree(model_dir) == before


# --------------------------------------------------------------------------- #
# REQ-CKPTREFRESH-07
# --------------------------------------------------------------------------- #


class TestNativeOnlyModelDirPassedAsIs:
    """REQ-CKPTREFRESH-07: without best_model_weights.pt, -m is the region model dir itself."""

    def test_native_ckpt_only(self, tmp_path):
        model_dir = make_model_dir(tmp_path)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        fake = run_region(tmp_path)

        assert fake.calls[0]["model_arg"] == str(model_dir)
        assert_evaluate_sees_only(fake.calls[0], NATIVE_STATE)


# --------------------------------------------------------------------------- #
# REQ-CKPTREFRESH-08
# --------------------------------------------------------------------------- #


def lines_naming(output: str, region: str, weights_path: str) -> list[str]:
    """Lines holding weights_path and, outside any path, the region name.

    The region name is usually a segment of the model paths themselves, so it
    is only counted in the words of the line that are not paths.
    """
    found = []
    for line in output.splitlines():
        if weights_path not in line:
            continue
        words = line.replace(weights_path, " ").split()
        if any(region in word for word in words if "/" not in word):
            found.append(line)
    return found


class TestWeightsFileLogged:
    """REQ-CKPTREFRESH-08: one console line per region names region + source weights file."""

    def test_best_weights_named_when_native_ckpt_present(self, tmp_path, capsys):
        model_dir = make_model_dir(tmp_path)
        weights = write_weights(model_dir, BEST_STATE)
        write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        run_region(tmp_path)

        assert len(lines_naming(capsys.readouterr().out, REGION, str(weights))) == 1

    def test_best_weights_named_when_only_best_weights(self, tmp_path, capsys):
        model_dir = make_model_dir(tmp_path)
        weights = write_weights(model_dir, BEST_STATE)

        run_region(tmp_path)

        assert len(lines_naming(capsys.readouterr().out, REGION, str(weights))) == 1

    def test_native_ckpt_named_when_no_best_weights(self, tmp_path, capsys):
        model_dir = make_model_dir(tmp_path)
        native = write_native_ckpt(model_dir, "epoch=80-step=106272.ckpt", NATIVE_STATE)

        run_region(tmp_path)

        assert len(lines_naming(capsys.readouterr().out, REGION, str(native))) == 1

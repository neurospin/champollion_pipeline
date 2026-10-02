#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydra config precedence of train_champollion.py (REQ-CFGPREC-01, REQ-CFGPREC-02).

Hydra 1.3 gives the primary ``config_path`` of champollion_V1's ``train.py``
(``<checkout>/champollion/configs``) precedence over ``hydra.searchpath``
entries, so a ``dataset/`` or ``dataset_localization/`` YAML left in the
checkout silently shadows the copy stage 3 generated under ``--config-dir``.

These tests are mechanism-agnostic: ``_CONTRASTIVE_DIR`` is redirected to a
tmp_path checkout whose ``train.py`` is a stub Hydra app with the same
decorator as champollion_V1 (``config_path="configs"``, ``version_base="1.1"``)
that dumps the composed config to JSON. ``execute_command`` is replaced by a
real subprocess run of the argv train_champollion builds (``python3`` swapped
for ``sys.executable``), so the assertion is on what Hydra actually loads, not
on which CLI flags carry the config dir. Nothing under external/ is touched.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from champollion_pipeline import train_champollion
from champollion_pipeline.train_champollion import TrainChampollion

pytest.importorskip("hydra")

DATASET = "CFGPREC_TEST"
REGION = "cingulate_left"
LOCALIZATION = "local"
DUMP_ENV = "CFGPREC_DUMP_PATH"

_STUB_TRAIN_PY = textwrap.dedent(
    """\
    import json
    import os

    import hydra
    from omegaconf import OmegaConf


    @hydra.main(config_name="config", version_base="1.1", config_path="configs")
    def main(cfg):
        with open(os.environ["{dump_env}"], "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=False), f)


    if __name__ == "__main__":
        main()
    """
).format(dump_env=DUMP_ENV)

# Mirrors the keys/groups champollion_V1's config.yaml must expose for the
# overrides train_champollion passes (platform=, mode=, load_sparse=).
_STUB_CONFIG_YAML = textwrap.dedent(
    """\
    defaults:
      - _self_
      - mode: encoder
      - platform: cuda
    load_sparse: false
    """
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_checkout(root):
    """Build a fake champollion_V1 checkout; return its champollion/ package dir."""
    pkg = root / "champollion_V1" / "champollion"
    configs = pkg / "configs"
    _write(pkg / "train.py", _STUB_TRAIN_PY)
    _write(configs / "config.yaml", _STUB_CONFIG_YAML)
    _write(configs / "mode" / "encoder.yaml", "mode_name: encoder\n")
    _write(configs / "platform" / "cuda.yaml", "platform_name: cuda\n")
    _write(configs / "platform" / "cpu.yaml", "platform_name: cpu\n")
    # champollion_V1 tracks dataset_localization/local.yaml in its checkout.
    _write(
        configs / "dataset_localization" / f"{LOCALIZATION}.yaml",
        "# @package _global_\ndataset_folder: /from/checkout\nlocalization_origin: checkout\n",
    )
    return pkg


def _run_training(tmp_path, monkeypatch, config_dir):
    """Run TrainChampollion end-to-end against the stub checkout; return Hydra's composed cfg."""
    pkg = _make_checkout(tmp_path)
    monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(pkg))
    dump = tmp_path / "composed_cfg.json"
    monkeypatch.setenv(DUMP_ENV, str(dump))

    script = TrainChampollion()
    script.parse_args(
        [
            "--dataset",
            DATASET,
            "--region",
            REGION,
            "--config-dir",
            str(config_dir),
            "--output_dir",
            str(tmp_path / "models"),
            "--localization",
            LOCALIZATION,
            "--cpu",
        ]
    )

    def real_execute(cmd, **_kwargs):
        assert cmd[0] == "python3", f"unexpected interpreter in {cmd}"
        proc = subprocess.run([sys.executable, *cmd[1:]], cwd=os.getcwd(), capture_output=True, text=True)
        assert proc.returncode == 0, f"stub train.py failed for argv {cmd}:\n{proc.stderr}"
        return proc.returncode

    monkeypatch.setattr(script, "execute_command", real_execute)
    assert script.run() == 0
    return json.loads(dump.read_text())


def _checkout_configs(tmp_path):
    return tmp_path / "champollion_V1" / "champollion" / "configs"


@pytest.mark.integration
class TestConfigDirTakesPrecedence:
    """REQ-CFGPREC-01: --config-dir copy wins over a same-path file in the checkout configs."""

    def test_dataset_region_yaml_loaded_from_config_dir(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "derivatives_configs"
        _write(config_dir / "dataset" / DATASET / f"{REGION}.yaml", "origin: config_dir\n")
        _write(_checkout_configs(tmp_path) / "dataset" / DATASET / f"{REGION}.yaml", "origin: checkout\n")

        cfg = _run_training(tmp_path, monkeypatch, config_dir)

        origin = cfg["dataset"][DATASET]["origin"]
        assert origin == "config_dir", (
            f"REQ-CFGPREC-01: Hydra loaded dataset/{DATASET}/{REGION}.yaml from the {origin!r} copy; "
            "the --config-dir copy must shadow the champollion_V1 checkout copy"
        )

    def test_localization_yaml_loaded_from_config_dir(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "derivatives_configs"
        _write(config_dir / "dataset" / DATASET / f"{REGION}.yaml", "origin: config_dir\n")
        _write(
            config_dir / "dataset_localization" / f"{LOCALIZATION}.yaml",
            "# @package _global_\ndataset_folder: /from/config_dir\nlocalization_origin: config_dir\n",
        )
        # The stub checkout already holds its own dataset_localization/local.yaml.

        cfg = _run_training(tmp_path, monkeypatch, config_dir)

        origin = cfg.get("localization_origin")
        assert origin == "config_dir", (
            f"REQ-CFGPREC-01: Hydra loaded dataset_localization/{LOCALIZATION}.yaml from the {origin!r} copy; "
            "the --config-dir copy must shadow the champollion_V1 checkout copy"
        )


@pytest.mark.integration
class TestCheckoutFallback:
    """REQ-CFGPREC-02: a region config absent from --config-dir still resolves from the checkout."""

    def test_dataset_region_yaml_falls_back_to_checkout(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "derivatives_configs"
        config_dir.mkdir()
        _write(_checkout_configs(tmp_path) / "dataset" / DATASET / f"{REGION}.yaml", "origin: checkout\n")

        cfg = _run_training(tmp_path, monkeypatch, config_dir)

        assert cfg["dataset"][DATASET]["origin"] == "checkout", (
            "REQ-CFGPREC-02: region config present only in the champollion_V1 checkout was not loaded"
        )

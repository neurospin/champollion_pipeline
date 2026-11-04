"""Shared helpers of the main.py stage tests.

main.py lives at the repo root, not in the installed package, so each test module loads it by
file path under its own private module name. The stage helpers replace the script class in its
own module by a MagicMock, run the stage and return the argv handed to ``parse_args``.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def load_main_module(module_name):
    """Load main.py under the private name ``module_name``."""
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option_values(argv, option):
    """Values given to ``option`` in argv, in either the '--opt=V' or the '--opt V' form."""
    values = []
    for i, token in enumerate(argv):
        if token == option:
            values.append(argv[i + 1] if i + 1 < len(argv) else None)
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def stage_paths_config(main_module):
    """PipelineConfig with every path the six stages read set to a distinct value."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.input_path = "/raw"  # noqa: V101
    dataset.morphologist_graphs = "/graphs"  # noqa: V101
    dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    dataset.crops_path = "/crops"  # noqa: V101
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.snapshots_path = "/snapshots"  # noqa: V101
    return config


def tiles_config(main_module):
    """PipelineConfig with the paths RunCorticalTilesStage reads set to distinct values."""
    config = main_module.PipelineConfig()
    config.dataset.morphologist_graphs = "/graphs"  # noqa: V101
    config.dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    return config


def masks_version_config(main_module, masks_version, hf_enabled):
    """PipelineConfig with the given masks_version and the requested model source."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.masks_version = masks_version  # noqa: V101
    dataset.hf_enabled = hf_enabled  # noqa: V101
    if hf_enabled:
        dataset.hf_repo_id = "neurospin/Champollion_V1"  # noqa: V101
    return config


def capture_stage_argv(stage_cls, script_path, stage_name, config, logger_name):
    """Run ``stage_cls.execute()`` with the class at ``script_path`` mocked; return the argv it built."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch(script_path, script_cls):
        stage = stage_cls(stage_name, config, logging.getLogger(logger_name))
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return list(script_cls.return_value.parse_args.call_args[0][0])


def tiles_stage_argv(main_module, config, logger_name):
    """Argv RunCorticalTilesStage builds, with RunCorticalTiles mocked."""
    return capture_stage_argv(
        main_module.RunCorticalTilesStage,
        "champollion_pipeline.run_cortical_tiles.RunCorticalTiles",
        "run_cortical_tiles",
        config,
        logger_name,
    )


def embeddings_stage_argv(main_module, config, stage_name, logger_name):
    """Argv GenerateEmbeddingsStage builds, with GenerateEmbeddings mocked."""
    return capture_stage_argv(
        main_module.GenerateEmbeddingsStage,
        "champollion_pipeline.generate_embeddings.GenerateEmbeddings",
        stage_name,
        config,
        logger_name,
    )

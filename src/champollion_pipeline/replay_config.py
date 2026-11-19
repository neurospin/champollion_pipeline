"""Replay config of a main.py run: the effective configuration written to <logs folder>/replay_<timestamp>.yaml.

main.py calls write_replay_config once per run (sequential, streaming, --dry-run), before it hands the PipelineConfig to
PipelineOrchestrator or run_parallel_pipeline, so that `main.py --config <replay file>` runs the same configuration
again without its command-line arguments. The file holds the command line, the inference and the --config YAML merged,
with path fields made absolute; root_path, dependencies, dataset.config_path and dataset.hf_token stay out of it (a run
that used --env needs --env again).

Inputs:
    PipelineConfig: the effective configuration of the run, from main.py (command line, inference, --config YAML).
    Working directory of main.py: the anchor of relative path fields.

Outputs:
    <log_dir>/replay_<YYYYmmdd-HHMMSS>.yaml, else <outputs_path>/logs/replay_<YYYYmmdd-HHMMSS>.yaml: the replay
    file, read back by main.py --config through ConfigLoader.load_from_yaml. When it cannot be written, one warning
    names its path and the error and the run goes on.

```mermaid
flowchart LR
    main[main.py] -.calls.-> this[replay_config.py]
    cfg[(effective PipelineConfig)] --> this
    this --> replay[(logs/replay_YYYYmmdd-HHMMSS.yaml)]
    replay --> rerun[main.py --config replay file]
```
"""

import copy
import logging
import os
from datetime import datetime
from pathlib import Path

from champollion_pipeline.config import ConfigLoader, PipelineConfig

logger = logging.getLogger(__name__)

# Path fields made absolute: PipelineConfig then DatasetConfig. path_to_graph and path_sk_with_hull stay relative
# (they are relative to each subject folder).
_PIPELINE_PATH_FIELDS = (
    "data_path",
    "models_path",
    "outputs_path",
    "champollion_v1_path",
    "log_dir",
    "log",
    "profile_dir",
)
_DATASET_PATH_FIELDS = (
    "datasets_root",
    "input_path",
    "morphologist_graphs",
    "cortical_tiles_output",
    "crops_path",
    "embeddings_path",
    "sk_qc_path",
    "models_cache",
    "splits_basedir",
    "snapshots_path",
    "reference_data_path",
    "champollion_data_root",
)
# Keys ConfigLoader.save_to_yaml would write but a replay leaves out (dataset.hf_token it never writes).
_EXEMPT_KEYS = ("root_path", "dependencies", "dataset.config_path")
_STAMP_FORMAT = "%Y%m%d-%H%M%S"


def logs_folder(config: PipelineConfig) -> Path:
    """The logs folder of the run: config.log_dir, else <outputs_path>/logs, made absolute against the working dir."""
    return Path(os.path.abspath(config.log_dir or os.path.join(config.outputs_path, "logs")))


def write_replay_config(config: PipelineConfig) -> str:  # noqa: V103 - used by main.py
    """Write the replay file of config to the run's logs folder and return its path; config is left unchanged.

    The name is replay_<local time of writing, seconds>.yaml: a second run of the same second and folder overwrites it.
    OSError (folder or file not writable) is not raised: one warning names the path and the error, the path tried is
    returned and the run goes on.
    """
    assert config is not None, "replay_config.write_replay_config pre: config is not None"
    effective = copy.deepcopy(config)
    _make_paths_absolute(effective)
    folder = logs_folder(effective)
    result = str(folder / f"replay_{datetime.now().strftime(_STAMP_FORMAT)}.yaml")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        # save_to_yaml drops dataset.hf_token and warns when one was set
        ConfigLoader.save_to_yaml(effective, result, omit=_EXEMPT_KEYS)
    except OSError as error:
        logger.warning("Replay config not written to %s: %s", result, error)
    else:
        logger.info("Replay config written to %s", result)
    assert len(result) > 0, "replay_config.write_replay_config post: len(result) > 0"
    return result


def _make_paths_absolute(config: PipelineConfig) -> None:
    """Make, in place, every relative non-empty path field absolute against the working directory."""
    for owner, names in ((config, _PIPELINE_PATH_FIELDS), (config.dataset, _DATASET_PATH_FIELDS)):
        for name in names:
            value = getattr(owner, name)
            if isinstance(value, str) and value:
                setattr(owner, name, os.path.abspath(value))

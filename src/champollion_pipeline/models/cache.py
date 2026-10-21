"""The models cache directory and the model_set.json marker of a model set."""

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from os.path import dirname, join

MODEL_SET_FILE_NAME = "model_set.json"


def ensure_models_cache_dir(cache_dir: str) -> str:
    """Create the models cache directory if needed and check it is writable.

    Args:
        cache_dir: absolute path of the models cache directory.

    Returns:
        cache_dir, unchanged.

    Raises:
        OSError: naming cache_dir, when it cannot be created or is not writable.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as err:
        raise OSError(f"Cannot create models cache directory {cache_dir}: {err}") from err
    if not os.access(cache_dir, os.W_OK):
        raise OSError(f"Models cache directory {cache_dir} is not writable")
    return cache_dir


@dataclass(frozen=True)
class ModelSet:
    """Content of a model set directory's model_set.json marker."""

    model_version: str
    masks: str
    checkpoint_epoch: int | None = None


def read_model_set(model_set_dir: str) -> ModelSet:
    """Read and validate <model_set_dir>/model_set.json.

    OSError (a missing marker included) propagates; any invalid content raises a ValueError whose message starts
    with the marker path. Unknown keys are ignored; a missing checkpoint_epoch gives None.
    """
    path = join(model_set_dir, MODEL_SET_FILE_NAME)
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except ValueError as err:
            raise ValueError(f"{path}: not valid JSON ({err})") from err
    if type(data) is not dict:
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    for key in ("model_version", "masks"):
        value = data.get(key)
        if not isinstance(value, str) or value == "":
            raise ValueError(f"{path}: {key!r} must be a non-empty string, got {value!r}")
    if "checkpoint_epoch" in data:
        epoch = data["checkpoint_epoch"]
        if not (type(epoch) is int and epoch >= 0):
            raise ValueError(f"{path}: 'checkpoint_epoch' must be a non-negative integer, got {epoch!r}")
    return ModelSet(data["model_version"], data["masks"], data.get("checkpoint_epoch"))


def write_json_atomic(path: str, data: dict) -> None:
    """Write data as indented JSON to path through a temporary file in the same directory, then rename it.
    The directory must exist; OSError propagates and leaves no temporary file behind."""
    stem = os.path.splitext(os.path.basename(path))[0]
    fd, tmp_path = tempfile.mkstemp(dir=dirname(path), prefix=f".{stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def write_model_set(model_set_dir: str, model_set: ModelSet) -> str:
    """Atomically write <model_set_dir>/model_set.json (model_version, masks, then checkpoint_epoch when set) and
    return its path. The directory must exist; OSError propagates and leaves no temporary file behind."""
    path = join(model_set_dir, MODEL_SET_FILE_NAME)
    content = {"model_version": model_set.model_version, "masks": model_set.masks}
    if model_set.checkpoint_epoch is not None:
        content["checkpoint_epoch"] = model_set.checkpoint_epoch
    write_json_atomic(path, content)
    return path

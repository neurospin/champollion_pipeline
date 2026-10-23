"""Weights of the region model directories of a model set: epoch-N checkpoint lookup, best-weights stand-in and the
temporary evaluation mirrors given to evaluate.py."""

import contextlib
import glob
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from os.path import abspath, exists, join
from pathlib import Path

import torch

_CONVERTED_CKPT_NAME = "best_model.ckpt"
_WEIGHTS_FILE_NAME = "best_model_weights.pt"
DEFAULT_CHECKPOINT_EPOCH = 80
_EPOCH_CKPT_PATTERN = r"epoch={epoch}-step=(?P<step>\d+)\.ckpt"
_VERSION_DIR_PATTERN = re.compile(r"version_(?P<index>\d+)")
_MIRROR_PREFIX = "champollion_eval_"


def _save_converted_ckpt(state_dict: dict, ckpt_path: Path) -> None:
    """Write state_dict as a converted Lightning ckpt (the only owner of its shape)."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "epoch": 0, "global_step": 0}, str(ckpt_path))


def _new_mirror(model_path: str, root: str) -> tuple[str, Path]:
    """Create <root>/<region dir name> with the region's .hydra (symlink) and an empty
    logs/lightning_logs/version_0/checkpoints/; return (mirror, ckpt_dir). Never writes under model_path.
    """
    mirror = Path(root) / os.path.basename(os.path.normpath(model_path))
    mirror.mkdir()
    hydra_dir = join(model_path, ".hydra")
    if exists(hydra_dir):
        os.symlink(abspath(hydra_dir), mirror / ".hydra")
    ckpt_dir = mirror / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    return str(mirror), ckpt_dir


@contextlib.contextmanager
def _temporary_mirror_root() -> Iterator[str]:
    """A temporary directory for an evaluation mirror, removed on exit (symlinks are not followed)."""
    root = tempfile.mkdtemp(prefix=_MIRROR_PREFIX)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _load_weights_state_dict(pt_path: Path) -> dict:
    """Load best_model_weights.pt, unwrapping a top-level state_dict if present."""
    loaded = torch.load(str(pt_path), map_location="cpu")
    return loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded


def _build_best_weights_mirror(model_path: str, pt_path: Path, root: str) -> str:
    """Mirror holding the region's .hydra and one ckpt converted from pt_path."""
    mirror, ckpt_dir = _new_mirror(model_path, root)
    _save_converted_ckpt(_load_weights_state_dict(pt_path), ckpt_dir / _CONVERTED_CKPT_NAME)
    return mirror


def _build_epoch_ckpt_mirror(model_path: str, ckpt: Path, root: str) -> str:
    """Mirror holding the region's .hydra and a symlink to the epoch-N checkpoint, under its own name."""
    mirror, ckpt_dir = _new_mirror(model_path, root)
    os.symlink(abspath(ckpt), ckpt_dir / ckpt.name)
    return mirror


def _compute_ckpt_rank(ckpt: Path, step: int) -> tuple[int, int, str]:
    """Sort key: step, then version_<n> index (-1 if unparsable), then path text."""
    match = _VERSION_DIR_PATTERN.fullmatch(ckpt.parent.parent.name)
    return step, int(match["index"]) if match else -1, str(ckpt)


def find_epoch_checkpoint(model_path: str, epoch: int) -> Path | None:
    """The epoch=<epoch>-step=<S>.ckpt under <model_path>/logs/lightning_logs/version_*/checkpoints/ with the
    highest S, or None when there is none."""
    pattern = re.compile(_EPOCH_CKPT_PATTERN.format(epoch=epoch))
    lightning_logs = Path(model_path) / "logs" / "lightning_logs"
    ranked = []
    for ckpt in lightning_logs.glob("version_*/checkpoints/*.ckpt"):
        match = pattern.fullmatch(ckpt.name)
        if match:
            ranked.append((_compute_ckpt_rank(ckpt, int(match["step"])), ckpt))
    return max(ranked)[1] if ranked else None


def has_region_weights(model_path: str, epoch: int) -> bool:
    """True when the region has an epoch-N checkpoint or logs/best_model_weights.pt."""
    return (
        find_epoch_checkpoint(model_path, epoch) is not None
        or (Path(model_path) / "logs" / _WEIGHTS_FILE_NAME).exists()
    )


def uses_best_weights_stand_in(model_path: str, epoch: int) -> bool:
    """True when best_model_weights.pt stands in for the missing epoch-N checkpoint."""
    if find_epoch_checkpoint(model_path, epoch) is not None:
        return False
    return (Path(model_path) / "logs" / _WEIGHTS_FILE_NAME).exists()


def check_region_weights(region_dirs: list[str], models_path: str, epoch: int) -> None:
    """Raise ValueError naming every region with neither an epoch-N checkpoint nor best weights."""
    missing = list(
        dict.fromkeys(resolve_region_name(d, models_path) for d in region_dirs if not has_region_weights(d, epoch))
    )
    if missing:
        raise ValueError(
            f"No weights for {len(missing)} region(s): {', '.join(missing)}. Each region model "
            f"directory needs logs/best_model_weights.pt or an epoch={epoch}-step=*.ckpt "
            "checkpoint under logs/lightning_logs/version_*/checkpoints/; no region was evaluated."
        )


@contextlib.contextmanager
def evaluation_model_dir(model_path: str, epoch: int) -> Iterator[str]:
    """Yield the directory to pass to evaluate.py as -m, in this order:

    1. an epoch-N checkpoint exists (any version_*): a temporary mirror holding .hydra and only
       that checkpoint, removed afterwards;
    2. no best_model_weights.pt: model_path itself, nothing written (the stage fails fast before this);
    3. otherwise: a temporary mirror holding .hydra and the converted best weights, which stand in
       for the missing epoch-N checkpoint.
    """
    epoch_ckpt = find_epoch_checkpoint(model_path, epoch)
    if epoch_ckpt is not None:
        with _temporary_mirror_root() as root:
            yield _build_epoch_ckpt_mirror(model_path, epoch_ckpt, root)
        return
    pt_path = Path(model_path) / "logs" / _WEIGHTS_FILE_NAME
    if not pt_path.exists():
        yield model_path
        return
    with _temporary_mirror_root() as root:
        yield _build_best_weights_mirror(model_path, pt_path, root)


def weights_source(model_path: str, epoch: int) -> str | None:
    """Return the user-dir weights file evaluate.py ends up using, or None."""
    epoch_ckpt = find_epoch_checkpoint(model_path, epoch)
    if epoch_ckpt is not None:
        return str(epoch_ckpt)
    pt_path = join(model_path, "logs", _WEIGHTS_FILE_NAME)
    if exists(pt_path):
        return pt_path
    ckpts = glob.glob(model_path + "/logs/lightning_logs/version_0/checkpoints/*.ckpt")
    return ckpts[0] if ckpts else None


def find_region_model_dirs(models_path: str) -> list[str]:
    """Return every directory, at any depth under models_path, that directly
    contains a logs/ subdirectory. A matched directory's own subtree is not
    searched further. models_path itself is never returned. Symlinks are not
    resolved when reporting a match, but realpath is used to guard against
    cycles. Sorted lexicographically.
    """
    if not os.path.isdir(models_path):
        raise ValueError(f"Models path not found: {models_path}")

    found: list[str] = []
    visited_real: set[str] = set()
    stack = [models_path]
    while stack:
        current = stack.pop()
        real_current = os.path.realpath(current)
        if real_current in visited_real:
            continue
        visited_real.add(real_current)
        try:
            children = sorted(os.listdir(current))
        except OSError:
            continue
        for child in children:
            if child.startswith("."):
                continue
            child_path = join(current, child)
            if not os.path.isdir(child_path):
                continue
            if os.path.isdir(join(child_path, "logs")):
                found.append(child_path)
            else:
                stack.append(child_path)
    return sorted(found)


def resolve_region_name(model_path: str, models_path: str) -> str:
    """Return the nearest path segment, at or above model_path and below
    models_path, whose name ends in '_left' or '_right'.

    A flat model layout (<models_path>/<Region>_<side>/logs/...) has
    model_path itself end in the suffix. A HuggingFace-cache layout with
    an extra fold-hash directory (<models_path>/<Region>_<side>/<hash>/logs/...)
    needs one step up. Falls back to the model directory's own basename
    if no ancestor matches (unchanged from prior behaviour).
    """
    models_path = os.path.normpath(models_path)
    segment = os.path.normpath(model_path)
    while segment and segment != models_path:
        name = os.path.basename(segment)
        if name.endswith("_left") or name.endswith("_right"):
            return name
        segment = os.path.dirname(segment)
    return os.path.basename(model_path)

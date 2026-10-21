#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate sulcal embeddings per region.
Calls champollion/evaluate.py for each region in models_path.
"""

import argparse
import contextlib
import cProfile
import glob
import gzip
import json
import os
import pstats
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from io import StringIO
from os.path import abspath, dirname, exists, join
from pathlib import Path
from urllib.parse import urlparse

import torch
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    compute_models_cache_dir,
    compute_region_embeddings_dir,
)
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

# Add champollion to path for CKA imports
_SCRIPT_DIR = dirname(abspath(__file__))
_CONVERTED_CKPT_NAME = "best_model.ckpt"
_WEIGHTS_FILE_NAME = "best_model_weights.pt"
DEFAULT_CHECKPOINT_EPOCH = 80
_EPOCH_CKPT_PATTERN = r"epoch={epoch}-step=(?P<step>\d+)\.ckpt"
_VERSION_DIR_PATTERN = re.compile(r"version_(?P<index>\d+)")
_MIRROR_PREFIX = "champollion_eval_"
MODEL_SET_FILE_NAME = "model_set.json"
PROVENANCE_FILE_NAME = "provenance.json"
_CHAMPOLLION_DIR = abspath(join(_SCRIPT_DIR, "..", "..", "external", "champollion_V1"))
if _CHAMPOLLION_DIR not in sys.path:
    sys.path.insert(0, _CHAMPOLLION_DIR)

from champollion.metrics.cka_coherence import test_models_coherence_from_directory  # noqa: E402


def _parse_non_negative_int(value: str) -> int:
    """argparse type: an integer >= 0 (Lightning numbers epochs from 0)."""
    try:
        number = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}") from err
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}")
    return number


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


def _write_json_atomic(path: str, data: dict) -> None:
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
    _write_json_atomic(path, content)
    return path


def read_pipeline_commit() -> str | None:
    """40-hex git commit of the champollion_pipeline source checkout, or None when git cannot report it."""
    repo_dir = dirname(abspath(__file__))
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    return commit


class ModelFetchStrategy(ABC):
    """Abstract base class for model fetching strategies."""

    @abstractmethod
    def can_handle(self, models_path: str) -> bool:
        """Check if this strategy can handle the given path."""
        pass

    @abstractmethod
    def fetch(self, models_path: str, extract_to: str, no_cache: bool = False) -> str:
        """Fetch models and return the local path."""
        pass


class LocalPathStrategy(ModelFetchStrategy):
    """Strategy for handling local file paths."""

    def can_handle(self, models_path: str) -> bool:
        """Check if path exists locally."""
        return exists(models_path)

    def fetch(self, models_path: str, extract_to: str, no_cache: bool = False) -> str:
        """Return the local path as-is, or extract if it's an archive."""
        # If it's a directory, use it directly
        if os.path.isdir(models_path):
            print(f"Models found at local path: {models_path}")
            return models_path

        # If it's an archive file, check cache or extract
        if any(models_path.endswith(ext) for ext in [".tar.xz", ".tar.gz", ".tgz", ".gz"]):
            # Compute cache path based on archive name
            # Extract to a subdirectory named after the archive
            archive_name = Path(models_path).stem
            if archive_name.endswith(".tar"):
                archive_name = Path(archive_name).stem
            cached_path = join(extract_to, archive_name)

            # Check if cache exists and should be reused
            if exists(cached_path) and os.path.isdir(cached_path) and not no_cache:
                # Verify cache has content (not just an empty directory)
                if any(os.scandir(cached_path)):
                    print(f"Using cached extraction: {cached_path}")
                    print("(Use --no-cache to force re-extraction)")
                    return cached_path

            # Extract archive
            print(f"Local archive found: {models_path}")
            print(f"Extracting to: {cached_path}")
            if no_cache and exists(cached_path):
                print(f"Removing existing cache: {cached_path}")
                shutil.rmtree(cached_path)
            extracted_path = self._extract_archive(models_path, cached_path)
            return extracted_path

        # Otherwise, assume it's a regular file/directory and return as-is
        print(f"Models found at local path: {models_path}")
        return models_path

    def _extract_archive(self, archive_path: str, extract_to: str) -> str:
        """Extract archive and return path to extracted content."""
        os.makedirs(extract_to, exist_ok=True)

        if archive_path.endswith((".tar.xz", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=extract_to)
                members = tar.getmembers()
                if members:
                    # Return the extract_to path where contents are
                    print(f"Extracted to: {extract_to}")
                    return extract_to
        elif archive_path.endswith(".gz"):
            output_path = join(extract_to, Path(archive_path).stem)
            with gzip.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to: {output_path}")
            return output_path

        return extract_to


class HuggingFaceStrategy(ModelFetchStrategy):
    """Strategy for handling Hugging Face repository IDs and URLs."""

    def __init__(self, subfolder: str = None):
        self.subfolder = subfolder

    def can_handle(self, models_path: str) -> bool:
        """Check if path looks like a Hugging Face repo ID or URL."""
        # Check if it's a huggingface.co URL
        parsed = urlparse(models_path)
        if parsed.scheme in ["http", "https"]:
            return "huggingface.co" in parsed.netloc
        # HF repo IDs are typically in format: username/repo-name
        # Check if it looks like a repo ID (no file extensions)
        return (
            "/" in models_path
            and not models_path.startswith("/")
            and not any(models_path.endswith(ext) for ext in [".tar.xz", ".tar.gz", ".tgz", ".gz", ".zip"])
        )

    def _extract_repo_id(self, models_path: str) -> str:
        """Extract repo ID from a Hugging Face URL or return as-is if already a repo ID."""
        parsed = urlparse(models_path)
        if parsed.scheme in ["http", "https"] and "huggingface.co" in parsed.netloc:
            # URL format: https://huggingface.co/username/repo-name
            # Path is /username/repo-name, strip leading slash
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 2:
                return "/".join(path_parts[:2])  # username/repo-name
            elif len(path_parts) == 1:
                return path_parts[0]  # Just repo-name for official repos
        return models_path  # Already a repo ID

    def fetch(self, models_path: str, extract_to: str, no_cache: bool = False) -> str:
        """Download from Hugging Face."""
        try:
            # Extract repo ID from URL if needed
            repo_id = self._extract_repo_id(models_path)
            print(f"Attempting to download from Hugging Face: {repo_id}")
            if self.subfolder:
                print(f"  model version subfolder: {self.subfolder}")
            from huggingface_hub import snapshot_download

            # Ensure system CA bundle is used when the env's certifi bundle is incomplete
            # (e.g. behind a corporate proxy with a custom root CA).
            if not os.environ.get("SSL_CERT_FILE") and not os.environ.get("REQUESTS_CA_BUNDLE"):
                _sys_ca = "/etc/ssl/certs/ca-certificates.crt"
                if os.path.exists(_sys_ca):
                    os.environ["SSL_CERT_FILE"] = _sys_ca

            # Cache per (repo, subfolder) to avoid collisions across model versions
            repo_name = repo_id.split("/")[-1]
            cache_name = f"{repo_name}_{self.subfolder}" if self.subfolder else repo_name
            local_path = join(extract_to, cache_name)

            # HuggingFace handles caching internally, but we can force redownload
            # snapshot_download does not accept subfolder; use allow_patterns instead
            allow_patterns = [f"{self.subfolder}/*"] if self.subfolder else None
            downloaded_path = snapshot_download(
                repo_id=repo_id, local_dir=local_path, allow_patterns=allow_patterns, force_download=no_cache
            )
            if self.subfolder:
                downloaded_path = join(downloaded_path, self.subfolder)

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Hugging Face downloads. Install it with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download from Hugging Face: {e}")
        if self.subfolder:
            self._ensure_model_set_marker(downloaded_path)
        print(f"Successfully downloaded from Hugging Face to: {downloaded_path}")
        return downloaded_path

    def _ensure_model_set_marker(self, model_set_dir: str) -> None:
        """Write model_set.json (model_version and masks = the subfolder) unless one is already there; a write
        failure only warns."""
        path = join(model_set_dir, MODEL_SET_FILE_NAME)
        if os.path.lexists(path):
            return
        try:
            write_model_set(model_set_dir, ModelSet(self.subfolder, self.subfolder))
        except OSError as err:
            warnings.warn(
                f"Could not write {path} ({err}); the model set is used without a model_set.json marker",
                UserWarning,
                stacklevel=2,
            )
            return
        print(f"  model set marker written: {path}")


class RemoteArchiveStrategy(ModelFetchStrategy):
    """Strategy for handling remote archive URLs."""

    def can_handle(self, models_path: str) -> bool:
        """Check if path is a remote URL to an archive."""
        parsed = urlparse(models_path)
        if parsed.scheme not in ["http", "https", "ftp"]:
            return False
        # Check if it ends with archive extensions
        return any(models_path.endswith(ext) for ext in [".tar.xz", ".tar.gz", ".tgz", ".gz", ".zip"])

    def fetch(self, models_path: str, extract_to: str, no_cache: bool = False) -> str:
        """Download and extract remote archive."""
        import urllib.request

        # Compute cache path based on archive name
        filename = Path(urlparse(models_path).path).name
        archive_name = Path(filename).stem
        if archive_name.endswith(".tar"):
            archive_name = Path(archive_name).stem
        cached_path = join(extract_to, archive_name)

        # Check if cache exists and should be reused
        if exists(cached_path) and os.path.isdir(cached_path) and not no_cache:
            print(f"Using cached extraction: {cached_path}")
            print("(Use --no-cache to force re-download)")
            return cached_path

        # Remove existing cache if --no-cache
        if no_cache and exists(cached_path):
            print(f"Removing existing cache: {cached_path}")
            shutil.rmtree(cached_path)

        print(f"Downloading archive from: {models_path}")

        # Download to temporary file
        local_archive = join(extract_to, filename)

        try:
            urllib.request.urlretrieve(models_path, local_archive)
            print(f"Downloaded to: {local_archive}")

            # Extract the archive
            extracted_path = self._extract_archive(local_archive, extract_to)

            # Clean up the archive file
            os.remove(local_archive)

            return extracted_path

        except Exception as e:
            raise RuntimeError(f"Failed to download or extract archive: {e}")

    def _extract_archive(self, archive_path: str, extract_to: str) -> str:
        """Extract archive and return path to extracted content."""
        print(f"Extracting {archive_path}...")

        if archive_path.endswith((".tar.xz", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=extract_to)
                members = tar.getmembers()
                if members:
                    top_dir = members[0].name.split("/")[0]
                    extracted_path = join(extract_to, top_dir)
                    print(f"Extracted to: {extracted_path}")
                    return extracted_path
        elif archive_path.endswith(".gz"):
            output_path = join(extract_to, Path(archive_path).stem)
            with gzip.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to: {output_path}")
            return output_path

        return extract_to


class InteractiveFallbackStrategy(ModelFetchStrategy):
    """Fallback strategy that asks user for local archive."""

    def can_handle(self, models_path: str) -> bool:
        """This strategy handles all cases as a fallback."""
        return True

    def fetch(self, models_path: str, extract_to: str, no_cache: bool = False) -> str:
        """Ask user for local archive path (only works in interactive mode)."""
        print(f"\nCannot automatically retrieve models from: {models_path}")

        # Check if we're in an interactive terminal
        if not sys.stdin.isatty():
            raise RuntimeError(
                f"Cannot find models at: {models_path}\n"
                f"When running non-interactively, please provide:\n"
                f"  - A valid local path to the models directory\n"
                f"  - A local archive file (.tar.xz, .tar.gz)\n"
                f"  - A HuggingFace repo ID (e.g., 'neurospin/Champollion_V1')"
            )

        print("\nPlease provide a local path to the models or archive.")
        response = input("Do you have a local copy? (yes/no): ").strip()

        if response.lower() not in ["yes", "y"]:
            print("Cannot proceed without models. Exiting gracefully.")
            sys.exit(0)

        archive_path = input("Path to models directory or archive (tar.xz, tar.gz, .gz): ").strip()

        if not exists(archive_path):
            raise FileNotFoundError(f"Path not found: {archive_path}")

        # If it's a directory, use it directly
        if os.path.isdir(archive_path):
            print(f"Using models from directory: {archive_path}")
            return archive_path

        # Check for cached extraction
        archive_name = Path(archive_path).stem
        if archive_name.endswith(".tar"):
            archive_name = Path(archive_name).stem
        cached_path = join(extract_to, archive_name)

        if exists(cached_path) and os.path.isdir(cached_path) and not no_cache:
            print(f"Using cached extraction: {cached_path}")
            print("(Use --no-cache to force re-extraction)")
            return cached_path

        # Remove existing cache if --no-cache
        if no_cache and exists(cached_path):
            print(f"Removing existing cache: {cached_path}")
            shutil.rmtree(cached_path)

        # Otherwise, try to extract it
        print(f"Extracting archive: {archive_path}")
        extracted_path = self._extract_archive(archive_path, extract_to)
        return extracted_path

    def _extract_archive(self, archive_path: str, extract_to: str) -> str:
        """Extract archive and return path to extracted content."""
        os.makedirs(extract_to, exist_ok=True)

        if archive_path.endswith((".tar.xz", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=extract_to)
                members = tar.getmembers()
                if members:
                    top_dir = members[0].name.split("/")[0]
                    extracted_path = join(extract_to, top_dir)
                    print(f"Extracted to: {extracted_path}")
                    return extracted_path
        elif archive_path.endswith(".gz"):
            output_path = join(extract_to, Path(archive_path).stem)
            with gzip.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to: {output_path}")
            return output_path

        return extract_to


class GenerateEmbeddings(ScriptBuilder):
    """Script for generating sulcal embeddings per region via champollion/evaluate.py."""

    def __init__(self):
        super().__init__(
            script_name="generate_embeddings",
            description="Generate sulcal embeddings per region using champollion/evaluate.py.",
        )
        (
            self.add_argument(
                "models_path", type=str, help="Path to the directory containing per-region model folders."
            )
            .add_argument(
                "datasets_root",
                type=str,
                help="Absolute path to the dataset root directory (e.g. /my/path/to/DATASET/).",
            )
            .add_flag("--overwrite", "Recompute embeddings that already exist on disk.")
            .add_flag("--cpu", "Force CPU usage (disable CUDA).")
            .add_flag("--profiling", "Enable Python profiling (cProfile).")
            .add_flag("--run-cka", "Run CKA coherence test after embeddings.")
            .add_flag("--no-cache", "Force re-extraction of archive (ignore cache).")
            .add_optional_argument(
                "--checkpoint-epoch",
                "Training epoch N (0-based) of the Lightning checkpoint epoch=N-step=*.ckpt to evaluate each region "
                "with. Default: 80.",
                default=DEFAULT_CHECKPOINT_EPOCH,
                type_=_parse_non_negative_int,
            )
            .add_optional_argument(
                "--cortical_version",
                f"Derivatives folder name (e.g. 'cortical_tiles-2027'). "
                f"Defaults to cortical_tiles-{CORTICAL_TILES_VERSION}.",
                default=f"cortical_tiles-{CORTICAL_TILES_VERSION}",
            )
            .add_flag(
                "--legacy",
                "Use deep_folding-2025 as the derivatives folder. Shorthand for --cortical_version deep_folding-2025.",
            )
            .add_optional_argument(
                "--model-version",
                "Model-set subfolder to download from the Hugging Face models repository (e.g. 'canonical_25'). "
                "Ignored when models_path is a local directory.",
                default=None,
            )
            .add_optional_argument(
                "--masks-version",
                "Deprecated alias of --model-version; same value and behaviour.",
                default=None,
            )
            .add_optional_argument(
                "--masks",
                "Cortical tiles mask version used as crops subdirectory (e.g. 'canonical_25'). "
                "Defaults to the --model-version value when given, else canonical_25.",
                default=None,
            )
            .add_flag(
                "--allow-version-mismatch",
                "Run even when --masks and --model-version differ (crops of one mask version evaluated with the "
                "model set of another); a warning is emitted.",
            )
            .add_argument(
                "--regions",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "Restrict embedding generation to specific region names "
                    "(e.g. SCsylv_left FIPFIPPoCinf_right). "
                    "By default all regions found in models_path are processed."
                ),
            )
            .add_optional_argument(
                "--output",
                "Output base directory override. "
                "Defaults to {datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/.",
                default=None,
            )
            .add_optional_argument(
                "--subjects",
                "Deprecated and ignored. Each region's subjects file is read from "
                "{crops}/<crop>/mask/{L|R}skeleton_subject.csv next to its skeleton.",
                default=None,
            )
            .add_optional_argument(
                "--models-cache",
                "Models cache directory; relative paths resolve against the cwd. "
                "Defaults to <datasets_root>/derivatives/champollion_V1/models_cache.",
                default=None,
            )
        )
        self._models_source: dict | None = None

    def _make_regions_tmpdir(self, models_path: str) -> str:
        """Return a temp dir containing symlinks to only the requested region subdirs."""
        missing = [r for r in self.args.regions if not os.path.isdir(join(models_path, r))]
        if missing:
            raise FileNotFoundError(f"Regions not found in {models_path}: {', '.join(missing)}")
        tmpdir = tempfile.mkdtemp(prefix="champollion_regions_")
        for region in self.args.regions:
            os.symlink(join(models_path, region), join(tmpdir, region))
        return tmpdir

    def _get_derivatives_folder(self) -> str:
        """Return the derivatives folder to use, resolving --legacy and --cortical_version."""
        if getattr(self.args, "legacy", False):
            return "deep_folding-2025"
        return self.args.cortical_version

    def _get_crops_2mm_dir(self, datasets_root: str) -> str:
        """Return the 2mm crops directory, resolving --legacy and --masks."""
        derivatives = join(datasets_root, "derivatives", self._get_derivatives_folder())
        if getattr(self.args, "legacy", False):
            return join(derivatives, "crops", "2mm")
        return join(derivatives, "crops", self.args.masks, "2mm")

    def _patch_config_paths(self, config_path: str, target_folder: str) -> None:
        """Rewrite the derivatives folder in every YAML under config_path in-place."""
        import re

        pattern = re.compile(r"(derivatives/)([^/\n]+)(/crops(?:/[^/\n]+)*/2mm)")
        patched = 0
        for yaml_file in Path(config_path).rglob("*.yaml"):
            text = yaml_file.read_text()
            new_text = pattern.sub(lambda m: f"{m.group(1)}{target_folder}{m.group(3)}", text)
            if new_text != text:
                yaml_file.write_text(new_text)
                patched += 1
        if patched:
            print(f"Patched {patched} YAML file(s): derivatives folder → {target_folder}")

    def _compute_extract_dir(self) -> str:
        """Return the absolute models cache dir: --models-cache if given, else the dataset-tree default."""
        override = getattr(self.args, "models_cache", None)
        if override is not None:
            return abspath(override)
        return compute_models_cache_dir(self.args.datasets_root)

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """Parse args, then resolve the deprecated --masks-version alias into model_version."""
        parsed = super().parse_args(args)
        self._resolve_model_version_alias()
        self._resolve_masks_default()
        self._check_version_mismatch()
        self._check_local_model_set()
        return parsed

    def _resolve_model_version_alias(self) -> None:
        """Copy --masks-version into model_version with a FutureWarning; exit 2 if the two flags differ."""
        alias = self.args.masks_version
        if alias is None:
            return
        model_version = self.args.model_version
        if model_version is not None and model_version != alias:
            self.parser.error(
                f"--model-version {model_version!r} and --masks-version {alias!r} differ; --masks-version is a "
                "deprecated alias of --model-version, pass only --model-version"
            )
        warnings.warn(
            "--masks-version is deprecated; use --model-version instead "
            "(same value, the Hugging Face model-set subfolder).",
            FutureWarning,
            stacklevel=2,
        )
        self.args.model_version = alias

    def _resolve_masks_default(self) -> None:
        """Fill --masks when it was not given: the model version if set, else DEFAULT_MASKS_VERSION."""
        if self.args.masks is None:
            model_version = self.args.model_version
            self.args.masks = model_version if model_version is not None else DEFAULT_MASKS_VERSION

    def _check_version_mismatch(self) -> None:
        """Exit 2 when --masks and --model-version differ, unless --allow-version-mismatch (then warn)."""
        model_version, masks = self.args.model_version, self.args.masks
        if model_version is None or masks == model_version:
            return
        if not self.args.allow_version_mismatch:
            self.parser.error(
                f"--masks {masks!r} and --model-version {model_version!r} differ: --masks picks the input crops, "
                "--model-version the model set; pass --allow-version-mismatch to run with both"
            )
        warnings.warn(
            f"--masks {masks!r} and --model-version {model_version!r} differ; running anyway "
            f"(--allow-version-mismatch): crops {masks!r}, models {model_version!r}",
            UserWarning,
            stacklevel=2,
        )

    def _check_local_model_set(self) -> None:
        """Exit 2 when a local model set's model_set.json is invalid, or names other masks than --masks unless
        --allow-version-mismatch (then warn). A local directory without a marker: warn that it cannot be checked.
        Not a local directory: nothing."""
        models_dir = self.args.models_path
        if not os.path.isdir(models_dir):
            return
        if not os.path.isfile(join(models_dir, MODEL_SET_FILE_NAME)):
            masks = self.args.masks
            warnings.warn(
                f"{join(models_dir, MODEL_SET_FILE_NAME)} not found: the local model set cannot be checked against "
                f"--masks {masks!r}; running anyway with --masks {masks!r}",
                UserWarning,
                stacklevel=2,
            )
            return
        try:
            model_set = read_model_set(models_dir)
        except ValueError as err:
            self.parser.error(str(err))
        path, marker_masks, masks = join(models_dir, MODEL_SET_FILE_NAME), model_set.masks, self.args.masks
        if marker_masks == masks:
            return
        if not self.args.allow_version_mismatch:
            self.parser.error(
                f"{path} masks {marker_masks!r} and --masks {masks!r} differ: the local model set was trained on "
                f"{marker_masks!r} crops; pass --masks {marker_masks!r}, or --allow-version-mismatch to run with both"
            )
        warnings.warn(
            f"{path} masks {marker_masks!r} and --masks {masks!r} differ; running anyway (--allow-version-mismatch): "
            f"crops {masks!r}, models trained on {marker_masks!r}",
            UserWarning,
            stacklevel=2,
        )

    def fetch_models(self, models_path):
        """
        Fetch models using strategy pattern.

        Tries strategies in order:
        1. LocalPathStrategy - Use if path exists locally
        2. HuggingFaceStrategy - Try if it looks like HF repo ID
        3. RemoteArchiveStrategy - Try if it's a URL to an archive
        4. InteractiveFallbackStrategy - Ask user for local copy

        Args:
            models_path: Path, URL, or HF repo ID for the models

        Returns:
            Local path to the models
        """
        # Downloaded/extracted models are cached under --models-cache, or inside
        # the dataset tree: <datasets_root>/derivatives/champollion_V1/models_cache
        extract_to = ensure_models_cache_dir(self._compute_extract_dir())

        # Get no_cache flag (force re-extraction)
        no_cache = getattr(self.args, "no_cache", False)
        self._models_source = None

        # Check HuggingFace and URL strategies first with the ORIGINAL path
        # These strategies look for semantic patterns (e.g., "user/repo", URLs)
        # that would be destroyed by path resolution
        model_version = getattr(self.args, "model_version", None)
        hf_strategy = HuggingFaceStrategy(subfolder=model_version)
        if hf_strategy.can_handle(models_path):
            try:
                fetched = hf_strategy.fetch(models_path, extract_to, no_cache)
                self._models_source = {
                    "type": "huggingface",
                    "repo_id": hf_strategy._extract_repo_id(models_path),
                    "subfolder": model_version,
                }
                return fetched
            except Exception as e:
                print(f"HuggingFace strategy failed: {e}")
                # Continue to other strategies

        remote_strategy = RemoteArchiveStrategy()
        if remote_strategy.can_handle(models_path):
            try:
                fetched = remote_strategy.fetch(models_path, extract_to, no_cache)
                self._models_source = {"type": "url", "url": models_path}
                return fetched
            except Exception as e:
                print(f"Remote archive strategy failed: {e}")
                # Continue to other strategies

        # Resolve relative paths to absolute for local file strategies
        # This ensures relative paths like ../../models.tar.xz work
        resolved_path = models_path
        if not urlparse(models_path).scheme:  # Not a URL
            if not models_path.startswith("/"):  # Relative path
                resolved_path = abspath(models_path)

        # Try local path strategy with resolved path
        local_strategy = LocalPathStrategy()
        if local_strategy.can_handle(resolved_path):
            try:
                fetched = local_strategy.fetch(resolved_path, extract_to, no_cache)
                self._models_source = {"type": "local", "path": os.path.abspath(resolved_path)}
                return fetched
            except Exception as e:
                print(f"Local path strategy failed: {e}")
                # Continue to fallback

        # Fallback: ask user interactively
        fallback_strategy = InteractiveFallbackStrategy()
        fetched = fallback_strategy.fetch(resolved_path, extract_to, no_cache)
        self._models_source = {"type": "local", "path": os.path.abspath(fetched)}
        return fetched

    def _read_marker_model_version(self, fetched_models_path: str) -> str | None:
        """The model_version of the fetched model set's model_set.json, or None when absent or unreadable."""
        path = join(fetched_models_path, MODEL_SET_FILE_NAME)
        if not os.path.isfile(path):
            return None
        try:
            return read_model_set(fetched_models_path).model_version
        except (ValueError, OSError) as err:
            warnings.warn(
                f"Could not read {path} for provenance.json ({err}); model_version falls back to --model-version",
                UserWarning,
                stacklevel=2,
            )
            return None

    def _build_provenance(self, fetched_models_path: str) -> dict:
        """masks, model_version, checkpoint_epoch, models_source and pipeline_commit of this run, in that order."""
        model_version = self._read_marker_model_version(fetched_models_path)
        if model_version is None:
            model_version = getattr(self.args, "model_version", None)
        return {
            "masks": self.args.masks,
            "model_version": model_version,
            "checkpoint_epoch": self.args.checkpoint_epoch,
            "models_source": self._models_source,
            "pipeline_commit": read_pipeline_commit(),
        }

    def _write_provenance(self, output_base: str, provenance: dict) -> str:
        """Atomically write <output_base>/provenance.json and return its path; OSError fails the stage."""
        os.makedirs(output_base, exist_ok=True)
        path = join(output_base, PROVENANCE_FILE_NAME)
        _write_json_atomic(path, provenance)
        print(f"Provenance written: {path}")
        return path

    def _validate_inputs(self):
        pass

    def run(self):
        """Execute the embeddings pipeline script."""
        self._validate_inputs()
        if self.args.profiling:
            return self._run_with_profiling()
        return self._run_normal()

    def _run_with_profiling(self):
        """Run pipeline with cProfile enabled."""
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = self._run_normal()
        finally:
            profiler.disable()
            # Save profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            stats.dump_stats("embeddings_profile.prof")
            print("\nProfiling results saved to embeddings_profile.prof")
            print("View with: python -m pstats embeddings_profile.prof")
            # Print top 20 functions
            stream = StringIO()
            stats_print = pstats.Stats(profiler, stream=stream)
            stats_print.sort_stats("cumulative")
            stats_print.print_stats(20)
            print(stream.getvalue())

        return result

    def _run_normal(self):
        """Execute the embeddings pipeline script (normal mode)."""
        local_dir = os.getcwd()

        # Save original CUDA_VISIBLE_DEVICES value for restoration later
        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Force CPU usage if requested
        if self.args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("CPU mode enabled: CUDA_VISIBLE_DEVICES set to empty string")

        try:
            return self._run_pipeline(local_dir)
        finally:
            # Restore original CUDA_VISIBLE_DEVICES value
            if original_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

    def _run_pipeline(self, local_dir):
        """Run embeddings generation using champollion/evaluate.py per region."""
        original_models_path = self.args.models_path
        self.args.models_path = self.fetch_models(original_models_path)
        provenance = self._build_provenance(self.args.models_path)

        tmpdir = None
        if self.args.regions:
            tmpdir = self._make_regions_tmpdir(self.args.models_path)
            self.args.models_path = tmpdir

        script_dir = dirname(abspath(__file__))
        evaluate_script = abspath(
            join(script_dir, "..", "..", "external", "champollion_V1", "champollion", "evaluate.py")
        )

        datasets_root = self.args.datasets_root.rstrip("/")
        output_base = self.args.output or compute_region_embeddings_dir(datasets_root, self.args.masks)

        crops_2mm_dir = self._get_crops_2mm_dir(datasets_root)

        if self.args.subjects:
            warnings.warn(
                "--subjects is deprecated and ignored: each region uses the "
                "{side}skeleton_subject.csv next to its {side}skeleton.npy.",
                FutureWarning,
                stacklevel=2,
            )

        try:
            result = self._run_per_region(evaluate_script, crops_2mm_dir, output_base)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir)
            self.args.models_path = original_models_path

        self._write_provenance(output_base, provenance)
        if self.args.run_cka:
            self._run_cka_test(output_base)

        return result

    def _find_crop_dir(self, crops_2mm_dir: str, region_model_name: str) -> str:
        """Map a model region name back to its crop directory name.

        Model dirs have dots stripped: crop 'SC-sylv.' → model 'SCsylv_left'.
        Search for a crop dir where dirname.replace('.', '') matches the base name.
        """
        base = region_model_name
        if base.endswith("_left"):
            base = base[:-5]
        elif base.endswith("_right"):
            base = base[:-6]

        if exists(crops_2mm_dir):
            for dname in os.listdir(crops_2mm_dir):
                if os.path.isdir(join(crops_2mm_dir, dname)):
                    if dname.replace(".", "") == base:
                        return dname
        return base

    @staticmethod
    def _save_converted_ckpt(state_dict: dict, ckpt_path: Path) -> None:
        """Write state_dict as a converted Lightning ckpt (the only owner of its shape)."""
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state_dict, "epoch": 0, "global_step": 0}, str(ckpt_path))

    @staticmethod
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

    @staticmethod
    @contextlib.contextmanager
    def _temporary_mirror_root() -> Iterator[str]:
        """A temporary directory for an evaluation mirror, removed on exit (symlinks are not followed)."""
        root = tempfile.mkdtemp(prefix=_MIRROR_PREFIX)
        try:
            yield root
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _build_best_weights_mirror(self, model_path: str, pt_path: Path, root: str) -> str:
        """Mirror holding the region's .hydra and one ckpt converted from pt_path."""
        mirror, ckpt_dir = self._new_mirror(model_path, root)
        self._save_converted_ckpt(self._load_weights_state_dict(pt_path), ckpt_dir / _CONVERTED_CKPT_NAME)
        return mirror

    @staticmethod
    def _build_epoch_ckpt_mirror(model_path: str, ckpt: Path, root: str) -> str:
        """Mirror holding the region's .hydra and a symlink to the epoch-N checkpoint, under its own name."""
        mirror, ckpt_dir = GenerateEmbeddings._new_mirror(model_path, root)
        os.symlink(abspath(ckpt), ckpt_dir / ckpt.name)
        return mirror

    @staticmethod
    def _compute_ckpt_rank(ckpt: Path, step: int) -> tuple[int, int, str]:
        """Sort key: step, then version_<n> index (-1 if unparsable), then path text."""
        match = _VERSION_DIR_PATTERN.fullmatch(ckpt.parent.parent.name)
        return step, int(match["index"]) if match else -1, str(ckpt)

    def _find_epoch_ckpt(self, model_path: str) -> Path | None:
        """The epoch-N checkpoint (N = --checkpoint-epoch) of any version_*/checkpoints/, highest step first."""
        pattern = re.compile(_EPOCH_CKPT_PATTERN.format(epoch=self.args.checkpoint_epoch))
        lightning_logs = Path(model_path) / "logs" / "lightning_logs"
        ranked = []
        for ckpt in lightning_logs.glob("version_*/checkpoints/*.ckpt"):
            match = pattern.fullmatch(ckpt.name)
            if match:
                ranked.append((self._compute_ckpt_rank(ckpt, int(match["step"])), ckpt))
        return max(ranked)[1] if ranked else None

    def _has_region_weights(self, model_path: str) -> bool:
        """True when the region has an epoch-N checkpoint or logs/best_model_weights.pt."""
        return (
            self._find_epoch_ckpt(model_path) is not None or (Path(model_path) / "logs" / _WEIGHTS_FILE_NAME).exists()
        )

    def _uses_best_weights_stand_in(self, model_path: str) -> bool:
        """True when best_model_weights.pt stands in for the missing epoch-N checkpoint."""
        if self._find_epoch_ckpt(model_path) is not None:
            return False
        return (Path(model_path) / "logs" / _WEIGHTS_FILE_NAME).exists()

    def _check_region_weights(self, region_dirs: list[str], models_path: str) -> None:
        """Raise ValueError naming every region with neither an epoch-N checkpoint nor best weights."""
        missing = list(
            dict.fromkeys(
                self._resolve_region_name(d, models_path) for d in region_dirs if not self._has_region_weights(d)
            )
        )
        if missing:
            raise ValueError(
                f"No weights for {len(missing)} region(s): {', '.join(missing)}. Each region model "
                f"directory needs logs/best_model_weights.pt or an epoch={self.args.checkpoint_epoch}-step=*.ckpt "
                "checkpoint under logs/lightning_logs/version_*/checkpoints/; no region was evaluated."
            )

    @contextlib.contextmanager
    def _evaluation_model_dir(self, model_path: str) -> Iterator[str]:
        """Yield the directory to pass to evaluate.py as -m, in this order:

        1. an epoch-N checkpoint exists (any version_*): a temporary mirror holding .hydra and only
           that checkpoint, removed afterwards;
        2. no best_model_weights.pt: model_path itself, nothing written (the stage fails fast before this);
        3. otherwise: a temporary mirror holding .hydra and the converted best weights, which stand in
           for the missing epoch-N checkpoint.
        """
        epoch_ckpt = self._find_epoch_ckpt(model_path)
        if epoch_ckpt is not None:
            with self._temporary_mirror_root() as root:
                yield self._build_epoch_ckpt_mirror(model_path, epoch_ckpt, root)
            return
        pt_path = Path(model_path) / "logs" / _WEIGHTS_FILE_NAME
        if not pt_path.exists():
            yield model_path
            return
        with self._temporary_mirror_root() as root:
            yield self._build_best_weights_mirror(model_path, pt_path, root)

    def _weights_source(self, model_path: str) -> str | None:
        """Return the user-dir weights file evaluate.py ends up using, or None."""
        epoch_ckpt = self._find_epoch_ckpt(model_path)
        if epoch_ckpt is not None:
            return str(epoch_ckpt)
        pt_path = join(model_path, "logs", _WEIGHTS_FILE_NAME)
        if exists(pt_path):
            return pt_path
        ckpts = glob.glob(model_path + "/logs/lightning_logs/version_0/checkpoints/*.ckpt")
        return ckpts[0] if ckpts else None

    @staticmethod
    def _load_weights_state_dict(pt_path: Path) -> dict:
        """Load best_model_weights.pt, unwrapping a top-level state_dict if present."""
        loaded = torch.load(str(pt_path), map_location="cpu")
        return loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded

    def _find_region_model_dirs(self, models_path: str) -> list[str]:
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

    def _resolve_region_name(self, model_path: str, models_path: str) -> str:
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

    def _run_per_region(self, evaluate_script: str, crops_2mm_dir: str, output_base: str) -> int:
        """Invoke champollion/evaluate.py for each region in models_path.

        Every region is attempted even after a failure. Returns the first non-zero
        evaluate code, or 0 when all evaluated regions succeed.
        """
        models_path = self.args.models_path
        region_dirs = self._find_region_model_dirs(models_path)

        if not region_dirs:
            raise ValueError(f"No region model directories (containing logs/) found under {models_path}")
        self._check_region_weights(region_dirs, models_path)

        print(f"\nGenerating embeddings for {len(region_dirs)} regions → {output_base}")

        failed_regions: list[str] = []
        first_failure = 0
        for model_path in region_dirs:
            region = self._resolve_region_name(model_path, models_path)
            side = "L" if region.endswith("_left") else "R"
            crop_name = self._find_crop_dir(crops_2mm_dir, region)
            mask_dir = join(crops_2mm_dir, crop_name, "mask")
            skels_path = join(mask_dir, f"{side}skeleton.npy")
            subjects_path = join(mask_dir, f"{side}skeleton_subject.csv")
            saving_path = join(output_base, region, "full_embeddings.csv")

            if exists(saving_path) and not getattr(self.args, "overwrite", False):
                print(f"  [SKIP] {region} — already exists (--overwrite to recompute)")
                continue

            with self._evaluation_model_dir(model_path) as eval_model_path:
                os.makedirs(os.path.dirname(saving_path), exist_ok=True)
                # A recompute must never leave the previous run's output behind.
                if exists(saving_path):
                    os.remove(saving_path)

                cmd = [
                    sys.executable,
                    evaluate_script,
                    "-m",
                    eval_model_path,
                    "-sk",
                    skels_path,
                    "-i",
                    subjects_path,
                    "-s",
                    saving_path,
                ]

                weights = self._weights_source(model_path) or "<none found>"
                print(f"\n[Region {region}] weights: {weights}")
                if self._uses_best_weights_stand_in(model_path):
                    epoch = self.args.checkpoint_epoch
                    print(
                        f"[Region {region}] WARNING: no epoch={epoch} checkpoint; "
                        f"best_model_weights.pt stands in for epoch {epoch} (interim)"
                    )
                code = self.execute_command(cmd, shell=False)
            if code != 0:
                failed_regions.append(region)
                first_failure = first_failure or code
                if exists(saving_path):
                    os.remove(saving_path)

        if failed_regions:
            print(f"\n[FAILED] {len(failed_regions)} region(s) failed: {', '.join(failed_regions)}")
        return first_failure

    def _run_cka_test(self, output_base: str) -> None:
        """Run CKA coherence test on the generated embeddings."""
        print("\n" + "=" * 60)
        print("Running CKA Coherence Test")
        print("=" * 60)

        cka_output = join(output_base, "cka_results")
        print(f"Embeddings dir: {output_base}")
        print(f"CKA output: {cka_output}")

        try:
            test_models_coherence_from_directory(
                models_dir=output_base,
                embedding_filename="full_embeddings.csv",
                output_dir=cka_output,
                subject_column="ID",
            )
            print("CKA coherence test completed.")
        except Exception as e:
            print(f"Warning: CKA test failed - {e}")


def main():
    """Main entry point."""
    init_pipeline_process()
    script = GenerateEmbeddings()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())

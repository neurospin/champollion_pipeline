#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate sulcal embeddings per region.
Calls champollion/evaluate.py for each region in models_path.
"""

import cProfile
import gzip
import os
import pstats
import shutil
import sys
import tarfile
import tempfile
from abc import ABC, abstractmethod
from io import StringIO
from os.path import abspath, dirname, exists, join
from pathlib import Path
from urllib.parse import urlparse

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

# Add champollion to path for CKA imports
_SCRIPT_DIR = dirname(abspath(__file__))
_CHAMPOLLION_DIR = abspath(join(_SCRIPT_DIR, "..", "..", "external", "champollion_V1"))
if _CHAMPOLLION_DIR not in sys.path:
    sys.path.insert(0, _CHAMPOLLION_DIR)

from champollion.metrics.cka_coherence import test_models_coherence_from_directory  # noqa: E402


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
            if "huggingface.co" in parsed.netloc:
                return True
            return False
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
                print(f"  mask version subfolder: {self.subfolder}")
            from huggingface_hub import snapshot_download

            # Cache per (repo, subfolder) to avoid collisions across mask versions
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
            print(f"Successfully downloaded from Hugging Face to: {downloaded_path}")
            return downloaded_path

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Hugging Face downloads. Install it with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download from Hugging Face: {e}")


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
                "--cortical_version",
                f"Derivatives folder name (e.g. 'cortical_tiles-2027'). "
                f"Defaults to cortical_tiles-{CORTICAL_TILES_VERSION}.",
                default=f"cortical_tiles-{CORTICAL_TILES_VERSION}",
            )
            .add_flag(
                "--legacy",
                "Use deep_folding-2025 as the derivatives folder. "
                "Shorthand for --cortical_version deep_folding-2025.",
            )
            .add_optional_argument(
                "--masks-version",
                "Mask version subfolder to download from HuggingFace (e.g. 'canonical_25'). "
                "Ignored when models_path is a local directory.",
                default=None,
            )
            .add_optional_argument(
                "--masks",
                "Cortical tiles mask version used as crops subdirectory (e.g. 'canonical_25').",
                default="canonical_25",
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
                "Output base directory override. Defaults to "
                "{parent_of_datasets_root}/{dataset_name}embeddings/.",
                default=None,
            )
            .add_optional_argument(
                "--subjects",
                "Path to subjects CSV file with a 'Subject' column. "
                "Defaults to {datasets_root}/participants.tsv.",
                default=None,
            )
        )

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
        # Define extraction directory (where to store downloaded/extracted)
        # Use data/{datasets_root}/derivatives/champollion_V1/models_cache
        script_dir = dirname(abspath(__file__))
        data_dir = join(
            script_dir,
            "..",
            "..",
            "data",
            self.args.datasets_root.lstrip("/"),
            "derivatives",
            "champollion_V1",
            "models_cache",
        )
        extract_to = abspath(data_dir)
        os.makedirs(extract_to, exist_ok=True)

        # Get no_cache flag (force re-extraction)
        no_cache = getattr(self.args, "no_cache", False)

        # Check HuggingFace and URL strategies first with the ORIGINAL path
        # These strategies look for semantic patterns (e.g., "user/repo", URLs)
        # that would be destroyed by path resolution
        masks_version = getattr(self.args, "masks_version", None)
        hf_strategy = HuggingFaceStrategy(subfolder=masks_version)
        if hf_strategy.can_handle(models_path):
            try:
                return hf_strategy.fetch(models_path, extract_to, no_cache)
            except Exception as e:
                print(f"HuggingFace strategy failed: {e}")
                # Continue to other strategies

        remote_strategy = RemoteArchiveStrategy()
        if remote_strategy.can_handle(models_path):
            try:
                return remote_strategy.fetch(models_path, extract_to, no_cache)
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
                return local_strategy.fetch(resolved_path, extract_to, no_cache)
            except Exception as e:
                print(f"Local path strategy failed: {e}")
                # Continue to fallback

        # Fallback: ask user interactively
        fallback_strategy = InteractiveFallbackStrategy()
        return fallback_strategy.fetch(resolved_path, extract_to, no_cache)

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

        tmpdir = None
        if self.args.regions:
            tmpdir = self._make_regions_tmpdir(self.args.models_path)
            self.args.models_path = tmpdir

        script_dir = dirname(abspath(__file__))
        evaluate_script = abspath(join(
            script_dir, "..", "..", "external", "champollion_V1", "champollion", "evaluate.py"
        ))

        datasets_root = self.args.datasets_root.rstrip("/")
        output_base = self.args.output or join(
            os.path.dirname(datasets_root), os.path.basename(datasets_root) + "embeddings"
        )

        target_folder = self._get_derivatives_folder()
        masks = getattr(self.args, "masks", "canonical_25")
        crops_2mm_dir = join(datasets_root, "derivatives", target_folder, "crops", masks, "2mm")

        subjects_path = self.args.subjects or self._find_subjects_file(datasets_root)

        try:
            result = self._run_per_region(evaluate_script, crops_2mm_dir, subjects_path, output_base)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir)
            self.args.models_path = original_models_path

        if self.args.run_cka:
            self._run_cka_test(output_base)

        return result

    def _find_subjects_file(self, datasets_root: str) -> str:
        """Return path to participants file in datasets_root."""
        for fname in ("participants.tsv", "participants.csv"):
            path = join(datasets_root, fname)
            if exists(path):
                return path
        raise FileNotFoundError(
            f"No participants file found in {datasets_root}. "
            "Pass --subjects to specify the path explicitly."
        )

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

    def _run_per_region(
        self, evaluate_script: str, crops_2mm_dir: str, subjects_path: str, output_base: str
    ) -> int:
        """Invoke champollion/evaluate.py for each region in models_path."""
        models_path = self.args.models_path
        try:
            region_dirs = sorted([
                d for d in os.listdir(models_path)
                if os.path.isdir(join(models_path, d))
            ])
        except FileNotFoundError as exc:
            raise ValueError(f"Models path not found: {models_path}") from exc

        if not region_dirs:
            raise ValueError(f"No region subdirectories found in {models_path}")

        print(f"\nGenerating embeddings for {len(region_dirs)} regions → {output_base}")

        last_result = 0
        for region in region_dirs:
            model_path = join(models_path, region)
            side = "L" if region.endswith("_left") else "R"
            crop_name = self._find_crop_dir(crops_2mm_dir, region)
            skels_path = join(crops_2mm_dir, crop_name, "mask", f"{side}skeleton.npy")
            saving_path = join(output_base, region, "full_embeddings.csv")

            if exists(saving_path) and not getattr(self.args, "overwrite", False):
                print(f"  [SKIP] {region} — already exists (--overwrite to recompute)")
                continue

            os.makedirs(os.path.dirname(saving_path), exist_ok=True)

            cmd = [sys.executable, evaluate_script,
                   "-m", model_path,
                   "-sk", skels_path,
                   "-i", subjects_path,
                   "-s", saving_path]

            print(f"\n[Region {region}]")
            last_result = self.execute_command(cmd, shell=False)

        return last_result

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
                subject_column="Subject",
            )
            print("CKA coherence test completed.")
        except Exception as e:
            print(f"Warning: CKA test failed - {e}")


def main():
    """Main entry point."""
    script = GenerateEmbeddings()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())

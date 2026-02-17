#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate embeddings and train classifiers.
This script manages user inputs and calls embeddings_pipeline.py
using subprocess.
"""

import cProfile
import gzip
import os
import pstats
import shutil
import sys
import tarfile
from abc import ABC, abstractmethod
from io import StringIO
from os.path import abspath, dirname, exists, join
from pathlib import Path
from urllib.parse import urlparse

from champollion_utils.script_builder import ScriptBuilder

# Add champollion to path for CKA imports
_SCRIPT_DIR = dirname(abspath(__file__))
_CHAMPOLLION_DIR = abspath(join(
    _SCRIPT_DIR, '..', 'external', 'champollion_V1', 'contrastive'
))
if _CHAMPOLLION_DIR not in sys.path:
    sys.path.insert(0, _CHAMPOLLION_DIR)

from contrastive.evaluation.cka_coherence import test_models_coherence_from_directory  # noqa: E402


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
        if any(models_path.endswith(ext) for ext in
               ['.tar.xz', '.tar.gz', '.tgz', '.gz']):
            # Compute cache path based on archive name
            # Extract to a subdirectory named after the archive
            archive_name = Path(models_path).stem
            if archive_name.endswith('.tar'):
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

        if archive_path.endswith(('.tar.xz', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(path=extract_to)
                members = tar.getmembers()
                if members:
                    # Return the extract_to path where contents are
                    print(f"Extracted to: {extract_to}")
                    return extract_to
        elif archive_path.endswith('.gz'):
            output_path = join(extract_to, Path(archive_path).stem)
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to: {output_path}")
            return output_path

        return extract_to


class HuggingFaceStrategy(ModelFetchStrategy):
    """Strategy for handling Hugging Face repository IDs and URLs."""

    def can_handle(self, models_path: str) -> bool:
        """Check if path looks like a Hugging Face repo ID or URL."""
        # Check if it's a huggingface.co URL
        parsed = urlparse(models_path)
        if parsed.scheme in ['http', 'https']:
            if 'huggingface.co' in parsed.netloc:
                return True
            return False
        # HF repo IDs are typically in format: username/repo-name
        # Check if it looks like a repo ID (no file extensions)
        return ('/' in models_path and
                not models_path.startswith('/') and
                not any(models_path.endswith(ext) for ext in
                        ['.tar.xz', '.tar.gz', '.tgz', '.gz', '.zip']))

    def _extract_repo_id(self, models_path: str) -> str:
        """Extract repo ID from a Hugging Face URL or return as-is if already a repo ID."""
        parsed = urlparse(models_path)
        if parsed.scheme in ['http', 'https'] and 'huggingface.co' in parsed.netloc:
            # URL format: https://huggingface.co/username/repo-name
            # Path is /username/repo-name, strip leading slash
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return '/'.join(path_parts[:2])  # username/repo-name
            elif len(path_parts) == 1:
                return path_parts[0]  # Just repo-name for official repos
        return models_path  # Already a repo ID

    def fetch(self, models_path: str, extract_to: str, no_cache: bool = False) -> str:
        """Download from Hugging Face."""
        try:
            # Extract repo ID from URL if needed
            repo_id = self._extract_repo_id(models_path)
            print(f"Attempting to download from Hugging Face: {repo_id}")
            from huggingface_hub import snapshot_download

            # Create local directory based on repo name
            repo_name = repo_id.split('/')[-1]
            local_path = join(extract_to, repo_name)

            # HuggingFace handles caching internally, but we can force redownload
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                force_download=no_cache
            )
            print(f"Successfully downloaded from Hugging Face to: "
                  f"{downloaded_path}")
            return downloaded_path

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Hugging Face downloads. "
                "Install it with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download from Hugging Face: {e}"
            )


class RemoteArchiveStrategy(ModelFetchStrategy):
    """Strategy for handling remote archive URLs."""

    def can_handle(self, models_path: str) -> bool:
        """Check if path is a remote URL to an archive."""
        parsed = urlparse(models_path)
        if parsed.scheme not in ['http', 'https', 'ftp']:
            return False
        # Check if it ends with archive extensions
        return any(models_path.endswith(ext) for ext in
                   ['.tar.xz', '.tar.gz', '.tgz', '.gz', '.zip'])

    def fetch(self, models_path: str, extract_to: str, no_cache: bool = False) -> str:
        """Download and extract remote archive."""
        import urllib.request

        # Compute cache path based on archive name
        filename = Path(urlparse(models_path).path).name
        archive_name = Path(filename).stem
        if archive_name.endswith('.tar'):
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
            extracted_path = self._extract_archive(
                local_archive, extract_to
            )

            # Clean up the archive file
            os.remove(local_archive)

            return extracted_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download or extract archive: {e}"
            )

    def _extract_archive(self, archive_path: str, extract_to: str) -> str:
        """Extract archive and return path to extracted content."""
        print(f"Extracting {archive_path}...")

        if archive_path.endswith(('.tar.xz', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(path=extract_to)
                members = tar.getmembers()
                if members:
                    top_dir = members[0].name.split('/')[0]
                    extracted_path = join(extract_to, top_dir)
                    print(f"Extracted to: {extracted_path}")
                    return extracted_path
        elif archive_path.endswith('.gz'):
            output_path = join(extract_to, Path(archive_path).stem)
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
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
        print(f"\nCannot automatically retrieve models from: "
              f"{models_path}")

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

        if response.lower() not in ['yes', 'y']:
            print("Cannot proceed without models. Exiting gracefully.")
            sys.exit(0)

        archive_path = input(
            "Path to models directory or archive "
            "(tar.xz, tar.gz, .gz): "
        ).strip()

        if not exists(archive_path):
            raise FileNotFoundError(
                f"Path not found: {archive_path}"
            )

        # If it's a directory, use it directly
        if os.path.isdir(archive_path):
            print(f"Using models from directory: {archive_path}")
            return archive_path

        # Check for cached extraction
        archive_name = Path(archive_path).stem
        if archive_name.endswith('.tar'):
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

        if archive_path.endswith(('.tar.xz', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(path=extract_to)
                members = tar.getmembers()
                if members:
                    top_dir = members[0].name.split('/')[0]
                    extracted_path = join(extract_to, top_dir)
                    print(f"Extracted to: {extracted_path}")
                    return extracted_path
        elif archive_path.endswith('.gz'):
            output_path = join(extract_to, Path(archive_path).stem)
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to: {output_path}")
            return output_path

        return extract_to


class GenerateEmbeddings(ScriptBuilder):
    """Script for generating embeddings and training classifiers."""

    def __init__(self):
        super().__init__(
            script_name="generate_embeddings",
            description=(
                "Generate embeddings and train classifiers for "
                "deep learning models."
            )
        )
        # Configure arguments using method chaining
        (self.add_argument(
            "models_path", type=str,
            help="Path to the directory containing model folders.")
         .add_argument(
            "dataset_localization", type=str,
            help="Key for dataset localization.")
         .add_argument(
            "datasets_root", type=str,
            help="Root path to the dataset YAML configs.")
         .add_argument(
            "short_name", type=str,
            help=(
                "Name of the directory where to store both "
                "embeddings and aucs."
            ))
         .add_argument(
            "--datasets", type=str, nargs="+",
            default=["toto"],
            help="List of dataset names (default: ['toto']).")
         .add_argument(
            "--labels", type=str, nargs="+",
            default=["Sex"],
            help="List of labels (default: ['Sex']).")
         .add_optional_argument(
            "--classifier_name", "Classifier name.",
            default="svm")
         .add_flag("--overwrite", "Overwrite existing embeddings.")
         .add_flag(
            "--embeddings_only",
            "Only compute embeddings (skip classifiers).")
         .add_flag(
            "--use_best_model",
            "Use the best model saved during training.")
         .add_argument(
            "--subsets", type=str, nargs="+",
            default=["full"],
            help="Subsets of data to train on (default: ['full']).")
         .add_argument(
            "--epochs", type=str, nargs="+",
            default=["None"],
            help="List of epochs to evaluate (default: [None]).")
         .add_optional_argument(
            "--config_path",
            "Path to dataset config directory.",
            default=None)
         .add_optional_argument(
            "--split",
            "Splitting strategy ('random' or 'custom').",
            default="random")
         .add_optional_argument(
            "--cv",
            "Number of cross-validation folds.",
            default=5,
            type_=int)
         .add_optional_argument(
            "--splits_basedir",
            "Directory for custom splits.", default="")
         .add_optional_argument(
            "--idx_region_evaluation",
            "Index of region to evaluate (multi-head models).",
            default=None,
            type_=int)
         .add_flag("--verbose", "Enable verbose output.")
         .add_flag("--cpu", "Force CPU usage (disable CUDA).")
         .add_flag("--profiling", "Enable Python profiling (cProfile).")
         .add_flag("--skip-cka", "Skip CKA coherence test after embeddings.")
         .add_flag("--no-cache", "Force re-extraction of archive (ignore cache).")
         .add_optional_argument(
            "--nb_jobs",
            "Number of CPU workers for DataLoader.",
            default=None, type_=int))

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
        data_dir = join(script_dir, '..', '..', 'data', self.args.datasets_root,
                        'derivatives', 'champollion_V1', 'models_cache')
        extract_to = abspath(data_dir)
        os.makedirs(extract_to, exist_ok=True)

        # Get no_cache flag (force re-extraction)
        no_cache = getattr(self.args, 'no_cache', False)

        # Check HuggingFace and URL strategies first with the ORIGINAL path
        # These strategies look for semantic patterns (e.g., "user/repo", URLs)
        # that would be destroyed by path resolution
        hf_strategy = HuggingFaceStrategy()
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
            if not models_path.startswith('/'):  # Relative path
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

    def run(self):
        """Execute the embeddings pipeline script."""
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
            stats.sort_stats('cumulative')
            stats.dump_stats('embeddings_profile.prof')
            print("\nProfiling results saved to embeddings_profile.prof")
            print("View with: python -m pstats embeddings_profile.prof")
            # Print top 20 functions
            stream = StringIO()
            stats_print = pstats.Stats(profiler, stream=stream)
            stats_print.sort_stats('cumulative')
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
        """Internal method to run the pipeline (called within try/finally)."""
        # Fetch models if they don't exist
        original_models_path = self.args.models_path
        self.args.models_path = self.fetch_models(original_models_path)

        # Get absolute path to champollion_V1/contrastive
        script_dir = dirname(abspath(__file__))
        champollion_dir = abspath(join(
            script_dir, '..', 'external', 'champollion_V1', 'contrastive'
        ))

        os.chdir(champollion_dir)

        # Use build_command to construct the command
        defaults = {
            "datasets": ["toto"],
            "labels": ["Sex"],
            "classifier_name": "svm",
            "overwrite": False,
            "embeddings_only": False,
            "use_best_model": False,
            "subsets": ["full"],
            "epochs": ["None"],
            "config_path": None,
            "split": "random",
            "cv": 5,
            "splits_basedir": "",
            "idx_region_evaluation": None,
            "verbose": False,
            "cpu": False,
            "nb_jobs": None
        }

        cmd = self.build_command(
            script_path="evaluation/embeddings_pipeline.py",
            required_args=[
                "models_path",
                "dataset_localization",
                "datasets_root",
                "short_name"
                ],
            defaults=defaults
        )

        result = self.execute_command(cmd, shell=False)

        # Run CKA coherence test by default (skip if --skip-cka)
        if not self.args.skip_cka:
            self._run_cka_test()

        os.chdir(local_dir)

        return result

    def _run_cka_test(self):
        """Run CKA coherence test comparing embeddings across all models."""
        print("\n" + "=" * 60)
        print("Running CKA Coherence Test")
        print("=" * 60)

        # CKA compares all embeddings found in models_path
        # Output goes to cka_results inside models_path
        cka_output = join(self.args.models_path, 'cka_results')

        print(f"Models path: {self.args.models_path}")
        print(f"CKA output: {cka_output}")

        try:
            test_models_coherence_from_directory(
                models_dir=self.args.models_path,
                embedding_filename='full_embeddings.csv',
                output_dir=cka_output,
                subject_column='Subject'
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

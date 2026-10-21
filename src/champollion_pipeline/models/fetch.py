"""Model set fetch strategies (local path, Hugging Face, remote archive, interactive fallback) and their order."""

import os
import sys
import warnings
from abc import ABC, abstractmethod
from os.path import abspath, exists, join
from pathlib import Path
from urllib.parse import urlparse

from champollion_pipeline.models.cache import MODEL_SET_FILE_NAME, ModelSet, write_model_set
from champollion_pipeline.utils.archive import extract_archive, remove_stale_cache, reuse_cached_extraction


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

            if reuse_cached_extraction(cached_path, no_cache, require_content=True):
                print("(Use --no-cache to force re-extraction)")
                return cached_path

            print(f"Local archive found: {models_path}")
            print(f"Extracting to: {cached_path}")
            remove_stale_cache(cached_path, no_cache)
            return self._extract_archive(models_path, cached_path)

        # Otherwise, assume it's a regular file/directory and return as-is
        print(f"Models found at local path: {models_path}")
        return models_path

    def _extract_archive(self, archive_path: str, extract_to: str) -> str:
        """Extract archive into extract_to and return extract_to (or the decompressed .gz file)."""
        return extract_archive(archive_path, extract_to, top_level=False)


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

        if reuse_cached_extraction(cached_path, no_cache):
            print("(Use --no-cache to force re-download)")
            return cached_path
        remove_stale_cache(cached_path, no_cache)

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
        """Extract archive and return path to its top-level entry."""
        print(f"Extracting {archive_path}...")
        return extract_archive(archive_path, extract_to, top_level=True)


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

        if reuse_cached_extraction(cached_path, no_cache):
            print("(Use --no-cache to force re-extraction)")
            return cached_path
        remove_stale_cache(cached_path, no_cache)

        print(f"Extracting archive: {archive_path}")
        return self._extract_archive(archive_path, extract_to)

    def _extract_archive(self, archive_path: str, extract_to: str) -> str:
        """Extract archive and return path to its top-level entry."""
        return extract_archive(archive_path, extract_to, top_level=True)


def fetch_model_set(models_path: str, extract_to: str, model_version: str | None, no_cache: bool) -> tuple[str, dict]:
    """Fetch the model set with the first strategy that succeeds and return (local path, models source).

    Hugging Face and remote archive URLs are tried first on the original path (path resolution would destroy their
    patterns), then a local path or archive, then the interactive fallback.
    """
    hf_strategy = HuggingFaceStrategy(subfolder=model_version)
    if hf_strategy.can_handle(models_path):
        try:
            fetched = hf_strategy.fetch(models_path, extract_to, no_cache)
            return fetched, {
                "type": "huggingface",
                "repo_id": hf_strategy._extract_repo_id(models_path),
                "subfolder": model_version,
            }
        except Exception as e:
            print(f"HuggingFace strategy failed: {e}")
            # Continue to other strategies

    remote_strategy = RemoteArchiveStrategy()
    if remote_strategy.can_handle(models_path):
        try:
            fetched = remote_strategy.fetch(models_path, extract_to, no_cache)
            return fetched, {"type": "url", "url": models_path}
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
            return fetched, {"type": "local", "path": os.path.abspath(resolved_path)}
        except Exception as e:
            print(f"Local path strategy failed: {e}")
            # Continue to fallback

    # Fallback: ask user interactively
    fallback_strategy = InteractiveFallbackStrategy()
    fetched = fallback_strategy.fetch(resolved_path, extract_to, no_cache)
    return fetched, {"type": "local", "path": os.path.abspath(fetched)}

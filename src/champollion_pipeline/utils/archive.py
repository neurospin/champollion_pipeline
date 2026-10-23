"""Archive extraction and extraction-cache helpers shared by the model fetch strategies."""

import gzip
import os
import shutil
import tarfile
from os.path import exists, join
from pathlib import Path

_TAR_SUFFIXES = (".tar.xz", ".tar.gz", ".tgz")


def extract_archive(archive_path: str, extract_to: str, *, top_level: bool) -> str:
    """Extract a tar or gzip archive under extract_to and return the extracted path.

    A tar archive returns its first top-level entry when top_level is set, else extract_to; a plain .gz returns the
    decompressed file; an unknown suffix or an empty tar returns extract_to.
    """
    os.makedirs(extract_to, exist_ok=True)
    if archive_path.endswith(_TAR_SUFFIXES):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=extract_to)
            members = tar.getmembers()
        if members:
            result = join(extract_to, members[0].name.split("/")[0]) if top_level else extract_to
            print(f"Extracted to: {result}")
            return result
    elif archive_path.endswith(".gz"):
        output_path = join(extract_to, Path(archive_path).stem)
        with gzip.open(archive_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Extracted to: {output_path}")
        return output_path
    return extract_to


def reuse_cached_extraction(cached_path: str, no_cache: bool, *, require_content: bool = False) -> bool:
    """Return True when the cached extraction at cached_path can be reused (and say so); require_content also needs
    the directory to be non-empty."""
    reuse = (
        exists(cached_path)
        and os.path.isdir(cached_path)
        and not no_cache
        and (not require_content or any(os.scandir(cached_path)))
    )
    if reuse:
        print(f"Using cached extraction: {cached_path}")
    return reuse


def remove_stale_cache(cached_path: str, no_cache: bool) -> None:
    """Delete the cached extraction at cached_path when no_cache is set."""
    if no_cache and exists(cached_path):
        print(f"Removing existing cache: {cached_path}")
        shutil.rmtree(cached_path)

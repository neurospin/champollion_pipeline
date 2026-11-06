"""A writable working copy of a packaged resource, whatever the mode of the packaged file."""

import os
import shutil
import stat
from contextlib import suppress


def grants_no_write(path: str) -> bool:
    """Return True when the mode bits of path hold no write bit; False when path is absent."""
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        return False
    return not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def replace_with_writable_copy(source: str, destination: str) -> None:
    """Put a copy of source at destination, in place of any file there, that its owner can write."""
    with suppress(FileNotFoundError):
        os.remove(destination)
    shutil.copyfile(source, destination)
    os.chmod(destination, stat.S_IMODE(os.stat(destination).st_mode) | stat.S_IWUSR)

"""Test helper: make the file system list each folder in reverse order."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch


class ReversedListing:
    """An os.scandir() result giving its entries last to first."""

    def __init__(self, entries: list) -> None:
        self._entries = iter(reversed(entries))

    def __iter__(self) -> "ReversedListing":
        return self

    def __next__(self) -> os.DirEntry:
        return next(self._entries)

    def __enter__(self) -> "ReversedListing":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def close(self) -> None:
        """Nothing to release: the real listing is already closed."""


@contextmanager
def file_system_listing_reversed() -> Iterator[None]:
    """Make the file system list each folder in reverse; os.walk and glob both read it through os.scandir."""
    real_scandir = os.scandir

    def reversed_scandir(*args, **kwargs) -> ReversedListing:
        with real_scandir(*args, **kwargs) as listing:
            return ReversedListing(list(listing))

    with patch("os.scandir", reversed_scandir):
        yield

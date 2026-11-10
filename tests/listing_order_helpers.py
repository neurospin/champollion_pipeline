"""Listing orders shared by the listing-order tests.

Each helper takes a list of directory entry names and returns them in one order, so a test can feed the code under
test the same entries in several orders and check the result does not change.
"""

from collections.abc import Callable

Order = Callable[[list[str]], list[str]]


def ascending(names: list[str]) -> list[str]:
    return sorted(names)


def descending(names: list[str]) -> list[str]:
    return sorted(names, reverse=True)


def rotated(names: list[str]) -> list[str]:
    """Sorted names with the first one moved to the end: neither ascending nor descending."""
    ordered = sorted(names)
    return ordered[1:] + ordered[:1]

"""Validation of a subject name used to build per-subject paths."""

import os


def subject_name_problem(subject: str) -> str | None:
    """Return why subject cannot name a per-subject directory, or None when it can.

    Pure check: no I/O, no logging, never raises.
    """
    if subject == "":
        return "it is empty"
    if subject in (os.curdir, os.pardir):
        return "it is '.' or '..'"
    if os.sep in subject or (os.altsep and os.altsep in subject):
        return "it contains a path separator"
    return None

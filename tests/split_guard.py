"""
Shared checks of the test-file split guards (tests/test_*_split_guard.py).

A split guard freezes the class-qualified test names of a reference test file and checks, over the files of one glob in
tests/, that each name is defined in exactly one file and that each file stays within max_file_lines. Each guard keeps
its own reference names, glob and budget quantity; this module holds the parsing, counting, measure writing and
assertions.
"""

import ast
import json
import os
from collections import Counter
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
MAX_FILE_LINES = 1000


def split_files(pattern):
    """Files of tests/ matching the glob pattern, sorted for a fixed read order."""
    return sorted(TESTS_DIR.glob(pattern))


def _is_test_function(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")


def _is_test_class(node):
    return isinstance(node, ast.ClassDef) and node.name.startswith("Test")


def _node_test_names(node):
    """Test names one top-level node defines: its own name, or its Test* class test methods, class-qualified."""
    if _is_test_function(node):
        return [node.name]
    if _is_test_class(node):
        return [f"{node.name}::{member.name}" for member in node.body if _is_test_function(member)]
    return []


def defined_test_names(path):
    """Module-level test functions and Test* class test methods of one file, class-qualified."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [name for node in tree.body for name in _node_test_names(node)]


def assert_each_name_defined_once(reference_names, expected_count, pattern):
    """Each of the expected_count distinct reference names is defined in exactly one file matching pattern."""
    counts = Counter()
    for path in split_files(pattern):
        counts.update(defined_test_names(path))
    wrong = {name: counts[name] for name in reference_names if counts[name] != 1}
    assert len(set(reference_names)) == expected_count
    assert wrong == {}, f"reference test names not defined exactly once: {wrong}"


def assert_files_within_limit(pattern, measure):
    """Each file matching pattern has at most MAX_FILE_LINES lines; the largest count goes to $ALM_MEASURES."""
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in split_files(pattern)}
    measures = os.environ.get("ALM_MEASURES")
    if measures and counts:
        with Path(measures).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({measure: max(counts.values())}) + "\n")
    over = {name: lines for name, lines in counts.items() if lines > MAX_FILE_LINES}
    assert over == {}, f"files over {MAX_FILE_LINES} lines: {over}"

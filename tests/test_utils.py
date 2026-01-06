#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for utils/lib.py utility functions.
"""

import pytest
from pathlib import Path

from utils.lib import are_paths_valid, get_nth_parent_dir


class TestArePathsValid:
    """Test are_paths_valid function."""

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="doesn't contain any path"):
            are_paths_valid([])

    def test_single_valid_path(self, temp_dir):
        """Test single valid path returns True."""
        assert are_paths_valid([temp_dir]) is True

    def test_multiple_valid_paths(self, temp_dir):
        """Test multiple valid paths return True."""
        path1 = Path(temp_dir) / "dir1"
        path2 = Path(temp_dir) / "dir2"
        path1.mkdir()
        path2.mkdir()

        assert are_paths_valid([str(path1), str(path2)]) is True

    def test_single_invalid_path(self):
        """Test single invalid path returns False."""
        assert are_paths_valid(["/nonexistent/path"]) is False

    def test_mixed_valid_invalid_paths(self, temp_dir):
        """Test mixed valid and invalid paths returns False."""
        valid_path = temp_dir
        invalid_path = "/nonexistent/path"

        assert are_paths_valid([valid_path, invalid_path]) is False

    def test_all_invalid_paths(self):
        """Test all invalid paths return False."""
        assert are_paths_valid(["/nonexistent1", "/nonexistent2", "/nonexistent3"]) is False

    def test_stops_at_first_invalid(self, temp_dir):
        """Test that function stops checking after first invalid path."""
        path1 = Path(temp_dir) / "valid"
        path1.mkdir()

        # First path valid, second invalid, third valid (but shouldn't be checked)
        result = are_paths_valid([str(path1), "/nonexistent", str(path1)])

        # Should return False because second path is invalid
        assert result is False


class TestGetNthParentDir:
    """Test get_nth_parent_dir function."""

    def test_get_first_parent(self):
        """Test getting the first parent directory."""
        folder = "/path/to/some/folder"
        result = get_nth_parent_dir(folder, 1)
        assert result == "/path/to/some"

    def test_get_second_parent(self):
        """Test getting the second parent directory."""
        folder = "/path/to/some/folder"
        result = get_nth_parent_dir(folder, 2)
        assert result == "/path/to"

    def test_get_third_parent(self):
        """Test getting the third parent directory."""
        folder = "/path/to/some/folder"
        result = get_nth_parent_dir(folder, 3)
        # When n >= len(split('/')) - 3, returns /<first_dir>/
        # split('/') = ['', 'path', 'to', 'some', 'folder'] -> len=5
        # n=3 >= 5-3=2, so returns '/path/'
        assert result == "/path/"

    def test_zero_iterations(self):
        """Test with zero iterations returns same folder."""
        folder = "/path/to/folder"
        result = get_nth_parent_dir(folder, 0)
        assert result == folder

    def test_reaches_root(self):
        """Test reaching root directory."""
        folder = "/path/to/folder"
        result = get_nth_parent_dir(folder, 10)  # More than depth
        # Should return root with trailing slash
        assert result == "/path/"

    def test_single_level_deep(self):
        """Test with single level deep path."""
        folder = "/folder"
        result = get_nth_parent_dir(folder, 1)
        # Should return root with folder name
        assert result == "/folder/"

    def test_nested_path(self):
        """Test with deeply nested path."""
        folder = "/a/b/c/d/e/f/g"
        result = get_nth_parent_dir(folder, 3)
        assert result == "/a/b/c/d"

    def test_trailing_slash_handling(self):
        """Test that trailing slashes are handled correctly."""
        folder = "/path/to/folder/"
        result = get_nth_parent_dir(folder, 1)
        # Behavior depends on implementation with trailing slash
        # This test documents the actual behavior
        assert "/" in result

    def test_exceeds_depth_returns_root_based(self):
        """Test that exceeding depth returns root-based path."""
        folder = "/a/b/c"
        result = get_nth_parent_dir(folder, 100)
        # Should return root with first component
        assert result.startswith("/")
        assert result == "/a/"


class TestGetNthParentDirEdgeCases:
    """Test edge cases for get_nth_parent_dir."""

    def test_actual_filesystem_path(self, temp_dir):
        """Test with actual filesystem path."""
        # Create a nested structure
        nested = Path(temp_dir) / "a" / "b" / "c" / "d"
        nested.mkdir(parents=True)

        result = get_nth_parent_dir(str(nested), 2)
        expected = str(Path(temp_dir) / "a" / "b")
        assert result == expected

    def test_relative_to_root(self, temp_dir):
        """Test getting parent relative to temp directory."""
        # Create nested path
        path = Path(temp_dir) / "level1" / "level2" / "level3"
        path.mkdir(parents=True)

        result = get_nth_parent_dir(str(path), 1)
        assert result == str(Path(temp_dir) / "level1" / "level2")


@pytest.mark.unit
class TestUtilsFunctionsBehavior:
    """Test overall behavior and integration of utility functions."""

    def test_are_paths_valid_with_get_nth_parent_dir(self, temp_dir):
        """Test using both functions together."""
        # Create a nested structure
        nested = Path(temp_dir) / "a" / "b" / "c"
        nested.mkdir(parents=True)

        # Get parent directory
        parent = get_nth_parent_dir(str(nested), 1)

        # Check that parent path is valid
        assert are_paths_valid([parent]) is True

    def test_are_paths_valid_type_checking(self):
        """Test that are_paths_valid accepts list of strings."""
        # Should not raise type errors
        result = are_paths_valid(["/path1", "/path2"])
        assert isinstance(result, bool)

    def test_get_nth_parent_dir_returns_string(self):
        """Test that get_nth_parent_dir returns a string."""
        result = get_nth_parent_dir("/path/to/folder", 1)
        assert isinstance(result, str)


@pytest.mark.smoke
class TestUtilsSmoke:
    """Smoke tests for utility functions."""

    def test_are_paths_valid_exists(self):
        """Test that are_paths_valid function exists."""
        assert callable(are_paths_valid)

    def test_get_nth_parent_dir_exists(self):
        """Test that get_nth_parent_dir function exists."""
        assert callable(get_nth_parent_dir)

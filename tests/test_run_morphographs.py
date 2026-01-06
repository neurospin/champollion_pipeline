import pytest

from os import getcwd
from os.path import exists

from src.generate_morphologist_graphs import GenerateMorphologistGraphs

path: str = "/my/path/example/"


def test_does_path_exists():
    """Test that invalid paths raise ValueError."""
    # Testing input path
    script = GenerateMorphologistGraphs()
    args = [path, path]
    script.parse_args(args)

    # This should raise an error because the path doesn't exist
    with pytest.raises(ValueError) as excinfo:
        script.run()

    assert "Please input valid paths" in str(excinfo.value)


def test_does_morphologist_create_outputs():
    """Test that morphologist creates output directory."""
    # This test would need valid input data to run properly
    # Skipping for now as it requires actual data
    pytest.skip("Requires valid input data to test")


def test_are_output_in_correct_location():
    """Test that outputs are created in the correct location."""
    # This test would need valid input data to run properly
    # Skipping for now as it requires actual data
    pytest.skip("Requires valid input data to test")

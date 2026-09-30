from pathlib import Path

# Version used for derivatives folder naming (e.g., cortical_tiles-2026)
CORTICAL_TILES_VERSION = "2026"
DERIVATIVES_FOLDER = f"cortical_tiles-{CORTICAL_TILES_VERSION}"


def find_dataset_folder(path: str, dataset_name: str) -> str:
    """Find the dataset_folder by locating dataset_name in the path.

    The dataset_folder is the parent directory of the dataset_name component.
    For example, if path is '/data/TEST01/derivatives/cortical_tiles/crops/2mm'
    and dataset_name is 'TEST01', returns '/data'.

    Args:
        path: Absolute path containing the dataset name as a component
        dataset_name: Name of the dataset directory to search for

    Returns:
        Absolute path to the parent of the dataset directory

    Raises:
        ValueError: If dataset_name is not found in the path
    """
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part == dataset_name:
            return str(Path(*parts[:i]))
    raise ValueError(f"Dataset name '{dataset_name}' not found in path: {path}")

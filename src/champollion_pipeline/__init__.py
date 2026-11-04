"""Champollion pipeline — end-to-end sulcal embedding pipeline."""

import importlib

__version__ = "0.2.32"

# Stage classes are exported lazily (PEP 562): importing the package, or a
# toolchain-free submodule such as config, loads none of soma, cortical_tiles
# or champollion. A class whose submodule cannot be imported in this pixi
# environment (e.g. embeddings-only on Jean-Zay) is simply absent.
_LAZY_EXPORTS: dict[str, str] = {
    "GenerateMorphologistGraphs": "generate_morphologist_graphs",
    "RunCorticalTiles": "run_cortical_tiles",
    "GenerateChampollionConfig": "generate_champollion_config",
    "GenerateEmbeddings": "generate_embeddings",
    "PutTogetherEmbeddings": "put_together_embeddings",
    "GenerateSnapshots": "generate_snapshots",
    "TrainChampollion": "train_champollion",
    "PruneFailedSubjects": "prune_failed_subjects",
    "PurgeSubject": "purge_subject",
    "GenerateMasks": "generate_masks",
    "GenerateLabellingQC": "generate_labelling_qc",
}


def __getattr__(name: str) -> object:  # noqa: V103 - PEP 562 module attribute hook
    """Import a stage class's submodule on first access and cache the class."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{_LAZY_EXPORTS[name]}", __name__)
    except ImportError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} ({_LAZY_EXPORTS[name]} unavailable: {exc})"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value

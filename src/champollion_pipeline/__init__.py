"""Champollion pipeline — end-to-end sulcal embedding pipeline."""

__version__ = "0.2.0"

from .generate_morphologist_graphs import GenerateMorphologistGraphs
from .run_cortical_tiles import RunCorticalTiles
from .generate_champollion_config import GenerateChampollionConfig
from .generate_embeddings import GenerateEmbeddings
from .put_together_embeddings import PutTogetherEmbeddings
from .generate_snapshots import GenerateSnapshots
from .train_champollion import TrainChampollion
from .prune_failed_subjects import PruneFailedSubjects
from .purge_subject import PurgeSubject
from .generate_masks import GenerateMasks

__all__ = [
    "GenerateMorphologistGraphs",
    "RunCorticalTiles",
    "GenerateChampollionConfig",
    "GenerateEmbeddings",
    "PutTogetherEmbeddings",
    "GenerateSnapshots",
    "TrainChampollion",
    "PruneFailedSubjects",
    "PurgeSubject",
    "GenerateMasks",
]

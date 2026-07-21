"""Champollion pipeline — end-to-end sulcal embedding pipeline."""

__version__ = "0.2.0"

# Each script has optional dependencies (brainvisa, dracopy, torch…) that may
# not be installed in every pixi environment. Guard each import so the package
# is importable in partial envs (e.g. embeddings-only on Jean-Zay).

try:
    from .generate_morphologist_graphs import GenerateMorphologistGraphs
except ImportError:
    pass

try:
    from .run_cortical_tiles import RunCorticalTiles
except ImportError:
    pass

try:
    from .generate_champollion_config import GenerateChampollionConfig
except ImportError:
    pass

try:
    from .generate_embeddings import GenerateEmbeddings
except ImportError:
    pass

try:
    from .put_together_embeddings import PutTogetherEmbeddings
except ImportError:
    pass

try:
    from .generate_snapshots import GenerateSnapshots
except ImportError:
    pass

try:
    from .train_champollion import TrainChampollion
except ImportError:
    pass

try:
    from .prune_failed_subjects import PruneFailedSubjects
except ImportError:
    pass

try:
    from .purge_subject import PurgeSubject
except ImportError:
    pass

try:
    from .generate_masks import GenerateMasks
except ImportError:
    pass

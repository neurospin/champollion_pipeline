"""Champollion pipeline — end-to-end sulcal embedding pipeline."""

import contextlib

__version__ = "0.2.32"

# Each script has optional dependencies (brainvisa, torch…) that may
# not be installed in every pixi environment. Guard each import so the package
# is importable in partial envs (e.g. embeddings-only on Jean-Zay).

with contextlib.suppress(ImportError):
    from .generate_morphologist_graphs import GenerateMorphologistGraphs

with contextlib.suppress(ImportError):
    from .run_cortical_tiles import RunCorticalTiles

with contextlib.suppress(ImportError):
    from .generate_champollion_config import GenerateChampollionConfig

with contextlib.suppress(ImportError):
    from .generate_embeddings import GenerateEmbeddings

with contextlib.suppress(ImportError):
    from .put_together_embeddings import PutTogetherEmbeddings

with contextlib.suppress(ImportError):
    from .generate_snapshots import GenerateSnapshots

with contextlib.suppress(ImportError):
    from .train_champollion import TrainChampollion

with contextlib.suppress(ImportError):
    from .prune_failed_subjects import PruneFailedSubjects

with contextlib.suppress(ImportError):
    from .purge_subject import PurgeSubject

with contextlib.suppress(ImportError):
    from .generate_masks import GenerateMasks

with contextlib.suppress(ImportError):
    from .generate_labelling_qc import GenerateLabellingQC

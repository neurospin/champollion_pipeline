"""The models of one stage 4 run: discovery, checkpoint epoch of each model, output folders and provenance entries."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from os.path import join

from champollion_pipeline.derivatives_layout import compute_model_output_dir
from champollion_pipeline.models.checkpoint import (
    ModelEntry,
    find_region_model_dirs,
    find_region_models,
    has_region_weights,
    weights_source,
)
from champollion_pipeline.models.embeddings_qc import EMBEDDINGS_FILENAME
from champollion_pipeline.models.epoch_spec import resolve_model_epochs


class ModelRunRefused(ValueError):
    """The run's models or --checkpoint-epoch ids are refused: the stage ends with status 2."""


@dataclass(frozen=True)
class ModelRun:
    """The models of a run, sorted by (region, model id), with the epoch of each model id."""

    entries: tuple[ModelEntry, ...]
    epochs: Mapping[str, int]
    per_model_folders: bool

    def epoch_of(self, entry: ModelEntry) -> int:
        """The checkpoint epoch of ``entry``."""
        return self.epochs[entry.model_id]

    def lists_models(self) -> bool:
        """True when provenance.json lists the models: a region holds two or more, or per_model_folders is set."""
        regions = [entry.region for entry in self.entries]
        return self.per_model_folders or len(regions) > len(set(regions))

    def compute_provenance_models(self) -> list[dict]:
        """One {region, model_id, checkpoint_epoch, weights_source} object per model."""
        return [
            {
                "region": entry.region,
                "model_id": entry.model_id,
                "checkpoint_epoch": self.epoch_of(entry),
                "weights_source": weights_source(entry.path, self.epoch_of(entry)),
            }
            for entry in self.entries
        ]

    def compute_output_folders(self, output_base: str) -> dict[str, str]:
        """The output folder of each model, by model directory."""
        counts = Counter(entry.region for entry in self.entries)
        return {
            entry.path: compute_model_output_dir(
                output_base, entry.region, entry.model_id, counts[entry.region], self.per_model_folders
            )
            for entry in self.entries
        }

    def compute_cka_files(self, output_base: str) -> dict[str, str] | None:
        """The embeddings file of each model by label; None when every model has an empty model id.

        Such a run keeps the folder naming of the CKA directory search, which is the label.
        """
        if all(not entry.model_id for entry in self.entries):
            return None
        folders = self.compute_output_folders(output_base)
        return {entry.label: join(folders[entry.path], EMBEDDINGS_FILENAME) for entry in self.entries}


def resolve_model_run(
    models_path: str, default_epoch: int, epochs_by_id: Mapping[str, int], per_model_folders: bool
) -> ModelRun:
    """Discover the models under ``models_path`` and give each its epoch.

    Raises:
        ModelRunRefused: when two models of a region share a model id, or an id of ``epochs_by_id`` names no model.
        ValueError: when models_path is missing or holds no model, or a model lacks weights (each named by label).

    Complexity: O(n log n) in the number of model directories.
    """
    try:
        entries = find_region_models(models_path)
    except ValueError as err:
        find_region_model_dirs(models_path)  # a missing models path raises its own ValueError
        raise ModelRunRefused(str(err)) from err
    if not entries:
        raise ValueError(f"No region model directories (containing logs/) found under {models_path}")
    try:
        epochs = resolve_model_epochs(default_epoch, epochs_by_id, (entry.model_id for entry in entries))
    except ValueError as err:
        raise ModelRunRefused(f"argument --checkpoint-epoch: {err}") from err
    missing = [entry.label for entry in entries if not has_region_weights(entry.path, epochs[entry.model_id])]
    if missing:
        raise ValueError(
            f"No weights for {len(missing)} region(s): {', '.join(missing)}. Each model directory needs "
            "logs/best_model_weights.pt or an epoch=<N>-step=*.ckpt checkpoint of its epoch under "
            "logs/lightning_logs/version_*/checkpoints/; no region was evaluated."
        )
    return ModelRun(tuple(entries), epochs, per_model_folders)

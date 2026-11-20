"""EmbeddingEngine: lazily loads each region encoder once, in-process, and embeds batches of scans.

The encoders are champollion_V1's own (load_model, load_data, generate_embeddings, imported read-only), so a
region embeds as evaluate.py would, without one Python subprocess and one model load per region and run.
Not thread-safe: one engine serves one consumer.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from os.path import join

import numpy as np
import pandas as pd
import torch
from champollion import evaluate
from champollion_utils.logger import LOG

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.models.checkpoint import (
    DEFAULT_CHECKPOINT_EPOCH,
    ModelEntry,
    check_region_weights,
    evaluation_model_dir,
    find_region_models,
    uses_best_weights_stand_in,
    weights_source,
)

logger = logging.getLogger("champollion_pipeline.embedding_engine")

_SKELETON_NDIM = 5


class EmbeddingEngine:
    """Encoders of the models of one models path, keyed by (region, model id), loaded on first use and kept resident."""

    def __init__(
        self,
        models_path: str,
        device: torch.device | str,
        *,
        epoch: int = DEFAULT_CHECKPOINT_EPOCH,
        epochs: Mapping[str, int] | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.models_path = models_path
        self.device = torch.device(device)
        self.epoch = epoch
        self.epochs: dict[str, int] = dict(epochs or {})
        self.batch_size = batch_size
        self._models: dict[tuple[str, str], ModelEntry] = {
            (entry.region, entry.model_id): entry for entry in find_region_models(models_path)
        }
        self._encoders: dict[tuple[str, str], tuple[torch.nn.Module, tuple[int, ...]]] = {}

    @property
    def regions(self) -> list[str]:
        """Distinct region names of the models under the models path, sorted."""
        return sorted({region for region, _ in self._models})

    @property
    def models(self) -> list[tuple[str, str]]:
        """(region name, model id) of each model under the models path, in the order find_region_models gives."""
        return list(self._models)

    def embed(
        self, region: str, skeletons: np.ndarray, scan_ids: Sequence[str], model_id: str | None = None
    ) -> pd.DataFrame:
        """Embed skeletons shaped (N, H, W, D, C) with the encoder of the model: columns ID, dim1..dimK.

        ``model_id`` None is the only model of the region (ValueError when it holds several).
        """
        entry = self._resolve(region, model_id)
        if skeletons.ndim != _SKELETON_NDIM:
            raise ValueError(f"Skeletons must be shaped (N, H, W, D, C); got shape {skeletons.shape}")
        if len(scan_ids) != skeletons.shape[0]:
            raise ValueError(f"{skeletons.shape[0]} skeleton(s) but {len(scan_ids)} scan id(s) for {region}")
        if skeletons.shape[0] == 0:
            raise ValueError(f"No skeleton to embed for {region}")
        buffer = io.BytesIO()
        np.save(buffer, skeletons)
        buffer.seek(0)
        loader, in_shape = evaluate.load_data(buffer, batch_size=self.batch_size)
        return self._embed_loader(entry, loader, in_shape, scan_ids)

    def embed_crops(self, region: str, crops_2mm: str, model_id: str | None = None) -> pd.DataFrame:
        """Embed the region skeletons of a crops directory, labelled by its subjects CSV."""
        entry = self._resolve(region, model_id)
        side = "L" if region.endswith("_left") else "R"
        mask_dir = join(crops_2mm, GenerateEmbeddings._find_crop_dir(None, crops_2mm, region), "mask")
        skels_path = join(mask_dir, f"{side}skeleton.npy")
        subjects_path = join(mask_dir, f"{side}skeleton_subject.csv")
        loader, in_shape = evaluate.load_data(skels_path, batch_size=self.batch_size)
        subjects = list(evaluate.load_subjects(subjects_path))
        if len(subjects) != len(loader.dataset):
            raise ValueError(
                f"{skels_path} holds {len(loader.dataset)} skeleton(s) but {subjects_path} lists "
                f"{len(subjects)} subject(s)"
            )
        return self._embed_loader(entry, loader, in_shape, subjects)

    def _resolve(self, region: str, model_id: str | None) -> ModelEntry:
        """The model of a region: the given id, else the region's only model."""
        ids = [mid for reg, mid in self._models if reg == region]
        if not ids:
            raise ValueError(f"Unknown region {region!r}: no region model directory for it under {self.models_path}")
        if model_id is None:
            if len(ids) > 1:
                raise ValueError(f"Region {region!r} holds several models; give one of the model ids {ids!r}")
            model_id = ids[0]
        if (region, model_id) not in self._models:
            raise ValueError(f"Region {region!r} holds no model with id {model_id!r}; model ids: {ids!r}")
        return self._models[(region, model_id)]

    def _embed_loader(
        self, entry: ModelEntry, loader, in_shape: tuple[int, ...], scan_ids: Sequence[str]
    ) -> pd.DataFrame:
        model = self._encoder(entry, in_shape)
        embeddings = evaluate.generate_embeddings(model, loader, self.device)
        table = pd.DataFrame(embeddings, columns=[f"dim{i}" for i in range(1, embeddings.shape[1] + 1)])
        table.insert(0, "ID", list(scan_ids))
        return table

    def _encoder(self, entry: ModelEntry, in_shape: tuple[int, ...]) -> torch.nn.Module:
        """The resident encoder of a model, loaded on first use for this input shape."""
        key = (entry.region, entry.model_id)
        label = entry.label
        if key in self._encoders:
            model, loaded_shape = self._encoders[key]
            if loaded_shape != in_shape:
                raise ValueError(f"{label}: skeletons shaped {in_shape} but its encoder was loaded for {loaded_shape}")
            return model
        model_dir = entry.path
        epoch = self.epochs.get(entry.model_id, self.epoch)
        check_region_weights([model_dir], self.models_path, epoch)
        start = time.perf_counter()
        captured = io.StringIO()
        with evaluation_model_dir(model_dir, epoch) as eval_dir, redirect_stdout(captured):
            model = evaluate.load_model(eval_dir, in_shape).to(self.device)
        seconds = time.perf_counter() - start
        if captured.getvalue():
            logger.debug("%s: load_model output:\n%s", label, captured.getvalue().rstrip())
        logger.log(LOG, "%s: encoder loaded in %.3f s from %s", label, seconds, weights_source(model_dir, epoch))
        if uses_best_weights_stand_in(model_dir, epoch):
            logger.warning("%s: no epoch-%d checkpoint; best_model_weights.pt stands in for it", label, epoch)
        self._encoders[key] = (model, in_shape)
        return model

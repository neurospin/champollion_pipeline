"""EmbeddingEngine: lazily loads each region encoder once, in-process, and embeds batches of scans.

The encoders are champollion_V1's own (load_model, load_data, generate_embeddings, imported read-only), so a
region embeds as evaluate.py would, without one Python subprocess and one model load per region and run.
Not thread-safe: one engine serves one consumer.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Sequence
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
    check_region_weights,
    evaluation_model_dir,
    find_region_model_dirs,
    resolve_region_name,
    uses_best_weights_stand_in,
    weights_source,
)

logger = logging.getLogger("champollion_pipeline.embedding_engine")

_SKELETON_NDIM = 5


class EmbeddingEngine:
    """Region encoders of one models path, loaded on first use and kept resident."""

    def __init__(
        self,
        models_path: str,
        device: torch.device | str,
        *,
        epoch: int = DEFAULT_CHECKPOINT_EPOCH,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.models_path = models_path
        self.device = torch.device(device)
        self.epoch = epoch
        self.batch_size = batch_size
        self._region_dirs: dict[str, str] = {}
        for model_dir in find_region_model_dirs(models_path):
            region = resolve_region_name(model_dir, models_path)
            if region in self._region_dirs:
                logger.warning(
                    "%s: duplicate region model directory %s ignored; using %s",
                    region,
                    model_dir,
                    self._region_dirs[region],
                )
                continue
            self._region_dirs[region] = model_dir
        self._encoders: dict[str, tuple[torch.nn.Module, tuple[int, ...]]] = {}

    @property
    def regions(self) -> list[str]:
        """Region names of the region model directories under the models path, sorted."""
        return sorted(self._region_dirs)

    def embed(self, region: str, skeletons: np.ndarray, scan_ids: Sequence[str]) -> pd.DataFrame:
        """Embed skeletons shaped (N, H, W, D, C) with the region encoder: columns ID, dim1..dimK."""
        self._check_region(region)
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
        return self._embed_loader(region, loader, in_shape, scan_ids)

    def embed_crops(self, region: str, crops_2mm: str) -> pd.DataFrame:
        """Embed the region skeletons of a crops directory, labelled by its subjects CSV."""
        self._check_region(region)
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
        return self._embed_loader(region, loader, in_shape, subjects)

    def _check_region(self, region: str) -> None:
        if region not in self._region_dirs:
            raise ValueError(f"Unknown region {region!r}: no region model directory for it under {self.models_path}")

    def _embed_loader(self, region: str, loader, in_shape: tuple[int, ...], scan_ids: Sequence[str]) -> pd.DataFrame:
        model = self._encoder(region, in_shape)
        embeddings = evaluate.generate_embeddings(model, loader, self.device)
        table = pd.DataFrame(embeddings, columns=[f"dim{i}" for i in range(1, embeddings.shape[1] + 1)])
        table.insert(0, "ID", list(scan_ids))
        return table

    def _encoder(self, region: str, in_shape: tuple[int, ...]) -> torch.nn.Module:
        """The resident encoder of a region, loaded on first use for this input shape."""
        if region in self._encoders:
            model, loaded_shape = self._encoders[region]
            if loaded_shape != in_shape:
                raise ValueError(f"{region}: skeletons shaped {in_shape} but its encoder was loaded for {loaded_shape}")
            return model
        model_dir = self._region_dirs[region]
        check_region_weights([model_dir], self.models_path, self.epoch)
        start = time.perf_counter()
        captured = io.StringIO()
        with evaluation_model_dir(model_dir, self.epoch) as eval_dir, redirect_stdout(captured):
            model = evaluate.load_model(eval_dir, in_shape).to(self.device)
        seconds = time.perf_counter() - start
        if captured.getvalue():
            logger.debug("%s: load_model output:\n%s", region, captured.getvalue().rstrip())
        logger.log(LOG, "%s: encoder loaded in %.3f s from %s", region, seconds, weights_source(model_dir, self.epoch))
        if uses_best_weights_stand_in(model_dir, self.epoch):
            logger.warning("%s: no epoch-%d checkpoint; best_model_weights.pt stands in for it", region, self.epoch)
        self._encoders[region] = (model, in_shape)
        return model

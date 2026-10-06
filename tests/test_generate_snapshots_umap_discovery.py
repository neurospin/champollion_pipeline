#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filename contract between the combine stage and UMAP snapshot discovery.

TASK-200: put_together_embeddings writes ``{region}_{hemi}_embeddings.csv``
but generate_snapshots.discover_umap_pairs only accepted
``{region}_{hemi}_{suffix}_embeddings.csv``, so stage 6 never produced a
UMAP plot from combine output.

Requirements: REQ-UMAPDISC-01 .. REQ-UMAPDISC-04.
"""

import pytest

from champollion_pipeline.generate_snapshots import discover_umap_pairs
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings

# Region names exercising hyphens, underscores and dots.
REGIONS = [
    "Lobule_parietal_sup",
    "SC-sylv",
    "FCLp-subsc-FCLa-INSULA",
    "S.Or.",
    "F.Coll.-S.Rh.",
]
HEMIS = ["left", "right"]


def _dirs(tmp_path):
    emb_dir = tmp_path / "embeddings"
    ref_dir = tmp_path / "reference"
    emb_dir.mkdir()
    ref_dir.mkdir()
    return emb_dir, ref_dir


def _write_reference(ref_dir, region, hemi):
    model = ref_dir / f"umap_{region}_{hemi}.pkl"
    coords = ref_dir / f"umap_{region}_{hemi}_coords.npy"
    model.touch()
    coords.touch()
    return model, coords


class TestCombineStageNamingPaired:
    """REQ-UMAPDISC-01: {region}_{hemi}_embeddings.csv is paired."""

    @pytest.mark.parametrize("hemi", HEMIS)
    def test_combine_name_paired_with_reference_files(self, tmp_path, hemi):
        emb_dir, ref_dir = _dirs(tmp_path)
        csv = emb_dir / f"FCLp-subsc-FCLa-INSULA_{hemi}_embeddings.csv"
        csv.write_text("ID,dim1\nsub01,0.1\n")
        model, coords = _write_reference(ref_dir, "FCLp-subsc-FCLa-INSULA", hemi)

        pairs = discover_umap_pairs(str(emb_dir), str(ref_dir))

        assert pairs == [(str(csv), str(model), str(coords), "FCLp-subsc-FCLa-INSULA", hemi)]


class TestSuffixedNamingStillPaired:
    """REQ-UMAPDISC-02: {region}_{hemi}_{suffix}_embeddings.csv stays paired."""

    @pytest.mark.parametrize("hemi", HEMIS)
    @pytest.mark.parametrize("suffix", ["ixi", "name01", "001"])
    def test_suffixed_name_paired_with_reference_files(self, tmp_path, suffix, hemi):
        emb_dir, ref_dir = _dirs(tmp_path)
        csv = emb_dir / f"FCLp-subsc-FCLa-INSULA_{hemi}_{suffix}_embeddings.csv"
        csv.write_text("ID,dim1\nsub01,0.1\n")
        model, coords = _write_reference(ref_dir, "FCLp-subsc-FCLa-INSULA", hemi)

        pairs = discover_umap_pairs(str(emb_dir), str(ref_dir))

        assert pairs == [(str(csv), str(model), str(coords), "FCLp-subsc-FCLa-INSULA", hemi)]


class TestRegionNameParsing:
    """REQ-UMAPDISC-03: region name is the exact text before _left_/_right_."""

    @pytest.mark.parametrize("hemi", HEMIS)
    @pytest.mark.parametrize("region", REGIONS)
    def test_region_from_combine_name(self, tmp_path, region, hemi):
        emb_dir, ref_dir = _dirs(tmp_path)
        (emb_dir / f"{region}_{hemi}_embeddings.csv").touch()
        _write_reference(ref_dir, region, hemi)

        pairs = discover_umap_pairs(str(emb_dir), str(ref_dir))

        assert [(p[3], p[4]) for p in pairs] == [(region, hemi)]

    @pytest.mark.parametrize("hemi", HEMIS)
    @pytest.mark.parametrize("region", REGIONS)
    def test_region_from_suffixed_name(self, tmp_path, region, hemi):
        emb_dir, ref_dir = _dirs(tmp_path)
        (emb_dir / f"{region}_{hemi}_ixi_embeddings.csv").touch()
        _write_reference(ref_dir, region, hemi)

        pairs = discover_umap_pairs(str(emb_dir), str(ref_dir))

        assert [(p[3], p[4]) for p in pairs] == [(region, hemi)]


class TestCombineToSnapshotsContract:
    """REQ-UMAPDISC-04: combine stage output feeds UMAP discovery end to end."""

    def test_combine_output_names_discovered_as_region_hemi_pairs(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        expected = set()
        for region in REGIONS:
            for hemi in HEMIS:
                region_dir = source / f"{region}_{hemi}"
                region_dir.mkdir()
                (region_dir / "full_embeddings.csv").write_text("ID,dim1\nsub01,0.1\n")
                expected.add((region, hemi))
        combined = tmp_path / "combined"
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        for region, hemi in expected:
            _write_reference(ref_dir, region, hemi)

        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(combined)])
        assert script.run() == 0

        pairs = discover_umap_pairs(str(combined), str(ref_dir))

        found = [(p[3], p[4]) for p in pairs]
        assert len(found) == len(expected)
        assert set(found) == expected

"""TASK-201: tiles snapshot data root (REQ-TILESROOT-01).

Hermetic per ARCHITECTURE.md#snapshot-mesh-fallback-hermetic: Anatomist,
PyAIMS and cortical_tiles are mocked through ``sys.modules``; every path the
code touches lives under ``tmp_path`` and ``ICBM_MESH_DIR_FALLBACK`` is
redirected there, so nothing depends on /neurospin being mounted or not.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import champollion_pipeline.generate_snapshots as gs
from champollion_pipeline.generate_snapshots import generate_tiles_snapshot

OPTION = "--champollion_data_root"


@pytest.fixture
def tiles_env(tmp_path, monkeypatch):
    """Mocked modules, an empty data root (no region graphs) and ICBM meshes on disk."""
    empty_root = tmp_path / "champollion_data_without_meshes"
    empty_root.mkdir()

    icbm = tmp_path / "icbm"
    icbm.mkdir()
    (icbm / "mni_icbm152_nlin_asym_09c_Lhemi.gii").touch()
    (icbm / "mni_icbm152_nlin_asym_09c_Rhemi.gii").touch()
    monkeypatch.setattr(gs, "ICBM_MESH_DIR_FALLBACK", str(icbm))

    anatomist = MagicMock()
    anatomist.createWindow.return_value = MagicMock()
    headless = MagicMock()
    headless.Anatomist.return_value = anatomist
    anatomist_pkg = MagicMock()
    anatomist_pkg.headless = headless

    aims = MagicMock()
    aims.carto.Paths.findResourceFile.side_effect = lambda *args, **kwargs: (  # noqa: V101
        str(icbm) if "icbm152" in args[0] else "/nomenclature.hie"
    )

    config_mod = MagicMock()
    config_mod.config.return_value.get_champollion_data_root_dir.return_value = str(empty_root)
    cortical_tiles = MagicMock()
    cortical_tiles.config = config_mod

    modules = {
        "anatomist": anatomist_pkg,
        "anatomist.headless": headless,
        "cortical_tiles": cortical_tiles,
        "cortical_tiles.config": config_mod,
        "soma": MagicMock(aims=aims),
        "soma.aims": aims,
    }

    crops = tmp_path / "crops"
    mask = crops / "S.Or." / "mask"
    mask.mkdir(parents=True)
    (mask / "Lmask_skeleton.nii.gz").touch()

    return modules, empty_root, crops


def _expected_graph_path(root, side="L", level=1):
    return f"{root}/mask/2mm/regions/meshes/{side}regions_model_{level}.arg"


@pytest.mark.unit
class TestTilesMissingRegionGraphMessage:
    """REQ-TILESROOT-01: missing region graph -> message names the file path and --champollion_data_root."""

    def test_explicit_root_message_names_path_and_option(self, tmp_path, tiles_env, capsys):
        """REQ-TILESROOT-01: explicit champollion_data_root lacking the region graph."""
        modules, empty_root, crops = tiles_env
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), champollion_data_root=str(empty_root)
            )
        out = capsys.readouterr().out
        assert snaps == []
        assert _expected_graph_path(empty_root) in out
        assert OPTION in out, f"no mention of {OPTION} in output:\n{out}"

    def test_config_default_root_message_names_path_and_option(self, tmp_path, tiles_env, capsys):
        """REQ-TILESROOT-01: cortical_tiles config default root lacking the region graph (TASK-201 case)."""
        modules, empty_root, crops = tiles_env
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(str(crops), str(tmp_path / "tiles.png"))
        out = capsys.readouterr().out
        assert snaps == []
        assert _expected_graph_path(empty_root) in out
        assert OPTION in out, f"no mention of {OPTION} in output:\n{out}"

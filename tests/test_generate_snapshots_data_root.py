"""TASK-201: tiles snapshot data root (REQ-TILESROOT-01).

Hermetic per ARCHITECTURE.md#snapshot-mesh-fallback-hermetic: Anatomist,
PyAIMS and cortical_tiles are patched as module attributes of snapshot_types.brainvisa_imports; every path the
code touches lives under ``tmp_path`` and ``ICBM_MESH_DIR_FALLBACK`` is
redirected there, so nothing depends on /neurospin being mounted or not.
The message is read as one WARNING log record through pipeline_caplog
(REQ-LOGGING-BDRABCZUK-047543C19136).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_snapshots import generate_tiles_snapshot
from champollion_pipeline.snapshot_types import brainvisa_imports
from champollion_pipeline.snapshot_types import tiles as tiles_module

_MODULE_ATTRS = {"anatomist.headless": "ana", "cortical_tiles.config": "config", "soma.aims": "aims"}


def _as_module_attrs(modules):
    """generate_snapshots attribute -> mock, for the guarded BrainVISA modules present in ``modules``."""
    return {attr: modules[key] for key, attr in _MODULE_ATTRS.items() if key in modules}


OPTION = "--champollion_data_root"


def _graph_warning(capture, graph_path):
    """Message of the one WARNING record naming ``graph_path``."""
    naming = [r for r in capture.records if r.levelno == logging.WARNING and graph_path in r.getMessage()]
    assert len(naming) == 1, [(r.levelname, r.getMessage()) for r in capture.records]
    return naming[0].getMessage()


@pytest.fixture
def tiles_env(tmp_path, monkeypatch):
    """Mocked modules, an empty data root (no region graphs) and ICBM meshes on disk."""
    empty_root = tmp_path / "champollion_data_without_meshes"
    empty_root.mkdir()

    icbm = tmp_path / "icbm"
    icbm.mkdir()
    (icbm / "mni_icbm152_nlin_asym_09c_Lhemi.gii").touch()
    (icbm / "mni_icbm152_nlin_asym_09c_Rhemi.gii").touch()
    monkeypatch.setattr(tiles_module, "ICBM_MESH_DIR_FALLBACK", str(icbm))

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

    def test_explicit_root_message_names_path_and_option(self, tmp_path, tiles_env, pipeline_caplog):
        """REQ-TILESROOT-01: explicit champollion_data_root lacking the region graph."""
        modules, empty_root, crops = tiles_env
        with patch.multiple(brainvisa_imports, **_as_module_attrs(modules)):
            snaps = generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), champollion_data_root=str(empty_root)
            )
        assert snaps == []
        message = _graph_warning(pipeline_caplog, _expected_graph_path(empty_root))
        assert OPTION in message, f"no mention of {OPTION} in the warning: {message}"

    def test_config_default_root_message_names_path_and_option(self, tmp_path, tiles_env, pipeline_caplog):
        """REQ-TILESROOT-01: cortical_tiles config default root lacking the region graph (TASK-201 case)."""
        modules, empty_root, crops = tiles_env
        with patch.multiple(brainvisa_imports, **_as_module_attrs(modules)):
            snaps = generate_tiles_snapshot(str(crops), str(tmp_path / "tiles.png"))
        assert snaps == []
        message = _graph_warning(pipeline_caplog, _expected_graph_path(empty_root))
        assert OPTION in message, f"no mention of {OPTION} in the warning: {message}"

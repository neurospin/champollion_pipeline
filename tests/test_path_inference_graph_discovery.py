"""Stage 2 graph discovery on the main.py side: no fixed graph folder for a Morphologist input given without
--path_to_graph, and no --path_to_graph option on the stage 2 command line then (TASK-ARGDISC-BDRABCZUK-1601F53514C5).

- REQ-ARGDISC-BDRABCZUK-3CA4DDF7F6C7: path inference sets no graph folder path, whatever the folder names.
- REQ-ARGDISC-BDRABCZUK-EDF830CF0DDF: with no graph folder path, stage 2 starts without --path_to_graph.

Trees are names only (tests/morphologist_layout_fixtures.py): classic t1mri, morphologist-cli (no t1mri folder) and
SENIOR (several acquisitions and labelling sessions per subject, which v1.0.0 refused with AmbiguousGraphFolders).
"""

import logging
from pathlib import Path

from champollion_pipeline.config import DatasetConfig, PipelineConfig
from champollion_pipeline.path_inference import infer_paths
from champollion_pipeline.pipeline_stages import RunCorticalTilesStage
from champollion_pipeline.utils.input_kind import InputKind
from tests.morphologist_layout_fixtures import build_classic_tree, build_morphologist_cli_tree, build_senior_tree

PATH_TO_GRAPH_OPTION = "--path_to_graph"


def _inferred_graph_folder(input_path: Path) -> object:
    """path_to_graph of infer_paths on a Morphologist ``input_path`` with a default config."""
    return infer_paths(PipelineConfig(), str(input_path), InputKind.MORPHOLOGIST).dataset.path_to_graph


def test_classic_tree_sets_no_graph_folder(tmp_path: Path) -> None:
    """A classic t1mri tree below derivatives/morphologist-6.0 gets no graph folder path: discovery finds the graphs."""
    root = tmp_path / "dataset"
    build_classic_tree(root / "derivatives" / "morphologist-6.0")
    assert _inferred_graph_folder(root) is None


def test_morphologist_cli_tree_sets_no_graph_folder(tmp_path: Path) -> None:
    """A morphologist-cli tree (no t1mri folder) gets no graph folder path, never the default t1mri one."""
    subjects = build_morphologist_cli_tree(tmp_path / "dataset" / "derivatives" / "morphologist")
    assert _inferred_graph_folder(subjects) is None


def test_senior_tree_sets_no_graph_folder(tmp_path: Path) -> None:
    """A SENIOR subjects folder, graphs under several acquisition folders, gets no graph folder path and no error."""
    build_senior_tree(tmp_path / "senior")
    assert _inferred_graph_folder(tmp_path / "senior" / "derivatives" / "subjects_f") is None


def test_stage2_argv_holds_no_path_to_graph(tmp_path: Path) -> None:
    """With no graph folder path in the config, the stage 2 argv carries no --path_to_graph option at all."""
    dataset = DatasetConfig(morphologist_graphs=str(tmp_path / "graphs"), cortical_tiles_output=str(tmp_path / "out"))
    dataset.path_to_graph = None  # what path inference sets for a Morphologist input given without --path_to_graph
    stage = RunCorticalTilesStage("run_cortical_tiles", PipelineConfig(dataset=dataset), logging.getLogger(__name__))
    argv = stage.build_args()
    assert not [arg for arg in argv if arg.startswith(PATH_TO_GRAPH_OPTION)], argv

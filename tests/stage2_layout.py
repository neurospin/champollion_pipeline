"""Stage 2 fixtures of tests that take run_cortical_tiles.py through its input QC and its mocked crop step.

run() needs a subject holding one graph per hemisphere at <input>/<subject>/<path_to_graph>/<side>*.arg.
"""

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cortical_tiles.brainvisa import remove_ventricle as remove_ventricle_module

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.stage2_skeletons import SIDES, SKELETON_FILE

GRAPH_DIR = "graphs"
SKELETON_DIR = "skeleton"
SUBJECT = "subj01"
TEMPLATE_CONFIG = Path(__file__).resolve().parents[1] / "src/champollion_pipeline/resources/pipeline_loop_2mm.json"
CROP_STEMS = {"foldlabel": "label"}  # crop kinds whose files carry another stem than the kind name
# Signature of the real cortical_tiles remove_ventricle(), captured before any
# test patches it, so fakes can read its arguments however they are passed.
REMOVE_VENTRICLE_SIGNATURE = inspect.signature(remove_ventricle_module.remove_ventricle)
CROP_MARKER = "crops_complete.marker"
QC_OPTION = "--sk_qc_path"


@dataclass
class StageRun:
    """What run() did with the crop subprocess mocked."""

    status: int
    crop_commands: list[list[str]]
    rows: dict[str, dict]
    crop_qc_path: Path
    config: dict

    @property
    def crop_calls(self) -> int:
        return len(self.crop_commands)

    def crop_qc_rows(self) -> list[tuple[str, int]]:
        assert self.crop_qc_path.is_file(), f"{self.crop_qc_path} was not written"
        return [(row["participant_id"], row["qc"]) for row in read_qc_tsv(self.crop_qc_path)]

    def crop_qc_option(self) -> Path:
        """The --sk_qc_path value of the one crop command."""
        (command,) = self.crop_commands
        assert QC_OPTION in command, command
        return Path(command[command.index(QC_OPTION) + 1])

    def qc(self, subject: str) -> int:
        return self.rows[subject]["qc"]

    def reason(self, subject: str) -> str:
        return self.rows[subject]["reason"] or ""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_whole_brain_functions(monkeypatch):
    """Prevent the real whole-brain submodule calls from running in tests that
    don't care about them (REQ-WHOLEBRAIN-01 runs unconditionally in run()).
    TestWholeBrainGeneration patches these itself, taking precedence within
    its own `with patch(...)` blocks.
    """
    monkeypatch.setattr(
        "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes",
        MagicMock(),
    )
    monkeypatch.setattr(
        "champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle",
        MagicMock(),
    )


def add_graphs(input_dir, subject=SUBJECT, graph_dir=GRAPH_DIR, sides="LR"):
    """Lay out ``subject`` under ``input_dir``: ``graph_dir`` holding a graph per hemisphere of ``sides``."""
    folder = Path(input_dir) / subject / graph_dir
    folder.mkdir(parents=True, exist_ok=True)
    for side in sides:
        (folder / f"{side}{subject}.arg").touch()


def add_skeletons(input_dir, subject=SUBJECT, skeleton_dir=SKELETON_DIR):
    """Lay both skeletons of ``subject`` in ``skeleton_dir``, for a run the conftest stub does not reach."""
    folder = Path(input_dir) / subject / skeleton_dir
    folder.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        (folder / SKELETON_FILE.format(side=side, subject=subject)).touch()


def write_whole_brain_volumes(*args, **kwargs):
    """What the real remove_ventricle leaves on disk: an output volume per fused volume of its source directory."""
    bound = REMOVE_VENTRICLE_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    params = bound.arguments
    side = params["side"]
    output_dir = Path(params["output_dir"]) / side
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{side}{params['src_filename']}"
    for path in (Path(params["src_dir"]) / side).glob(f"{stem}*.nii.gz"):
        subject = path.name[len(stem) : -len(".nii.gz")]
        (output_dir / f"{side}{params['output_filename']}{subject}.nii.gz").touch()


def crop_array(crops_dir, region, side, kind):
    """Path of the <side><stem>.npy file of ``kind`` in the mask folder of ``region``."""
    return Path(crops_dir) / region / "mask" / f"{side}{CROP_STEMS.get(kind, kind)}.npy"


def crop_csv(crops_dir, region, side, kind):
    """Path of the <side><stem>_subject.csv file of ``kind`` in the mask folder of ``region``."""
    array = crop_array(crops_dir, region, side, kind)
    return array.with_name(f"{array.stem}_subject.csv")


def add_crops(crops_dir, subjects, regions, kinds=("skeleton",), sides="LR"):
    """Lay the array of each of ``kinds`` per region and side, with a subject CSV naming ``subjects``.

    The kind mask gets its array alone: _generate_mask_npys writes no subject CSV.
    """
    for region in regions:
        for side in sides:
            for kind in kinds:
                array = crop_array(crops_dir, region, side, kind)
                array.parent.mkdir(parents=True, exist_ok=True)
                np.save(array, np.zeros((len(subjects), 1)))
                if kind != "mask":
                    crop_csv(crops_dir, region, side, kind).write_text("\n".join(["Subject", *subjects]) + "\n")


def add_crop_marker(crops_dir):
    """Lay the completion marker of a finished crop run in ``crops_dir``; return its path."""
    marker = Path(crops_dir) / CROP_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return marker


def stage_dirs(root, graph_dir=GRAPH_DIR):
    """Create <root>/output and <root>/input, holding one subject: skeletons, and graphs in ``graph_dir``."""
    input_dir, output_dir = Path(root) / "input", Path(root) / "output"
    add_graphs(input_dir, graph_dir=graph_dir)
    add_skeletons(input_dir)
    output_dir.mkdir()
    return input_dir, output_dir


def stage_dirs_on_template(root, config_overrides=None):
    """stage_dirs, plus <root>/output/pipeline_loop_2mm.json seeded from the shipped template; return all three."""
    input_dir, output_dir = stage_dirs(root)
    config = json.loads(TEMPLATE_CONFIG.read_text())
    config.update(config_overrides or {})
    config_path = output_dir / TEMPLATE_CONFIG.name
    config_path.write_text(json.dumps(config))
    return input_dir, output_dir, config_path


def parsed_script(input_dir, output_dir, *extra_args):
    """A RunCorticalTiles parsed on the two paths, the graph and skeleton options and ``extra_args``."""
    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            GRAPH_DIR,
            "--path_sk_with_hull",
            SKELETON_DIR,
            *extra_args,
        ]
    )
    return script


def run_mocked(script):
    """Run ``script`` with its paths taken as valid and its commands mocked; return the execute_command mock."""
    with (
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", return_value=0) as mock_exec,
    ):
        script.run()
    return mock_exec


def run_on_config(input_dir, output_dir, config_data, *extra_args):
    """run_mocked on a pipeline_loop_2mm.json holding ``config_data``; return that file's content after the run."""
    config_path = Path(output_dir) / TEMPLATE_CONFIG.name
    config_path.write_text(json.dumps(config_data))
    run_mocked(parsed_script(input_dir, output_dir, *extra_args))
    return json.loads(config_path.read_text())

#!/usr/bin/env python3
"""Stage 2 input QC of the skeletons (TASK-PRECHECK-BDRABCZUK-145D053A8EEE).

User decision 2026-11-05, "L and R; add skeleton QC". Morphologist writes the skeleton of hemisphere S to
<path_sk_with_hull>/Sskeleton_<subject>.nii.gz, the name cortical_tiles reads. Each test goes through
RunCorticalTiles.run() with the crop step mocked and reads input_qc.tsv, so it holds wherever the check sits.

REQ-PRECHECK-BDRABCZUK-22CEB8ED1855: no file matches: missing_skeleton_S.
REQ-PRECHECK-BDRABCZUK-81244C77FFC6: no match holds a non-zero voxel: unusable_skeleton_S.
REQ-PRECHECK-BDRABCZUK-BE65CA1049B5: a file aims.read refuses holds no non-zero voxel.
REQ-PRECHECK-BDRABCZUK-1A141207E27C: order of the reasons in the reason cell.
REQ-PRECHECK-BDRABCZUK-A93EBB1FDFDC: the pre-check and input_qc.tsv name the same missing skeletons.
REQ-PRECHECK-BDRABCZUK-65453DD1CA00: crop_qc.tsv, written exactly when a skeleton reason excludes a subject.
REQ-PRECHECK-BDRABCZUK-855E37DF2C71: pipeline_loop_2mm.json names crop_qc.tsv.
REQ-PRECHECK-BDRABCZUK-9F5806895C50: the crop command names crop_qc.tsv, so the exclusion holds whatever the config.
REQ-NTUI-BDRABCZUK-E78CDFDFBF9A: a run writing no crop_qc.tsv and given no --sk_qc_path passes no such option.
REQ-NTUI-BDRABCZUK-0401777EE012: a relative --sk_qc_path stays the input QC file of such a run.
REQ-PRECHECK-BDRABCZUK-042A92B1A567: such a run ends partial (user decision "Exclude from crops").
REQ-PRECHECK-BDRABCZUK-8B2157BA956B: the DatasetConfig default names the segmentation folder.
"""

import importlib
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.file_indexer.pipeline_checks import SubjectEligibilityChecker
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles, aims
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.stage2_layout import QC_OPTION, TEMPLATE_CONFIG, StageRun, add_graphs
from tests.stage2_skeletons import (  # noqa: F401 - the fixture overrides the conftest stub in this file
    SIDES,
    SKELETON_FILE,
    _stub_stage2_skeletons,
    skeleton_volume,
)

REPO_DIR = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_DIR / "main.py"
CROP_SCRIPT_DIR = REPO_DIR / "external/cortical_tiles/cortical_tiles/brainvisa"

LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"
CROP_SCRIPT = "generate_sulcal_regions.py"
QC_FILE = "input_qc.tsv"
CROP_QC_FILE = "crop_qc.tsv"
GRAPH_FOLDER = "graphs"
SKELETON_FOLDER = "segmentation"
SUBJECT = "sub-002"
OTHER_SUBJECT = "sub-001"
# The tree of the IXI run on rosette.
ANALYSIS_PATH = "t1mri/default_acquisition/default_analysis"
REAL_GRAPH_FOLDER = f"{ANALYSIS_PATH}/folds/3.1"
REAL_SKELETON_FOLDER = f"{ANALYSIS_PATH}/segmentation"
ANY_ACQUISITION_SKELETON_PATH = "t1mri/*/default_analysis/segmentation"


@dataclass
class _Stage:
    """A stage 2 input tree, and how aims.read answers on its skeletons (by file name)."""

    root: Path
    unreadable: dict[str, Exception] = field(default_factory=dict)
    empty: set[str] = field(default_factory=set)

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    def add_subject(
        self,
        subject: str,
        *,
        graphs: str = SIDES,
        skeletons: str = SIDES,
        graph_folder: str = GRAPH_FOLDER,
        skeleton_folder: str = SKELETON_FOLDER,
    ) -> None:
        """Lay ``subject``: a graph per hemisphere of ``graphs``, a skeleton per hemisphere of ``skeletons``."""
        add_graphs(self.input_dir, subject, graph_folder, graphs)
        (self.input_dir / subject).mkdir(parents=True, exist_ok=True)
        for side in skeletons:
            self.add_file(subject, skeleton_folder, SKELETON_FILE.format(side=side, subject=subject))

    def add_file(self, subject: str, folder: str, name: str) -> None:
        path = self.input_dir / subject / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def read_volume(self, path: str, *_args, **_kwargs):
        """aims.read on this tree."""
        name = Path(path).name
        if name in self.unreadable:
            raise self.unreadable[name]
        return skeleton_volume(0) if name in self.empty else skeleton_volume()

    def run(  # noqa: PLR0913 - one keyword per stage case
        self,
        extra_args: tuple[str, ...] = (),
        *,
        path_to_graph: str = GRAPH_FOLDER,
        path_sk_with_hull: str = SKELETON_FOLDER,
        crop_status: int = 0,
        has_finished_crops: bool = False,
        config_overrides: dict | None = None,
    ) -> StageRun:
        """Run the stage on the template config with ``config_overrides``.

        The crop subprocess returns ``crop_status``, ``has_finished_crops`` skips it.
        """
        output_dir = self.root / "output"
        output_dir.mkdir(exist_ok=True)
        config_path = output_dir / TEMPLATE_CONFIG.name
        template = json.loads(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
        config_path.write_text(json.dumps({**template, **(config_overrides or {})}), encoding="utf-8")
        crop_commands = []

        def execute(command, *_args, **_kwargs):
            if any(CROP_SCRIPT in str(part) for part in command):
                crop_commands.append([str(part) for part in command])
            return crop_status

        script = RunCorticalTiles()
        script.parse_args(
            [
                str(self.input_dir),
                str(output_dir),
                "--path_to_graph",
                path_to_graph,
                "--path_sk_with_hull",
                path_sk_with_hull,
                *extra_args,
            ]
        )
        with (
            patch.object(aims, "read", side_effect=self.read_volume),
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "_preflight_check", return_value=not has_finished_crops),
            patch.object(script, "execute_command", side_effect=execute),
            patch.object(script, "_generate_mask_npys"),
            patch("champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes"),
            patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle"),
        ):
            status = script.run()
        derivatives = output_dir / DERIVATIVES_FOLDER
        rows = {record["participant_id"]: record for record in read_qc_tsv(derivatives / QC_FILE)}
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return StageRun(status, crop_commands, rows, derivatives / CROP_QC_FILE, config)


@pytest.fixture
def stage(tmp_path) -> _Stage:
    return _Stage(tmp_path)


def _skeleton(side: str, subject: str = SUBJECT) -> str:
    return SKELETON_FILE.format(side=side, subject=subject)


def _other_side(side: str) -> str:
    return SIDES.replace(side, "")


def _sk_qc_file(folder: Path, rows: dict[str, int]) -> Path:
    """Write a --sk_qc_path file in ``folder``, ``rows`` mapping each participant_id to its qc."""
    path = folder / "sk_qc.tsv"
    lines = ["participant_id\tqc", *(f"{subject}\t{qc}" for subject, qc in rows.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ====================== REQ-PRECHECK-BDRABCZUK-22CEB8ED1855: missing skeleton ======================


@pytest.mark.parametrize("side", SIDES)
def test_absent_skeleton_gets_its_reason(stage, side):
    """The other hemisphere's skeleton does not stand in, and an absent file is not reported unusable."""
    stage.add_subject(SUBJECT, skeletons=_other_side(side))
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run()

    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, f"missing_skeleton_{side}")


def test_both_skeletons_absent(stage):
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run()

    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, "missing_skeleton_L;missing_skeleton_R")


def test_other_hemisphere_volume_is_not_a_skeleton(stage):
    """Morphologist writes other R*.nii.gz volumes beside the skeleton: none of them stands in for it."""
    stage.add_subject(SUBJECT, skeletons="L")
    stage.add_subject(OTHER_SUBJECT)
    stage.add_file(SUBJECT, SKELETON_FOLDER, f"Rroots_{SUBJECT}.nii.gz")
    stage.add_file(SUBJECT, SKELETON_FOLDER, f"Rcortex_{SUBJECT}.nii.gz")

    run = stage.run()

    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, "missing_skeleton_R")


def test_wildcard_skeleton_path(stage):
    """path_sk_with_hull reads '*' as a wildcard, as the streaming stages pass it."""
    stage.add_subject(OTHER_SUBJECT, skeleton_folder="t1mri/acq-1/default_analysis/segmentation")
    stage.add_subject(SUBJECT, skeleton_folder="t1mri/acq-1/other_analysis/segmentation")

    run = stage.run(path_sk_with_hull=ANY_ACQUISITION_SKELETON_PATH)

    assert (run.qc(OTHER_SUBJECT), run.reason(OTHER_SUBJECT)) == (1, "")
    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, "missing_skeleton_L;missing_skeleton_R")


def test_glob_characters_in_the_skeleton_path_are_literal(stage):
    """'*' is the only wildcard: seg[1] names the folder seg[1], never seg1."""
    stage.add_subject(OTHER_SUBJECT, skeleton_folder="seg[1]")
    stage.add_subject(SUBJECT, skeleton_folder="seg1")

    run = stage.run(path_sk_with_hull="seg[1]")

    assert (run.qc(OTHER_SUBJECT), run.reason(OTHER_SUBJECT)) == (1, "")
    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, "missing_skeleton_L;missing_skeleton_R")


def test_real_tree_passes(stage):
    """The tree of the IXI run: segmentation/Lskeleton_sub-002.nii.gz and its R twin."""
    stage.add_subject(SUBJECT, graph_folder=REAL_GRAPH_FOLDER, skeleton_folder=REAL_SKELETON_FOLDER)

    run = stage.run(path_to_graph=REAL_GRAPH_FOLDER, path_sk_with_hull=REAL_SKELETON_FOLDER)

    assert (run.status, run.qc(SUBJECT), run.reason(SUBJECT)) == (0, 1, "")


def test_missing_skeleton_emits_an_error_record(stage, pipeline_caplog):
    stage.add_subject(SUBJECT, skeletons="R")
    stage.add_subject(OTHER_SUBJECT)

    stage.run()

    messages = [r.getMessage() for r in pipeline_caplog.records if r.name == LOGGER_NAME and r.levelno == logging.ERROR]
    assert [text for text in messages if SUBJECT in text and "missing_skeleton_L" in text]


def test_no_subject_with_a_skeleton_stops_the_stage(stage):
    stage.add_subject(SUBJECT, skeletons="")

    run = stage.run()

    assert (run.status, run.crop_calls) == (1, 0)


def test_subject_with_skeletons_keeps_the_stage_running(stage):
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run()

    assert (run.status, run.crop_calls) == (3, 1)
    assert (run.qc(SUBJECT), run.qc(OTHER_SUBJECT)) == (0, 1)


# ====================== REQ-PRECHECK-BDRABCZUK-81244C77FFC6: no non-zero voxel ======================


@pytest.mark.parametrize("side", SIDES)
def test_empty_volume_is_unusable(stage, side):
    stage.add_subject(SUBJECT)
    stage.add_subject(OTHER_SUBJECT)
    stage.empty.add(_skeleton(side))

    run = stage.run()

    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, f"unusable_skeleton_{side}")
    assert (run.qc(OTHER_SUBJECT), run.reason(OTHER_SUBJECT)) == (1, "")


def test_one_usable_match_is_enough(stage):
    """Beside an empty match, a match holding a non-zero voxel passes the hemisphere."""
    stage.add_subject(SUBJECT, skeleton_folder="t1mri/acq-1/default_analysis/segmentation")
    stage.add_file(SUBJECT, "t1mri/acq-2/default_analysis/segmentation", "Lskeleton_empty.nii.gz")
    stage.empty.add("Lskeleton_empty.nii.gz")

    run = stage.run(path_sk_with_hull=ANY_ACQUISITION_SKELETON_PATH)

    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (1, "")


# ====================== REQ-PRECHECK-BDRABCZUK-BE65CA1049B5: a file aims.read refuses ======================


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("error", [OSError("truncated file"), RuntimeError("unknown format")], ids=["os", "runtime"])
def test_unreadable_skeleton_is_unusable(stage, side, error):
    stage.add_subject(SUBJECT)
    stage.add_subject(OTHER_SUBJECT)
    stage.unreadable[_skeleton(side)] = error

    run = stage.run()

    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, f"unusable_skeleton_{side}")


def test_unreadable_and_empty_matches_are_unusable(stage):
    stage.add_subject(SUBJECT, skeletons="R")
    stage.add_subject(OTHER_SUBJECT)
    stage.add_file(SUBJECT, SKELETON_FOLDER, "Lskeleton_a.nii.gz")
    stage.add_file(SUBJECT, SKELETON_FOLDER, "Lskeleton_b.nii.gz")
    stage.unreadable["Lskeleton_a.nii.gz"] = OSError("truncated file")
    stage.empty.add("Lskeleton_b.nii.gz")

    run = stage.run()

    assert (run.qc(SUBJECT), run.reason(SUBJECT)) == (0, "unusable_skeleton_L")


def test_unreadable_skeleton_leaves_the_other_subjects(stage):
    """The refusal stays on its subject: the stage goes on with the others."""
    stage.add_subject(SUBJECT)
    stage.add_subject(OTHER_SUBJECT)
    stage.unreadable[_skeleton("L")] = OSError("truncated file")

    run = stage.run()

    assert (run.status, run.crop_calls) == (3, 1)
    assert (run.qc(SUBJECT), run.qc(OTHER_SUBJECT)) == (0, 1)


# ====================== REQ-PRECHECK-BDRABCZUK-1A141207E27C: order of the reasons ======================


@pytest.mark.parametrize(
    ("graphs", "skeletons", "empty", "expected"),
    [
        pytest.param(
            "",
            "",
            "",
            "missing_graph_L;missing_graph_R;missing_skeleton_L;missing_skeleton_R;excluded_by_sk_qc",
            id="each-check-fails",
        ),
        pytest.param("LR", "R", "R", "missing_skeleton_L;unusable_skeleton_R;excluded_by_sk_qc", id="missing-unusable"),
        pytest.param("LR", "L", "L", "unusable_skeleton_L;missing_skeleton_R;excluded_by_sk_qc", id="unusable-missing"),
        pytest.param("L", "", "", "missing_graph_R;missing_skeleton_L;missing_skeleton_R;excluded_by_sk_qc", id="one"),
    ],
)
def test_reason_order(stage, graphs, skeletons, empty, expected):
    stage.add_subject(SUBJECT, graphs=graphs, skeletons=skeletons)
    stage.add_subject(OTHER_SUBJECT)
    stage.empty.update(_skeleton(side) for side in empty)
    sk_qc = _sk_qc_file(stage.root, {SUBJECT: 0, OTHER_SUBJECT: 1})

    run = stage.run((QC_OPTION, str(sk_qc)))

    assert run.reason(SUBJECT) == expected


# ====================== REQ-PRECHECK-BDRABCZUK-A93EBB1FDFDC: agreement with the pre-check ======================


@pytest.fixture
def pre_check_patterns(tmp_path) -> list[str]:
    """Stage 2 patterns of main.py on the real tree."""
    spec = importlib.util.spec_from_file_location("_champollion_main_skeleton_qc_under_test", MAIN_PY)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(tmp_path / "outputs"),
    )
    config.dataset.path_to_graph = REAL_GRAPH_FOLDER  # noqa: V101
    config.dataset.path_sk_with_hull = REAL_SKELETON_FOLDER  # noqa: V101
    stage_under_test = main_module.RunCorticalTilesStage("run_cortical_tiles", config, logging.getLogger(LOGGER_NAME))
    return stage_under_test.required_file_patterns()


@pytest.mark.parametrize("skeletons", ["", "L", "R"], ids=["no-skeleton", "left-only", "right-only"])
def test_pre_check_agrees_with_input_qc_on_skeletons(stage, pre_check_patterns, skeletons):
    """Each absent skeleton is named by both, in a folder where other volumes of each hemisphere sit."""
    folders = {"graph_folder": REAL_GRAPH_FOLDER, "skeleton_folder": REAL_SKELETON_FOLDER}
    stage.add_subject(OTHER_SUBJECT, **folders)
    stage.add_subject(SUBJECT, skeletons=skeletons, **folders)
    for side in SIDES:
        stage.add_file(SUBJECT, REAL_SKELETON_FOLDER, f"{side}roots_{SUBJECT}.nii.gz")

    report = SubjectEligibilityChecker(str(stage.input_dir), pre_check_patterns).check_subject_patterns()
    run = stage.run(path_to_graph=REAL_GRAPH_FOLDER, path_sk_with_hull=REAL_SKELETON_FOLDER)

    missing = {scan.subject: patterns for scan, patterns in report.ineligible.items()}
    absent = set(SIDES) - set(skeletons)
    assert set(missing) == {SUBJECT}
    assert set(missing[SUBJECT]) == {f"{REAL_SKELETON_FOLDER}/{side}skeleton*.nii.gz" for side in absent}
    assert set(run.reason(SUBJECT).split(";")) == {f"missing_skeleton_{side}" for side in absent}


# ====================== REQ-PRECHECK-BDRABCZUK-65453DD1CA00: crop_qc.tsv ======================


@pytest.mark.parametrize("has_finished_crops", [False, True], ids=["crop-step", "kept-crops"])
def test_crop_qc_holds_the_qc_1_rows(stage, has_finished_crops):
    """Finished crops skip the crop step: the run still writes crop_qc.tsv."""
    stage.add_subject(SUBJECT, skeletons="L")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run(has_finished_crops=has_finished_crops)

    assert run.crop_qc_rows() == [(OTHER_SUBJECT, 1)]


def test_crop_qc_leaves_out_each_qc_0_subject(stage):
    """Beside the skeleton reason, the graph and the --sk_qc_path file reasons keep a subject out too."""
    no_graph, excluded = "sub-003", "sub-004"
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(no_graph, graphs="L")
    stage.add_subject(excluded)
    stage.add_subject(OTHER_SUBJECT)
    sk_qc = _sk_qc_file(stage.root, {SUBJECT: 1, no_graph: 1, excluded: 0, OTHER_SUBJECT: 1})

    run = stage.run((QC_OPTION, str(sk_qc)))

    assert run.crop_qc_rows() == [(OTHER_SUBJECT, 1)]


def test_no_crop_qc_without_a_skeleton_reason(stage):
    """A run with no skeleton reason hands cortical_tiles what it handed it before, a missing graph included."""
    stage.add_subject(SUBJECT, graphs="L")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run()

    assert not run.crop_qc_path.exists()
    assert run.config["skel_qc_path"] == ""
    assert QC_OPTION not in run.crop_commands[0]


# ====================== REQ-PRECHECK-BDRABCZUK-855E37DF2C71: pipeline_loop_2mm.json ======================


@pytest.mark.parametrize("has_finished_crops", [False, True], ids=["crop-step", "kept-crops"])
def test_config_names_crop_qc(stage, has_finished_crops):
    """Finished crops skip the crop step: the config of the run still names crop_qc.tsv."""
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run(has_finished_crops=has_finished_crops)

    assert Path(run.config["skel_qc_path"]) == run.crop_qc_path


# ====================== REQ-PRECHECK-BDRABCZUK-9F5806895C50: crop command ======================


def _user_qc_args(stage: _Stage, has_user_file: bool) -> tuple[str, ...]:
    """The --sk_qc_path arguments of a run: a file passing both subjects, or none."""
    return (QC_OPTION, str(_sk_qc_file(stage.root, {SUBJECT: 1, OTHER_SUBJECT: 1}))) if has_user_file else ()


@pytest.mark.parametrize("has_user_file", [True, False], ids=["user-file", "no-user-file"])
def test_crop_command_names_crop_qc(stage, has_user_file):
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run(_user_qc_args(stage, has_user_file))

    assert run.crop_qc_option() == run.crop_qc_path


@pytest.fixture
def crop_script(monkeypatch):
    """The real generate_sulcal_regions module, its regions left unrun; it imports its sibling as a top-level module."""
    monkeypatch.syspath_prepend(str(CROP_SCRIPT_DIR))
    module = importlib.import_module("generate_sulcal_regions")
    monkeypatch.setattr(module, "_run_regions", lambda *_args: set())
    return module


def _subjects_kept_by_crop_step(crop_script, run: StageRun, subjects: list[str]) -> list[str]:
    """Resolve the config as generate_sulcal_regions.py does on the crop command, then select ``subjects`` by QC."""
    (command,) = run.crop_commands
    sk_qc_path = command[command.index(QC_OPTION) + 1] if QC_OPTION in command else ""
    config, _ = crop_script.generate_sulcal_regions(
        regions=[],
        sides=[],
        input_types=[],
        path_dataset=command[command.index("-d") + 1],
        verbose=0,
        output_dir=None,
        path_to_graph=GRAPH_FOLDER,
        path_sk_with_hull=SKELETON_FOLDER,
        sk_qc_path=sk_qc_path,
        njobs=1,
    )
    return crop_script.select_good_qc(subjects, config["skel_qc_path"])


@pytest.mark.parametrize("has_user_file", [True, False], ids=["user-file", "no-user-file"])
@pytest.mark.parametrize(
    "config_overrides",
    [
        pytest.param({}, id="template-with-local"),
        pytest.param({"supervised_output_dir": "/synthetic/supervised"}, id="config-without-local"),
    ],
)
def test_exclusion_reaches_cortical_tiles_whatever_the_config(stage, crop_script, config_overrides, has_user_file):
    """With no "$local" left, cortical_tiles overwrites skel_qc_path by its option: the subject must stay out."""
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run(_user_qc_args(stage, has_user_file), config_overrides=config_overrides)

    assert _subjects_kept_by_crop_step(crop_script, run, [SUBJECT, OTHER_SUBJECT]) == [OTHER_SUBJECT]


# ====================== REQ-NTUI-BDRABCZUK-0401777EE012: relative --sk_qc_path ======================


def test_relative_sk_qc_path_on_a_skeleton_exclusion(stage, monkeypatch):
    """The relative file still feeds the input QC; cortical_tiles reads crop_qc.tsv, by its absolute path."""
    launch_dir = stage.root / "launch"
    launch_dir.mkdir()
    _sk_qc_file(launch_dir, {SUBJECT: 0, OTHER_SUBJECT: 1})
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)
    monkeypatch.chdir(launch_dir)

    run = stage.run((QC_OPTION, "sk_qc.tsv"))

    assert run.reason(SUBJECT) == "missing_skeleton_L;missing_skeleton_R;excluded_by_sk_qc"
    assert run.crop_qc_option() == run.crop_qc_path


# ====================== REQ-PRECHECK-BDRABCZUK-042A92B1A567: the run ends partial ======================


@pytest.mark.parametrize(
    ("skeletons", "empty", "reason"),
    [
        pytest.param("L", "", "missing_skeleton_R", id="missing"),
        pytest.param("LR", "R", "unusable_skeleton_R", id="unusable"),
    ],
)
def test_skeleton_exclusion_ends_partial(stage, skeletons, empty, reason):
    """Exit 3, with the reason in input_qc.tsv."""
    stage.add_subject(SUBJECT, skeletons=skeletons)
    stage.add_subject(OTHER_SUBJECT)
    stage.empty.update(_skeleton(side) for side in empty)

    run = stage.run()

    assert (run.status, run.reason(SUBJECT)) == (3, reason)


def test_skeleton_exclusion_ends_partial_on_existing_crops(stage):
    """Finished crops skip the crop step; the run on an excluded subject stays partial."""
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run(has_finished_crops=True)

    assert (run.status, run.crop_calls) == (3, 0)


def test_failed_crop_status_stays(stage):
    """A failed crop step keeps its status: only a run that would return 0 turns partial."""
    stage.add_subject(SUBJECT, skeletons="")
    stage.add_subject(OTHER_SUBJECT)

    run = stage.run(crop_status=1)

    assert run.status == 1


# ====================== REQ-PRECHECK-BDRABCZUK-8B2157BA956B: default skeleton folder ======================


def test_default_skeleton_path_is_the_segmentation_folder():
    assert DatasetConfig().path_sk_with_hull == REAL_SKELETON_FOLDER

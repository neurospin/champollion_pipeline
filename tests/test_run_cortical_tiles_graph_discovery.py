"""Stage 2 hands each subject's graph files to cortical_tiles (TASK-ARGDISC-BDRABCZUK-1601F53514C5).

Without --path_to_graph, run_cortical_tiles lists each subject folder's graphs by graph discovery and passes them to
generate_sulcal_regions as a --graphs map file (cortical_tiles dev 6e01c5b: {subject folder name: {"L": [.arg],
"R": [.arg]}}, exclusive with a path_to_graph). A given --path_to_graph keeps today's behaviour.

- REQ-ARGDISC-BDRABCZUK-86F0D7690C07: a command line without --path_to_graph is accepted.
- REQ-ARGDISC-BDRABCZUK-50ACE43F2815: a --graphs file maps each subject folder name to absolute graph paths per side.
- REQ-ARGDISC-BDRABCZUK-B11227A6DA6D: no graph folder path on the crop command line nor in its JSON configuration.
- REQ-ARGDISC-BDRABCZUK-238665BDCA47: a labelled graph is listed in preference to an unlabelled one.
- REQ-ARGDISC-BDRABCZUK-44BACEC2B222: one WARNING names the subject folder whose listed graph is unlabelled.
- REQ-ARGDISC-BDRABCZUK-4D0B8D27B9A6: for now, the first graph of a side in natural path order (3TV2 before 3TV10).
- REQ-ARGDISC-BDRABCZUK-1B165C63FF65: several labelling sets without --graph_label stop with status 2 before cropping.
- REQ-ARGDISC-BDRABCZUK-04208DE4587D: that stop logs an ERROR naming the labelling sets.
- REQ-ARGDISC-BDRABCZUK-10F7B3E6AF8C: --graph_label picks the graphs of its labelling set.
- REQ-ARGDISC-BDRABCZUK-69B4D8B125B3: a given --path_to_graph reaches the crop command in place of a --graphs file.
- REQ-ARGDISC-BDRABCZUK-E26273C27C70: the input QC reads missing_graph_<side> from graph discovery.
- REQ-ARGDISC-BDRABCZUK-70D2199E103B: the ventricle removal gets no graph folder path.

Trees are names only (tests/morphologist_layout_fixtures.py); the crop subprocess is mocked, so no data is read.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.morphologist_layout_fixtures import (
    CLASSIC_SUBJECTS,
    CLI_SUBJECTS,
    build_classic_tree,
    build_morphologist_cli_tree,
    build_senior_tree,
)
from tests.stage2_layout import (  # noqa: F401
    GRAPH_DIR,
    TEMPLATE_CONFIG,
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture
    stage_dirs,
)
from tests.whole_brain_step import _make_skeletons, _RemoveVentricleRecorder

SKELETON_DIR = "skeleton"
CROP_SCRIPT = "generate_sulcal_regions.py"
GRAPHS_OPTION = "--graphs"
PATH_TO_GRAPH_OPTION = "--path_to_graph"
SIDES = ("L", "R")
CLI_LABELLED, CLI_UNLABELLED = CLI_SUBJECTS  # sub-01 holds sul-0_auto graphs, sub-02 unlabelled graphs only
SENIOR_SUBJECT = "sf0001"
SENIOR_SETS = ("default_session_auto", "long_session_auto")


@dataclass
class CropRun:
    """What run() did with the crop subprocess mocked: status, crop commands, and per command the --graphs map and the
    pipeline JSON configuration as they were when the command started."""

    status: int
    commands: list[list[str]] = field(default_factory=list)
    graphs_maps: list[dict | None] = field(default_factory=list)
    configs: list[dict] = field(default_factory=list)

    def single_map(self) -> dict:
        """The --graphs map of the one crop command."""
        assert len(self.commands) == 1, self.commands
        (graphs,) = self.graphs_maps
        assert graphs is not None, f"no {GRAPHS_OPTION} option in {self.commands[0]}"
        return graphs


def _parsed(input_dir: Path, output_dir: Path, *extra_args: str) -> RunCorticalTiles:
    """A RunCorticalTiles parsed on the two folders, a skeleton folder and ``extra_args``; no --path_to_graph unless
    ``extra_args`` holds one."""
    script = RunCorticalTiles()
    script.parse_args([str(input_dir), str(output_dir), "--path_sk_with_hull", SKELETON_DIR, *extra_args])
    return script


def _option_value(command: list[str], option: str) -> str | None:
    """The value after ``option`` in ``command`` (``option value`` or ``option=value``), None without the option."""
    for index, arg in enumerate(command):
        if arg == option:
            return command[index + 1]
        if arg.startswith(option + "="):
            return arg.partition("=")[2]
    return None


def _run(script: RunCorticalTiles, recorder: _RemoveVentricleRecorder | None = None) -> CropRun:
    """Run ``script`` with its paths taken as valid, the crop subprocess mocked and the mask arrays skipped.

    A SystemExit gives its code as the status. ``recorder`` replaces cortical_tiles' remove_ventricle when given.
    """
    run = CropRun(status=-1)
    config_path = Path(script.args.output).resolve() / TEMPLATE_CONFIG.name

    def execute(cmd, *args, **kwargs):
        command = [str(arg) for arg in cmd]
        if any(arg.endswith(CROP_SCRIPT) for arg in command):
            run.commands.append(command)
            graphs_file = _option_value(command, GRAPHS_OPTION)
            run.graphs_maps.append(json.loads(Path(graphs_file).read_text()) if graphs_file else None)
            run.configs.append(json.loads(config_path.read_text()) if config_path.is_file() else {})
        return 0

    patches = [
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", side_effect=execute),
        patch.object(script, "_generate_mask_npys"),
    ]
    if recorder is not None:
        patches.append(
            patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle", side_effect=recorder)
        )
    for active in patches:
        active.start()
    try:
        run.status = script.run()
    except SystemExit as error:
        run.status = error.code
    finally:
        for active in reversed(patches):
            active.stop()
    return run


def _output(tmp_path: Path) -> Path:
    """An empty output folder below tmp_path."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def _cli_input(tmp_path: Path) -> Path:
    """A morphologist-cli subjects folder: qc/qc.tsv, sub-01 (labelled and unlabelled graphs), sub-02 (unlabelled)."""
    return build_morphologist_cli_tree(tmp_path / "morphologist")


def _cli_graph(subjects: Path, subject: str, side: str, labelled: bool) -> str:
    """Absolute path of a morphologist-cli graph of the fixture tree."""
    folds = subjects / subject / "ses-1" / "run-0" / "ana-0" / "folds" / "3.1"
    prefix = f"{subject}_ses-1_run-0"
    if labelled:
        return str((folds / "sul-0_auto" / f"{prefix}_sul-0_hemi-{side}_auto.arg").resolve())
    return str((folds / f"{prefix}_hemi-{side}.arg").resolve())


def _senior_input(tmp_path: Path) -> Path:
    """The subjects_f folder of a SENIOR tree: sf0001, acquisitions 3TV0 to 3TV10, two labelling sets each."""
    build_senior_tree(tmp_path / "senior")
    return tmp_path / "senior" / "derivatives" / "subjects_f"


def _records(pipeline_caplog, caplog, level: int) -> list[str]:
    """Messages of the records at ``level`` from the champollion_pipeline loggers and any propagating logger."""
    records = [*pipeline_caplog.records, *caplog.records]
    return [record.getMessage() for record in records if record.levelno == level]


def test_command_line_without_path_to_graph_parses(tmp_path: Path) -> None:
    """--path_to_graph is optional: a command line without it parses, and leaves no graph folder path."""
    script = _parsed(tmp_path / "input", tmp_path / "output")
    assert script.args.path_to_graph is None


def test_graphs_map_names_each_cli_subject_folder(tmp_path: Path) -> None:
    """The map has one key per subject folder of a morphologist-cli input; the qc folder holding no graph is none."""
    graphs = _run(_parsed(_cli_input(tmp_path), _output(tmp_path))).single_map()
    assert sorted(graphs) == sorted(CLI_SUBJECTS)


def test_graphs_map_lists_classic_graphs_of_both_hemispheres(tmp_path: Path) -> None:
    """Each subject of a classic tree gets one absolute, existing graph path per hemisphere."""
    subjects = build_classic_tree(tmp_path / "morphologist-6.0")
    graphs = _run(_parsed(subjects, _output(tmp_path))).single_map()
    assert sorted(graphs) == sorted(CLASSIC_SUBJECTS)
    for subject in CLASSIC_SUBJECTS:
        for side in SIDES:
            (listed,) = graphs[subject][side]
            assert Path(listed).is_absolute(), listed
            assert Path(listed).is_file(), listed
            assert Path(listed).suffix == ".arg"
            assert (subjects / subject).resolve() in Path(listed).resolve().parents


def test_crop_command_holds_no_path_to_graph(tmp_path: Path) -> None:
    """The crop command carries no --path_to_graph: cortical_tiles refuses one beside a graphs map."""
    run = _run(_parsed(_cli_input(tmp_path), _output(tmp_path)))
    run.single_map()
    assert _option_value(run.commands[0], PATH_TO_GRAPH_OPTION) is None, run.commands[0]


def test_crop_config_holds_no_path_to_graph(tmp_path: Path) -> None:
    """The pipeline JSON configuration the crop command reads holds no graph folder path either."""
    run = _run(_parsed(_cli_input(tmp_path), _output(tmp_path)))
    run.single_map()
    (config,) = run.configs
    assert config.get("path_to_graph") is None, config.get("path_to_graph")


def test_labelled_graph_listed_in_preference(tmp_path: Path) -> None:
    """sub-01 holds an unlabelled and a sul-0_auto graph per side: the labelled one is listed, alone."""
    subjects = _cli_input(tmp_path)
    graphs = _run(_parsed(subjects, _output(tmp_path))).single_map()
    for side in SIDES:
        assert graphs[CLI_LABELLED][side] == [_cli_graph(subjects, CLI_LABELLED, side, labelled=True)]


def test_unlabelled_graph_listed_without_a_labelled_graph(tmp_path: Path) -> None:
    """sub-02 holds unlabelled graphs only: they are listed."""
    subjects = _cli_input(tmp_path)
    graphs = _run(_parsed(subjects, _output(tmp_path))).single_map()
    for side in SIDES:
        assert graphs[CLI_UNLABELLED][side] == [_cli_graph(subjects, CLI_UNLABELLED, side, labelled=False)]


def test_unlabelled_listed_graph_logs_a_warning(tmp_path: Path, pipeline_caplog, caplog) -> None:
    """A WARNING names sub-02, listed with unlabelled graphs; none names sub-01, listed with labelled graphs."""
    caplog.set_level(logging.INFO)
    run = _run(_parsed(_cli_input(tmp_path), _output(tmp_path)))
    run.single_map()
    warnings = _records(pipeline_caplog, caplog, logging.WARNING)
    assert [message for message in warnings if CLI_UNLABELLED in message], warnings
    assert not [message for message in warnings if CLI_LABELLED in message], warnings


def test_first_acquisition_in_natural_order_is_listed(tmp_path: Path) -> None:
    """Graphs of one labelling set under acquisitions 3TV2 and 3TV10: 3TV2 comes first in natural order, alone."""
    subjects, subject, session = tmp_path / "subjects", "s001", "default_session_auto"
    expected = {}
    for acquisition in ("3TV10", "3TV2"):
        folder = subjects / subject / "t1mri" / acquisition / "default_analysis" / "folds" / "3.1" / session
        folder.mkdir(parents=True)
        for side in SIDES:
            graph = folder / f"{side}{subject}_{session}.arg"
            graph.touch()
            if acquisition == "3TV2":
                expected[side] = [str(graph.resolve())]
    graphs = _run(_parsed(subjects, _output(tmp_path))).single_map()
    for side in SIDES:
        assert graphs[subject][side] == expected[side]


def test_several_labelling_sets_stop_with_status_2_before_cropping(tmp_path: Path) -> None:
    """A SENIOR subject labelled by two sets, no --graph_label: status 2 and no crop command."""
    run = _run(_parsed(_senior_input(tmp_path), _output(tmp_path)))
    assert run.status == 2
    assert run.commands == []


def test_several_labelling_sets_error_names_the_sets(tmp_path: Path, pipeline_caplog, caplog) -> None:
    """The stop logs an ERROR naming both labelling sets of the SENIOR subject."""
    caplog.set_level(logging.INFO)
    _run(_parsed(_senior_input(tmp_path), _output(tmp_path)))
    errors = _records(pipeline_caplog, caplog, logging.ERROR)
    assert [message for message in errors if all(name in message for name in SENIOR_SETS)], errors


def test_graph_label_picks_its_labelling_set(tmp_path: Path) -> None:
    """--graph_label long_session_auto lists that set's graph of the first acquisition, 3TV0."""
    subjects = _senior_input(tmp_path)
    graphs = _run(_parsed(subjects, _output(tmp_path), "--graph_label", "long_session_auto")).single_map()
    folds = subjects / SENIOR_SUBJECT / "t1mri" / "3TV0" / "default_analysis" / "folds" / "3.1"
    for side in SIDES:
        expected = folds / "long_session_auto" / f"{side}{SENIOR_SUBJECT}_long_session_auto.arg"
        assert graphs[SENIOR_SUBJECT][side] == [str(expected.resolve())]


def test_given_path_to_graph_reaches_the_crop_command(tmp_path: Path) -> None:
    """A given --path_to_graph is passed to generate_sulcal_regions unchanged."""
    input_dir, output_dir = stage_dirs(tmp_path)
    run = _run(_parsed(input_dir, output_dir, PATH_TO_GRAPH_OPTION, GRAPH_DIR))
    (command,) = run.commands
    assert _option_value(command, PATH_TO_GRAPH_OPTION) == GRAPH_DIR


def test_given_path_to_graph_passes_no_graphs_file(tmp_path: Path) -> None:
    """With a given --path_to_graph the crop command carries no --graphs file."""
    input_dir, output_dir = stage_dirs(tmp_path)
    run = _run(_parsed(input_dir, output_dir, PATH_TO_GRAPH_OPTION, GRAPH_DIR))
    (command,) = run.commands
    assert _option_value(command, GRAPHS_OPTION) is None, command


def _input_qc(output_dir: Path) -> dict[str, dict]:
    """input_qc.tsv rows of a run, by participant_id."""
    rows = read_qc_tsv(output_dir / DERIVATIVES_FOLDER / "input_qc.tsv")
    return {str(row["participant_id"]): row for row in rows}


def test_input_qc_names_the_hemisphere_without_graph(tmp_path: Path) -> None:
    """sub-02 without its right graph gets missing_graph_R; sub-01 holding both passes."""
    subjects = _cli_input(tmp_path)
    Path(_cli_graph(subjects, CLI_UNLABELLED, "R", labelled=False)).unlink()
    output_dir = _output(tmp_path)
    _run(_parsed(subjects, output_dir))
    rows = _input_qc(output_dir)
    assert rows[CLI_UNLABELLED]["qc"] == 0
    assert "missing_graph_R" in str(rows[CLI_UNLABELLED]["reason"])
    assert "missing_graph_L" not in str(rows[CLI_UNLABELLED]["reason"])
    assert rows[CLI_LABELLED]["qc"] == 1


def test_input_qc_passes_cli_subjects_holding_both_graphs(tmp_path: Path) -> None:
    """Both morphologist-cli subjects hold a graph per side below their folder: both pass the input QC."""
    output_dir = _output(tmp_path)
    _run(_parsed(_cli_input(tmp_path), output_dir))
    rows = _input_qc(output_dir)
    assert {subject: rows[subject]["qc"] for subject in CLI_SUBJECTS} == dict.fromkeys(CLI_SUBJECTS, 1)


@pytest.mark.usefixtures("_stub_whole_brain_functions")
def test_ventricle_removal_gets_no_graph_folder(tmp_path: Path) -> None:
    """On a classic tree labelled by default_session_auto, each remove_ventricle call gets no graph folder path."""
    subjects = build_classic_tree(tmp_path / "morphologist-6.0")
    output_dir = _output(tmp_path)
    for subject in CLASSIC_SUBJECTS:
        _make_skeletons(output_dir, subject)
    recorder = _RemoveVentricleRecorder()
    _run(_parsed(subjects, output_dir), recorder)
    assert recorder.path_to_graph_by_subject, "remove_ventricle processed no subject"
    for subject, values in recorder.path_to_graph_by_subject.items():
        assert values == [None] * len(values), subject

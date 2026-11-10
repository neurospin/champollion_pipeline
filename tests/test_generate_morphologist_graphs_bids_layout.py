"""--bids chain from stage 1 to stage 2 (TASK-INPUTKIND-BDRABCZUK-277A9E5DC923, DEF-INPUTKIND-BDRABCZUK-48F830CF3A22).

No Morphologist run here. Each test takes the arguments stage 1 builds, hands them to the real morphologist-cli with
--dry-run, and asks the installed file organisation model (soma.fom) where the graphs of that request go. The trees
stage 2 reads are built from those computed paths, so they follow what the installed tools ask for, not a copy of it.
"""

import ast
import csv
import json
import shutil
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import pytest
from hypothesis import Phase, example, given, settings
from hypothesis import strategies as st

from champollion_pipeline.file_indexer import ScanId, SubjectEligibilityChecker
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.utils.input_kind import InputKind, detect_input_kind
from champollion_pipeline.utils.qc_report import compute_participant_id, write_qc_tsv
from tests.test_generate_morphologist_graphs import _input_qc_path, _run_stage, _write_nifti

pytestmark = pytest.mark.unit  # noqa: V107

_FORMAT_ROOT = "/subjects"
_DEFAULT_ACQUISITION = "default_acquisition"
_NONOVERLAP = "morphologist-auto-nonoverlap-1.0"
_GRAPHS = "t1mri/*/default_analysis/folds/3.1"
_MORPHOLOGIST_CLI = "morphologist-cli"
_LABELS = st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=6)
_ANSWER = "PATHS "
_FORMAT_SERVER = f"""
import json
import sys

from soma.application import Application
from soma.fom import AttributesToPaths

application = Application("capsul", plugin_modules=["soma.fom"])
application.initialize()
rules = {{}}
for line in sys.stdin:
    name, query = json.loads(line)
    if name not in rules:
        rules[name] = AttributesToPaths(
            application.fom_manager.load_foms(name),
            selection={{}},
            directories={{"output": "{_FORMAT_ROOT}", "input": "{_FORMAT_ROOT}"}},
            preferred_formats=set(),
        )
    print({_ANSWER!r} + json.dumps([path for path, _ in rules[name].find_paths(query)]), flush=True)
"""


class _InstalledFormats:
    """Graph paths as the installed Morphologist file organisation models compute them, from a child interpreter."""

    def __init__(self, server):
        self._server = server

    def _paths(self, name, query):
        self._server.stdin.write(json.dumps([name, query]) + "\n")
        self._server.stdin.flush()
        for line in self._server.stdout:
            if line.startswith(_ANSWER):
                return json.loads(line.removeprefix(_ANSWER))
        raise AssertionError("the soma.fom child process stopped")

    def graphs(self, name, attributes):
        """Relative left then right graph paths of one scan; empty when the format has no rule for ``attributes``."""
        found = []
        for parameter in ("left_graph", "right_graph"):
            query = {"fom_process": "Morphologist", "fom_parameter": parameter, "center": "", **attributes}
            found += [PurePosixPath(path).relative_to(_FORMAT_ROOT) for path in self._paths(name, query)]
        return found


@pytest.fixture(scope="module")
def formats():
    """The installed file organisation models, loaded once per worker (about ten seconds).

    In a child process: tests/conftest.py replaces soma in this one, and soma.fom is then out of reach.
    """
    if subprocess.run([sys.executable, "-c", "import soma.fom"], capture_output=True, check=False).returncode:
        pytest.skip("soma.fom is not installed in this environment")
    with subprocess.Popen(
        [sys.executable, "-c", _FORMAT_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) as server:
        yield _InstalledFormats(server)
        server.stdin.close()


@pytest.fixture
def dirs(tmp_path):
    """Existing input and output directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


@dataclass(frozen=True)
class _Scan:
    """One file of a Morphologist request and the completion attributes that come with it."""

    file_name: str
    attributes: dict


def _write_inputs(input_dir, relative_paths):
    """Write one valid NIfTI file per path."""
    for relative_path in relative_paths:
        path = input_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_nifti(path)


def _qc_rows(output_dir):
    """Rows of the stage 1 input_qc.tsv as text, in file order: read_qc_tsv refuses a repeated participant_id."""
    with _input_qc_path(output_dir).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _stage_command(input_dir, output_dir, extra_args=("--bids",)):
    """The morphologist-cli argv stage 1 builds; morphologist-cli itself does not run."""
    _, execute = _run_stage(input_dir, output_dir, extra_args=extra_args)
    assert execute.called, f"stage 1 ran no morphologist-cli, see {_qc_rows(output_dir)}"
    return execute.call_args.args[0]


def _capsul_command(command, input_dir):
    """The capsul command the real morphologist-cli prints for ``command`` under --dry-run."""
    if shutil.which(_MORPHOLOGIST_CLI) is None:
        pytest.skip("morphologist-cli is not installed in this environment")
    separator = len(command) - 1 - command[::-1].index("--")
    dry_run = [*command[:separator], "--dry-run", *command[separator:]]
    printed = subprocess.run(dry_run, cwd=input_dir, capture_output=True, text=True, check=True).stdout
    return ast.literal_eval(printed.strip().splitlines()[-1])


def _as_values(text, count):
    """A capsul value as one entry per scan: an iterated list, or the same text ``count`` times."""
    value = ast.literal_eval(text) if text.startswith("[") else text
    return value if isinstance(value, list) else [value] * count


def _read_request(capsul):
    """(output format, scans) of a capsul command; a later --of or -a wins, as in capsul's option parser."""
    output_format = next(capsul[i + 1] for i in range(len(capsul) - 1, -1, -1) if capsul[i] == "--of")
    files = next(arg.partition("=")[2] for arg in capsul if arg.startswith("t1mri="))
    files = _as_values(files, 1)
    attributes = dict(capsul[i + 1].split("=", 1) for i, arg in enumerate(capsul) if arg == "-a")
    per_scan = {key: _as_values(value, len(files)) for key, value in attributes.items()}
    return output_format, [
        _Scan(Path(file).name, {key: values[i] for key, values in per_scan.items()}) for i, file in enumerate(files)
    ]


def _graphs_by_file(formats, input_dir, output_dir):
    """{input file name: [left graph, right graph]} of a --bids stage 1 run, relative to the subjects folder."""
    capsul = _capsul_command(_stage_command(input_dir, output_dir), input_dir)
    output_format, scans = _read_request(capsul)
    return {scan.file_name: formats.graphs(output_format, scan.attributes) for scan in scans}


def _graph_tree(formats, dirs, relative_paths):
    """Run stage 1 on ``relative_paths`` and write the graphs it asks for; return the subjects folder."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, relative_paths)
    subjects = output_dir.parent / "subjects"
    subjects.mkdir()
    for graphs in _graphs_by_file(formats, input_dir, output_dir).values():
        for graph in graphs:
            (subjects / graph).parent.mkdir(parents=True, exist_ok=True)
            (subjects / graph).write_bytes(b"")
    return subjects


def _folders_of(graphs):
    """(subject, acquisition) folders of graph paths in the <subject>/t1mri/<acquisition> layout, else what is there."""
    assert graphs, "the output format has no rule for the attributes stage 1 gave"
    return {graph.parts[:3] for graph in graphs}


@st.composite
def _bids_names(draw):
    """(subject label, entities text) of a BIDS file name with any of the ses, acq and run entities."""
    entities = [f"{key}-{draw(_LABELS)}" for key in ("ses", "acq", "run") if draw(st.booleans())]
    return draw(_LABELS), "_".join(entities)


# REQ-INPUTKIND-BDRABCZUK-A1300CDB5901, and the rule of candidate INV-312C55B42725 (the Defect must not recur).
@settings(max_examples=12, deadline=None, database=None, phases=(Phase.explicit, Phase.generate))
@example(name=("01", "ses-V1"))
@example(name=("01", ""))
@given(name=_bids_names())
def test_graphs_lie_below_subject_t1mri(formats, name):
    """Whatever the entities, nested or flat, both graphs of a file lie below sub-<label>/t1mri, with no sub-sub-."""
    label, entities = name
    stem = "_".join(filter(None, (f"sub-{label}", entities)))
    with tempfile.TemporaryDirectory() as folder:
        for relative_path in (f"{stem}_T1w.nii.gz", f"sub-{label}/anat/{stem}_T1w.nii.gz"):
            input_dir = Path(folder) / f"input{len(relative_path)}"
            output_dir = Path(folder) / f"output{len(relative_path)}"
            input_dir.mkdir()
            output_dir.mkdir()
            _write_inputs(input_dir, [relative_path])
            graphs = _graphs_by_file(formats, input_dir, output_dir)[f"{stem}_T1w.nii.gz"]
            assert {folders[:2] for folders in _folders_of(graphs)} == {(f"sub-{label}", "t1mri")}


@pytest.mark.parametrize(
    ("relative_path", "acquisition"),
    [
        ("sub-01_ses-V1_T1w.nii.gz", "ses-V1"),
        ("sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz", "ses-V1"),
        ("sub-01_ses-V1_run-1_T1w.nii.gz", "ses-V1_run-1"),
        ("sub-01/ses-V1/anat/sub-01_ses-V1_run-1_T1w.nii.gz", "ses-V1_run-1"),
        ("sub-01_acq-mprage_T1w.nii", "acq-mprage"),
        ("sub-01/anat/sub-01_acq-mprage_T1w.nii", "acq-mprage"),
        ("sub-01_ses-V1_acq-x_run-2_T1w.nii.gz", "ses-V1_acq-x_run-2"),
        ("sub-01_ses-pre.nii", "ses-pre"),
    ],
)
def test_acquisition_folder_is_the_file_name_text(formats, dirs, relative_path, acquisition):
    """REQ-INPUTKIND-BDRABCZUK-A6458DA2FBA1: the text between sub-<label>_ and _T1w or the extension, verbatim.

    A name without _T1w is flat only: below a sub-* folder it is no T1 file (REQ-INPUTKIND-BDRABCZUK-AE205BE00702).
    """
    input_dir, output_dir = dirs
    _write_inputs(input_dir, [relative_path])
    graphs = _graphs_by_file(formats, input_dir, output_dir)[Path(relative_path).name]
    assert _folders_of(graphs) == {("sub-01", "t1mri", acquisition)}


def test_each_file_keeps_its_own_acquisition_folder(formats, dirs):
    """REQ-INPUTKIND-BDRABCZUK-A6458DA2FBA1: in one call of several files, no file gets the folder of another."""
    input_dir, output_dir = dirs
    _write_inputs(
        input_dir,
        [
            "sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz",
            "sub-01/ses-V2/anat/sub-01_ses-V2_T1w.nii.gz",
            "sub-02/anat/sub-02_T1w.nii.gz",
        ],
    )
    graphs = _graphs_by_file(formats, input_dir, output_dir)
    assert {name: _folders_of(paths) for name, paths in graphs.items()} == {
        "sub-01_ses-V1_T1w.nii.gz": {("sub-01", "t1mri", "ses-V1")},
        "sub-01_ses-V2_T1w.nii.gz": {("sub-01", "t1mri", "ses-V2")},
        "sub-02_T1w.nii.gz": {("sub-02", "t1mri", _DEFAULT_ACQUISITION)},
    }


@pytest.mark.parametrize("file_name", ["sub-01_T1w.nii.gz", "sub-01_T1w.nii", "sub-01.nii.gz", "sub-01.nii"])
def test_no_entity_gives_default_acquisition(formats, dirs, file_name):
    """REQ-INPUTKIND-BDRABCZUK-7D0F40A034D8: no text between sub-<label> and _T1w or the extension."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, [file_name])
    graphs = _graphs_by_file(formats, input_dir, output_dir)[file_name]
    assert _folders_of(graphs) == {("sub-01", "t1mri", _DEFAULT_ACQUISITION)}


@pytest.mark.parametrize(
    "files",
    [
        ("sub-01_ses-V1_run-1_T1w.nii.gz", "sub-01_ses-V1_run-2_T1w.nii.gz"),
        ("sub-01_ses-V1_T1w.nii.gz", "sub-01_ses-V2_T1w.nii.gz"),
        ("sub-01_ses-V1_T1w.nii.gz", "sub-01_ses-V1_run-2_T1w.nii.gz"),
    ],
    ids=["two_runs_of_one_session", "two_sessions", "session_beside_one_of_its_runs"],
)
def test_different_acquisition_folders_both_reach_cli(dirs, files):
    """REQ-INPUTKIND-BDRABCZUK-63E61E6C1B60: neither file of one subject is lost to the other."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, files)
    command = _stage_command(input_dir, output_dir)
    assert [name for name in files if name not in command] == []


@pytest.mark.parametrize(
    "files",
    [
        ("a/sub-01_ses-V1_T1w.nii.gz", "b/sub-01_ses-V1_T1w.nii.gz"),
        ("sub-01_ses-V1_T1w.nii", "sub-01_ses-V1_T1w.nii.gz"),
    ],
    ids=["one_name_in_two_folders", "two_extensions"],
)
def test_same_scan_folder_twice_is_a_duplicate(dirs, files):
    """REQ-INPUTKIND-BDRABCZUK-185D602AA38D: one <subject>/t1mri/<acquisition> cannot hold two files."""
    input_dir, output_dir = dirs
    kept = "sub-02_T1w.nii.gz"
    _write_inputs(input_dir, [*files, kept])
    command = _stage_command(input_dir, output_dir)
    assert [row["qc"] for row in _qc_rows(output_dir) if row["reason"] == "duplicate_subject"] == ["0"]
    assert kept in command
    assert [name for name in files if name in command] == []


@pytest.mark.parametrize(
    "file_name",
    [
        "sub-01_['a']_T1w.nii.gz",
        "sub-01_FLAIR.nii.gz",
        "sub-01_ses-V+1_T1w.nii.gz",
        "sub-01_x[1]-2_T1w.nii.gz",
    ],
    ids=["python_literal", "no_key", "value_with_a_symbol", "key_with_a_symbol"],
)
def test_part_that_is_no_entity_gives_qc_0(dirs, file_name):
    """REQ-INPUTKIND-BDRABCZUK-E3C113C319C3: capsul evaluates a value that starts with [, so it never gets one."""
    input_dir, output_dir = dirs
    kept = "sub-02_T1w.nii.gz"
    _write_inputs(input_dir, [file_name, kept])
    command = _stage_command(input_dir, output_dir)
    assert [row["qc"] for row in _qc_rows(output_dir) if row["reason"] == "no_bids_entity"] == ["0"]
    assert kept in command
    assert file_name not in command


@pytest.mark.parametrize(
    "file_name",
    ["sub-01_T1w.nii.gz", "sub-01_ses-V1_T1w.nii.gz", "sub-01_ses-V1_run-1_T1w.nii.gz"],
    ids=["no_session", "session", "session_and_run"],
)
def test_stage_1_id_equals_stage_2_id(formats, dirs, file_name):
    """REQ-INPUTKIND-BDRABCZUK-57C46D5C7CC4: one file, one scan, one id at both stages."""
    subjects = _graph_tree(formats, dirs, [file_name])
    scans = SubjectEligibilityChecker(str(subjects), [], bids=True).list_scans()
    assert len(scans) == 1, f"no scan in {sorted(path.name for path in subjects.iterdir())}"
    assert {row["participant_id"] for row in _qc_rows(dirs[1])} == {compute_participant_id(str(scans[0]), True)}


def test_list_scans_finds_each_kept_file(formats, dirs):
    """REQ-INPUTKIND-BDRABCZUK-91CB93BECCF9: one ScanId per kept file, in the layout stage 2 reads."""
    subjects = _graph_tree(
        formats,
        dirs,
        [
            "sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz",
            "sub-01/ses-V2/anat/sub-01_ses-V2_T1w.nii.gz",
            "sub-002_ses-V1_run-1_T1w.nii.gz",
            "sub-03/anat/sub-03_T1w.nii.gz",
        ],
    )
    assert SubjectEligibilityChecker(str(subjects), [], bids=True).list_scans() == [
        ScanId("sub-002", "ses-V1_run-1", bids=True),
        ScanId("sub-01", "ses-V1", bids=True),
        ScanId("sub-01", "ses-V2", bids=True),
        ScanId("sub-03", _DEFAULT_ACQUISITION, bids=True),
    ]


def test_detect_input_kind_reads_morphologist(formats, dirs):
    """REQ-INPUTKIND-BDRABCZUK-1B91FE248179: the output of stage 1 is an input stages 2+ take."""
    subjects = _graph_tree(formats, dirs, ["sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz"])
    assert detect_input_kind(str(subjects)) is InputKind.MORPHOLOGIST


def test_graphs_match_required_patterns(formats, dirs):
    """REQ-INPUTKIND-BDRABCZUK-0072F0971EE1: L*.arg and R*.arg of the stage 2 pre-check match each scan's graphs."""
    subjects = _graph_tree(formats, dirs, ["sub-01_ses-V1_T1w.nii.gz", "sub-01_ses-V2_T1w.nii.gz", "sub-02_T1w.nii.gz"])
    report = SubjectEligibilityChecker(
        str(subjects), [f"{_GRAPHS}/L*.arg", f"{_GRAPHS}/R*.arg"], bids=True
    ).check_subject_patterns()
    assert len(report.eligible) == 3
    assert report.ineligible == {}


def test_without_bids_no_subject_is_passed(dirs):
    """REQ-INPUTKIND-BDRABCZUK-D36697F4F8BD: Morphologist names the subject; the pipeline composes nothing."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, ["sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz"])
    command = _stage_command(input_dir, output_dir, extra_args=())
    assert not {"-s", "--subject", "--participant_label"} & set(command)
    assert [arg for arg in command if arg.startswith("subject=")] == []


def test_without_bids_no_acquisition_is_passed(dirs):
    """REQ-INPUTKIND-BDRABCZUK-487FA6140CD1: the classic tree keeps default_acquisition."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, ["sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz"])
    command = _stage_command(input_dir, output_dir, extra_args=())
    assert [arg for arg in command if arg.startswith("acquisition=")] == []


@pytest.mark.parametrize(
    ("file_name", "participant_id"),
    [
        ("sub-01_ses-1_T1w.nii.gz", "sub-01"),
        ("sub-ABC12_acq-mprage_T1w.nii", "sub-ABC12"),
        ("sub-7.nii.gz", "sub-7"),
        ("sub-8_ses-pre.nii", "sub-8"),
        ("sub-01_ses-V1_run-1_T1w.nii.gz", "sub-01"),
    ],
)
def test_bids_participant_id_is_the_subject_entity(dirs, file_name, participant_id):
    """REQ-NTUI-BDRABCZUK-4D7E056586B4: sub-<label> alone, whatever follows it in the file name."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, [file_name])
    _run_stage(input_dir, output_dir, extra_args=("--bids",))
    assert [row["participant_id"] for row in _qc_rows(output_dir)] == [participant_id]


def test_bids_passes_the_nonoverlap_formats(dirs):
    """REQ-BIDS-BDRABCZUK-579916807967: morphologist-bids-2.0 has no t1mri folder in any setting."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, ["sub-01_ses-V1_T1w.nii.gz"])
    command = _stage_command(input_dir, output_dir)
    assert command[command.index("--if") + 1] == _NONOVERLAP
    assert command[command.index("--of") + 1] == _NONOVERLAP


def test_input_qc_has_one_row_per_scan(dirs):
    """REQ-INPUTKIND-BDRABCZUK-541C32F20DBD: sessions and runs of one participant are rows of their own."""
    input_dir, output_dir = dirs
    _write_inputs(
        input_dir,
        [
            "sub-01_ses-V1_run-1_T1w.nii.gz",
            "sub-01_ses-V1_run-2_T1w.nii.gz",
            "sub-01_ses-V2_T1w.nii.gz",
            "sub-02_T1w.nii.gz",
        ],
    )
    _run_stage(input_dir, output_dir, extra_args=("--bids",))
    assert sorted((row["participant_id"], row.get("acquisition"), row["qc"]) for row in _qc_rows(output_dir)) == [
        ("sub-01", "ses-V1_run-1", "1"),
        ("sub-01", "ses-V1_run-2", "1"),
        ("sub-01", "ses-V2", "1"),
        ("sub-02", _DEFAULT_ACQUISITION, "1"),
    ]


def test_input_qc_has_an_acquisition_column(dirs):
    """REQ-INPUTKIND-BDRABCZUK-8D389326F92F: a run with one scan per participant has the column too."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, ["sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz"])
    _run_stage(input_dir, output_dir, extra_args=("--bids",))
    assert [row.get("acquisition") for row in _qc_rows(output_dir)] == ["ses-V1"]


def test_failed_scan_is_excluded_alone(dirs):
    """REQ-INPUTKIND-BDRABCZUK-162871EE68F4: a bad session shows on its own row, the other one goes on."""
    input_dir, output_dir = dirs
    bad = "sub-01_ses-V1_T1w.nii.gz"
    good = "sub-01_ses-V2_T1w.nii.gz"
    _write_nifti(input_dir / bad, magic=b"bad\0")
    _write_nifti(input_dir / good)
    command = _stage_command(input_dir, output_dir)
    assert {row.get("acquisition"): row["qc"] for row in _qc_rows(output_dir)} == {"ses-V1": "0", "ses-V2": "1"}
    assert good in command
    assert bad not in command


def test_without_bids_no_acquisition_column(dirs):
    """REQ-INPUTKIND-BDRABCZUK-A5276AFE8FF1: the classic file keeps its columns."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, ["sub-01_ses-V1_T1w.nii.gz"])
    _run_stage(input_dir, output_dir)
    assert [list(row) for row in _qc_rows(output_dir)] == [["participant_id", "qc", "reason"]]


def test_write_accepts_one_id_with_two_acquisitions(tmp_path):
    """REQ-NTUI-BDRABCZUK-3AEC5DC5A260: stage 1 names the column, so one participant may have one row per scan."""
    path = tmp_path / "input_qc.tsv"
    rows = [
        {"participant_id": "sub-01", "qc": 1, "acquisition": "ses-V1"},
        {"participant_id": "sub-01", "qc": 0, "acquisition": "ses-V2", "reason": "duplicate_subject"},
    ]
    write_qc_tsv(path, rows, extra_columns=("acquisition",))
    assert path.read_text(encoding="utf-8").splitlines() == [
        "participant_id\tqc\tacquisition\treason",
        "sub-01\t1\tses-V1\t",
        "sub-01\t0\tses-V2\tduplicate_subject",
    ]


def test_write_refuses_one_id_with_one_acquisition(tmp_path):
    """REQ-NTUI-BDRABCZUK-3AEC5DC5A260: one scan is one row, and nothing is written."""
    path = tmp_path / "input_qc.tsv"
    rows = [{"participant_id": "sub-01", "qc": 1, "acquisition": "ses-V1"}] * 2
    with pytest.raises(ValueError, match="sub-01"):
        write_qc_tsv(path, rows, extra_columns=("acquisition",))
    assert not path.exists()


def test_write_refuses_a_repeated_id_without_acquisition_column(tmp_path):
    """REQ-NTUI-BDRABCZUK-3AEC5DC5A260: the rule stays for each file that does not name the column."""
    path = tmp_path / "output_qc.tsv"
    rows = [
        {"participant_id": "sub-01", "qc": 1, "acquisition": "ses-V1"},
        {"participant_id": "sub-01", "qc": 1, "acquisition": "ses-V2"},
    ]
    with pytest.raises(ValueError, match="sub-01"):
        write_qc_tsv(path, rows)
    assert not path.exists()


@pytest.mark.parametrize(
    ("file_name", "text"),
    [
        ("sub-01_['a']_T1w.nii.gz", "['a']"),
        ("sub-01_FLAIR.nii.gz", "FLAIR"),
        ("sub-01_ses-V+1_T1w.nii.gz", "ses-V+1"),
        ("sub-01_x[1]-2_T1w.nii.gz", "x[1]-2"),
    ],
    ids=["python_literal", "no_key", "value_with_a_symbol", "key_with_a_symbol"],
)
def test_no_entity_row_acquisition_is_the_file_name_text(dirs, file_name, text):
    """REQ-INPUTKIND-BDRABCZUK-2AE7D67F1378: a refused name keeps its own text, so its row stays unique."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, [file_name, "sub-02_T1w.nii.gz"])
    _run_stage(input_dir, output_dir, extra_args=("--bids",))
    refused = [row for row in _qc_rows(output_dir) if row["reason"] == "no_bids_entity"]
    assert [(row["participant_id"], row.get("acquisition")) for row in refused] == [("sub-01", text)]


def test_no_subject_row_acquisition_is_empty(dirs):
    """REQ-INPUTKIND-BDRABCZUK-7310EA9C177B: no subject, so no acquisition folder."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, ["scan.nii.gz", "sub-02_T1w.nii.gz"])
    _run_stage(input_dir, output_dir, extra_args=("--bids",))
    refused = [row for row in _qc_rows(output_dir) if row["reason"] == "no_bids_subject"]
    assert [(row["participant_id"], row.get("acquisition")) for row in refused] == [("scan", "")]


@pytest.mark.parametrize(
    ("files", "acquisitions"),
    [
        (("sub-01_rec-a_T1w.nii.gz", "sub-01_rec-b_T1w.nii.gz"), ["rec-a", "rec-b"]),
        (("sub-01_ses-V1_rec-a_T1w.nii.gz", "sub-01_ses-V1_rec-b_T1w.nii.gz"), ["ses-V1_rec-a", "ses-V1_rec-b"]),
        (("sub-01_T1w.nii.gz", "sub-01_rec-a_T1w.nii.gz"), [_DEFAULT_ACQUISITION, "rec-a"]),
        (("sub-01_ses-V1_run-1_T1w.nii.gz", "sub-01_run-1_ses-V1_T1w.nii.gz"), ["run-1_ses-V1", "ses-V1_run-1"]),
    ],
    ids=["rec_a_and_rec_b", "rec_a_and_rec_b_of_one_session", "no_entity_and_rec", "two_entity_orders"],
)
def test_same_entity_key_in_two_folders_is_a_duplicate(dirs, files, acquisitions):
    """REQ-INPUTKIND-BDRABCZUK-B4A2D5BA12EB: stage 2 would give both scans one key and drop one almost silently."""
    input_dir, output_dir = dirs
    kept = "sub-02_T1w.nii.gz"
    _write_inputs(input_dir, [*files, kept])
    command = _stage_command(input_dir, output_dir)
    rows = sorted(
        (row["participant_id"], row.get("acquisition"), row["qc"], row["reason"]) for row in _qc_rows(output_dir)
    )
    assert rows == [
        *[("sub-01", acquisition, "0", "duplicate_subject") for acquisition in acquisitions],
        ("sub-02", _DEFAULT_ACQUISITION, "1", ""),
    ]
    assert kept in command
    assert [name for name in files if name in command] == []


@pytest.mark.parametrize(
    "files",
    [("sub-01_rec-a_T1w.nii.gz",), ("sub-01_rec-a_T1w.nii.gz", "sub-01_ses-V1_rec-a_T1w.nii.gz")],
    ids=["alone", "beside_another_session"],
)
def test_lone_rec_file_is_kept(dirs, files):
    """REQ-INPUTKIND-BDRABCZUK-B4A2D5BA12EB, REQ-INPUTKIND-BDRABCZUK-63E61E6C1B60: a rec- file of its own key runs."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, files)
    command = _stage_command(input_dir, output_dir)
    assert {row.get("acquisition"): row["qc"] for row in _qc_rows(output_dir)} == {
        Path(name).name.removeprefix("sub-01_").removesuffix("_T1w.nii.gz"): "1" for name in files
    }
    assert [name for name in files if name not in command] == []


def test_input_qc_is_written_before_cli(dirs):
    """REQ-NTUI-BDRABCZUK-88390900F3E6: with --bids too, the file is on disk when morphologist-cli starts."""
    input_dir, output_dir = dirs
    _write_inputs(input_dir, ["sub-01_ses-V1_T1w.nii.gz", "sub-01_ses-V2_T1w.nii.gz"])
    seen_at_call = []
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir), "--bids"])
    with patch.object(
        script,
        "execute_command",
        side_effect=lambda *a, **k: seen_at_call.append(_input_qc_path(output_dir).is_file()) or 0,
    ):
        script.run()
    assert seen_at_call == [True]

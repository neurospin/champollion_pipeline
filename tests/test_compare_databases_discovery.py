"""Graph discovery in the compare databases subcommand (TASK-ARGDISC-BDRABCZUK-6FFA97286BC0).

Without --path_to_graph_a / --path_to_graph_b, a campaign's graphs come from the graph discovery of
--labeled_subjects_dir: the graphs whose labelling set is the campaign's label (--label_a, --label_b), counted for
the subject discovery gives them, one graph per subject and hemisphere. A given sub-path keeps today's behaviour.

The trees come from tests/morphologist_layout_fixtures.py (empty files, names only); a second labelling set is added
below them where a test needs two campaigns. ``soma.aims`` is replaced by a fake reader that records each path read,
so no graph is opened. --njobs 1 keeps the joblib workers in this process.

Requirements (area ARGDISC, author BDRABCZUK, root ids listed by the Task):
given sub-path without discovery; labelling set of the campaign's label; discovered subject; --side;
first of several graphs; warning on several graphs; exit 2 on a label without graph; error naming the sets;
no cortical_tiles needed. Area COMPARE: the graph sub-path options are optional.
"""

import csv
import logging
import os
import re
import sys
from pathlib import Path

import pytest

import compare_sulcal.brainvisa
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from compare import Compare
from compare_databases import CompareDatabases
from tests.log_assertions import COMPARE_LOGGER_NAME
from tests.morphologist_layout_fixtures import (
    CLASSIC_SUBJECTS,
    CLI_LABELLED_SUBJECTS,
    CLI_SUBJECTS,
    HEMISPHERES,
    build_classic_tree,
    build_morphologist_cli_tree,
    build_senior_tree,
)

SULCUS = "S.C._left"
CLI_SET = "sul-0_auto"
SECOND_SET = "base_manual"
CLASSIC_SET = "default_session_auto"
CLASSIC_FOLDS = "t1mri/default_acquisition/default_analysis/folds/3.1"


class _FakeVertex:
    def __init__(self):
        self._data = {"name": SULCUS, "aims_ss": [{"v0": 1}]}

    def get(self, key):
        return self._data.get(key)


class _FakeGraph:
    def vertices(self):
        return [_FakeVertex()]


class _FakeAims:
    """Fake ``soma.aims``: every graph holds one voxel of SULCUS; records the paths read."""

    def __init__(self):
        self.read_paths: list[str] = []

    def read(self, path):
        self.read_paths.append(os.path.abspath(str(path)))
        return _FakeGraph()


@pytest.fixture
def fake_aims(monkeypatch):
    """Install the fake reader where the compare databases mode reads graphs."""
    fake = _FakeAims()
    monkeypatch.setattr(compare_sulcal.brainvisa, "aims", fake)
    return fake


def _fake_get_all_subjects_as_dictionary(src_dir_list, graph_file_list, side):
    """cortical_tiles' subject lookup: one entry per sub-folder of each source folder, the pattern filled in."""
    subjects = []
    for src_dir, graph_file in zip(src_dir_list, graph_file_list, strict=True):
        for name in sorted(os.listdir(src_dir)):
            if os.path.isdir(os.path.join(src_dir, name)):
                subjects.append(
                    {
                        "subject": name,
                        "side": side,
                        "dir": src_dir,
                        "graph_file": graph_file % {"side": side, "subject": name},
                    }
                )
    return subjects


@pytest.fixture  # noqa: V103 - used through usefixtures
def cortical_tiles_lookup(monkeypatch):
    """Stand in for cortical_tiles' subject lookup, which the sub-path mode uses."""
    monkeypatch.setattr(
        compare_sulcal.brainvisa, "get_all_subjects_as_dictionary", _fake_get_all_subjects_as_dictionary
    )


def _touch_graph(path: Path) -> None:
    """Create one empty .arg file and its empty .data folder, as the shared fixtures do."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.with_suffix(".data").mkdir(exist_ok=True)


def _add_cli_labelling_set(root: Path, subjects, set_name: str) -> None:
    """Add a labelling set folder holding both hemisphere graphs below each morphologist-cli subject named."""
    for subject in subjects:
        folds = root / subject / "ses-1" / "run-0" / "ana-0" / "folds" / "3.1" / set_name
        for hemi in HEMISPHERES:
            _touch_graph(folds / f"{subject}_ses-1_run-0_hemi-{hemi}.arg")


def _cli_graph(subject: str, hemi: str, set_name: str) -> str:
    """Relative path of a morphologist-cli graph of a labelling set (the fixture's sul-0_auto naming or ours)."""
    folds = f"{subject}/ses-1/run-0/ana-0/folds/3.1/{set_name}"
    if set_name == CLI_SET:
        return f"{folds}/{subject}_ses-1_run-0_sul-0_hemi-{hemi}_auto.arg"
    return f"{folds}/{subject}_ses-1_run-0_hemi-{hemi}.arg"


def _senior_graph(group: str, subject: str, acquisition: str, hemi: str) -> str:
    folds = f"derivatives/{group}/{subject}/t1mri/{acquisition}/default_analysis/folds/3.1"
    return f"{folds}/{CLASSIC_SET}/{hemi}{subject}_{CLASSIC_SET}.arg"


def _two_set_cli_tree(root: Path) -> Path:
    """morphologist-cli tree: sub-01 holds sul-0_auto and base_manual graphs, sub-02 only base_manual graphs."""
    build_morphologist_cli_tree(root)
    _add_cli_labelling_set(root, CLI_SUBJECTS, SECOND_SET)
    return root


def _parse(script, argv):
    """Parse argv with script's parser; a refusal fails the test instead of exiting."""
    try:
        script.args = script.parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"{type(script).__name__} refused {argv!r} with exit status {exc.code}")
    return script


def _status(script) -> int:
    """Exit status of script.run(), whether returned or raised as SystemExit."""
    try:
        return script.run()
    except SystemExit as exc:
        return exc.code


def _databases(root: Path, output: Path, *extra: str) -> Compare:
    argv = ["databases", "--labeled_subjects_dir", str(root), "--njobs", "1", "--output", str(output), *extra]
    return _parse(Compare(), argv)


def _relative_reads(fake: _FakeAims, root: Path) -> list[str]:
    return sorted(Path(os.path.relpath(path, root.resolve())).as_posix() for path in fake.read_paths)


def _csv_row(output: Path, sulcus: str) -> dict:
    with open(output, newline="") as handle:
        return {row["sulcus"]: row for row in csv.DictReader(handle)}[sulcus]


def _classic_set_and_folds_reads() -> list[str]:
    """Expected reads: campaign A from <folds>/<set>, campaign B from <folds>, both sides of each classic subject."""
    expected = [
        f"{subject}/{CLASSIC_FOLDS}/{CLASSIC_SET}/{hemi}{subject}_{CLASSIC_SET}.arg"
        for subject in CLASSIC_SUBJECTS
        for hemi in HEMISPHERES
    ] + [f"{subject}/{CLASSIC_FOLDS}/{hemi}{subject}.arg" for subject in CLASSIC_SUBJECTS for hemi in HEMISPHERES]
    return sorted(expected)


def _messages(caplog, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(COMPARE_LOGGER_NAME) and record.levelno == level
    ]


# --------------------------------------------------------------------------- #
# Graph sub-path options optional (supersedes REQ-COMPARE-BDRABCZUK-9948C646D2E5)
# --------------------------------------------------------------------------- #


def test_graph_sub_path_options_are_optional():
    """Both entry points parse a command line without --path_to_graph_a and --path_to_graph_b."""
    compare_args = _parse(Compare(), ["databases", "--labeled_subjects_dir", "/subjects"]).args
    script_args = _parse(CompareDatabases(), ["--labeled_subjects_dir", "/subjects"]).args
    assert (compare_args.path_to_graph_a, compare_args.path_to_graph_b) == (None, None)
    assert (script_args.path_to_graph_a, script_args.path_to_graph_b) == (None, None)


# --------------------------------------------------------------------------- #
# A given sub-path keeps today's behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("cortical_tiles_lookup")
def test_given_sub_paths_read_without_graph_discovery(tmp_path, fake_aims, monkeypatch):
    """Each campaign reads <subject>/<its sub-path>/<side><subject>*.arg; graph discovery is never called."""

    def _no_discovery(_self, folder):
        raise AssertionError(f"graph discovery called on {folder!r}")

    monkeypatch.setattr(FileIndexStageIndexer, "discover_graphs", _no_discovery)
    root = build_classic_tree(tmp_path / "subjects")
    script = _databases(
        root,
        tmp_path / "db.csv",
        f"--path_to_graph_a={CLASSIC_FOLDS}/{CLASSIC_SET}",
        f"--path_to_graph_b={CLASSIC_FOLDS}",
    )

    assert _status(script) == 0
    assert _relative_reads(fake_aims, root) == _classic_set_and_folds_reads()


# --------------------------------------------------------------------------- #
# Without a sub-path: graph discovery, by labelling set
# --------------------------------------------------------------------------- #


def test_discovery_reads_the_labelling_sets_of_label_a_and_label_b(tmp_path, fake_aims):
    """Campaign A reads the sul-0_auto graphs, campaign B the base_manual graphs; unlabelled graphs are not read."""
    root = _two_set_cli_tree(tmp_path / "subjects")
    output = tmp_path / "db.csv"
    script = _databases(root, output, f"--label_a={CLI_SET}", f"--label_b={SECOND_SET}")

    assert _status(script) == 0
    expected = [_cli_graph(subject, hemi, CLI_SET) for subject in CLI_LABELLED_SUBJECTS for hemi in HEMISPHERES] + [
        _cli_graph(subject, hemi, SECOND_SET) for subject in CLI_SUBJECTS for hemi in HEMISPHERES
    ]
    assert _relative_reads(fake_aims, root) == sorted(expected)
    row = _csv_row(output, SULCUS)
    assert (row[f"N_{CLI_SET}"], row[f"N_{SECOND_SET}"]) == (str(len(CLI_LABELLED_SUBJECTS)), str(len(CLI_SUBJECTS)))


@pytest.mark.usefixtures("cortical_tiles_lookup")
def test_discovery_for_campaign_a_and_sub_path_for_campaign_b(tmp_path, fake_aims):
    """Only campaign B has a sub-path: A comes from graph discovery, B from its sub-path."""
    root = build_classic_tree(tmp_path / "subjects")
    script = _databases(root, tmp_path / "db.csv", f"--label_a={CLASSIC_SET}", f"--path_to_graph_b={CLASSIC_FOLDS}")

    assert _status(script) == 0
    assert _relative_reads(fake_aims, root) == _classic_set_and_folds_reads()


def test_compare_databases_script_reads_campaigns_from_graph_discovery(tmp_path, fake_aims):
    """src/compare_databases.py without sub-paths runs the discovery comparison and writes the CSV."""
    root = _two_set_cli_tree(tmp_path / "subjects")
    output = tmp_path / "db.csv"
    argv = ["--labeled_subjects_dir", str(root), "--njobs", "1", "--output", str(output)]
    script = _parse(CompareDatabases(), [*argv, f"--label_a={CLI_SET}", f"--label_b={SECOND_SET}"])

    assert _status(script) == 0
    assert _csv_row(output, SULCUS)[f"N_{SECOND_SET}"] == str(len(CLI_SUBJECTS))


def test_discovery_counts_a_graph_for_its_discovered_subject(tmp_path, fake_aims, caplog):
    """SENIOR subjects sit three folders below --labeled_subjects_dir; each graph counts for sf0001 or sh0001."""
    caplog.set_level(1, logger=COMPARE_LOGGER_NAME)
    root = build_senior_tree(tmp_path / "senior")
    _touch_graph(root / _senior_graph("subjects_f", "sf0001", "3TV0", "L").replace(CLASSIC_SET, SECOND_SET))
    output = tmp_path / "db.csv"
    script = _databases(root, output, f"--label_a={CLASSIC_SET}", f"--label_b={SECOND_SET}")

    assert _status(script) == 0
    row = _csv_row(output, SULCUS)
    assert (row[f"N_{CLASSIC_SET}"], row[f"N_{SECOND_SET}"]) == ("2", "1")
    joined = "\n".join(_messages(caplog, 25))
    assert re.search(r"Subjects in both:\s+1\b", joined), joined


def test_discovery_reads_only_the_side_selected(tmp_path, fake_aims):
    """--side L: only the hemi-L graphs of both labelling sets are read."""
    root = _two_set_cli_tree(tmp_path / "subjects")
    script = _databases(root, tmp_path / "db.csv", f"--label_a={CLI_SET}", f"--label_b={SECOND_SET}", "--side", "L")

    assert _status(script) == 0
    expected = [_cli_graph(subject, "L", CLI_SET) for subject in CLI_LABELLED_SUBJECTS] + [
        _cli_graph(subject, "L", SECOND_SET) for subject in CLI_SUBJECTS
    ]
    assert _relative_reads(fake_aims, root) == sorted(expected)


# --------------------------------------------------------------------------- #
# Several graphs of one subject and hemisphere
# --------------------------------------------------------------------------- #


def _senior_one_side(tmp_path: Path) -> tuple[Path, Compare]:
    """SENIOR tree (four acquisitions per subject, each with a default_session_auto graph), both campaigns on it."""
    root = build_senior_tree(tmp_path / "senior")
    script = _databases(
        root, tmp_path / "db.csv", f"--label_a={CLASSIC_SET}", f"--label_b={CLASSIC_SET}", "--side", "L"
    )
    return root, script


def test_several_graphs_read_the_first_in_natural_path_order(tmp_path, fake_aims):
    """Of acquisitions 3TV0, 3TV1, 3TV2 and 3TV10, only the 3TV0 graph is read, once per campaign."""
    root, script = _senior_one_side(tmp_path)

    assert _status(script) == 0
    first = [_senior_graph("subjects_f", "sf0001", "3TV0", "L"), _senior_graph("subjects_h", "sh0001", "3TV0", "L")]
    assert _relative_reads(fake_aims, root) == sorted(first * 2)


def test_several_graphs_log_a_warning_naming_subject_and_graph(tmp_path, fake_aims, caplog):
    """The warning names the subject and the graph read (its 3TV0 path)."""
    caplog.set_level(logging.WARNING, logger=COMPARE_LOGGER_NAME)
    _root, script = _senior_one_side(tmp_path)

    _status(script)
    warnings = _messages(caplog, logging.WARNING)
    graph_name = Path(_senior_graph("subjects_f", "sf0001", "3TV0", "L"))
    assert any("sf0001" in message and "3TV0/" in message and graph_name.name in message for message in warnings), (
        warnings
    )


# --------------------------------------------------------------------------- #
# A label without any discovered graph
# --------------------------------------------------------------------------- #


def test_label_without_discovered_graph_stops_with_status_2(tmp_path, fake_aims):
    """--label_a names no labelling set of the tree: exit status 2, no graph read."""
    root = build_morphologist_cli_tree(tmp_path / "subjects")
    script = _databases(root, tmp_path / "db.csv", "--label_a=base2018_manual", f"--label_b={CLI_SET}")

    assert _status(script) == 2
    assert fake_aims.read_paths == []
    assert not (tmp_path / "db.csv").exists()


def test_label_without_discovered_graph_logs_label_and_labelling_sets(tmp_path, fake_aims, caplog):
    """The ERROR record names the label asked for and the labelling set found (sul-0_auto)."""
    caplog.set_level(logging.ERROR, logger=COMPARE_LOGGER_NAME)
    root = build_morphologist_cli_tree(tmp_path / "subjects")
    script = _databases(root, tmp_path / "db.csv", "--label_a=base2018_manual", f"--label_b={CLI_SET}")

    _status(script)
    errors = _messages(caplog, logging.ERROR)
    assert any("base2018_manual" in message and CLI_SET in message for message in errors), errors


# --------------------------------------------------------------------------- #
# No cortical_tiles needed
# --------------------------------------------------------------------------- #


def test_discovery_compares_without_cortical_tiles(tmp_path, fake_aims, monkeypatch):
    """cortical_tiles cannot be imported: the discovery comparison still writes its CSV."""
    monkeypatch.setattr(compare_sulcal.brainvisa, "get_all_subjects_as_dictionary", None)
    monkeypatch.setattr(compare_sulcal.brainvisa, "_CORTICAL_TILES_IMPORT_ERROR", ImportError("no cortical_tiles"))
    monkeypatch.setitem(sys.modules, "cortical_tiles", None)
    root = _two_set_cli_tree(tmp_path / "subjects")
    output = tmp_path / "db.csv"
    script = _databases(root, output, f"--label_a={CLI_SET}", f"--label_b={SECOND_SET}")

    assert _status(script) == 0
    assert output.is_file()

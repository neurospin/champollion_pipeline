"""Stage 6 snapshots find each subject's graphs and white meshes through the graph discovery.

TASK-ARGDISC-BDRABCZUK-AACEA4B5776D: no <subject>/t1mri/<acquisition> folder is needed; requirements named per class.
Synthetic trees of empty files only (tests/morphologist_layout_fixtures.py and small trees built here).
"""

import logging
import shutil
from pathlib import Path
from unittest.mock import patch

from champollion_pipeline import generate_snapshots as gs
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.snapshot_types.input_qc import SnapshotsInputQC, check_sulcal_inputs
from champollion_pipeline.snapshot_types.sulcal import find_sulcal_graphs_by_hemi, list_sulcal_subjects
from tests.morphologist_layout_fixtures import build_mixed_tree, build_morphologist_cli_tree, build_senior_tree

SNAPSHOTS_LOGGER = "champollion_pipeline.generate_snapshots"
SULCAL_LOGGER = "champollion_pipeline.snapshot_types.sulcal"
CLI_PREFIX = "sub-01_ses-1_run-0"
CLI_ANALYSIS = Path("sub-01", "ses-1", "run-0", "ana-0")


def _touch(root, *relative_names):
    """Create each relative name below root as an empty file; return the created paths."""
    paths = []
    for name in relative_names:
        path = Path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(path)
    return paths


def _cli_graph(root, side, labelled=False):
    """Path of a morphologist-cli graph of sub-01 below root (built by build_morphologist_cli_tree)."""
    folds = Path(root, CLI_ANALYSIS, "folds", "3.1")
    if labelled:
        return folds / "sul-0_auto" / f"{CLI_PREFIX}_sul-0_hemi-{side}_auto.arg"
    return folds / f"{CLI_PREFIX}_hemi-{side}.arg"


def _script(tmp_path, morpho, extra=None):
    args = ["--output_dir", str(tmp_path / "out"), "--morphologist_dir", str(morpho)] + (extra or [])
    script = GenerateSnapshots()
    script.parse_args(args)
    return script


def _run(script):
    """Run the sulcal step with a stand-in render; return the render mock."""
    with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]) as render:
        script._run_sulcal((800, 600))
    return render


def _rendered(render):
    """{graph path: mesh_path} of each stand-in render call."""
    return {str(call.args[0]): call.kwargs.get("mesh_path") for call in render.call_args_list}


def _run_stage(tmp_path, morpho, extra=None):
    """Run the whole stage with stand-in renders; return (status, render mock, tiles mock, umap mock)."""
    script = _script(tmp_path, morpho, extra)
    with (
        patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]) as render,
        patch.object(script, "_run_tiles", return_value=[]) as tiles,
        patch.object(script, "_run_umap", return_value=[]) as umap,
    ):
        status = script.run()
    return status, render, tiles, umap


def _stopped_counts(tmp_path, render, tiles, umap):
    """(sulcal renders, tiles calls, UMAP calls, manifest written): all zero or False after an early stop."""
    return (
        render.call_count,
        tiles.call_count,
        umap.call_count,
        (tmp_path / "out" / "snapshots_manifest.json").exists(),
    )


def _written_input_qc(tmp_path):
    """[(participant_id, qc, reason)] of <output_dir>/input_qc.tsv written by GenerateSnapshots.run."""
    with open(tmp_path / "out" / "input_qc.tsv", encoding="utf-8") as handle:
        header, *rows = [line.split("\t") for line in handle.read().splitlines()]
    return [(row[0], row[header.index("qc")], row[header.index("reason")]) for row in rows]


def _rendered_side(render, side):
    """Graph paths rendered for side L or R, read from the hemisphere of the output file name."""
    hemi = {"L": "_left.png", "R": "_right.png"}[side]
    return [str(call.args[0]) for call in render.call_args_list if str(call.args[1]).endswith(hemi)]


class TestSulcalSubjectsByDiscovery:
    """REQ-ARGDISC-BDRABCZUK-1DE93AD336E4 (supersedes -38D75C69C9AC, REQ-INPUTKIND-BDRABCZUK-F9B3F4DDFCCD):
    list_sulcal_subjects gives the discovered subject attributes and the subject folder names, once each, in natural
    order."""

    def test_subjects_below_a_folder_that_is_no_subject(self, tmp_path):
        build_mixed_tree(tmp_path)
        assert list_sulcal_subjects(str(tmp_path)) == ["sub-01", "sub-02"]

    def test_subject_folder_without_graph_is_listed(self, tmp_path):
        """User 2026-11-26: a subject folder with no .arg keeps its input QC row, so it stays listed."""
        build_morphologist_cli_tree(tmp_path)
        _touch(tmp_path, "sub-03/ses-V1/anat/sub-03_ses-V1_T1w.nii.gz")
        assert list_sulcal_subjects(str(tmp_path)) == ["sub-01", "sub-02", "sub-03"]

    def test_subjects_come_in_natural_order(self, tmp_path):
        _touch(tmp_path, "sub-10/ana/folds/Lsub-10.arg", "sub-2/ana/folds/Lsub-2.arg")
        assert list_sulcal_subjects(str(tmp_path)) == ["sub-2", "sub-10"]

    def test_graph_less_subject_folder_takes_its_natural_place(self, tmp_path):
        _touch(tmp_path, "sub-10/ana/folds/Lsub-10.arg", "sub-9/ses-V1/anat/sub-9_ses-V1_T1w.nii.gz")
        assert list_sulcal_subjects(str(tmp_path)) == ["sub-9", "sub-10"]


class TestSubjectFolders:
    """Which folders are subject folders: the shared subject-folder rule (find_graph_subject_folder through
    resolve_subjects_dir) decides, never a rule of stage 6 (user decision Q2 2026-11-26)."""

    def test_children_of_the_resolved_folder_are_subject_folders(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-D9E2FFAD5710 (supersedes -845981A67A25): resolve_subjects_dir gives site, whose
        children are the subject folders; the sub-01_..._hemi-L.data folders inside sub-01 are none, a graph-less
        child keeps its place."""
        build_morphologist_cli_tree(tmp_path / "site")
        _touch(tmp_path, "site/sub-03/ses-V1/anat/sub-03_ses-V1_T1w.nii.gz")
        assert list_sulcal_subjects(str(tmp_path)) == ["sub-01", "sub-02", "sub-03"]

    def test_graph_less_folder_beside_the_resolved_folder_is_no_subject_folder(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-D9E2FFAD5710: site_b holds no graph, so resolve_subjects_dir gives site_a only."""
        build_morphologist_cli_tree(tmp_path / "site_a")
        _touch(tmp_path, "site_b/sub-05/ses-V1/anat/sub-05_ses-V1_T1w.nii.gz")
        assert list_sulcal_subjects(str(tmp_path)) == ["sub-01", "sub-02"]

    def test_nested_sub_folder_is_the_subject_folder(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-D9E2FFAD5710: the nearest sub- folder of the graph is its subject folder, not the
        outermost one."""
        _touch(tmp_path, "sub-01/sub-01a/ana/folds/Lsub-01a.arg")
        assert list_sulcal_subjects(str(tmp_path)) == ["sub-01a"]

    def test_given_sub_folder_is_the_only_subject_folder(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-92CFF849158A (supersedes -2958B102C35F): sub-01 given alone; ses-1 is none."""
        build_morphologist_cli_tree(tmp_path)
        assert list_sulcal_subjects(str(tmp_path / "sub-01")) == ["sub-01"]

    def test_single_subject_folder_without_sub_name_is_the_only_subject_folder(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-92CFF849158A: SENIOR sf0001 given alone is the subject folder; t1mri is none."""
        subjects = _senior_two_acquisitions(tmp_path)
        assert list_sulcal_subjects(str(subjects / "sf0001")) == ["sf0001"]

    def test_no_subject_folder_without_any_graph(self, tmp_path):
        """Supersedes REQ-ARGDISC-BDRABCZUK-5022AB917E47 (user decision 2026-11-26 15:40): with no .arg anywhere, no
        child folder is guessed as a subject folder."""
        _no_graph_tree(tmp_path)
        assert list_sulcal_subjects(str(tmp_path)) == []


def _no_graph_tree(root):
    """Folders below root holding a T1, a Morphologist acquisition volume and a QC table, but no .arg file."""
    _touch(root, "sub-01/anat/sub-01_T1w.nii.gz", "sf0099/t1mri/3TV0/sf0099.nii.gz", "qc/report.tsv")


def _ambiguous_tree(root):
    """Two subjects folders holding graphs, site_a and site_b; returns (site_a, site_b)."""
    build_morphologist_cli_tree(root / "site_a")
    build_morphologist_cli_tree(root / "site_b")
    return root / "site_a", root / "site_b"


class TestAmbiguousSubjectsFolders:
    """User decision Q2 2026-11-26: stage 6 stops with an ERROR when resolve_subjects_dir finds several subjects
    folders."""

    def test_error_names_both_candidate_folders(self, tmp_path, pipeline_caplog):
        """REQ-ARGDISC-BDRABCZUK-A81898C97104: one ERROR record names both candidate folders."""
        site_a, site_b = _ambiguous_tree(tmp_path / "morpho")
        with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]):
            _script(tmp_path, tmp_path / "morpho", ["--sulcal-only"]).run()
        naming = [
            r
            for r in pipeline_caplog.records
            if r.levelno == logging.ERROR
            and r.name == SNAPSHOTS_LOGGER
            and str(site_a) in r.getMessage()
            and str(site_b) in r.getMessage()
        ]
        assert len(naming) == 1

    def test_run_returns_1_without_rendering(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-D2C37C0DAA30: status 1, no snapshot of any step rendered, no manifest."""
        _ambiguous_tree(tmp_path / "morpho")
        status, render, tiles, umap = _run_stage(tmp_path, tmp_path / "morpho")
        stopped = _stopped_counts(tmp_path, render, tiles, umap)
        assert (status, stopped) == (gs.FAILED_STATUS, (0, 0, 0, False))

    def test_input_qc_has_one_failed_row_naming_the_folders(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-41F175D2A994 (user decision 2026-11-26 16:10 a): one qc 0 row names both folders."""
        site_a, site_b = _ambiguous_tree(tmp_path / "morpho")
        _run_stage(tmp_path, tmp_path / "morpho")
        ((_, qc, reason),) = _written_input_qc(tmp_path)
        assert (qc, str(site_a) in reason, str(site_b) in reason) == ("0", True, True)


def _input_qc_rows(tmp_path, morpho):
    """{participant_id: (qc, reason)} of the input_qc.tsv that check_sulcal_inputs gives for morpho."""
    report = SnapshotsInputQC()
    check_sulcal_inputs(report, str(morpho))
    out = tmp_path / "out_qc"
    out.mkdir(exist_ok=True)
    with open(report.write(str(out)), encoding="utf-8") as handle:
        header, *rows = [line.split("\t") for line in handle.read().splitlines()]
    return {row[0]: (row[header.index("qc")], row[header.index("reason")]) for row in rows}


class TestNoGraphFoundRow:
    """REQ-ARGDISC-BDRABCZUK-FDD579E8EFF1: a subject folder with no graph keeps a failed row, reason no graph found."""

    def test_subject_folder_without_graph_has_a_failed_row(self, tmp_path):
        build_morphologist_cli_tree(tmp_path)
        _touch(tmp_path, "sub-03/ses-V1/anat/sub-03_ses-V1_T1w.nii.gz")
        rows = _input_qc_rows(tmp_path, tmp_path)
        assert rows == {"sub-01": ("1", ""), "sub-02": ("1", ""), "sub-03": ("0", "no graph found")}

    def test_input_without_any_graph_has_one_failed_row_for_its_path(self, tmp_path):
        """User decision 2026-11-26 15:40: one failed row for the input path, no row per child folder."""
        morpho = tmp_path / "morpho"
        _no_graph_tree(morpho)
        assert _input_qc_rows(tmp_path, morpho) == {str(morpho): ("0", "no graph found")}


class TestNoGraphAnywhere:
    """User decision 2026-11-26 16:10 (b): with no .arg anywhere in the input, only the sulcal snapshots are skipped."""

    def test_no_sulcal_snapshot_is_rendered(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-1976ACF09AD5 (reworded): no sulcal render. Guard: passes before the change too."""
        _no_graph_tree(tmp_path / "morpho")
        _, render, _, _ = _run_stage(tmp_path, tmp_path / "morpho")
        assert render.call_count == 0

    def test_tiles_and_umap_steps_still_run(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-C6BC0A0541C8: the tiles step and the UMAP step run, the input QC keeps its row."""
        morpho = tmp_path / "morpho"
        _no_graph_tree(morpho)
        _, _, tiles, umap = _run_stage(tmp_path, morpho)
        assert (tiles.call_count, umap.call_count, _written_input_qc(tmp_path)) == (
            1,
            1,
            [(str(morpho), "0", "no graph found")],
        )


def _loose_graph_tree(root):
    """A morphologist-cli tree below root (no sub- name) plus a graph file directly in root; returns that file."""
    build_morphologist_cli_tree(root)
    (loose,) = _touch(root, "Lsub-09.arg")
    return loose


class TestLooseGraph:
    """User decision 2026-11-26 15:40: a .arg directly in a --morphologist_dir folder not named sub- stops stage 6 like
    stage 1->2's LooseGraphInput."""

    def test_error_names_the_loose_graph(self, tmp_path, pipeline_caplog):
        """One ERROR record on the generate_snapshots logger names the loose graph file."""
        loose = _loose_graph_tree(tmp_path / "morpho")
        with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]):
            _script(tmp_path, tmp_path / "morpho", ["--sulcal-only"]).run()
        naming = [
            r
            for r in pipeline_caplog.records
            if r.levelno == logging.ERROR and r.name == SNAPSHOTS_LOGGER and str(loose) in r.getMessage()
        ]
        assert len(naming) == 1

    def test_run_returns_2_without_rendering(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-E52EFFD2ABC6: status 2 (input error), no snapshot of any step, no manifest."""
        _loose_graph_tree(tmp_path / "morpho")
        status, render, tiles, umap = _run_stage(tmp_path, tmp_path / "morpho")
        stopped = _stopped_counts(tmp_path, render, tiles, umap)
        assert (status, stopped) == (2, (0, 0, 0, False))

    def test_input_qc_has_one_failed_row_naming_the_graph(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-FF80999721CA (user decision 2026-11-26 16:10 a): one qc 0 row names the loose graph."""
        loose = _loose_graph_tree(tmp_path / "morpho")
        _run_stage(tmp_path, tmp_path / "morpho")
        ((_, qc, reason),) = _written_input_qc(tmp_path)
        assert (qc, str(loose) in reason) == ("0", True)


class TestGraphWithoutHemisphere:
    """A discovered graph whose hemisphere attribute is None (no hemi-L/hemi-R entity, no leading L or R)."""

    GRAPH = "sub-01/ana-0/sub-01_ses-1.arg"

    def test_graph_is_in_neither_hemisphere_list(self, tmp_path):
        """REQ-ARGDISC-BDRABCZUK-686A9DDDA471 (supersedes REQ-PATTERNS-BDRABCZUK-1511DE259EA2). Guard: passes now."""
        _touch(tmp_path, self.GRAPH)
        assert find_sulcal_graphs_by_hemi(str(tmp_path), "sub-01") == {"left": [], "right": []}

    def test_graph_gets_one_warning(self, tmp_path, pipeline_caplog):
        """REQ-ARGDISC-BDRABCZUK-0E9ED683BC6B (supersedes REQ-PATTERNS-BDRABCZUK-C6CAEB2FB7C7)."""
        (graph,) = _touch(tmp_path, self.GRAPH)
        find_sulcal_graphs_by_hemi(str(tmp_path), "sub-01")
        warnings = [
            r
            for r in pipeline_caplog.records
            if r.levelno == logging.WARNING and r.name == SULCAL_LOGGER and str(graph) in r.getMessage()
        ]
        assert len(warnings) == 1


class TestSubjectGraphsByDiscovery:
    """REQ-ARGDISC-BDRABCZUK-BFAAA28B57E1: a subject's graphs are the discovered .arg files, whatever the folders."""

    def test_graph_outside_any_folds_or_sulci_folder_is_taken(self, tmp_path):
        (graph,) = _touch(tmp_path, "sub-01/ana/Lsub-01.arg")
        assert find_sulcal_graphs_by_hemi(str(tmp_path), "sub-01") == {"left": [str(graph)], "right": []}


class TestHemisphereFromDiscovery:
    """REQ-ARGDISC-BDRABCZUK-146BCB3D8C05 (supersedes REQ-PATTERNS-BDRABCZUK-FBD2685DECAC): the hemisphere lists follow
    the discovery hemisphere attribute (hemi-L / hemi-R entity).
    """

    def test_cli_hemi_l_graphs_are_left(self, tmp_path):
        build_morphologist_cli_tree(tmp_path)
        by_hemi = find_sulcal_graphs_by_hemi(str(tmp_path), "sub-01")
        assert sorted(by_hemi["left"]) == sorted(str(_cli_graph(tmp_path, "L", labelled)) for labelled in (False, True))

    def test_cli_hemi_r_graphs_are_right(self, tmp_path):
        build_morphologist_cli_tree(tmp_path)
        by_hemi = find_sulcal_graphs_by_hemi(str(tmp_path), "sub-01")
        assert sorted(by_hemi["right"]) == sorted(
            str(_cli_graph(tmp_path, "R", labelled)) for labelled in (False, True)
        )


class TestWhiteMeshNames:
    """REQ-ARGDISC-BDRABCZUK-41423A43A2E2: a hemi-L_white.gii file is a left white mesh."""

    def test_hemi_entity_mesh_name_is_found(self, tmp_path):
        analysis = "morpho/sub-01/t1mri/default_acquisition/default_analysis"
        graph, mesh = _touch(
            tmp_path, f"{analysis}/folds/3.1/Lsub-01.arg", f"{analysis}/segmentation/mesh/sub-01_hemi-L_white.gii"
        )
        render = _run(_script(tmp_path, tmp_path / "morpho"))
        assert _rendered(render) == {str(graph): str(mesh)}


class TestWhiteMeshNearestFolder:
    """REQ-ARGDISC-BDRABCZUK-800135D0C33B: the white mesh comes from the nearest folder above the graph, within its
    subject folder.
    """

    def test_mesh_folder_beside_the_folds_folder(self, tmp_path):
        graph, mesh = _touch(
            tmp_path, "morpho/sub-01/ana-0/folds/3.1/Lsub-01.arg", "morpho/sub-01/ana-0/mesh/sub-01_Lwhite.gii"
        )
        render = _run(_script(tmp_path, tmp_path / "morpho"))
        assert _rendered(render) == {str(graph): str(mesh)}

    def test_cli_tree_meshes_are_found(self, tmp_path):
        build_morphologist_cli_tree(tmp_path)
        render = _run(_script(tmp_path, tmp_path))
        meshes = sorted(str(mesh) for mesh in _rendered(render).values())
        expected = sorted(
            str(Path(tmp_path, CLI_ANALYSIS, "mesh", f"{CLI_PREFIX}_hemi-{side}_white.gii")) for side in "LR"
        )
        assert meshes == expected

    def test_cli_tree_input_qc_has_no_missing_mesh(self, tmp_path):
        build_morphologist_cli_tree(tmp_path)
        report = SnapshotsInputQC()
        check_sulcal_inputs(report, str(tmp_path))
        (tmp_path / "out_qc").mkdir()
        written = report.write(str(tmp_path / "out_qc"))
        with open(written, encoding="utf-8") as handle:
            header, *rows = [line.split("\t") for line in handle.read().splitlines()]
        assert [(row[0], row[header.index("qc")]) for row in rows] == [("sub-01", "1"), ("sub-02", "1")]
        assert "missing_white_mesh" not in header

    def test_mesh_above_the_subject_folder_is_not_taken(self, tmp_path):
        """Guard: the search stops at the subject folder (passes before the change too)."""
        (graph,) = _touch(tmp_path, "morpho/sub-01/ana-0/folds/3.1/Lsub-01.arg")
        _touch(tmp_path, "morpho/templates/icbm_Lwhite.gii")
        render = _run(_script(tmp_path, tmp_path / "morpho", ["--subject", "sub-01"]))
        assert _rendered(render) == {str(graph): None}


class TestLabelledPreference:
    """REQ-ARGDISC-BDRABCZUK-5DF3F8C49BF1: outside the candidate session folders a labelled (labelling_set) graph is
    preferred.
    """

    def test_cli_labelled_graph_is_rendered(self, tmp_path):
        build_morphologist_cli_tree(tmp_path)
        render = _run(_script(tmp_path, tmp_path))
        assert _rendered_side(render, "L") == [str(_cli_graph(tmp_path, "L", labelled=True))]

    def test_unlabelled_graph_is_rendered_without_labelled_one(self, tmp_path):
        build_morphologist_cli_tree(tmp_path)
        render = _run(_script(tmp_path, tmp_path, ["--subject", "sub-02"]))
        expected = Path(tmp_path, "sub-02", "ses-1", "run-0", "ana-0", "folds", "3.1", "sub-02_ses-1_run-0_hemi-L.arg")
        assert _rendered_side(render, "L") == [str(expected)]


def _senior_two_acquisitions(root):
    """SENIOR subjects_f folder of sf0001 holding acquisitions 3TV2 and 3TV10 only; returns that folder."""
    build_senior_tree(root)
    subjects = root / "derivatives" / "subjects_f"
    for acquisition in ("3TV0", "3TV1"):
        shutil.rmtree(subjects / "sf0001" / "t1mri" / acquisition)
    return subjects


def _senior_session_graph(subjects, acquisition, side="L"):
    folds = subjects / "sf0001" / "t1mri" / acquisition / "default_analysis" / "folds" / "3.1"
    return folds / "default_session_auto" / f"{side}sf0001_default_session_auto.arg"


class TestFirstGraphInNaturalOrder:
    """REQ-ARGDISC-BDRABCZUK-B149FCFAD703 (temporary; supersedes REQ-PATTERNS-BDRABCZUK-C3BF41ED43D3): of several
    graphs of a hemisphere, the first in natural path order is rendered.
    """

    def test_3tv2_is_rendered_before_3tv10(self, tmp_path):
        subjects = _senior_two_acquisitions(tmp_path)
        render = _run(_script(tmp_path, subjects))
        assert _rendered_side(render, "L") == [str(_senior_session_graph(subjects, "3TV2"))]


class TestUnrenderedGraphsLogged:
    """REQ-ARGDISC-BDRABCZUK-E578619328F7 (temporary; supersedes REQ-PATTERNS-BDRABCZUK-0D651D3AE998, REQ-SNAPSULC-
    BDRABCZUK-9E011521A120): each graph the natural order choice leaves out is logged at INFO level.
    """

    def test_left_out_graph_path_is_logged_at_info(self, tmp_path, pipeline_caplog):
        subjects = _senior_two_acquisitions(tmp_path)
        _run(_script(tmp_path, subjects))
        left_out = str(_senior_session_graph(subjects, "3TV10"))
        assert [
            r
            for r in pipeline_caplog.records
            if r.levelno == logging.INFO and r.name == SNAPSHOTS_LOGGER and left_out in r.getMessage()
        ]


def _write_snapshot(graph_path, output_path, *args, **kwargs):
    """Stand-in render writing a one-byte file at output_path; returns that path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(b"\x89")
    return output_path


class TestFirstGraphOutputQc:
    """REQ-ARGDISC-BDRABCZUK-4F013D0F2F44 (temporary; supersedes REQ-SNAPSULC-BDRABCZUK-0422374638E8)."""

    def test_first_graph_snapshot_row_holds_qc_1(self, tmp_path):
        subjects = _senior_two_acquisitions(tmp_path)
        script = _script(tmp_path, subjects)
        with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=_write_snapshot):
            script._run_sulcal((800, 600))
        rows = {row["check"]: (row["qc"], row["reason"]) for row in script._output_qc.compute_rows()}
        assert rows["sulcal_graph_left.png"] == (1, "")

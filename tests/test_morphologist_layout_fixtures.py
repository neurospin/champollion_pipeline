"""Self-tests of the Morphologist layout fixtures (TASK-ARGDISC-BDRABCZUK-5F7CDC3F12EA).

tests/morphologist_layout_fixtures.py builds, below a folder the test gives (usually
tmp_path), trees of empty files and folders named after the layouts seen on rosette:
the classic BrainVISA layout (IXI), the morphologist-cli layout (IDEASII), the
longitudinal SENIOR layout, derived volumes without graphs, a raw T1 BIDS tree and a
mixed tree. Names only: no file holds data. Each test pins the full file listing of
one builder, so the later ARGDISC Tasks test against a known tree.

Expected listings are written out here from the layouts, independently of the builders.

* REQ-ARGDISC-BDRABCZUK-F7AE27359C40 - test_classic_tree
* REQ-ARGDISC-BDRABCZUK-24722D799F88 - test_morphologist_cli_tree
* REQ-ARGDISC-BDRABCZUK-761268DDDBB4 - test_senior_tree
* REQ-ARGDISC-BDRABCZUK-5C10929FD09A - test_segmentation_volumes_tree
* REQ-ARGDISC-BDRABCZUK-1B18750C2029 - test_suffix_volumes_tree
* REQ-ARGDISC-BDRABCZUK-542AD7347634 - test_raw_t1_bids_tree
* REQ-ARGDISC-BDRABCZUK-256C6BCF7154 - test_mixed_tree
"""

from pathlib import Path

from tests.morphologist_layout_fixtures import (
    build_classic_tree,
    build_mixed_tree,
    build_morphologist_cli_tree,
    build_raw_t1_bids_tree,
    build_segmentation_volumes_tree,
    build_senior_tree,
    build_suffix_volumes_tree,
)

HEMIS = ("L", "R")
CLI_VOLUMES = ("brain", "skull_stripped", "nobias", "hfiltered", "variance", "edges", "voronoi", "whiteridge")
CLI_HEMI_VOLUMES = ("cortex", "grey_white", "gw_interface", "roots", "skeleton")
SENIOR_ACQUISITIONS = ("3TV0", "3TV1", "3TV2", "3TV10")
SENIOR_SESSIONS = ("default_session_auto", "long_session_auto")
SENIOR_SEGMENTATION = ("skull_stripped", "Lskeleton", "Rskeleton")


def files_below(root: Path) -> list[str]:
    """Sorted paths, relative to root, of the files below root."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def folders_below(root: Path, suffix: str) -> list[str]:
    """Sorted paths, relative to root, of the folders below root whose name ends with suffix."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob(f"*{suffix}") if path.is_dir())


def data_folders(graphs: list[str]) -> list[str]:
    """The <graph>.data folder that goes with each .arg graph, sorted."""
    return sorted(graph.removesuffix(".arg") + ".data" for graph in graphs if graph.endswith(".arg"))


def classic_subject(subject: str) -> list[str]:
    """Files of one subject of the classic BrainVISA layout (IXI)."""
    acquisition = f"{subject}/t1mri/default_acquisition"
    analysis = f"{acquisition}/default_analysis"
    folds = f"{analysis}/folds/3.1"
    graphs = [f"{folds}/{hemi}{subject}.arg" for hemi in HEMIS]
    labelled = [f"{folds}/default_session_auto/{hemi}{subject}_default_session_auto.arg" for hemi in HEMIS]
    volumes = [
        f"{analysis}/segmentation/{name}_{subject}.nii.gz"
        for name in ("brain", "skull_stripped", "Lskeleton", "Rskeleton", "voronoi")
    ]
    meshes = [f"{analysis}/segmentation/mesh/{subject}_{hemi}white.gii" for hemi in HEMIS]
    t1_and_bias = [f"{acquisition}/{subject}.nii.gz", f"{analysis}/nobias_{subject}.nii.gz"]
    return graphs + labelled + volumes + meshes + t1_and_bias


def cli_segmentation(subject: str) -> list[str]:
    """Derived volumes of one subject in a morphologist-cli segmentation folder."""
    stem = f"{subject}/ses-1/run-0/ana-0/segmentation/{subject}_ses-1_run-0"
    volumes = [f"{stem}_{name}.nii.gz" for name in CLI_VOLUMES]
    return volumes + [f"{stem}_hemi-{hemi}_{name}.nii.gz" for hemi in HEMIS for name in CLI_HEMI_VOLUMES]


def cli_subject(subject: str, labelled: bool) -> list[str]:
    """Files of one subject of the morphologist-cli layout (IDEASII); labelled adds the sul-0_auto graphs."""
    analysis = f"{subject}/ses-1/run-0/ana-0"
    stem = f"{subject}_ses-1_run-0"
    graphs = [f"{analysis}/folds/3.1/{stem}_hemi-{hemi}.arg" for hemi in HEMIS]
    if labelled:
        graphs += [f"{analysis}/folds/3.1/sul-0_auto/{stem}_sul-0_hemi-{hemi}_auto.arg" for hemi in HEMIS]
    meshes = [f"{analysis}/mesh/{stem}_hemi-{hemi}_white.gii" for hemi in HEMIS]
    return graphs + meshes + cli_segmentation(subject)


def cli_tree() -> list[str]:
    """Files of the whole morphologist-cli tree: sub-01 labelled, sub-02 with unlabelled graphs alone."""
    return ["qc/qc.tsv", *cli_subject("sub-01", labelled=True), *cli_subject("sub-02", labelled=False)]


def senior_acquisition(group: str, subject: str, acquisition: str) -> list[str]:
    """Files of one SENIOR acquisition, each .arg and .nii.gz with its .minf sibling."""
    root = f"derivatives/{group}/{subject}/t1mri/{acquisition}"
    analysis = f"{root}/default_analysis"
    folds = f"{analysis}/folds/3.1"
    graphs = [f"{folds}/{hemi}{subject}.arg" for hemi in HEMIS]
    graphs += [f"{folds}/{session}/{hemi}{subject}_{session}.arg" for session in SENIOR_SESSIONS for hemi in HEMIS]
    volumes = [f"{root}/{subject}.nii.gz", f"{analysis}/nobias_{subject}.nii.gz"]
    volumes += [f"{analysis}/segmentation/{name}_{subject}.nii.gz" for name in SENIOR_SEGMENTATION]
    named = graphs + volumes
    return named + [f"{name}.minf" for name in named]


def raw_t1_tree() -> list[str]:
    """Files of the raw T1 BIDS tree."""
    return [f"{subject}/ses-V1/anat/{subject}_ses-V1_T1w.nii.gz" for subject in ("sub-01", "sub-02")]


def test_classic_tree(tmp_path: Path) -> None:
    """Two IXI-like subjects: unlabelled and default_session_auto graphs, segmentation volumes, meshes, T1."""
    assert build_classic_tree(tmp_path) == tmp_path
    expected = classic_subject("IXI001-Guys-0001") + classic_subject("IXI002-HH-0002")
    assert files_below(tmp_path) == sorted(expected)
    assert folders_below(tmp_path, ".data") == data_folders(expected)


def test_morphologist_cli_tree(tmp_path: Path) -> None:
    """qc/qc.tsv, sub-01 with hemi and sul-0_auto graphs, sub-02 with unlabelled graphs alone."""
    assert build_morphologist_cli_tree(tmp_path) == tmp_path
    expected = cli_tree()
    assert files_below(tmp_path) == sorted(expected)
    assert folders_below(tmp_path, ".data") == data_folders(expected)


def test_senior_tree(tmp_path: Path) -> None:
    """Two subjects folders, four longitudinal acquisitions each, two labelling sessions, .minf siblings."""
    assert build_senior_tree(tmp_path) == tmp_path
    expected = [
        name
        for group, subject in (("subjects_f", "sf0001"), ("subjects_h", "sh0001"))
        for acquisition in SENIOR_ACQUISITIONS
        for name in senior_acquisition(group, subject, acquisition)
    ]
    assert files_below(tmp_path) == sorted(expected)
    assert folders_below(tmp_path, ".data") == data_folders(expected)


def test_segmentation_volumes_tree(tmp_path: Path) -> None:
    """The morphologist-cli segmentation volumes of sub-01 and no graph."""
    assert build_segmentation_volumes_tree(tmp_path) == tmp_path
    assert files_below(tmp_path) == sorted(cli_segmentation("sub-01"))


def test_suffix_volumes_tree(tmp_path: Path) -> None:
    """Volumes named with Morphologist derived suffixes, outside any segmentation folder, and no graph."""
    assert build_suffix_volumes_tree(tmp_path) == tmp_path
    suffixes = ("brain", "skull_stripped", "nobias", "voronoi", "hemi-L_skeleton")
    assert files_below(tmp_path) == sorted(f"sub-01/sub-01_{suffix}.nii.gz" for suffix in suffixes)


def test_raw_t1_bids_tree(tmp_path: Path) -> None:
    """Two raw T1 files in a BIDS tree and nothing else."""
    assert build_raw_t1_bids_tree(tmp_path) == tmp_path
    assert files_below(tmp_path) == sorted(raw_t1_tree())


def test_mixed_tree(tmp_path: Path) -> None:
    """The raw T1 BIDS tree below rawdata and the morphologist-cli tree below morphologist."""
    assert build_mixed_tree(tmp_path) == tmp_path
    expected = [f"rawdata/{name}" for name in raw_t1_tree()] + [f"morphologist/{name}" for name in cli_tree()]
    assert files_below(tmp_path) == sorted(expected)
    assert folders_below(tmp_path, ".data") == data_folders(expected)

"""Morphologist output layout fixtures for the ARGDISC epic (TASK-ARGDISC-BDRABCZUK-5F7CDC3F12EA).

Each builder creates, below the folder it is given (usually a test's tmp_path), a tree of
empty files and folders named after a layout seen on rosette, and returns that folder.
Names only: no file holds data. Shared by the tests of epic TASK-ARGDISC-BDRABCZUK-D91B3C7ACD31;
tests/test_morphologist_layout_fixtures.py pins each tree.
"""

from pathlib import Path

HEMISPHERES = ("L", "R")
CLASSIC_SUBJECTS = ("IXI001-Guys-0001", "IXI002-HH-0002")
CLASSIC_SEGMENTATION = ("brain", "skull_stripped", "Lskeleton", "Rskeleton", "voronoi")
CLI_SUBJECTS = ("sub-01", "sub-02")
CLI_LABELLED_SUBJECTS = ("sub-01",)
CLI_VOLUMES = ("brain", "skull_stripped", "nobias", "hfiltered", "variance", "edges", "voronoi", "whiteridge")
CLI_HEMI_VOLUMES = ("cortex", "grey_white", "gw_interface", "roots", "skeleton")
SENIOR_SUBJECTS = (("subjects_f", "sf0001"), ("subjects_h", "sh0001"))
SENIOR_ACQUISITIONS = ("3TV0", "3TV1", "3TV2", "3TV10")
SENIOR_SESSIONS = ("default_session_auto", "long_session_auto")
SENIOR_SEGMENTATION = ("skull_stripped", "Lskeleton", "Rskeleton")
DERIVED_SUFFIXES = ("brain", "skull_stripped", "nobias", "voronoi", "hemi-L_skeleton")
RAW_SESSION = "ses-V1"


def _create(root: Path, names: list[str]) -> Path:
    """Create each relative name below root as an empty file, in sorted order; a .arg graph also
    gets its empty <stem>.data folder. Returns root."""
    for name in sorted(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        if path.suffix == ".arg":
            path.with_suffix(".data").mkdir(exist_ok=True)
    return root


def _classic_names(subject: str) -> list[str]:
    """Relative names of one subject of the classic BrainVISA layout."""
    acquisition = f"{subject}/t1mri/default_acquisition"
    analysis = f"{acquisition}/default_analysis"
    folds = f"{analysis}/folds/3.1"
    names = [f"{acquisition}/{subject}.nii.gz", f"{analysis}/nobias_{subject}.nii.gz"]
    for hemi in HEMISPHERES:
        names.append(f"{folds}/{hemi}{subject}.arg")
        names.append(f"{folds}/default_session_auto/{hemi}{subject}_default_session_auto.arg")
        names.append(f"{analysis}/segmentation/mesh/{subject}_{hemi}white.gii")
    names.extend(f"{analysis}/segmentation/{volume}_{subject}.nii.gz" for volume in CLASSIC_SEGMENTATION)
    return names


def _cli_segmentation_names(subject: str) -> list[str]:
    """Relative names of the derived volumes in one morphologist-cli segmentation folder."""
    prefix = f"{subject}/ses-1/run-0/ana-0/segmentation/{subject}_ses-1_run-0"
    names = [f"{prefix}_{volume}.nii.gz" for volume in CLI_VOLUMES]
    for hemi in HEMISPHERES:
        names.extend(f"{prefix}_hemi-{hemi}_{volume}.nii.gz" for volume in CLI_HEMI_VOLUMES)
    return names


def _cli_names(subject: str) -> list[str]:
    """Relative names of one subject of the morphologist-cli layout; labelled subjects add sul-0_auto graphs."""
    analysis = f"{subject}/ses-1/run-0/ana-0"
    prefix = f"{subject}_ses-1_run-0"
    names = _cli_segmentation_names(subject)
    for hemi in HEMISPHERES:
        names.append(f"{analysis}/folds/3.1/{prefix}_hemi-{hemi}.arg")
        names.append(f"{analysis}/mesh/{prefix}_hemi-{hemi}_white.gii")
        if subject in CLI_LABELLED_SUBJECTS:
            names.append(f"{analysis}/folds/3.1/sul-0_auto/{prefix}_sul-0_hemi-{hemi}_auto.arg")
    return names


def _senior_names(group: str, subject: str, acquisition: str) -> list[str]:
    """Relative names of one SENIOR acquisition; every graph and volume has a .minf sibling."""
    base = f"derivatives/{group}/{subject}/t1mri/{acquisition}"
    analysis = f"{base}/default_analysis"
    folds = f"{analysis}/folds/3.1"
    names = [f"{base}/{subject}.nii.gz", f"{analysis}/nobias_{subject}.nii.gz"]
    names.extend(f"{analysis}/segmentation/{volume}_{subject}.nii.gz" for volume in SENIOR_SEGMENTATION)
    for hemi in HEMISPHERES:
        names.append(f"{folds}/{hemi}{subject}.arg")
        names.extend(f"{folds}/{session}/{hemi}{subject}_{session}.arg" for session in SENIOR_SESSIONS)
    return names + [f"{name}.minf" for name in names]


def build_classic_tree(root: Path) -> Path:
    """Classic BrainVISA layout (IXI): <subject>/t1mri/default_acquisition/default_analysis/..."""
    return _create(root, [name for subject in CLASSIC_SUBJECTS for name in _classic_names(subject)])


def build_morphologist_cli_tree(root: Path) -> Path:
    """morphologist-cli layout (IDEASII): qc/qc.tsv and sub-<label>/ses-1/run-0/ana-0/{folds,mesh,segmentation}."""
    return _create(root, ["qc/qc.tsv", *(name for subject in CLI_SUBJECTS for name in _cli_names(subject))])


def build_senior_tree(root: Path) -> Path:
    """SENIOR layout: derivatives/{subjects_f,subjects_h}/<subject>/t1mri/<acquisition>/default_analysis/..."""
    names = [
        name
        for group, subject in SENIOR_SUBJECTS
        for acquisition in SENIOR_ACQUISITIONS
        for name in _senior_names(group, subject, acquisition)
    ]
    return _create(root, names)


def build_segmentation_volumes_tree(root: Path) -> Path:
    """Volumes of a morphologist-cli segmentation folder and no graph."""
    return _create(root, _cli_segmentation_names(CLI_SUBJECTS[0]))


def build_suffix_volumes_tree(root: Path) -> Path:
    """Volumes named with Morphologist derived suffixes outside any segmentation folder, and no graph."""
    subject = CLI_SUBJECTS[0]
    return _create(root, [f"{subject}/{subject}_{suffix}.nii.gz" for suffix in DERIVED_SUFFIXES])


def build_raw_t1_bids_tree(root: Path) -> Path:
    """Raw T1 BIDS tree: sub-<label>/ses-V1/anat/sub-<label>_ses-V1_T1w.nii.gz."""
    names = [f"{subject}/{RAW_SESSION}/anat/{subject}_{RAW_SESSION}_T1w.nii.gz" for subject in CLI_SUBJECTS]
    return _create(root, names)


def build_mixed_tree(root: Path) -> Path:
    """Raw T1 BIDS tree below rawdata and morphologist-cli tree below morphologist."""
    build_raw_t1_bids_tree(root / "rawdata")
    build_morphologist_cli_tree(root / "morphologist")
    return root

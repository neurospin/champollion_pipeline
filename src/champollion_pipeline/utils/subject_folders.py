"""Which entries of a Morphologist subjects folder are subjects.

A dot-folder never is, a folder named exactly qc is only when it holds a t1mri folder, and every other
folder is, so that a failed subject keeps its row in the input QC. Stdlib only, so that importing the
rule stays light. cortical_tiles cannot import the pipeline and keeps a verbatim mirror of it in
subjects.py.
"""

import glob
import os

QC_FOLDER = "qc"
T1MRI_FOLDER = "t1mri"
SESSION_FOLDERS = "ses-*"


def has_t1mri_folder(folder: str | os.PathLike) -> bool:
    """Tell whether a t1mri folder sits directly in folder or directly in one of its ses- folders."""
    base = glob.escape(os.fspath(folder))
    patterns = (os.path.join(base, T1MRI_FOLDER), os.path.join(base, SESSION_FOLDERS, T1MRI_FOLDER))
    return any(os.path.isdir(match) for pattern in patterns for match in sorted(glob.glob(pattern)))


def is_non_subject(path: str | os.PathLike) -> bool:
    """Tell whether the folder at path is a dot-folder or a qc folder that holds no t1mri folder.

    A trailing separator changes nothing: the path is normalised before its last component is taken.
    """
    name = os.path.basename(os.path.normpath(os.fspath(path)))
    return name.startswith(".") or (name == QC_FOLDER and not has_t1mri_folder(path))


def list_morphologist_subjects(subjects_dir: str | os.PathLike) -> list[str]:
    """Return the names of the subject folders directly under subjects_dir, in ascending order.

    A subjects_dir that is no folder gives an empty list; an OSError from listing it reaches the caller.
    """
    if not os.path.isdir(subjects_dir):
        return []
    subjects = []
    for name in sorted(os.listdir(subjects_dir)):
        path = os.path.join(subjects_dir, name)
        if os.path.isdir(path) and not is_non_subject(path):
            subjects.append(name)
    return subjects

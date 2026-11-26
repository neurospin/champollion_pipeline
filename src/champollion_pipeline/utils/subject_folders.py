"""The subject folder of a Morphologist graph, and which entries of a Morphologist subjects folder are subjects.

Single owner of "subject folder" in the pipeline: every module that needs the subject folder of a graph calls
find_graph_subject_folder, and every lister of a subjects folder calls list_morphologist_subjects. No folder name such
as t1mri or ses-* plays a part. A dot-folder is never a subject, a folder named exactly qc is one only when it holds a
.arg file, and every other folder is, so that a failed subject keeps its row in the input QC. Stdlib only, so that
importing the rule stays light. cortical_tiles cannot import the pipeline and keeps its own mirror in subjects.py.
"""

import os
from collections.abc import Callable

QC_FOLDER = "qc"
T1MRI_FOLDER = "t1mri"
SESSION_FOLDERS = "ses-*"
SUBJECT_PREFIX = "sub-"
GRAPH_SUFFIX = ".arg"
SIDE_LETTERS: tuple[str, ...] = ("", "L", "R")
NAME_SEPARATORS: tuple[str, ...] = (".", "_")


def _is_named_after(folder: str, name: str) -> bool:
    """Tell whether a graph file ``name`` starts with ``folder``, an optional side letter before, a separator after."""
    return any(name.startswith(side + folder + separator) for side in SIDE_LETTERS for separator in NAME_SEPARATORS)


def _nearest(chain: tuple[str, ...], test: Callable[[str], bool]) -> int | None:
    """Return the index of the last folder of ``chain`` that passes ``test``, None when none does."""
    return next((index for index in range(len(chain) - 1, -1, -1) if test(chain[index])), None)


def find_graph_subject_folder(root_name: str, parts: tuple[str, ...], name: str) -> tuple[str, ...] | None:
    """Return the subject folder of the graph ``name`` in the folder ``parts`` below an input named ``root_name``.

    The nearest folder whose name starts with sub-, else the nearest folder named N with ``name`` starting N., N_,
    LN., LN_, RN. or RN_, else the folder directly in the input. Pure: no I/O.

    Args:
        root_name: the input folder's own name, ``os.path.basename(os.path.abspath(input_path))``.
        parts: folder names from the input down to the graph's folder, as ``input_kind.graph_files`` yields them.
        name: the graph file name.

    Returns:
        The subject folder as a prefix of ``parts`` (``()`` = the input itself), or None when the graph sits directly
        in an input whose name does not start with sub- (a loose graph: the caller raises LooseGraphError).

    Complexity: O(len(parts)) string compares (6 per folder at most).
    """
    if not parts:
        return () if root_name.startswith(SUBJECT_PREFIX) else None
    chain = (root_name, *parts)
    index = _nearest(chain, lambda folder: folder.startswith(SUBJECT_PREFIX))
    if index is None:
        index = _nearest(chain, lambda folder: _is_named_after(folder, name))
    result = parts[:1] if index is None else parts[:index]
    assert result == parts[: len(result)], "find_graph_subject_folder post: result is a prefix of parts"
    return result


def has_graph(folder: str | os.PathLike) -> bool:
    """Tell whether a file named *.arg, not starting with a dot, sits in folder or below it, outside dot folders.

    A missing folder gives False. Complexity: O(entries below folder) until the first graph.
    """
    pending = [os.fspath(folder)]
    while pending:
        try:
            entries = sorted(os.scandir(pending.pop()), key=lambda entry: entry.name)
        except OSError:
            continue
        subfolders = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if not entry.is_symlink():
                    subfolders.append(entry.path)
            elif entry.name.endswith(GRAPH_SUFFIX):
                return True
        pending.extend(reversed(subfolders))
    return False


def is_non_subject(path: str | os.PathLike) -> bool:
    """Tell whether the folder at path is a dot-folder or a qc folder that holds no .arg file.

    A trailing separator changes nothing: the path is normalised before its last component is taken.
    """
    name = os.path.basename(os.path.normpath(os.fspath(path)))
    return name.startswith(".") or (name == QC_FOLDER and not has_graph(path))


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

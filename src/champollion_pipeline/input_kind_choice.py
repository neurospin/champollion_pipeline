"""Choice of the kind main.py routes dataset.input_path as: detection, dataset.input_kind, a terminal prompt or a stop.

Every stop is an InputChoiceError whose text is the one ERROR record main.py logs before it ends with the error's
exit_status (1; 2 for NeitherInput, its subclass DerivedVolumesOnly, and LooseGraphInput).
Detection, the sourcedata filter and the subjects folder search stay in champollion_pipeline.utils.input_kind; this
module only decides and asks.
"""

import logging
import sys

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.utils.graph_item import LooseGraphError
from champollion_pipeline.utils.input_kind import (
    AmbiguousSubjectsFolders,
    InputKind,
    InputScan,
    derived_volumes,
    resolve_subjects_dir,
    scan_input,
    sourcedata_t1_files,
)

INPUT_KINDS: tuple[str, ...] = (InputKind.T1.value, InputKind.MORPHOLOGIST.value)

_KIND_QUESTION = "Input kind (t1 or morphologist): "
_FOLDER_QUESTION = "Subjects folder (one path as listed): "

logger = logging.getLogger(__name__)


class InputChoiceError(ValueError):
    """The run stops with exit status exit_status; str(error) is the ERROR message it logs."""

    exit_status: int = 1


class NeitherInput(InputChoiceError):
    """The input holds neither T1 files nor Morphologist graphs."""

    exit_status = 2  # the usage-error status of cli (_USAGE_ERROR_STATUS)


class DerivedVolumesOnly(NeitherInput):
    """The input holds Morphologist derived volumes and no T1 file or graph."""


class LooseGraphInput(InputChoiceError):
    """A .arg file sits directly in an input whose name does not start with sub-; the text names that file."""

    exit_status = 2  # the usage-error status of cli (_USAGE_ERROR_STATUS)


def check_input_kind_value(dataset: DatasetConfig) -> None:  # noqa: V103 - used by main.py
    """Raise InputChoiceError when dataset.input_kind is not None, t1 or morphologist."""
    value = dataset.input_kind
    if value is not None and value not in INPUT_KINDS:
        raise InputChoiceError(
            f"dataset.input_kind is {value!r}; it must be t1 or morphologist (--input-kind t1|morphologist)."
        )


def choose_input_kind(dataset: DatasetConfig) -> InputKind | None:  # noqa: V103 - used by main.py
    """Return the kind to route dataset.input_path as, None when input_path and input_kind are both empty.

    A prompt answer is stored in dataset.input_kind. Raises InputChoiceError for every stop.
    """
    input_path = dataset.input_path
    if not input_path:
        if dataset.input_kind is None:
            return None
        raise InputChoiceError(
            f"dataset.input_kind is {dataset.input_kind} but dataset.input_path is empty; "
            f"set --input-path to the folder holding the {dataset.input_kind} input."
        )
    scan = scan_input(input_path)
    if scan.kind is InputKind.NEITHER:
        if scan.first_derived is not None:
            raise DerivedVolumesOnly(_derived_only_text(input_path, scan.first_derived))
        raise NeitherInput(
            f"{input_path} holds neither T1 files nor Morphologist outputs (or is not a folder); point --input-path "
            "at a folder holding one kind (--input-kind only chooses between kinds that are there)."
        )
    kind = scan.kind
    if dataset.input_kind is None:
        answer = _undecided_kind(input_path, scan)
        if answer is not None:
            dataset.input_kind = answer
    if dataset.input_kind is not None:
        kind = _checked_kind(dataset, scan)
    if kind is InputKind.T1 and scan.first_graph is None:
        _warn_derived_volumes(input_path)
    return kind


def choose_subjects_folder(dataset: DatasetConfig) -> str | None:  # noqa: V103 - used by main.py
    """Return the subjects folder of a Morphologist dataset.input_path, as resolve_subjects_dir or a prompt gives it.

    None only when input_kind is None and the input is itself one subject. Raises InputChoiceError for every stop,
    LooseGraphInput (exit status 2) for a .arg file loose in the input.
    """
    try:
        folder = resolve_subjects_dir(dataset.input_path)
    except AmbiguousSubjectsFolders as error:
        return _ask_subjects_folder(error)
    except LooseGraphError as error:
        raise LooseGraphInput(str(error)) from error
    if folder is None and dataset.input_kind == InputKind.MORPHOLOGIST.value:
        raise InputChoiceError(_no_subjects_folder_text(dataset.input_path, None))
    return folder


def _undecided_kind(input_path: str, scan: InputScan) -> str | None:
    """The answer of the input kind prompt for both kinds or sourcedata T1 files; None when nothing is ambiguous."""
    if scan.kind is InputKind.BOTH:
        lines = [
            f"{input_path} holds both kinds of input:",
            f"  t1: T1 files, e.g. {scan.first_t1}",
            f"  morphologist: Morphologist outputs, e.g. {scan.first_graph}",
        ]
        batch = (
            f"{input_path} holds both T1 files ({scan.first_t1}) and Morphologist outputs ({scan.first_graph}); "
            "choose one with --input-kind t1 or --input-kind morphologist, or point --input-path at a folder "
            "holding one kind."
        )
    else:
        found = sourcedata_t1_files(input_path) if scan.kind is InputKind.T1 else []
        if not found:
            return None
        lines = [
            f"{input_path} holds T1 files below a sourcedata folder, e.g. {found[0]}:",
            "  t1: run Morphologist on every T1 file, sourcedata included",
            f"  morphologist: use the Morphologist outputs below {input_path}",
        ]
        batch = (
            f"{input_path} holds T1 files below a sourcedata folder ({found[0]}); choose with --input-kind t1 "
            "(run Morphologist on every T1 file, sourcedata included) or --input-kind morphologist, or point "
            "--input-path at the folder to use."
        )
    if not _is_terminal():
        raise InputChoiceError(batch)
    return _ask(lines, _KIND_QUESTION, INPUT_KINDS, f"No input kind was read for {input_path}; set --input-kind.")


def _checked_kind(dataset: DatasetConfig, scan: InputScan) -> InputKind:
    """The kind dataset.input_kind names, once the input is known to hold it."""
    input_path = dataset.input_path
    if dataset.input_kind == InputKind.T1.value:
        if scan.first_t1 is None:
            raise InputChoiceError(
                f"dataset.input_kind is t1 but {input_path} holds no T1 file, only Morphologist outputs "
                f"({scan.first_graph}); use --input-kind morphologist or point --input-path at the T1 files."
            )
        return InputKind.T1
    if scan.first_graph is None:
        raise InputChoiceError(_no_subjects_folder_text(input_path, scan.first_t1))
    return InputKind.MORPHOLOGIST


def _derived_only_text(input_path: str, first_derived: str) -> str:
    return (
        f"{input_path} holds Morphologist derived volumes (e.g. {first_derived}) and no T1 file or Morphologist "
        "graph: it looks like a Morphologist output layout without its graphs (.arg). Point --input-path at the "
        "folder holding the raw T1 files, or at the Morphologist subjects folder once its graphs exist."
    )


def _warn_derived_volumes(input_path: str) -> None:
    """Log one WARNING with the number of derived volumes left out of the T1 file list, when there are any."""
    found = derived_volumes(input_path)
    if found:
        logger.warning(
            "%d Morphologist derived volumes below %s are left out of the T1 file list (a folder named "
            "segmentation or a derived-volume name, e.g. %s).",
            len(found),
            input_path,
            found[0],
        )


def _no_subjects_folder_text(input_path: str, first_t1: str | None) -> str:
    held = f"; it holds T1 files ({first_t1})" if first_t1 is not None else ""
    return (
        f"dataset.input_kind is morphologist but no subjects folder found below {input_path}{held}; point "
        "--input-path at the folder holding the subject folders, set --morphologist-graphs, or use --input-kind t1."
    )


def _ask_subjects_folder(error: AmbiguousSubjectsFolders) -> str:
    """The subjects folder, among error.candidates, read at a terminal prompt."""
    if not _is_terminal():
        raise InputChoiceError(
            f"{error.path} holds several Morphologist subjects folders: {', '.join(error.candidates)}; "
            "pass the one to use as --input-path or --morphologist-graphs."
        )
    lines = [f"{error.path} holds several Morphologist subjects folders:", *(f"  {c}" for c in error.candidates)]
    return _ask(lines, _FOLDER_QUESTION, error.candidates, f"No subjects folder was read for {error.path}.")


def _is_terminal() -> bool:
    """Whether standard input and standard output are both terminals, read at call time."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(lines: list[str], question: str, choices: tuple[str, ...], on_eof: str) -> str:
    """One answer read after the prompt lines; it must be one of choices, as typed."""
    try:
        answer = input("\n".join([*lines, question]))
    except EOFError:
        raise InputChoiceError(f"{on_eof} main.py stops.") from None
    if answer not in choices:
        raise InputChoiceError(f"{answer!r} is not one of {', '.join(choices)}; main.py stops.")
    return answer

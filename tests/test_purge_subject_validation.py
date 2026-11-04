"""--subject validation of purge_subject.py (TASK-PATTERNS-BDRABCZUK-74D101E95145, audit H2).

purge_subject joins the derivatives path, a stage directory and --subject, then deletes the result.
Each unsafe or unknown --subject is refused before any deletion: an ERROR record from the
champollion_pipeline.purge_subject logger, a non-zero exit status, and an unchanged tree.

REQ-PATTERNS-BDRABCZUK-F4977EA12F53: an empty --subject is refused.
REQ-PATTERNS-BDRABCZUK-E509AB20771A: a --subject containing a path separator is refused.
REQ-PATTERNS-BDRABCZUK-43FBC52198ED: a --subject equal to '.' or '..' is refused.
REQ-PATTERNS-BDRABCZUK-8D99A13D7FAA: a --subject matching no per-subject directory, crop file or
aggregated subject CSV row is refused.
REQ-PATTERNS-BDRABCZUK-DD8EDC62611E: a per-subject directory resolving outside its stage directory
is refused.

Synthetic subject IDs and a synthetic derivatives tree under tmp_path only.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from champollion_pipeline.purge_subject import PurgeSubject

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.purge_subject"
STAGE_DIRS = ("skeletons", "foldlabels", "extremities", "transforms", "distmaps")
KEPT_SUBJECT = "sub-02"


def _tree(derivatives: Path) -> None:
    """A derivatives tree holding KEPT_SUBJECT in every stage directory, one crop file and one array."""
    for stage in STAGE_DIRS:
        subject_dir = derivatives / stage / KEPT_SUBJECT
        subject_dir.mkdir(parents=True)
        (subject_dir / "data.nii.gz").touch()
    mask_dir = derivatives / "crops" / "canonical_25" / "2mm" / "region_a" / "mask"
    (mask_dir / "Rcrops").mkdir(parents=True)
    (mask_dir / "Rcrops" / f"{KEPT_SUBJECT}.nii.gz").touch()
    np.save(mask_dir / "Rskeleton.npy", np.zeros((1, 1, 4, 4, 4), dtype=np.float32))
    pd.DataFrame({"participant_id": [KEPT_SUBJECT]}).to_csv(mask_dir / "Rskeleton_subject.csv", index=False)


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file and directory under root, with the file contents."""
    return {
        str(path.relative_to(root)): (b"<dir>" if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


def _exit_status(derivatives: Path, subject: str) -> int:
    """Exit status of purge_subject on derivatives: run()'s return value, or 1 for a refusal raised as an error."""
    script = PurgeSubject()
    script.args = script.parse_args([str(derivatives), "--subject", subject])
    try:
        return script.run()
    except SystemExit as exc:
        return 0 if exc.code in (None, 0) else 1
    except (ValueError, OSError):
        return 1


def _assert_refused(pipeline_caplog, status: int) -> None:
    assert status != 0
    errors = [record for record in pipeline_caplog.records if record.levelno >= logging.ERROR]
    assert errors, [(record.levelname, record.getMessage()) for record in pipeline_caplog.records]
    assert errors[0].name == LOGGER_NAME


def test_empty_subject_is_refused(tmp_path, pipeline_caplog):
    """REQ-PATTERNS-BDRABCZUK-F4977EA12F53."""
    derivatives = tmp_path / "derivatives"
    _tree(derivatives)
    before = _snapshot(derivatives)

    status = _exit_status(derivatives, "")

    assert _snapshot(derivatives) == before
    _assert_refused(pipeline_caplog, status)


@pytest.mark.parametrize("subject", ["absolute", "../../outside", "sub-02/x"])
def test_subject_with_path_separator_is_refused(tmp_path, pipeline_caplog, subject):
    """REQ-PATTERNS-BDRABCZUK-E509AB20771A."""
    derivatives = tmp_path / "derivatives"
    _tree(derivatives)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (derivatives / "skeletons" / KEPT_SUBJECT / "x").mkdir()
    if subject == "absolute":
        subject = str(outside)
    before = _snapshot(tmp_path)

    status = _exit_status(derivatives, subject)

    assert _snapshot(tmp_path) == before
    _assert_refused(pipeline_caplog, status)


@pytest.mark.parametrize("subject", [".", ".."])
def test_dot_subject_is_refused(tmp_path, pipeline_caplog, subject):
    """REQ-PATTERNS-BDRABCZUK-43FBC52198ED."""
    derivatives = tmp_path / "derivatives"
    _tree(derivatives)
    before = _snapshot(tmp_path)

    status = _exit_status(derivatives, subject)

    assert _snapshot(tmp_path) == before
    _assert_refused(pipeline_caplog, status)


def test_unknown_subject_is_refused(tmp_path, pipeline_caplog):
    """REQ-PATTERNS-BDRABCZUK-8D99A13D7FAA."""
    derivatives = tmp_path / "derivatives"
    _tree(derivatives)
    before = _snapshot(derivatives)

    status = _exit_status(derivatives, "sub-99")

    assert _snapshot(derivatives) == before
    _assert_refused(pipeline_caplog, status)


def test_subject_dir_resolving_outside_stage_dir_is_refused(tmp_path, pipeline_caplog):
    """REQ-PATTERNS-BDRABCZUK-DD8EDC62611E: transforms/<subject> is a symlink to a directory outside transforms/."""
    derivatives = tmp_path / "derivatives"
    _tree(derivatives)
    subject = "sub-01"
    skeleton_dir = derivatives / "skeletons" / subject
    skeleton_dir.mkdir()
    (skeleton_dir / "data.nii.gz").touch()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    (derivatives / "transforms" / subject).symlink_to(elsewhere, target_is_directory=True)
    before = _snapshot(tmp_path)

    status = _exit_status(derivatives, subject)

    assert _snapshot(tmp_path) == before
    _assert_refused(pipeline_caplog, status)

"""Transform QC of run_cortical_tiles.py (TASK-CORTILES-BDRABCZUK-81D525E049C9).

TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): cortical_tiles' generate_ICBM2009c_transforms.py
globs ``{subject}*/`` and can give a subject the ICBM2009c transform of another subject whose ID it
prefixes. After the cortical_tiles run, run_cortical_tiles.py compares the transform files
``<output>/cortical_tiles-<VERSION>/transforms/<side>/<side>transform_to_ICBM2009c_<ID>.trm`` by
content and reports the result in ``<output>/cortical_tiles-<VERSION>/transform_qc.tsv``. This file
goes with the check once cortical_tiles is fixed upstream.

REQ-CORTILES-BDRABCZUK-7A8278F78932: qc 0 row for each ID whose transform is byte-identical to
    another ID's transform of the same side.
REQ-CORTILES-BDRABCZUK-DF22FA179E58: the reason of a qc 0 row names the side and the other ID.
REQ-CORTILES-BDRABCZUK-0C949DBA7F1F: qc 1 row for each ID with a transform file and no qc 0 row.
REQ-CORTILES-BDRABCZUK-8C4C580A9194: WARNING record with the report path and the qc 0 row count.
"""

import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import read_qc_tsv

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"
CORTICAL_TILES_SCRIPT = "generate_sulcal_regions.py"
QC_FILENAME = "transform_qc.tsv"
SIDES = ("L", "R")
MINF_BYTES = b"attributes = {'format': 'TRM'}\n"  # identical for every ID, as BrainVISA writes it


def _transform_bytes(tag: str) -> bytes:
    """Content of one .trm file; equal tags give byte-identical files."""
    return f"0 0 0\n1 0 0\n0 1 0\n0 0 1  # {tag}\n".encode()


def _write_transforms(output_dir: Path, transforms: dict[tuple[str, str], str]) -> None:
    """Write <side>transform_to_ICBM2009c_<ID>.trm (content from its tag) and an identical .trm.minf."""
    for (side, subject), tag in transforms.items():
        side_dir = output_dir / DERIVATIVES_FOLDER / "transforms" / side
        side_dir.mkdir(parents=True, exist_ok=True)
        stem = side_dir / f"{side}transform_to_ICBM2009c_{subject}.trm"
        stem.write_bytes(_transform_bytes(tag))
        Path(f"{stem}.minf").write_bytes(MINF_BYTES)


def _run(temp_dir: str, transforms: dict[tuple[str, str], str], exit_code: int = 0):
    """Run RunCorticalTiles.run with every external step mocked.

    The mocked generate_sulcal_regions.py writes ``transforms`` ({(side, ID): content tag}) and
    exits ``exit_code``. Returns run()'s code and the path of transform_qc.tsv.
    """
    input_dir = Path(temp_dir) / "input"
    output_dir = Path(temp_dir) / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    for subject in sorted({subject for _, subject in transforms}):
        (input_dir / subject).mkdir()

    def execute(command, *_args, **_kwargs):
        if any(str(part).endswith(CORTICAL_TILES_SCRIPT) for part in command):
            _write_transforms(output_dir, transforms)
            return exit_code
        return 0

    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            "graphs",
            "--path_sk_with_hull",
            "skeleton",
            "--allow-prefix-ids",
        ]
    )
    with (
        patch.object(script, "execute_command", side_effect=execute),
        patch.object(script, "_generate_mask_npys"),
        patch(
            "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes",
            MagicMock(),
        ),
        patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle", MagicMock()),
    ):
        result = script.run()
    return result, output_dir / DERIVATIVES_FOLDER / QC_FILENAME


def _rows(qc_path: Path) -> dict[str, dict]:
    assert qc_path.is_file(), f"{qc_path} was not written"
    return {record["participant_id"]: record for record in read_qc_tsv(qc_path)}


def _warnings(pipeline_caplog, text: str) -> list[str]:
    return [
        record.getMessage()
        for record in pipeline_caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING and text in record.getMessage()
    ]


# sub-1 received sub-10's left transform; every other file is distinct.
LEFT_DUPLICATE = {
    ("L", "sub-1"): "sub-10 left",
    ("R", "sub-1"): "sub-1 right",
    ("L", "sub-10"): "sub-10 left",
    ("R", "sub-10"): "sub-10 right",
    ("L", "sub-2"): "sub-2 left",
    ("R", "sub-2"): "sub-2 right",
}

# sub-1 received sub-10's right transform; every other file is distinct.
RIGHT_DUPLICATE = {
    ("L", "sub-1"): "sub-1 left",
    ("R", "sub-1"): "sub-10 right",
    ("L", "sub-10"): "sub-10 left",
    ("R", "sub-10"): "sub-10 right",
    ("L", "sub-2"): "sub-2 left",
    ("R", "sub-2"): "sub-2 right",
}

ALL_DISTINCT = {(side, subject): f"{subject} {side}" for side in SIDES for subject in ("sub-1", "sub-10", "sub-2")}


class TestDuplicateTransformFlagged:
    """REQ-CORTILES-BDRABCZUK-7A8278F78932."""

    def test_both_ids_of_a_byte_identical_pair_get_qc_0(self, temp_dir):
        result, qc_path = _run(temp_dir, LEFT_DUPLICATE)

        rows = _rows(qc_path)
        assert result == 0
        assert rows["sub-1"]["qc"] == 0
        assert rows["sub-10"]["qc"] == 0

    def test_report_has_the_qc_report_columns(self, temp_dir):
        _, qc_path = _run(temp_dir, LEFT_DUPLICATE)

        assert qc_path.is_file(), f"{qc_path} was not written"
        header = qc_path.read_text(encoding="utf-8").splitlines()[0]
        assert header.split("\t") == ["participant_id", "qc", "reason"]

    def test_left_and_right_files_of_one_id_are_not_compared(self, temp_dir):
        transforms = dict(ALL_DISTINCT)
        transforms[("R", "sub-2")] = transforms[("L", "sub-2")]

        _, qc_path = _run(temp_dir, transforms)

        assert _rows(qc_path)["sub-2"]["qc"] == 1

    def test_identical_minf_sidecars_are_not_compared(self, temp_dir):
        _, qc_path = _run(temp_dir, ALL_DISTINCT)

        assert [subject for subject, row in _rows(qc_path).items() if row["qc"] == 0] == []

    def test_no_report_when_cortical_tiles_fails(self, temp_dir):
        result, qc_path = _run(temp_dir, LEFT_DUPLICATE, exit_code=1)

        assert result == 1
        assert not qc_path.exists()


class TestDuplicateTransformReason:
    """REQ-CORTILES-BDRABCZUK-DF22FA179E58."""

    @pytest.mark.parametrize(
        ("transforms", "side"), [(LEFT_DUPLICATE, "L"), (RIGHT_DUPLICATE, "R")], ids=["left", "right"]
    )
    def test_reason_names_side_and_other_id(self, temp_dir, transforms, side):
        _, qc_path = _run(temp_dir, transforms)

        rows = _rows(qc_path)
        for subject, other in (("sub-1", "sub-10"), ("sub-10", "sub-1")):
            reason = rows[subject]["reason"]
            assert re.search(rf"(?<![\w-]){re.escape(other)}(?![\w-])", reason), (subject, reason)
            assert re.search(rf"\b{side}\b", reason), (subject, reason)


class TestDistinctTransformKept:
    """REQ-CORTILES-BDRABCZUK-0C949DBA7F1F."""

    def test_ids_with_distinct_transforms_get_qc_1(self, temp_dir):
        _, qc_path = _run(temp_dir, ALL_DISTINCT)

        rows = _rows(qc_path)
        assert {subject: row["qc"] for subject, row in rows.items()} == {"sub-1": 1, "sub-10": 1, "sub-2": 1}

    def test_id_outside_a_duplicate_pair_gets_qc_1(self, temp_dir):
        _, qc_path = _run(temp_dir, LEFT_DUPLICATE)

        assert _rows(qc_path)["sub-2"]["qc"] == 1

    def test_id_with_one_side_only_gets_a_row(self, temp_dir):
        transforms = dict(ALL_DISTINCT)
        transforms[("L", "sub-3")] = "sub-3 left"

        _, qc_path = _run(temp_dir, transforms)

        assert _rows(qc_path)["sub-3"]["qc"] == 1


class TestDuplicateTransformWarning:
    """REQ-CORTILES-BDRABCZUK-8C4C580A9194."""

    def test_warning_names_report_path_and_qc_0_count(self, temp_dir, pipeline_caplog):
        transforms = dict(ALL_DISTINCT)
        transforms[("L", "sub-100")] = "sub-100 left"
        transforms[("R", "sub-100")] = transforms[("R", "sub-10")]
        transforms[("R", "sub-1")] = transforms[("R", "sub-10")]

        _, qc_path = _run(temp_dir, transforms)

        warnings = _warnings(pipeline_caplog, str(qc_path))
        assert len(warnings) == 1, warnings
        assert re.search(r"(?<![\w-])3(?![\w-])", warnings[0].replace(str(qc_path), "")), warnings[0]

    def test_no_warning_without_qc_0_row(self, temp_dir, pipeline_caplog):
        _run(temp_dir, ALL_DISTINCT)

        assert _warnings(pipeline_caplog, QC_FILENAME) == []

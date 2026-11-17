"""Listing order of generate_embeddings.py and of stage 1's reader (TASK-READORDER-BDRABCZUK-BB3841162154).

User rule: files must always be read in the same order, for reproducibility. The order control comes from
test_listing_order_src_fixes.py: it replaces what os.scandir and os.listdir yield, below glob and pathlib.

* REQ-READORDER-BDRABCZUK-F2A33460AB76 - _find_crop_dir takes the sorted-first match.
* REQ-READORDER-BDRABCZUK-12AA4FF2D371 - _patch_config_paths does not depend on listing order.
* REQ-READORDER-BDRABCZUK-F72EFE212710 - the morphologist-cli arguments do not depend on listing order.
* REQ-READORDER-BDRABCZUK-0A5BFBBF3CA4 - the stage 1 input_qc.tsv does not depend on listing order.
"""

from pathlib import Path

import pytest

from tests.test_generate_embeddings_strategies import make_region_logs, make_script
from tests.test_generate_morphologist_graphs import _input_qc_path, _run_stage, _write_nifti
from tests.test_listing_order_src_fixes import _ascending, _listing_in, unsorted_orders

MODEL_REGION = "SOr_left"
CROP_DIRS = ("S.O.r.", "S.Or.", "SOr")  # three names that reduce to the same model name once their dots are stripped
SORTED_FIRST_CROP_DIR = "S.O.r."

CONFIG_TARGET = "cortical_tiles-2026"
CONFIG_FILES = ("a.yaml", "m/b.yaml", "m/n/c.yaml", "z.yaml")


# A tmp models dir with an arbitrary name is not a Mask version: the parser needs --masks
# (REQ-MODELVER-BDRABCZUK-1B06D337ECC0); canonical_25 keeps the masks these tests always ran with.
MASKS_ARGS = ("--masks", "canonical_25")


@pytest.mark.unit
class TestRglobOrderControl:
    """The first audit's guards cover glob and Path.glob; _patch_config_paths lists with Path.rglob."""

    @unsorted_orders
    def test_scandir_order_reaches_rglob(self, tmp_path, order):
        names = ["a.yaml", "b.yaml", "c.yaml"]
        for name in names:
            (tmp_path / name).touch()
        with _listing_in(order):
            found = [path.name for path in tmp_path.rglob("*.yaml")]
        assert found == order(names)


def _make_crop_dirs(crops: Path) -> None:
    for name in CROP_DIRS:
        (crops / name).mkdir(parents=True)


@pytest.mark.unit
class TestFindCropDirOrder:
    """generate_embeddings.py _find_crop_dir: the first match of an unsorted listing names the crop that is read."""

    @unsorted_orders
    def test_sorted_first_crop_dir_is_returned(self, tmp_path, order):
        _make_crop_dirs(tmp_path)
        script = make_script(["/m", "/d"])
        with _listing_in(order):
            found = script._find_crop_dir(str(tmp_path), MODEL_REGION)
        assert found == SORTED_FIRST_CROP_DIR

    @unsorted_orders
    def test_evaluate_reads_skeleton_of_sorted_first_crop_dir(self, tmp_path, order):
        models = tmp_path / "models"
        make_region_logs(models / MODEL_REGION)
        crops = tmp_path / "crops"
        _make_crop_dirs(crops)
        script = make_script([str(models), "/d", *MASKS_ARGS])
        calls = []
        script.execute_command = lambda cmd, shell=False: calls.append(cmd) or 0
        with _listing_in(order):
            script._run_per_region("evaluate.py", str(crops), str(tmp_path / "out"))
        command = calls[0]
        assert command[command.index("-sk") + 1] == str(crops / SORTED_FIRST_CROP_DIR / "mask" / "Lskeleton.npy")


def _config_text(folder: str) -> str:
    return f"dataset: /data/derivatives/{folder}/crops/2mm/S.Or./mask\n"


@pytest.mark.unit
class TestPatchConfigPathsOrder:
    """generate_embeddings.py _patch_config_paths: each file is rewritten on its own, so its rglob stays harmless."""

    @unsorted_orders
    def test_yaml_content_does_not_depend_on_listing_order(self, tmp_path, order):
        paths = [tmp_path / name for name in CONFIG_FILES]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_config_text("deep_folding-2025"))
        script = make_script(["/m", "/d"])
        with _listing_in(order):
            script._patch_config_paths(str(tmp_path), CONFIG_TARGET)
        assert [path.read_text() for path in paths] == [_config_text(CONFIG_TARGET)] * len(paths)


def _add_nifti(input_dir: Path, relative_path: str) -> None:
    path = input_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_nifti(path)


def _stage_dirs(tmp_path: Path, relative_paths: tuple[str, ...]) -> tuple[Path, Path]:
    """Return (input_dir, output_dir), both existing, the input holding a NIfTI file at each relative path."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    for relative_path in relative_paths:
        _add_nifti(input_dir, relative_path)
    return input_dir, output_dir


@pytest.mark.unit
class TestStage1InputOrder:
    """generate_morphologist_graphs.py lists nothing itself: find_t1_files reads the folders and sorts the result."""

    @unsorted_orders
    def test_cli_arguments_do_not_depend_on_listing_order(self, tmp_path, order):
        input_dir, output_dir = _stage_dirs(tmp_path, ("a/s3.nii.gz", "b/c/s4.nii.gz", "b/s2.nii.gz", "s1.nii.gz"))
        with _listing_in(_ascending):
            _, expected = _run_stage(input_dir, output_dir)
        with _listing_in(order):
            _, found = _run_stage(input_dir, output_dir)
        assert found.call_args.args[0] == expected.call_args.args[0]

    @unsorted_orders
    def test_input_qc_file_does_not_depend_on_listing_order(self, tmp_path, order):
        input_dir, output_dir = _stage_dirs(tmp_path, ("a/s9.nii.gz", "b/s2.nii.gz", "c/s2.nii.gz", "s1.nii.gz"))
        with _listing_in(_ascending):
            _run_stage(input_dir, output_dir)
        expected = _input_qc_path(output_dir).read_text()
        with _listing_in(order):
            _run_stage(input_dir, output_dir)
        assert _input_qc_path(output_dir).read_text() == expected

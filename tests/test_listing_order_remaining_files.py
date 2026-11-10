"""Listing order of the nine remaining pipeline files (TASK-READORDER-BDRABCZUK-FB2B86367B2E).

User rule: files must always be read in the same order, for reproducibility. A real directory
gives one arbitrary order, so these tests replace what os.scandir and os.listdir yield, below
glob, pathlib and os.walk, and stay valid whichever listing call the code uses.

Listings to fix:

* REQ-READORDER-BDRABCZUK-AC2131ED9F9F - PurgeSubject._per_subject_crop_files returns sorted paths.
* REQ-READORDER-BDRABCZUK-248BBEB29661 - PurgeSubject._array_csvs yields in sorted region order.
* REQ-READORDER-BDRABCZUK-20374E007DC9 - the prune file sequence does not depend on listing order.
* REQ-READORDER-BDRABCZUK-1CFF9EB6A428 - a run_cka model-ID tie takes the sorted-first CSV.

Pins of an order already fixed (they pass before the fix; waiver of the user, 2026-11-06):

* REQ-READORDER-BDRABCZUK-905938836E42 - transform_qc_rows does not depend on listing order.
* REQ-READORDER-BDRABCZUK-6812AF787F26 - the run_cka command sequence does not depend on it.
* REQ-READORDER-BDRABCZUK-108B61633DD9 - _list_crop_regions returns sorted names.
* REQ-READORDER-BDRABCZUK-CBC2EA80C057 - _list_embedding_pairs returns sorted names per side.
* REQ-READORDER-BDRABCZUK-4676AF8F2584 - the prune subject line does not depend on it.
* REQ-READORDER-BDRABCZUK-9CE0B3327009 - the LISTING_PRESENCE_CHECKS do not depend on it.
* REQ-READORDER-BDRABCZUK-2A5A30356BDF - the run_combine region count does not depend on it.

Synthetic subject IDs and synthetic trees only.
"""

import contextlib
import glob
import importlib.util
import os
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.purge_subject import PurgeSubject
from champollion_pipeline.streaming import combine
from champollion_pipeline.utils.archive import reuse_cached_extraction
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir, versioned_crops_exist
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.transform_qc import TRANSFORM_STEM, transform_qc_rows
from compare_sulcal.crops import _list_crop_regions
from compare_sulcal.embeddings import _EMBEDDINGS_SUFFIX, _HEMI_BY_SIDE, _list_embedding_pairs
from run_cka import RunCKA
from tests.listing_order_helpers import Order
from tests.listing_order_helpers import ascending as _ascending
from tests.listing_order_helpers import descending as _descending
from tests.listing_order_helpers import rotated as _rotated
from tests.log_assertions import OUTPUT
from tests.prune_layout import crop_file, crops_2mm, make_aggregate, make_crop_files, run_prune, skeleton_file, touch

pytestmark = pytest.mark.unit  # noqa: V107

INSTALL_HEALTH = Path(__file__).resolve().parents[1] / "scripts" / "install_health.py"

REGIONS = ("region_a", "region_b", "region_c")
SIDES = ("L", "R")
SUBJECT = "sub-01"
OTHER_SUBJECT = "sub-02"
MODEL_ID = "name07-58-00--111"
TIED_STEMS = ("run_a", "run_b", "run_c")

EVERY_ORDER: tuple[Order, ...] = (_ascending, _descending, _rotated)
unsorted_orders = pytest.mark.parametrize("order", [_descending, _rotated], ids=["descending", "rotated"])


class _OrderedEntries:
    """What os.scandir returns, in a chosen order: a context manager that is also an iterator (os.walk needs both)."""

    def __init__(self, entries: list[os.DirEntry]) -> None:
        self._entries = iter(entries)

    def __enter__(self) -> "_OrderedEntries":
        return self

    def __exit__(self, *_exc_info) -> None:
        return None

    def __iter__(self) -> "_OrderedEntries":
        return self

    def __next__(self) -> os.DirEntry:
        return next(self._entries)


@contextlib.contextmanager
def _listing_in(order: Order) -> Iterator[None]:
    """Make os.scandir and os.listdir yield their entries in the given order of names."""
    real_scandir = os.scandir
    real_listdir = os.listdir

    def scandir(path="."):
        with real_scandir(path) as entries:
            by_name = {entry.name: entry for entry in entries}
        return _OrderedEntries([by_name[name] for name in order(list(by_name))])

    def listdir(path="."):
        return order(real_listdir(path))

    with patch("os.scandir", scandir), patch("os.listdir", listdir):
        yield


def _under_each_order(call: Callable[[], object]) -> dict[str, object]:
    """What call() gives under each listing order, by order name."""
    results = {}
    for order in EVERY_ORDER:
        with _listing_in(order):
            results[order.__name__] = call()
    return results


def _same_under_each_order(expected: object) -> dict[str, object]:
    return {order.__name__: expected for order in EVERY_ORDER}


def _populated(directory: Path) -> Path:
    """Create directory with three empty files; return it."""
    for name in ("a.txt", "b.txt", "c.txt"):
        touch(directory / name)
    return directory


def _output_messages(capture, marker: str) -> list[str]:
    """Messages of the OUTPUT records that contain marker, in logging order."""
    return [
        record.getMessage() for record in capture.records if record.levelno == OUTPUT and marker in record.getMessage()
    ]


class TestListingOrderControl:
    """The order control above must really reach the calls, or every test below proves nothing."""

    @unsorted_orders
    def test_scandir_order_reaches_glob_and_walk(self, tmp_path, order):
        names = ["a.csv", "b.csv", "c.csv"]
        for name in names:
            touch(tmp_path / "flat" / name)
            touch(tmp_path / "tree" / name / "file.txt")
        with _listing_in(order):
            from_glob = [os.path.basename(p) for p in glob.glob(str(tmp_path / "flat" / "*.csv"))]
            from_path_glob = [p.name for p in (tmp_path / "flat").glob("*.csv")]
            _root, _dirs, walked_files = next(os.walk(tmp_path / "flat"))
            _root, walked_dirs, _files = next(os.walk(tmp_path / "tree"))
        assert from_glob == order(names)
        assert from_path_glob == order(names)
        assert walked_files == order(names)
        assert walked_dirs == order(names)

    @unsorted_orders
    def test_listdir_order_reaches_iterdir(self, tmp_path, order):
        _populated(tmp_path)
        with _listing_in(order):
            from_iterdir = [p.name for p in tmp_path.iterdir()]
        assert from_iterdir == order(["a.txt", "b.txt", "c.txt"])

    def test_entries_serve_each_scandir_shape(self, tmp_path):
        _populated(tmp_path)
        with _listing_in(_descending):
            with os.scandir(tmp_path) as entries:
                through_with = [entry.name for entry in entries]
            through_any = any(os.scandir(tmp_path))
            through_next = next(os.scandir(tmp_path)).name
        assert through_with == ["c.txt", "b.txt", "a.txt"]
        assert through_any is True
        assert through_next == "c.txt"


def _purge_script(base: Path, dry_run: bool = False) -> PurgeSubject:
    script = PurgeSubject()
    argv = [str(base / DERIVATIVES_FOLDER), "--subject", SUBJECT] + (["--dry-run"] if dry_run else [])
    script.args = script.parse_args(argv)
    return script


def _subject_crop_files(base: Path) -> list[Path]:
    """Three files of SUBJECT in each of three crop subdirectories of each region; return their paths."""
    return [
        touch(crops_2mm(base) / region / "mask" / subdir / f"{SUBJECT}_{name}.nii.gz")
        for region in REGIONS
        for subdir in ("Lcrops", "Rcrops", "Zcrops")
        for name in ("a", "b", "c")
    ]


class TestPurgeCropFileOrder:
    """REQ-READORDER-BDRABCZUK-AC2131ED9F9F."""

    def test_crop_files_are_sorted(self, tmp_path):
        created = _subject_crop_files(tmp_path)
        script = _purge_script(tmp_path)
        found = _under_each_order(lambda: script._per_subject_crop_files(str(crops_2mm(tmp_path))))
        assert found == _same_under_each_order(sorted(created))

    def test_dry_run_delete_lines_are_sorted(self, tmp_path, caplog, pipeline_caplog):
        created = _subject_crop_files(tmp_path)
        script = _purge_script(tmp_path, dry_run=True)

        def delete_lines() -> list[str]:
            caplog.clear()
            assert script.run() == 0
            return _output_messages(pipeline_caplog, "Delete file: ")

        expected = [f"[dry-run] Delete file: {path}" for path in sorted(created)]
        assert _under_each_order(delete_lines) == _same_under_each_order(expected)


class TestPurgeArrayOrder:
    """REQ-READORDER-BDRABCZUK-248BBEB29661."""

    @staticmethod
    def _arrays(base: Path) -> None:
        for region in REGIONS:
            make_aggregate(base, region, "Lskeleton", [SUBJECT, OTHER_SUBJECT])

    def test_array_entries_are_sorted(self, tmp_path):
        self._arrays(tmp_path)
        script = _purge_script(tmp_path)

        def regions() -> list[str]:
            return [region_dir.name for region_dir, _csv, _npy in script._array_csvs(str(crops_2mm(tmp_path)))]

        assert _under_each_order(regions) == _same_under_each_order(list(REGIONS))

    def test_dry_run_filter_lines_are_sorted(self, tmp_path, caplog, pipeline_caplog):
        self._arrays(tmp_path)
        script = _purge_script(tmp_path, dry_run=True)

        def filtered_regions() -> list[str]:
            caplog.clear()
            assert script.run() == 0
            lines = _output_messages(pipeline_caplog, "Filter row(s)")
            return [re.search(r"\(region (\S+)\)", line).group(1) for line in lines]

        assert _under_each_order(filtered_regions) == _same_under_each_order(list(REGIONS))


class TestPruneDeletionOrder:
    """REQ-READORDER-BDRABCZUK-20374E007DC9."""

    @staticmethod
    def _tree(base: Path) -> int:
        """Files of a passing and of a failing subject; return how many belong to the failing one."""
        count = 0
        for region in REGIONS:
            for side in SIDES:
                make_crop_files(base, region, side, [SUBJECT, OTHER_SUBJECT])
                crop = crop_file(base, region, side, OTHER_SUBJECT)
                touch(crop.with_name(f"{crop.name}.minf"))
                touch(crop.with_name(f"{crop.name}.sha256"))
                count += 3
        for side in SIDES:
            touch(skeleton_file(base, side, OTHER_SUBJECT))
            count += 1
        return count

    @pytest.mark.parametrize(
        ("dry_run", "marker"),
        [(True, "[dry-run] Would delete: "), (False, "Removed: ")],
        ids=["dry_run", "real_run"],
    )
    def test_file_sequence_does_not_depend_on_listing_order(self, tmp_path, caplog, pipeline_caplog, dry_run, marker):
        sequences = {}
        for order in EVERY_ORDER:
            base = tmp_path / order.__name__
            expected_count = self._tree(base)
            caplog.clear()
            with _listing_in(order):
                assert run_prune(base, [(SUBJECT, 1), (OTHER_SUBJECT, 0)], dry_run=dry_run) == 0
            lines = _output_messages(pipeline_caplog, marker)
            assert len(lines) == expected_count
            sequences[order.__name__] = [str(Path(line.split(marker)[1]).relative_to(base)) for line in lines]
        assert sequences == _same_under_each_order(sequences[_ascending.__name__])


def _cka_csv(directory: Path, name: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ID,dim0\nsub-01,1.0\n")
    return path


def _cka_commands(path_a: Path, path_b: Path, output_dir: Path) -> list[tuple[str, str, str]]:
    """(first input, second input, output directory) of each CKA command of a run, in launch order."""
    script = RunCKA()
    script.args = script.parse_args(["--path_a", str(path_a), "--path_b", str(path_b), "--output_dir", str(output_dir)])
    commands: list[tuple[str, str, str]] = []

    def record(cmd, **_options) -> int:
        commands.append((cmd[3], cmd[4], cmd[6]))
        return 0

    script.execute_command = record
    assert script.run() == 0
    return commands


def _cka_command(file_a: Path, file_b: Path, output_dir: Path, label: str) -> tuple[str, str, str]:
    return (f"A:{file_a}", f"B:{file_b}", str(output_dir / label))


def _tied_name(stem: str) -> str:
    return f"{stem}_{MODEL_ID}_embeddings.csv"


class TestCkaModelIdTie:
    """REQ-READORDER-BDRABCZUK-1CFF9EB6A428."""

    def test_path_a_tie_pairs_sorted_first(self, tmp_path):
        path_a, path_b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
        tied = [_cka_csv(path_a, _tied_name(stem)) for stem in TIED_STEMS]
        single = _cka_csv(path_b, _tied_name("single"))
        expected = [_cka_command(tied[0], single, out, f"run_a_{MODEL_ID}")]
        commands = _under_each_order(lambda: _cka_commands(path_a, path_b, out))
        assert commands == _same_under_each_order(expected)

    def test_path_b_tie_pairs_sorted_first(self, tmp_path):
        path_a, path_b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
        single = _cka_csv(path_a, _tied_name("single"))
        tied = [_cka_csv(path_b, _tied_name(stem)) for stem in TIED_STEMS]
        expected = [_cka_command(single, tied[0], out, f"single_{MODEL_ID}")]
        commands = _under_each_order(lambda: _cka_commands(path_a, path_b, out))
        assert commands == _same_under_each_order(expected)


class TestCkaCommandOrder:
    """REQ-READORDER-BDRABCZUK-6812AF787F26 (pin)."""

    def test_nested_mode_commands(self, tmp_path):
        path_a, path_b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
        expected = [
            _cka_command(
                _cka_csv(path_a / region, "full_embeddings.csv"),
                _cka_csv(path_b / region, "full_embeddings.csv"),
                out,
                region,
            )
            for region in REGIONS
        ]
        commands = _under_each_order(lambda: _cka_commands(path_a, path_b, out))
        assert commands == _same_under_each_order(expected)

    def test_flat_exact_name_commands(self, tmp_path):
        path_a, path_b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
        expected = [
            _cka_command(
                _cka_csv(path_a, f"{region}_embeddings.csv"),
                _cka_csv(path_b, f"{region}_embeddings.csv"),
                out,
                region,
            )
            for region in REGIONS
        ]
        commands = _under_each_order(lambda: _cka_commands(path_a, path_b, out))
        assert commands == _same_under_each_order(expected)

    def test_flat_model_id_commands(self, tmp_path):
        path_a, path_b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
        expected = [
            _cka_command(
                _cka_csv(path_a, f"first_name07-58-00--{index}_embeddings.csv"),
                _cka_csv(path_b, f"second_name07-58-00--{index}_embeddings.csv"),
                out,
                f"first_name07-58-00--{index}",
            )
            for index in (111, 222, 333)
        ]
        commands = _under_each_order(lambda: _cka_commands(path_a, path_b, out))
        assert commands == _same_under_each_order(expected)


class TestTransformQcOrder:
    """REQ-READORDER-BDRABCZUK-905938836E42 (pin)."""

    @staticmethod
    def _transforms(transforms_dir: Path, side: str, content_by_subject: dict[str, bytes]) -> None:
        side_dir = transforms_dir / side
        side_dir.mkdir(parents=True)
        for subject, content in content_by_subject.items():
            (side_dir / f"{side}{TRANSFORM_STEM}{subject}.trm").write_bytes(content)

    def test_rows_do_not_depend_on_listing_order(self, tmp_path):
        self._transforms(tmp_path, "L", {"sub-01": b"same", "sub-02": b"same", "sub-03": b"same", "sub-04": b"own"})
        self._transforms(tmp_path, "R", {"sub-01": b"same", "sub-02": b"same", "sub-03": b"own", "sub-04": b"apart"})
        expected = [
            {
                "participant_id": "sub-01",
                "qc": 0,
                "reason": "L transform byte-identical to sub-02, sub-03; R transform byte-identical to sub-02",
            },
            {
                "participant_id": "sub-02",
                "qc": 0,
                "reason": "L transform byte-identical to sub-01, sub-03; R transform byte-identical to sub-01",
            },
            {"participant_id": "sub-03", "qc": 0, "reason": "L transform byte-identical to sub-01, sub-02"},
            {"participant_id": "sub-04", "qc": 1, "reason": ""},
        ]
        assert _under_each_order(lambda: transform_qc_rows(str(tmp_path))) == _same_under_each_order(expected)


class TestCompareListings:
    """REQ-READORDER-BDRABCZUK-108B61633DD9 and -CBC2EA80C057 (pins)."""

    def test_crop_regions_are_sorted(self, tmp_path):
        set_a, set_b = tmp_path / "a", tmp_path / "b"
        for region in REGIONS:
            (set_a / region).mkdir(parents=True)
            (set_b / region).mkdir(parents=True)
        (set_a / "region_only_in_a").mkdir()
        regions = _under_each_order(lambda: _list_crop_regions(set_a, set_b, None))
        assert regions == _same_under_each_order(list(REGIONS))

    def test_embedding_pairs_are_sorted(self, tmp_path):
        set_a, set_b = tmp_path / "a", tmp_path / "b"
        for side in SIDES:
            for region in REGIONS:
                name = f"{region}_{_HEMI_BY_SIDE[side]}{_EMBEDDINGS_SUFFIX}"
                touch(set_a / name)
                touch(set_b / name)
        expected = [(region, side) for side in SIDES for region in REGIONS]
        pairs = _under_each_order(lambda: _list_embedding_pairs(set_a, set_b, None, list(SIDES)))
        assert pairs == _same_under_each_order(expected)


class TestPruneSubjectLine:
    """REQ-READORDER-BDRABCZUK-4676AF8F2584 (pin)."""

    def test_subject_line_does_not_depend_on_listing_order(self, tmp_path, caplog, pipeline_caplog):
        failing = ["sub-02", "sub-03", "sub-04"]
        for region in REGIONS:
            for side in SIDES:
                make_crop_files(tmp_path, region, side, [SUBJECT, *failing])

        def subject_lines() -> list[str]:
            caplog.clear()
            assert run_prune(tmp_path, [(SUBJECT, 1)], dry_run=True) == 0
            return _output_messages(pipeline_caplog, "Subjects to remove")

        expected = [f"Subjects to remove (3): {failing}"]
        assert _under_each_order(subject_lines) == _same_under_each_order(expected)


@pytest.fixture(scope="module")
def install_health() -> Iterator[ModuleType]:
    """scripts/install_health.py as a module (scripts/ is no package), registered while the tests run."""
    spec = importlib.util.spec_from_file_location("install_health", INSTALL_HEALTH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"install_health": module}):
        spec.loader.exec_module(module)
        yield module


def _crops_exist(tmp_path: Path, _health: ModuleType) -> bool:
    _populated(Path(compute_versioned_crops_dir(str(tmp_path), "masks_a", 2.0)))
    return versioned_crops_exist(str(tmp_path), "masks_a", 2.0)


def _cached_extraction_reused(tmp_path: Path, _health: ModuleType) -> bool:
    return reuse_cached_extraction(str(_populated(tmp_path / "cache")), False, require_content=True)


def _package_source_found(tmp_path: Path, health: ModuleType) -> bool:
    meta = {"src": _populated(tmp_path / "package"), "fix_cmd": ["true"]}
    with (
        patch.object(health, "_pkg_importable", return_value=False),
        patch.object(health, "_pkg_version", return_value=None),
    ):
        result = health.check_package("package", meta, health.State(), False)
    return result.fixable


def _submodule_initialized(tmp_path: Path, health: ModuleType) -> bool:
    with patch.object(health.subprocess, "run", return_value=MagicMock(returncode=1, stdout="")):
        return health.check_submodule("submodule", _populated(tmp_path / "submodule"), health.State()).ok


LISTING_PRESENCE_CHECKS: dict[str, Callable[[Path, ModuleType], bool]] = {
    "versioned_crops_exist": _crops_exist,
    "reuse_cached_extraction": _cached_extraction_reused,
    "check_package": _package_source_found,
    "check_submodule": _submodule_initialized,
}


class TestPresenceChecks:
    """REQ-READORDER-BDRABCZUK-9CE0B3327009 (pin)."""

    @pytest.mark.parametrize("check", sorted(LISTING_PRESENCE_CHECKS))
    def test_presence_check_does_not_depend_on_listing_order(self, tmp_path, install_health, check):
        found = _under_each_order(lambda: LISTING_PRESENCE_CHECKS[check](tmp_path, install_health))
        assert found == _same_under_each_order(True)


class TestCombineCount:
    """REQ-READORDER-BDRABCZUK-2A5A30356BDF (pin)."""

    def test_count_does_not_depend_on_listing_order(self, tmp_path, monkeypatch, caplog, pipeline_caplog):
        source = _populated(tmp_path / "source")
        combined = tmp_path / "combined"
        for region in REGIONS:
            touch(combined / f"{region}_embeddings.csv")
        touch(combined / "notes.txt")
        monkeypatch.setattr(combine, "_resolve_embeddings_source", lambda _dataset: str(source))
        monkeypatch.setattr(combine, "_resolve_combined_dir", lambda _settings, _dataset: str(combined))
        monkeypatch.setattr(combine, "PutTogetherEmbeddings", MagicMock(return_value=MagicMock(run=lambda: 0)))
        settings = SimpleNamespace(dry_run=False)
        dataset = SimpleNamespace(regions=[])

        def combined_lines() -> list[str]:
            caplog.clear()
            combine.run_combine(settings, dataset)
            return _output_messages(pipeline_caplog, "Combined ")

        expected = [f"Combined 3 region embeddings files into {combined}"]
        assert _under_each_order(combined_lines) == _same_under_each_order(expected)

#!/usr/bin/env python3
"""
Per-region Mask versions in run_cortical_tiles.py (TASK-MODELVER-BDRABCZUK-9127B66D5C8E): given --region-masks
pairs <region model name>=<Mask version>, each hemisphere of a Sulcal region is cropped with the Mask version paired
with its region model name, into crops/<version>/2mm/<region> (user decisions 2026-11-17 and 2026-11-23: one Mask
version per region model name, shared by every model id of that region). The crops done-marker
crops/<version>/2mm/crops_complete.marker lists the region model names cropped there, one per line; a marker naming
none, written before this change, counts as listing the 56 (user decisions R1 and of the stage design re-run,
2026-11-23).

The map is plain input data here: its producer (TASK-MODELVER-BDRABCZUK-B9820AF55783) is not imported.

Synthetic fixtures only. The crop subprocess is faked by ``_FakeCropStep``, which follows the command line contract
generate_sulcal_regions.py has today: it crops the regions of -r (default: the 28 regions), the sides of -i (default:
L R), with the Mask version of --masks, else the masks_version of the <-d>/pipeline_loop_2mm.json it reads at call
time; it empties the skip list of each side it crops, as cortical_tiles does at each invocation; it writes
crops/<version>/2mm/<region>/mask/<side><kind>.npy for the kinds of -y and the mask, and skip-lists a subject whose
skeleton crop it finds empty. The output QC is the real one; whole-brain fusion and ventricle removal are mocked.

The REQ comments cite root ids (``alm get --root``): they stay valid when a Requirement is reworded.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
from cortical_tiles.brainvisa.utils.crop_subjects import EmptyCropError
from cortical_tiles.brainvisa.utils.skip_list import clear_skip_list, record_skip
from hypothesis import given
from hypothesis import strategies as st

from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    DEFAULT_REGIONS,
    MASK_VERSIONS,
    compute_region_model_name,
)
from champollion_pipeline.utils.cortical_tiles_config import (
    compute_missing_hemispheres,
    compute_versioned_crops_dir,
    plan_crop_passes,
)
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import read_qc_tsv
from champollion_pipeline.utils.region_masks import REGION_MODEL_NAMES
from tests.stage2_layout import (
    CROP_MARKER,
    TEMPLATE_CONFIG,
    add_crop_marker,
    add_crops,
    add_graphs,
    crop_array,
    parsed_script,
)

_MODULE = "champollion_pipeline.run_cortical_tiles"
_CROP_SCRIPT = "generate_sulcal_regions.py"
_CROP_STEP = "generate_crops"
_USAGE_STATUS = 2
_FAILED_STATUS = 1
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_KINDS = ("skeleton", "foldlabel")
_MASK_KIND = "mask"
_SIDE_OF = {"left": "L", "right": "R"}
_SYLV = "S.C.-sylv."
_STS = "S.T.s."
_REGIONS = (_SYLV, _STS)
# The default Mask version is one of the two, so that run() as it is today, which crops every region with it, puts
# the hemisphere paired with the other one in the wrong folder.
_CORRECTED = DEFAULT_MASKS_VERSION
_CANONICAL_25 = "canonical_25"
_MAP = {
    (_SYLV, "left"): _CORRECTED,
    (_SYLV, "right"): _CORRECTED,
    (_STS, "left"): _CORRECTED,
    (_STS, "right"): _CANONICAL_25,
}
_UNKNOWN_MODEL_NAME = "NotARegion_left"


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage2_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


def _pairs(version_map):
    """--region-masks values of a {(region, hemisphere): version} map."""
    return [
        f"{compute_region_model_name(region, hemisphere)}={version}"
        for (region, hemisphere), version in version_map.items()
    ]


def _placements(version_map):
    """{(version, region, side)} of a {(region, hemisphere): version} map."""
    return {(version, region, _SIDE_OF[hemisphere]) for (region, hemisphere), version in version_map.items()}


def _names_paired_with(version_map, version):
    """Region model names that ``version_map`` pairs with ``version``, in map order."""
    return [
        compute_region_model_name(region, hemisphere)
        for (region, hemisphere), paired in version_map.items()
        if paired == version
    ]


def _crops_dir(output_dir, version):
    return Path(compute_versioned_crops_dir(str(Path(output_dir) / DERIVATIVES_FOLDER), version))


def _crop_placements(output_dir):
    """{(version, region, side)} of each skeleton crop array under <output>/<derivatives>/crops."""
    crops_root = Path(output_dir) / DERIVATIVES_FOLDER / "crops"
    return {
        (path.parents[3].name, path.parents[1].name, path.name[0])
        for path in crops_root.glob(f"*/2mm/*/{_MASK_KIND}/*skeleton.npy")
    }


def _subject_csvs(output_dir):
    """The subject CSV of each crop array under <output>/<derivatives>/crops, sorted."""
    return sorted((Path(output_dir) / DERIVATIVES_FOLDER / "crops").glob(f"*/2mm/*/{_MASK_KIND}/*_subject.csv"))


def _listed_names(output_dir, version):
    """Region model names listed in crops/<version>/2mm/crops_complete.marker; None when there is no marker."""
    marker = _crops_dir(output_dir, version) / CROP_MARKER
    if not marker.is_file():
        return None
    return {line.strip() for line in marker.read_text().splitlines() if line.strip()}


def _option_values(command, *names):
    """Values following the first of ``names`` found in ``command``, up to the next option; None when absent."""
    for name in names:
        if name in command:
            values = []
            for part in command[command.index(name) + 1 :]:
                if part.startswith("-"):
                    break
                values.append(part)
            return values
    return None


def _crop_version(command, dataset):
    """The Mask version a crop command crops with: its --masks, else the masks_version of <dataset>'s config."""
    version = (_option_values(command, "--masks") or [None])[0]
    if version is None:
        config = json.loads((Path(dataset) / TEMPLATE_CONFIG.name).read_text())
        version = config.get("masks_version", DEFAULT_MASKS_VERSION)
    return version


@dataclass
class _FakeCropStep:
    """generate_sulcal_regions.py as run() calls it.

    ``missing``: (version, region, side) crops it does not write; ``empty``: (version, side, subject) skeleton crops
    it finds empty, so it skip-lists the subject and leaves it out of the arrays of that invocation; ``failing``:
    (version, side) invocations that return status 1 before writing anything.
    """

    missing: frozenset = frozenset()
    empty: frozenset = frozenset()
    failing: frozenset = frozenset()
    subjects: tuple = (_SUBJECT,)
    commands: list = field(default_factory=list)
    passes: list = field(default_factory=list)

    def __call__(self, command, *_args, **_kwargs):
        command = [str(part) for part in command]
        if not any(_CROP_SCRIPT in part for part in command):
            return 0
        self.commands.append(command)
        (dataset,) = _option_values(command, "-d", "--path_dataset")
        version = _crop_version(command, dataset)
        regions = _option_values(command, "-r", "--regions") or list(DEFAULT_REGIONS)
        sides = _option_values(command, "-i", "--sides") or ["L", "R"]
        kinds = (*(_option_values(command, "-y", "--input_types") or _KINDS), _MASK_KIND)
        self.passes.append((version, tuple(regions), tuple(sides)))
        skip_dir = Path(dataset) / DERIVATIVES_FOLDER
        for side in sides:
            clear_skip_list(skip_dir, side)
        if any((version, side) in self.failing for side in sides):
            return _FAILED_STATUS
        for side in sides:
            self._crop_side(_crops_dir(dataset, version), skip_dir, (version, side), regions, kinds)
        return 0

    def _crop_side(self, crops_dir, skip_dir, version_side, regions, kinds):
        """Write the crops of one side, skip-listing the subjects whose skeleton crop is empty."""
        version, side = version_side
        kept = [subject for subject in self.subjects if (version, side, subject) not in self.empty]
        for region in regions:
            if (version, region, side) not in self.missing:
                add_crops(crops_dir, kept, [region], kinds, side)
        for subject in sorted(set(self.subjects) - set(kept)):
            record_skip(skip_dir, side, _CROP_STEP, subject, EmptyCropError(f"empty crop in {crops_dir}"))


@dataclass
class _StageRun:
    """What ``_run_stage`` observed."""

    status: int | None
    output_dir: Path
    crop_step: _FakeCropStep
    placements_at_fusion: list = field(default_factory=list)


def _lay_version(output_dir, version, crop_keys, marker_names, subjects=(_SUBJECT,)):
    """Lay the crops of the (region, hemisphere) ``crop_keys`` in crops/<version>/2mm and its marker.

    ``marker_names``: the region model names the marker lists, one per line; [] lays an empty marker, None none.
    """
    crops_dir = _crops_dir(output_dir, version)
    for region, hemisphere in crop_keys:
        add_crops(crops_dir, list(subjects), [region], (*_KINDS, _MASK_KIND), _SIDE_OF[hemisphere])
    if marker_names is not None:
        marker = add_crop_marker(crops_dir)
        marker.write_text("".join(f"{name}\n" for name in marker_names))


def _seed_marked_crops(output_dir, version, version_map):
    """Lay the crops of the hemispheres ``version_map`` pairs with ``version``, and a marker listing them."""
    keys = [key for key, paired in version_map.items() if paired == version]
    _lay_version(output_dir, version, keys, _names_paired_with(version_map, version))


def _make_dirs(root, subjects):
    input_dir, output_dir = Path(root) / "input", Path(root) / "output"
    for subject in subjects:
        add_graphs(input_dir, subject)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def _seeding(seed):
    """A ``prepare`` laying the marked crops of each {version: version_map} of ``seed``."""

    def prepare(output_dir):
        for version, version_map in seed.items():
            _seed_marked_crops(output_dir, version, version_map)

    return prepare


def _run_stage(root, pairs, *, prepare=None, extra=(), crop_step=None):
    """Run run() on the two regions with ``pairs`` as --region-masks (no --region-masks when None).

    ``prepare(output_dir)`` lays the earlier outputs; ``extra``: further arguments; ``crop_step``: the fake crop
    subprocess (default: one cropping the one subject). A SystemExit gives its code as status.
    """
    crop_step = crop_step or _FakeCropStep()
    input_dir, output_dir = _make_dirs(root, crop_step.subjects)
    if prepare is not None:
        prepare(output_dir)
    run = _StageRun(None, output_dir, crop_step)

    def fuse(*_args, **_kwargs):
        run.placements_at_fusion.append(_crop_placements(output_dir))

    map_args = () if pairs is None else ("--region-masks", *pairs)
    try:
        script = parsed_script(input_dir, output_dir, "--regions", *_REGIONS, *map_args, *extra)
        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", side_effect=crop_step),
            patch.object(script, "_generate_mask_npys"),
            patch(f"{_MODULE}.add_left_and_right_volumes.add_left_and_right_volumes", side_effect=fuse),
            patch(f"{_MODULE}.remove_ventricle.remove_ventricle"),
        ):
            run.status = script.run()
    except SystemExit as error:
        run.status = error.code
    return run


def _messages(capsys, pipeline_caplog):
    """Text of stderr and of the champollion_pipeline records."""
    records = "\n".join(record.getMessage() for record in pipeline_caplog.records)
    return capsys.readouterr().err + "\n" + records


def _lay_partly_listed_crops(output_dir):
    """Crops of the four hemispheres in the default Mask version folder, its marker listing the S.C.-sylv. ones."""
    _lay_version(
        output_dir,
        _CORRECTED,
        list(_MAP),
        [compute_region_model_name(_SYLV, "left"), compute_region_model_name(_SYLV, "right")],
    )


# REQ-MODELVER-BDRABCZUK-1C661F5BC143, REQ-MODELVER-BDRABCZUK-2DC06EF8E6B6
def test_crops_land_in_the_paired_version_folder(temp_dir):
    run = _run_stage(temp_dir, _pairs(_MAP))

    missing = _placements(_MAP) - _crop_placements(run.output_dir)
    assert not missing, f"crops not under crops/<paired version>/2mm/<region>: {sorted(missing)}"


# REQ-MODELVER-BDRABCZUK-19644D186E08
def test_status_0_run_lists_each_cropped_hemisphere_in_its_version_marker(temp_dir):
    run = _run_stage(temp_dir, _pairs(_MAP))

    assert run.status == 0
    unlisted = {
        version: sorted(set(_names_paired_with(_MAP, version)) - (_listed_names(run.output_dir, version) or set()))
        for version in sorted(set(_MAP.values()))
    }
    assert not any(unlisted.values()), f"region model names not in crops/<version>/2mm/{CROP_MARKER}: {unlisted}"


# REQ-MODELVER-BDRABCZUK-B4CB816A4911
def test_crop_missing_from_its_paired_version_folder_gives_qc_0(temp_dir):
    run = _run_stage(temp_dir, _pairs(_MAP), crop_step=_FakeCropStep(missing=frozenset({(_CANONICAL_25, _STS, "R")})))

    qc_values = [
        record["qc"]
        for path in sorted(run.output_dir.rglob("output_qc.tsv"))
        for record in read_qc_tsv(path)
        if record["participant_id"] == _SUBJECT
    ]
    assert 0 in qc_values, f"{_SUBJECT} kept qc 1 in every output_qc.tsv: {qc_values}"


# REQ-MODELVER-BDRABCZUK-66D18502DA19
def test_whole_brain_fusion_runs_once_after_the_crops_of_each_version(temp_dir):
    run = _run_stage(temp_dir, _pairs(_MAP))

    assert len(run.placements_at_fusion) == 1
    (placements,) = run.placements_at_fusion
    missing = _placements(_MAP) - placements
    assert not missing, f"crops not written when the fusion started: {sorted(missing)}"


# REQ-MODELVER-BDRABCZUK-4F0E6D1DFAD2
def test_marked_paired_version_folders_start_no_crop(temp_dir):
    one_version = dict.fromkeys(_MAP, _CANONICAL_25)
    crop_commands = {}
    for name, version_map in (("one_version", one_version), ("two_versions", _MAP)):
        seed = {version: version_map for version in set(version_map.values())}
        run = _run_stage(Path(temp_dir) / name, _pairs(version_map), prepare=_seeding(seed))
        crop_commands[name] = len(run.crop_step.commands)

    assert crop_commands == {"one_version": 0, "two_versions": 0}


# REQ-MODELVER-BDRABCZUK-267EC8EF0516
def test_unmarked_paired_version_is_cropped_beside_a_marked_one(temp_dir):
    run = _run_stage(temp_dir, _pairs(_MAP), prepare=_seeding({_CORRECTED: _MAP}))

    assert (_CANONICAL_25, _STS, "R") in _crop_placements(run.output_dir)
    assert crop_array(_crops_dir(run.output_dir, _CANONICAL_25), _STS, "R", "skeleton").is_file()


# REQ-MODELVER-BDRABCZUK-267EC8EF0516
def test_hemisphere_missing_from_its_version_marker_is_cropped(temp_dir):
    def lay_earlier_outputs(output_dir):
        _seed_marked_crops(output_dir, _CORRECTED, _MAP)
        _lay_version(output_dir, _CANONICAL_25, [], [compute_region_model_name(_SYLV, "left")])

    run = _run_stage(temp_dir, _pairs(_MAP), prepare=lay_earlier_outputs)

    assert crop_array(_crops_dir(run.output_dir, _CANONICAL_25), _STS, "R", "skeleton").is_file()


# REQ-MODELVER-BDRABCZUK-899198D2FB96
def test_region_listed_in_its_version_marker_is_not_cropped_again(temp_dir):
    run = _run_stage(temp_dir, _pairs(_MAP), prepare=_seeding({_CORRECTED: _MAP}))

    assert run.crop_step.passes, f"no crop subprocess, though crops/{_CANONICAL_25}/2mm holds no marker"
    recropped = [crop for crop in run.crop_step.passes if crop[0] == _CORRECTED and _SYLV in crop[1]]
    assert not recropped, f"{_SYLV}, listed in the {_CORRECTED} marker, cropped again: {recropped}"


# REQ-MODELVER-BDRABCZUK-34E91B3D8C1F
def test_empty_marker_counts_as_listing_each_region_model_name(temp_dir):
    one_version = dict.fromkeys(_MAP, _CANONICAL_25)

    def lay_empty_marker(output_dir):
        _lay_version(output_dir, _CANONICAL_25, list(one_version), [])

    run = _run_stage(temp_dir, _pairs(one_version), prepare=lay_empty_marker)

    assert run.crop_step.commands == []


# REQ-MODELVER-BDRABCZUK-17CA6E7AF98E
def test_empty_crop_in_one_crop_subprocess_drops_the_subject_from_the_run(temp_dir):
    crop_step = _FakeCropStep(
        empty=frozenset({(_CANONICAL_25, "R", _OTHER_SUBJECT)}), subjects=(_SUBJECT, _OTHER_SUBJECT)
    )

    run = _run_stage(temp_dir, _pairs(_MAP), crop_step=crop_step)

    csvs = _subject_csvs(run.output_dir)
    assert csvs, "the run wrote no crop"
    keeping = [str(path.relative_to(run.output_dir)) for path in csvs if _OTHER_SUBJECT in path.read_text().split()]
    assert not keeping, f"{_OTHER_SUBJECT}, empty in crops/{_CANONICAL_25}, still in: {keeping}"


# REQ-MODELVER-BDRABCZUK-F07F9A0061C6
def test_failed_run_leaves_its_cropped_names_out_of_the_marker(temp_dir):
    crop_step = _FakeCropStep(failing=frozenset({(_CORRECTED, "R")}))

    run = _run_stage(temp_dir, _pairs(_MAP), crop_step=crop_step)

    assert run.status not in (0, 3)
    listed = {
        version: sorted(set(_names_paired_with(_MAP, version)) & (_listed_names(run.output_dir, version) or set()))
        for version in sorted(set(_MAP.values()))
    }
    assert not any(listed.values()), f"names of a failed run listed in crops/<version>/2mm/{CROP_MARKER}: {listed}"


# REQ-CTFAIL-BDRABCZUK-510660CF05B5
def test_marker_listing_each_hemisphere_of_the_run_starts_no_crop(temp_dir):
    def lay_fully_listed_crops(output_dir):
        listed = [compute_region_model_name(region, hemisphere) for region, hemisphere in _MAP]
        _lay_version(output_dir, _CORRECTED, list(_MAP), listed)

    run = _run_stage(temp_dir, None, prepare=lay_fully_listed_crops)

    assert run.crop_step.commands == []


# REQ-CTFAIL-BDRABCZUK-6D1B86D97D42
def test_marker_missing_a_hemisphere_of_the_run_starts_the_crop_step(temp_dir):
    run = _run_stage(temp_dir, None, prepare=_lay_partly_listed_crops)

    assert run.crop_step.commands, f"no crop subprocess, though the {CROP_MARKER} lists no {_STS} hemisphere"


# REQ-CTFAIL-BDRABCZUK-E29D83F6E106
def test_marker_missing_a_hemisphere_of_the_run_logs_a_warning(temp_dir, pipeline_caplog):
    _run_stage(temp_dir, None, prepare=_lay_partly_listed_crops)

    assert pipeline_caplog.assert_logged(logging.WARNING, CROP_MARKER).name == _MODULE


# REQ-MODELVER-BDRABCZUK-98414A65980D
def test_unknown_region_model_name_exits_2_naming_it(temp_dir, capsys, pipeline_caplog):
    run = _run_stage(temp_dir, [*_pairs(_MAP), f"{_UNKNOWN_MODEL_NAME}={_CANONICAL_25}"])

    assert run.status == _USAGE_STATUS
    assert _UNKNOWN_MODEL_NAME in _messages(capsys, pipeline_caplog)


# REQ-MODELVER-BDRABCZUK-B419D25D1BBE
def test_hemisphere_without_a_paired_version_exits_2_naming_it(temp_dir, capsys, pipeline_caplog):
    partial = {key: version for key, version in _MAP.items() if key != (_STS, "right")}

    run = _run_stage(temp_dir, _pairs(partial))

    assert run.status == _USAGE_STATUS
    assert compute_region_model_name(_STS, "right") in _messages(capsys, pipeline_caplog)


# REQ-MODELVER-BDRABCZUK-60B9708697EF (the backbone's rule, through stage 2)
def test_masks_beside_region_masks_exits_2_naming_masks(temp_dir, capsys, pipeline_caplog):
    run = _run_stage(temp_dir, _pairs(_MAP), extra=("--masks", _CANONICAL_25))

    assert run.status == _USAGE_STATUS
    assert "--masks" in _messages(capsys, pipeline_caplog)
    assert run.crop_step.commands == []


# REQ-INV-BDRABCZUK-197A06B6C111
@given(
    wanted=st.lists(st.sampled_from(REGION_MODEL_NAMES), unique=True),
    covered=st.none() | st.frozensets(st.sampled_from(REGION_MODEL_NAMES)),
    is_overwrite=st.booleans(),
)
def test_missing_hemispheres_are_no_more_than_the_wanted_ones(wanted, covered, is_overwrite):
    assert len(compute_missing_hemispheres(wanted, covered, is_overwrite)) <= len(wanted)


# REQ-INV-BDRABCZUK-EF427802A0CE
@given(
    to_crop=st.dictionaries(
        st.sampled_from(MASK_VERSIONS), st.lists(st.sampled_from(REGION_MODEL_NAMES), min_size=1, unique=True)
    )
)
def test_at_most_two_crop_passes_per_mask_version(to_crop):
    assert len(plan_crop_passes(to_crop, DEFAULT_REGIONS)) <= 2 * len(to_crop)

#!/usr/bin/env python3
"""
Unit tests for generate_masks.py:
  - get_sulci_for_regions helper
  - _compute_one_sulcus: public_use flag, makedirs exist_ok, error handling
  - _load_and_extract_subject: missing graph file, sulci filtering
  - GenerateMasks CLI: --public_use flag, --njobs auto-buffered, vox_str format
"""

import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from champollion_pipeline.generate_masks import (
    RETURN_DICTIONARY,
    BufferedRunner,
    GenerateMasks,
    RunConfig,
    SerialRunner,
    _compute_one_sulcus,
    _load_and_extract_subject,
    get_sulci_for_regions,
)
from tests.log_assertions import LOG, OUTPUT

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeMask:
    """Minimal Volume-like object supporting numpy array protocol."""

    def __init__(self):
        self._data = np.zeros((96, 114, 96, 1), dtype=np.int16)

    def __array__(self, dtype=None):
        return self._data

    def header(self):
        return {"voxel_size": [2.0, 2.0, 2.0, 1.0]}

    def copyHeaderFrom(self, other):
        pass


def _mock_aims_env(fake_mask=None):
    """Return (modules_dict, mock_aims, mock_compute_mask, fake_mask).

    The mocks replace the names generate_masks binds at import time
    (aims, initialize_mask, write_mask) through _patched_masks_module(modules),
    so the workers never touch the real PyAIMS library.
    """
    if fake_mask is None:
        fake_mask = _FakeMask()
    mock_aims = MagicMock()
    mock_compute_mask = MagicMock()
    mock_compute_mask.initialize_mask.return_value = fake_mask
    modules = {
        "soma": MagicMock(aims=mock_aims),
        "soma.aims": mock_aims,
        "compute_mask": mock_compute_mask,
    }
    return modules, mock_aims, mock_compute_mask, fake_mask


def _patched_masks_module(modules):
    """Patch the PyAIMS and compute_mask names bound in generate_masks with the mocks of _mock_aims_env."""
    compute_mask_mod = modules["compute_mask"]
    return patch.multiple(
        "champollion_pipeline.generate_masks",
        aims=modules["soma.aims"],
        initialize_mask=compute_mask_mod.initialize_mask,
        write_mask=compute_mask_mod.write_mask,
    )


def _two_subject_voxels(sulcus="S.C._left"):
    """Per-subject voxel dict with two subjects, distinct voxel positions."""
    return {
        "sub01": {sulcus: np.array([[10, 20, 30]], dtype=np.int32)},
        "sub02": {sulcus: np.array([[15, 25, 35]], dtype=np.int32)},
    }


@pytest.fixture
def json_path(tmp_path):
    """Write a minimal sulci_regions JSON to tmp_path and return its path."""
    data = {
        "brain": {
            "S.C.-sylv._left": {
                "S.C._left": ["S.C._left"],
                "S.C.sylvian._left": ["S.C.sylvian._left"],
            },
            "S.C.-sylv._right": {
                "S.C._right": ["S.C._right"],
                "S.C.sylvian._right": ["S.C.sylvian._right"],
            },
            "S.C.-S.Pe.C._left": {
                "S.C._left": ["S.C._left"],  # overlaps with S.C.-sylv.
                "S.Pe.C.inf._left": ["S.Pe.C.inf._left"],
            },
            "S.C.-S.Pe.C._right": {
                "S.C._right": ["S.C._right"],
                "S.Pe.C.inf._right": ["S.Pe.C.inf._right"],
            },
        }
    }
    p = tmp_path / "sulci_regions.json"
    p.write_text(json.dumps(data))
    return str(p)


class TestGetSulciForRegions:
    def test_single_region_single_side(self, json_path):
        """Single region + single side returns correct bare sulcus names."""
        result = get_sulci_for_regions(["S.C.-sylv."], ["L"], json_path)
        assert result == {"S.C.", "S.C.sylvian."}

    def test_single_region_both_sides(self, json_path):
        """Both sides are looked up and merged into one set."""
        result = get_sulci_for_regions(["S.C.-sylv."], ["L", "R"], json_path)
        assert result == {"S.C.", "S.C.sylvian."}

    def test_side_suffix_stripped(self, json_path):
        """No sulcus name in the result contains a side suffix."""
        result = get_sulci_for_regions(["S.C.-sylv."], ["L", "R"], json_path)
        for sulcus in result:
            assert "_left" not in sulcus
            assert "_right" not in sulcus

    def test_overlapping_sulci_deduplicated(self, json_path):
        """Sulci shared across regions appear only once."""
        result = get_sulci_for_regions(["S.C.-sylv.", "S.C.-S.Pe.C."], ["L"], json_path)
        # S.C. is in both regions — must appear only once
        assert "S.C." in result
        sulci_list = list(result)
        assert sulci_list.count("S.C.") == 1

    def test_two_regions_union(self, json_path):
        """Two regions produce the union of their sulci."""
        result = get_sulci_for_regions(["S.C.-sylv.", "S.C.-S.Pe.C."], ["L"], json_path)
        assert result == {"S.C.", "S.C.sylvian.", "S.Pe.C.inf."}

    def test_unknown_region_skipped(self, json_path):
        """Unknown region names are silently skipped."""
        result = get_sulci_for_regions(["DOES_NOT_EXIST"], ["L"], json_path)
        assert result == set()

    def test_mixed_known_unknown_regions(self, json_path):
        """Unknown regions are skipped; known ones still processed."""
        result = get_sulci_for_regions(["S.C.-sylv.", "DOES_NOT_EXIST"], ["L"], json_path)
        assert result == {"S.C.", "S.C.sylvian."}

    def test_returns_set(self, json_path):
        """Return type is always a set."""
        result = get_sulci_for_regions(["S.C.-sylv."], ["L"], json_path)
        assert isinstance(result, set)

    def test_empty_regions_returns_empty_set(self, json_path):
        """Empty region list returns empty set."""
        result = get_sulci_for_regions([], ["L", "R"], json_path)
        assert result == set()


# ---------------------------------------------------------------------------
# _compute_one_sulcus
# ---------------------------------------------------------------------------


class TestComputeOneSulcus:
    SULCUS = "S.C._left"
    VT = (2.0, 2.0, 2.0)

    def test_public_use_false_creates_sample_dir(self, tmp_path):
        """public_use=False: per-subject subdirectory is created."""
        modules, _, _, _ = _mock_aims_env()
        with _patched_masks_module(modules):
            _compute_one_sulcus(
                self.SULCUS,
                _two_subject_voxels(),
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
                public_use=False,
            )
        assert (tmp_path / "L" / self.SULCUS).is_dir()

    def test_public_use_true_skips_sample_dir(self, tmp_path):
        """public_use=True: per-subject subdirectory is NOT created."""
        modules, _, _, _ = _mock_aims_env()
        with _patched_masks_module(modules):
            _compute_one_sulcus(
                self.SULCUS,
                _two_subject_voxels(),
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
                public_use=True,
            )
        assert not (tmp_path / "L" / self.SULCUS).is_dir()

    def test_public_use_false_calls_aims_write_per_subject(self, tmp_path):
        """public_use=False: aims.write called once per subject."""
        modules, mock_aims, _, _ = _mock_aims_env()
        with _patched_masks_module(modules):
            _compute_one_sulcus(
                self.SULCUS,
                _two_subject_voxels(),
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
                public_use=False,
            )
        assert mock_aims.write.call_count == 2

    def test_public_use_true_never_calls_aims_write(self, tmp_path):
        """public_use=True: aims.write is never called."""
        modules, mock_aims, _, _ = _mock_aims_env()
        with _patched_masks_module(modules):
            _compute_one_sulcus(
                self.SULCUS,
                _two_subject_voxels(),
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
                public_use=True,
            )
        mock_aims.write.assert_not_called()

    def test_aggregate_mask_written_regardless_of_public_use(self, tmp_path):
        """write_mask is called for both public_use=True and False."""
        for flag in (False, True):
            modules, _, mock_cm, _ = _mock_aims_env()
            with _patched_masks_module(modules):
                _compute_one_sulcus(
                    self.SULCUS,
                    _two_subject_voxels(),
                    self.VT,
                    str(tmp_path / str(flag)),
                    "L",
                    "/fake_bv",
                    public_use=flag,
                )
            mock_cm.write_mask.assert_called_once()

    def test_preexisting_side_dir_does_not_raise(self, tmp_path):
        """Pre-existing side directory (EEXIST race) does not cause failure."""
        (tmp_path / "L").mkdir()  # simulate another thread already created it
        modules, _, _, _ = _mock_aims_env()
        with _patched_masks_module(modules):
            result = _compute_one_sulcus(
                self.SULCUS,
                {},
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
                public_use=False,
            )
        assert result == "ok"

    def test_returns_ok_on_success(self, tmp_path):
        """Worker returns the string 'ok' on success."""
        modules, _, _, _ = _mock_aims_env()
        with _patched_masks_module(modules):
            result = _compute_one_sulcus(
                self.SULCUS,
                {},
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
            )
        assert result == "ok"

    def test_returns_failed_string_on_exception(self, tmp_path):
        """Exception inside worker is caught and returned as 'failed: …'."""
        modules, _, mock_cm, _ = _mock_aims_env()
        mock_cm.initialize_mask.side_effect = RuntimeError("boom")  # noqa: V101
        with _patched_masks_module(modules):
            result = _compute_one_sulcus(
                self.SULCUS,
                {},
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
            )
        assert result.startswith("failed:")
        assert "boom" in result

    def test_voxels_accumulated_into_aggregate(self, tmp_path):
        """Both subjects' voxel positions appear in the aggregate array."""
        fake_mask = _FakeMask()
        modules, _, _, _ = _mock_aims_env(fake_mask)
        with _patched_masks_module(modules):
            _compute_one_sulcus(
                self.SULCUS,
                _two_subject_voxels(),
                self.VT,
                str(tmp_path),
                "L",
                "/fake_bv",
                public_use=True,
            )
        arr = fake_mask._data
        assert arr[10, 20, 30, 0] == 1  # sub01 voxel
        assert arr[15, 25, 35, 0] == 1  # sub02 voxel


# ---------------------------------------------------------------------------
# _load_and_extract_subject
# ---------------------------------------------------------------------------


class TestLoadAndExtractSubject:
    VT = (2.0, 2.0, 2.0)

    def _sub(self, path):
        return {
            "subject": "sub01",
            "dir": str(path),
            "graph_file": "%(subject)s.arg",
        }

    def _mock_graph(self, mock_aims, vertices=()):
        mock_graph = MagicMock()
        mock_graph.__getitem__ = MagicMock(return_value=[2.0, 2.0, 2.0, 1.0])  # noqa: V101
        mock_graph.vertices.return_value = list(vertices)
        mock_aims.read.return_value = mock_graph
        mock_transform = MagicMock()
        mock_transform.transform.side_effect = lambda c: c  # noqa: V101
        mock_aims.GraphManip.getICBM2009cTemplateTransform.return_value = mock_transform

    def test_missing_graph_file_returns_none(self, tmp_path):
        """No matching .arg file → returns (sub_name, None)."""
        modules, _, _, _ = _mock_aims_env()
        with _patched_masks_module(modules), patch("glob.glob", return_value=[]):
            name, data = _load_and_extract_subject(
                self._sub(tmp_path),
                set(),
                self.VT,
                "/fake_bv",
            )
        assert name == "sub01"
        assert data is None

    def test_found_graph_returns_dict(self, tmp_path):
        """Found graph file → returns (sub_name, dict)."""
        modules, mock_aims, _, _ = _mock_aims_env()
        self._mock_graph(mock_aims)
        graph_path = [str(tmp_path / "sub01.arg")]
        with _patched_masks_module(modules), patch("glob.glob", return_value=graph_path):
            name, data = _load_and_extract_subject(
                self._sub(tmp_path),
                {"S.C._left"},
                self.VT,
                "/fake_bv",
            )
        assert name == "sub01"
        assert isinstance(data, dict)

    def test_only_requested_sulci_in_result(self, tmp_path):
        """Vertices outside sulci_full_set are not included in sub_data."""

        def make_vertex(name):
            v = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.__getitem__ = MagicMock(return_value={(0, 0, 0): None})  # noqa: V101

            def _get(k):
                if k == "name":
                    return name
                if k == "aims_ss":
                    return mock_bucket
                return None

            v.get.side_effect = _get  # noqa: V101
            return v

        modules, mock_aims, _, _ = _mock_aims_env()
        self._mock_graph(
            mock_aims,
            vertices=[make_vertex("S.C._left"), make_vertex("OTHER._left")],
        )
        graph_path = [str(tmp_path / "sub01.arg")]
        with _patched_masks_module(modules), patch("glob.glob", return_value=graph_path):
            _, data = _load_and_extract_subject(
                self._sub(tmp_path),
                {"S.C._left"},
                self.VT,
                "/fake_bv",
            )
        assert "S.C._left" in data
        assert "OTHER._left" not in data

    def test_empty_sulci_set_returns_empty_sub_data(self, tmp_path):
        """Empty sulci_full_set → sub_data is an empty dict."""
        modules, mock_aims, _, _ = _mock_aims_env()
        self._mock_graph(mock_aims)
        graph_path = [str(tmp_path / "sub01.arg")]
        with _patched_masks_module(modules), patch("glob.glob", return_value=graph_path):
            _, data = _load_and_extract_subject(
                self._sub(tmp_path),
                set(),
                self.VT,
                "/fake_bv",
            )
        assert data == {}


# ---------------------------------------------------------------------------
# GenerateMasks CLI: --public_use flag
# ---------------------------------------------------------------------------


class TestPublicUseFlag:
    def _parse(self, extra_args=()):
        gm = GenerateMasks()
        argv = [
            "generate_masks",
            "--labeled_subjects_dir",
            "/x",
            "--path_to_graph_supervised",
            "p",
            "--output_dir",
            "/y",
            *extra_args,
        ]
        with patch("sys.argv", argv):
            gm.parse_args()
        return gm

    def test_public_use_defaults_to_false(self):
        """--public_use defaults to False when not supplied."""
        gm = self._parse()
        assert gm.args.public_use is False

    def test_public_use_flag_sets_true(self):
        """Passing --public_use sets the attribute to True."""
        gm = self._parse(["--public_use"])
        assert gm.args.public_use is True


# ---------------------------------------------------------------------------
# GenerateMasks CLI: --njobs implies --buffered
# ---------------------------------------------------------------------------


class TestNjobsAutoBuffered:
    def test_njobs_logs_auto_buffered_message(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-66FDAFAEC932: --njobs without --buffered logs the auto-enable notice at LOG."""
        gm = GenerateMasks()
        with patch(
            "sys.argv",
            [
                "generate_masks",
                "--labeled_subjects_dir",
                "/x",
                "--path_to_graph_supervised",
                "p",
                "--output_dir",
                str(tmp_path),
                "--njobs",
                "4",
            ],
        ):
            gm.parse_args()

        mock_runner = MagicMock()
        mock_runner.return_value = iter([])
        with (
            patch("champollion_pipeline.generate_masks.define_njobs", return_value=4),
            patch("champollion_pipeline.generate_masks.get_sulci_for_regions", return_value=set()),
            patch("champollion_pipeline.generate_masks.MaskRunner.create", return_value=mock_runner),
        ):
            gm.run()

        pipeline_caplog.assert_logged(LOG, "--njobs implies --buffered")

    def test_njobs_without_buffered_uses_buffered_path(self, tmp_path):
        """--njobs alone triggers the buffered runner."""
        gm = GenerateMasks()
        with patch(
            "sys.argv",
            [
                "generate_masks",
                "--labeled_subjects_dir",
                "/x",
                "--path_to_graph_supervised",
                "p",
                "--output_dir",
                str(tmp_path),
                "--njobs",
                "2",
            ],
        ):
            gm.parse_args()

        runner_called = []

        mock_runner = MagicMock()
        mock_runner.side_effect = lambda config: runner_called.append(True) or iter([])  # noqa: V101

        with (
            patch("champollion_pipeline.generate_masks.define_njobs", return_value=2),
            patch("champollion_pipeline.generate_masks.get_sulci_for_regions", return_value=set()),
            patch("champollion_pipeline.generate_masks.MaskRunner.create", return_value=mock_runner),
        ):
            gm.run()

        assert runner_called, "buffered runner was not called"


# ---------------------------------------------------------------------------
# GenerateMasks: output path uses mm-suffix format
# ---------------------------------------------------------------------------


class TestVoxStrOutputPath:
    def test_vox_str_uses_mm_suffix(self, tmp_path):
        """mask_dir in RunConfig contains '2mm', not '2.0'."""
        gm = GenerateMasks()
        with patch(
            "sys.argv",
            [
                "generate_masks",
                "--labeled_subjects_dir",
                "/x",
                "--path_to_graph_supervised",
                "p",
                "--output_dir",
                str(tmp_path),
                "--njobs",
                "1",
            ],
        ):
            gm.parse_args()

        captured = {}

        def fake_runner_call(config):
            captured["mask_dir"] = config.mask_dir
            return iter([])

        mock_runner = MagicMock()
        mock_runner.side_effect = fake_runner_call  # noqa: V101

        with (
            patch("champollion_pipeline.generate_masks.define_njobs", return_value=1),
            patch("champollion_pipeline.generate_masks.get_sulci_for_regions", return_value=set()),
            patch("champollion_pipeline.generate_masks.MaskRunner.create", return_value=mock_runner),
        ):
            gm.run()

        mask_dir = captured.get("mask_dir", "")
        assert "2mm" in mask_dir
        assert "2.0" not in mask_dir

    def test_vox_str_with_masks_tag(self, tmp_path):
        """When --masks is set, mask_dir contains tag then mm-suffix."""
        gm = GenerateMasks()
        with patch(
            "sys.argv",
            [
                "generate_masks",
                "--labeled_subjects_dir",
                "/x",
                "--path_to_graph_supervised",
                "p",
                "--output_dir",
                str(tmp_path),
                "--masks",
                "canonical_25",
                "--njobs",
                "1",
            ],
        ):
            gm.parse_args()

        captured = {}

        def fake_runner_call(config):
            captured["mask_dir"] = config.mask_dir
            return iter([])

        mock_runner = MagicMock()
        mock_runner.side_effect = fake_runner_call  # noqa: V101

        with (
            patch("champollion_pipeline.generate_masks.define_njobs", return_value=1),
            patch("champollion_pipeline.generate_masks.get_sulci_for_regions", return_value=set()),
            patch("champollion_pipeline.generate_masks.MaskRunner.create", return_value=mock_runner),
        ):
            gm.run()

        mask_dir = captured.get("mask_dir", "")
        assert "canonical_25" in mask_dir
        assert "2mm" in mask_dir


# ---------------------------------------------------------------------------
# LOGGING convention: messages go through the champollion_pipeline.generate_masks
# logger at the mapped levels (TASK-LOGGING-BDRABCZUK-FC3904F71983)
# ---------------------------------------------------------------------------

MASKS_LOGGER = "champollion_pipeline.generate_masks"


def _parsed_script(tmp_path):
    """GenerateMasks with parsed arguments, serial unbuffered, writing under tmp_path."""
    gm = GenerateMasks()
    argv = [
        "generate_masks",
        "--labeled_subjects_dir",
        "/x",
        "--path_to_graph_supervised",
        "p",
        "--output_dir",
        str(tmp_path / "masks_out"),
    ]
    with patch("sys.argv", argv):
        gm.parse_args()
    return gm


def _run_with_results(gm, results):
    """Run gm with a runner yielding one side 'L' with the given {sulcus: result}."""
    mock_runner = MagicMock()
    mock_runner.return_value = iter([("L", results)])
    with (
        patch("champollion_pipeline.generate_masks.get_sulci_for_regions", return_value=set(results)),
        patch("champollion_pipeline.generate_masks.MaskRunner.create", return_value=mock_runner),
    ):
        return gm.run()


def _assert_masks_record(pipeline_caplog, level, text):
    record = pipeline_caplog.assert_logged(level, text)
    assert record.name == MASKS_LOGGER


class TestRunResultLogging:
    def test_skipped_sulcus_logged_at_log(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-AC183340EDDB: a skipped sulcus is logged at LOG."""
        _run_with_results(_parsed_script(tmp_path), {"S.C.-sylv.": RETURN_DICTIONARY["skipped"]})

        _assert_masks_record(pipeline_caplog, LOG, "S.C.-sylv.")

    def test_ok_sulcus_logged_at_output(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-9B48A06BD3EC: a computed sulcus is logged at OUTPUT."""
        _run_with_results(_parsed_script(tmp_path), {"S.Pe.C.": RETURN_DICTIONARY["ok"]})

        _assert_masks_record(pipeline_caplog, OUTPUT, "S.Pe.C.")

    def test_invalid_foldlabel_sulcus_logged_at_warning(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-B0FC11C97093: an invalid-foldlabel sulcus is logged at WARNING."""
        _run_with_results(_parsed_script(tmp_path), {"S.Po.C.": RETURN_DICTIONARY["invalid_foldlabel"]})

        _assert_masks_record(pipeline_caplog, logging.WARNING, "S.Po.C.")

    def test_failed_sulcus_logged_at_error(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-CFD9EE54B936: a failed sulcus is logged at ERROR with the failure message."""
        exit_code = _run_with_results(_parsed_script(tmp_path), {"S.T.s.": "failed: graph unreadable"})

        _assert_masks_record(pipeline_caplog, logging.ERROR, "failed: graph unreadable")
        assert exit_code == 1

    def test_summary_logged_at_output(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-E91566B285D8: the run summary is logged at OUTPUT."""
        _run_with_results(_parsed_script(tmp_path), {"S.Pe.C.": RETURN_DICTIONARY["ok"]})

        _assert_masks_record(pipeline_caplog, OUTPUT, "MASK GENERATION SUMMARY")

    def test_output_dir_logged_at_log(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-AC85AC34CD0D: the --output_dir value is logged at LOG when run() starts."""
        _run_with_results(_parsed_script(tmp_path), {})

        _assert_masks_record(pipeline_caplog, LOG, str(tmp_path / "masks_out"))


def _run_config(tmp_path, **overrides):
    values = {
        "sulci": {"S.C."},
        "sides": ["L"],
        "mask_dir": str(tmp_path / "2mm"),
        "labeled_subjects_dir": str(tmp_path / "labelled"),
        "path_to_graph_supervised": "p",
        "nb_subjects": -1,
        "voxel_size": 2.0,
        "force": True,
        "brainvisa_dir": str(tmp_path),
    }
    values.update(overrides)
    return RunConfig(**values)


def _serial_parallel(*_args, **_kwargs):
    """Stand-in for joblib.Parallel that runs the delayed calls in this process."""
    return lambda calls: [func(*args, **kwargs) for func, args, kwargs in calls]


class TestRunnerLogging:
    def test_serial_invalid_foldlabel_logged_at_warning(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-78D091E02EB7: SerialRunner logs the invalid_subjects.log path at WARNING."""
        config = _run_config(tmp_path)
        with patch(
            "champollion_pipeline.generate_masks.compute_mask",
            side_effect=RuntimeError("too many simple surfaces in graph"),
        ):
            outcomes = dict(SerialRunner()(config))

        assert outcomes["L"] == {"S.C.": RETURN_DICTIONARY["invalid_foldlabel"]}
        _assert_masks_record(pipeline_caplog, logging.WARNING, "invalid_subjects.log")

    def test_serial_verbose_call_logged_at_log(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-A9975E7557BB: verbose SerialRunner logs the compute_mask call at LOG."""
        config = _run_config(tmp_path)
        with patch("champollion_pipeline.generate_masks.compute_mask"):
            list(SerialRunner(verbose=True)(config))

        _assert_masks_record(pipeline_caplog, LOG, "calling compute_mask")

    def _run_buffered(self, tmp_path, load_result, subject_names=("sub-07",)):
        config = _run_config(tmp_path)
        subjects = [{"subject": name} for name in subject_names]
        with (
            patch("champollion_pipeline.generate_masks.get_all_subjects_as_dictionary", return_value=subjects),
            patch("champollion_pipeline.generate_masks.select_subjects_int_if_list_of_dict", return_value=subjects),
            patch("champollion_pipeline.generate_masks.Parallel", _serial_parallel),
            patch("champollion_pipeline.generate_masks._load_and_extract_subject", return_value=load_result),
            patch("champollion_pipeline.generate_masks._compute_one_sulcus", return_value=RETURN_DICTIONARY["ok"]),
        ):
            return dict(BufferedRunner(1)(config))

    def test_buffered_missing_graph_logged_at_warning(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-98A88DD87C69: BufferedRunner logs a subject without graph at WARNING."""
        self._run_buffered(tmp_path, ("sub-07", None))

        _assert_masks_record(pipeline_caplog, logging.WARNING, "sub-07")

    def test_buffered_missing_mask_logged_at_warning(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-E87C13722E8D: BufferedRunner logs a mask file absent after the run at WARNING."""
        # _compute_one_sulcus is stubbed, so S.C._left.nii.gz is never written.
        self._run_buffered(tmp_path, ("sub-07", {}))

        _assert_masks_record(pipeline_caplog, logging.WARNING, "S.C._left.nii.gz")

    def test_buffered_loading_start_logged_at_log(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-625A7222ABCC: BufferedRunner logs the subject count at LOG when loading starts."""
        # Neither subject has a graph, so only the loading-start message can carry "2 subjects".
        self._run_buffered(tmp_path, ("sub-07", None), subject_names=("sub-07", "sub-08"))

        _assert_masks_record(pipeline_caplog, LOG, "2 subjects")

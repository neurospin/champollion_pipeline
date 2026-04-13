#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_masks.py:
  - get_sulci_for_regions helper
  - _compute_one_sulcus: public_use flag, makedirs exist_ok, error handling
  - _load_and_extract_subject: missing graph file, sulci filtering
  - GenerateMasks CLI: --public_use flag, --njobs auto-buffered, vox_str format
"""

import json
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from generate_masks import (
    get_sulci_for_regions,
    _compute_one_sulcus,
    _load_and_extract_subject,
    GenerateMasks,
)


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
        return {'voxel_size': [2.0, 2.0, 2.0, 1.0]}

    def copyHeaderFrom(self, other):
        pass


def _mock_aims_env(fake_mask=None):
    """Return (modules_dict, mock_aims, mock_compute_mask, fake_mask).

    Injects fake soma.aims and compute_mask into sys.modules so that the
    lazy imports inside _compute_one_sulcus / _load_and_extract_subject
    find mocks instead of the real PyAIMS library.
    """
    if fake_mask is None:
        fake_mask = _FakeMask()
    mock_aims = MagicMock()
    mock_compute_mask = MagicMock()
    mock_compute_mask.initialize_mask.return_value = fake_mask
    modules = {
        'soma': MagicMock(aims=mock_aims),
        'soma.aims': mock_aims,
        'compute_mask': mock_compute_mask,
    }
    return modules, mock_aims, mock_compute_mask, fake_mask


def _two_subject_voxels(sulcus='S.C._left'):
    """Per-subject voxel dict with two subjects, distinct voxel positions."""
    return {
        'sub01': {sulcus: np.array([[10, 20, 30]], dtype=np.int32)},
        'sub02': {sulcus: np.array([[15, 25, 35]], dtype=np.int32)},
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
                "S.C._left": ["S.C._left"],          # overlaps with S.C.-sylv.
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
        result = get_sulci_for_regions(
            ["S.C.-sylv.", "S.C.-S.Pe.C."], ["L"], json_path
        )
        # S.C. is in both regions — must appear only once
        assert "S.C." in result
        sulci_list = list(result)
        assert sulci_list.count("S.C.") == 1

    def test_two_regions_union(self, json_path):
        """Two regions produce the union of their sulci."""
        result = get_sulci_for_regions(
            ["S.C.-sylv.", "S.C.-S.Pe.C."], ["L"], json_path
        )
        assert result == {"S.C.", "S.C.sylvian.", "S.Pe.C.inf."}

    def test_unknown_region_skipped(self, json_path):
        """Unknown region names are silently skipped."""
        result = get_sulci_for_regions(["DOES_NOT_EXIST"], ["L"], json_path)
        assert result == set()

    def test_mixed_known_unknown_regions(self, json_path):
        """Unknown regions are skipped; known ones still processed."""
        result = get_sulci_for_regions(
            ["S.C.-sylv.", "DOES_NOT_EXIST"], ["L"], json_path
        )
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

    SULCUS = 'S.C._left'
    VT = (2.0, 2.0, 2.0)

    def test_public_use_false_creates_sample_dir(self, tmp_path):
        """public_use=False: per-subject subdirectory is created."""
        modules, mock_aims, _, _ = _mock_aims_env()
        with patch.dict(sys.modules, modules):
            _compute_one_sulcus(
                self.SULCUS, _two_subject_voxels(), self.VT,
                str(tmp_path), 'L', '/fake_bv', public_use=False,
            )
        assert (tmp_path / 'L' / self.SULCUS).is_dir()

    def test_public_use_true_skips_sample_dir(self, tmp_path):
        """public_use=True: per-subject subdirectory is NOT created."""
        modules, mock_aims, _, _ = _mock_aims_env()
        with patch.dict(sys.modules, modules):
            _compute_one_sulcus(
                self.SULCUS, _two_subject_voxels(), self.VT,
                str(tmp_path), 'L', '/fake_bv', public_use=True,
            )
        assert not (tmp_path / 'L' / self.SULCUS).is_dir()

    def test_public_use_false_calls_aims_write_per_subject(self, tmp_path):
        """public_use=False: aims.write called once per subject."""
        modules, mock_aims, _, _ = _mock_aims_env()
        with patch.dict(sys.modules, modules):
            _compute_one_sulcus(
                self.SULCUS, _two_subject_voxels(), self.VT,
                str(tmp_path), 'L', '/fake_bv', public_use=False,
            )
        assert mock_aims.write.call_count == 2

    def test_public_use_true_never_calls_aims_write(self, tmp_path):
        """public_use=True: aims.write is never called."""
        modules, mock_aims, _, _ = _mock_aims_env()
        with patch.dict(sys.modules, modules):
            _compute_one_sulcus(
                self.SULCUS, _two_subject_voxels(), self.VT,
                str(tmp_path), 'L', '/fake_bv', public_use=True,
            )
        mock_aims.write.assert_not_called()

    def test_aggregate_mask_written_regardless_of_public_use(self, tmp_path):
        """write_mask is called for both public_use=True and False."""
        for flag in (False, True):
            modules, _, mock_cm, _ = _mock_aims_env()
            with patch.dict(sys.modules, modules):
                _compute_one_sulcus(
                    self.SULCUS, _two_subject_voxels(), self.VT,
                    str(tmp_path / str(flag)), 'L', '/fake_bv',
                    public_use=flag,
                )
            mock_cm.write_mask.assert_called_once()

    def test_preexisting_side_dir_does_not_raise(self, tmp_path):
        """Pre-existing side directory (EEXIST race) does not cause failure."""
        (tmp_path / 'L').mkdir()  # simulate another thread already created it
        modules, _, _, _ = _mock_aims_env()
        with patch.dict(sys.modules, modules):
            result = _compute_one_sulcus(
                self.SULCUS, {}, self.VT,
                str(tmp_path), 'L', '/fake_bv', public_use=False,
            )
        assert result == "ok"

    def test_returns_ok_on_success(self, tmp_path):
        """Worker returns the string 'ok' on success."""
        modules, _, _, _ = _mock_aims_env()
        with patch.dict(sys.modules, modules):
            result = _compute_one_sulcus(
                self.SULCUS, {}, self.VT,
                str(tmp_path), 'L', '/fake_bv',
            )
        assert result == "ok"

    def test_returns_failed_string_on_exception(self, tmp_path):
        """Exception inside worker is caught and returned as 'failed: …'."""
        modules, _, mock_cm, _ = _mock_aims_env()
        mock_cm.initialize_mask.side_effect = RuntimeError("boom")
        with patch.dict(sys.modules, modules):
            result = _compute_one_sulcus(
                self.SULCUS, {}, self.VT,
                str(tmp_path), 'L', '/fake_bv',
            )
        assert result.startswith("failed:")
        assert "boom" in result

    def test_voxels_accumulated_into_aggregate(self, tmp_path):
        """Both subjects' voxel positions appear in the aggregate array."""
        fake_mask = _FakeMask()
        modules, _, _, _ = _mock_aims_env(fake_mask)
        with patch.dict(sys.modules, modules):
            _compute_one_sulcus(
                self.SULCUS, _two_subject_voxels(), self.VT,
                str(tmp_path), 'L', '/fake_bv', public_use=True,
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
            'subject': 'sub01',
            'dir': str(path),
            'graph_file': '%(subject)s.arg',
        }

    def _mock_graph(self, mock_aims, vertices=()):
        mock_graph = MagicMock()
        mock_graph.__getitem__ = MagicMock(return_value=[2.0, 2.0, 2.0, 1.0])
        mock_graph.vertices.return_value = list(vertices)
        mock_aims.read.return_value = mock_graph
        mock_transform = MagicMock()
        mock_transform.transform.side_effect = lambda c: c
        mock_aims.GraphManip.getICBM2009cTemplateTransform.return_value = (
            mock_transform)

    def test_missing_graph_file_returns_none(self, tmp_path):
        """No matching .arg file → returns (sub_name, None)."""
        modules, _, _, _ = _mock_aims_env()
        with patch.dict(sys.modules, modules):
            with patch('glob.glob', return_value=[]):
                name, data = _load_and_extract_subject(
                    self._sub(tmp_path), set(), self.VT, '/fake_bv',
                )
        assert name == 'sub01'
        assert data is None

    def test_found_graph_returns_dict(self, tmp_path):
        """Found graph file → returns (sub_name, dict)."""
        modules, mock_aims, _, _ = _mock_aims_env()
        self._mock_graph(mock_aims)
        graph_path = [str(tmp_path / 'sub01.arg')]
        with patch.dict(sys.modules, modules):
            with patch('glob.glob', return_value=graph_path):
                name, data = _load_and_extract_subject(
                    self._sub(tmp_path), {'S.C._left'}, self.VT, '/fake_bv',
                )
        assert name == 'sub01'
        assert isinstance(data, dict)

    def test_only_requested_sulci_in_result(self, tmp_path):
        """Vertices outside sulci_full_set are not included in sub_data."""
        def make_vertex(name):
            v = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.__getitem__ = MagicMock(return_value={(0, 0, 0): None})
            def _get(k):
                if k == 'name':
                    return name
                if k == 'aims_ss':
                    return mock_bucket
                return None
            v.get.side_effect = _get
            return v

        modules, mock_aims, _, _ = _mock_aims_env()
        self._mock_graph(
            mock_aims,
            vertices=[make_vertex('S.C._left'), make_vertex('OTHER._left')],
        )
        graph_path = [str(tmp_path / 'sub01.arg')]
        with patch.dict(sys.modules, modules):
            with patch('glob.glob', return_value=graph_path):
                _, data = _load_and_extract_subject(
                    self._sub(tmp_path), {'S.C._left'}, self.VT, '/fake_bv',
                )
        assert 'S.C._left' in data
        assert 'OTHER._left' not in data

    def test_empty_sulci_set_returns_empty_sub_data(self, tmp_path):
        """Empty sulci_full_set → sub_data is an empty dict."""
        modules, mock_aims, _, _ = _mock_aims_env()
        self._mock_graph(mock_aims)
        graph_path = [str(tmp_path / 'sub01.arg')]
        with patch.dict(sys.modules, modules):
            with patch('glob.glob', return_value=graph_path):
                _, data = _load_and_extract_subject(
                    self._sub(tmp_path), set(), self.VT, '/fake_bv',
                )
        assert data == {}


# ---------------------------------------------------------------------------
# GenerateMasks CLI: --public_use flag
# ---------------------------------------------------------------------------

class TestPublicUseFlag:

    def _parse(self, extra_args=()):
        gm = GenerateMasks()
        argv = [
            'generate_masks',
            '--labeled_subjects_dir', '/x',
            '--path_to_graph_supervised', 'p',
            '--output_dir', '/y',
        ] + list(extra_args)
        with patch('sys.argv', argv):
            gm.parse_args()
        return gm

    def test_public_use_defaults_to_false(self):
        """--public_use defaults to False when not supplied."""
        gm = self._parse()
        assert gm.args.public_use is False

    def test_public_use_flag_sets_true(self):
        """Passing --public_use sets the attribute to True."""
        gm = self._parse(['--public_use'])
        assert gm.args.public_use is True


# ---------------------------------------------------------------------------
# GenerateMasks CLI: --njobs implies --buffered
# ---------------------------------------------------------------------------

class TestNjobsAutoBuffered:

    def _bv_modules(self, njobs=4):
        """sys.modules patch dict covering deep_folding parallel util."""
        mock_parallel = MagicMock()
        mock_parallel.define_njobs.return_value = njobs
        return {
            'compute_mask': MagicMock(),
            'deep_folding': MagicMock(),
            'deep_folding.brainvisa': MagicMock(),
            'deep_folding.brainvisa.utils': MagicMock(),
            'deep_folding.brainvisa.utils.parallel': mock_parallel,
        }

    def test_njobs_prints_auto_buffered_message(self, tmp_path, capsys):
        """--njobs without --buffered prints the auto-enable notice."""
        gm = GenerateMasks()
        with patch('sys.argv', [
            'generate_masks',
            '--labeled_subjects_dir', '/x',
            '--path_to_graph_supervised', 'p',
            '--output_dir', str(tmp_path),
            '--njobs', '4',
        ]):
            gm.parse_args()

        mock_runner = MagicMock()
        mock_runner.return_value = iter([])
        with patch.dict(sys.modules, self._bv_modules(4)), \
             patch('generate_masks.get_sulci_for_regions',
                   return_value=set()), \
             patch('generate_masks.MaskRunner.create', return_value=mock_runner):
            gm.run()

        out = capsys.readouterr().out
        assert '--njobs implies --buffered' in out

    def test_njobs_without_buffered_uses_buffered_path(self, tmp_path):
        """--njobs alone triggers the buffered runner."""
        gm = GenerateMasks()
        with patch('sys.argv', [
            'generate_masks',
            '--labeled_subjects_dir', '/x',
            '--path_to_graph_supervised', 'p',
            '--output_dir', str(tmp_path),
            '--njobs', '2',
        ]):
            gm.parse_args()

        runner_called = []

        mock_runner = MagicMock()
        mock_runner.side_effect = lambda config: (runner_called.append(True) or iter([]))

        with patch.dict(sys.modules, self._bv_modules(2)), \
             patch('generate_masks.get_sulci_for_regions',
                   return_value=set()), \
             patch('generate_masks.MaskRunner.create', return_value=mock_runner):
            gm.run()

        assert runner_called, "buffered runner was not called"


# ---------------------------------------------------------------------------
# GenerateMasks: output path uses mm-suffix format
# ---------------------------------------------------------------------------

class TestVoxStrOutputPath:

    def _bv_modules(self, njobs=1):
        mock_parallel = MagicMock()
        mock_parallel.define_njobs.return_value = njobs
        return {
            'compute_mask': MagicMock(),
            'deep_folding': MagicMock(),
            'deep_folding.brainvisa': MagicMock(),
            'deep_folding.brainvisa.utils': MagicMock(),
            'deep_folding.brainvisa.utils.parallel': mock_parallel,
        }

    def test_vox_str_uses_mm_suffix(self, tmp_path):
        """mask_dir in RunConfig contains '2mm', not '2.0'."""
        gm = GenerateMasks()
        with patch('sys.argv', [
            'generate_masks',
            '--labeled_subjects_dir', '/x',
            '--path_to_graph_supervised', 'p',
            '--output_dir', str(tmp_path),
            '--njobs', '1',
        ]):
            gm.parse_args()

        captured = {}

        def fake_runner_call(config):
            captured['mask_dir'] = config.mask_dir
            return iter([])

        mock_runner = MagicMock()
        mock_runner.side_effect = fake_runner_call

        with patch.dict(sys.modules, self._bv_modules()), \
             patch('generate_masks.get_sulci_for_regions',
                   return_value=set()), \
             patch('generate_masks.MaskRunner.create', return_value=mock_runner):
            gm.run()

        mask_dir = captured.get('mask_dir', '')
        assert '2mm' in mask_dir
        assert '2.0' not in mask_dir

    def test_vox_str_with_masks_tag(self, tmp_path):
        """When --masks is set, mask_dir contains tag then mm-suffix."""
        gm = GenerateMasks()
        with patch('sys.argv', [
            'generate_masks',
            '--labeled_subjects_dir', '/x',
            '--path_to_graph_supervised', 'p',
            '--output_dir', str(tmp_path),
            '--masks', 'canonical_25',
            '--njobs', '1',
        ]):
            gm.parse_args()

        captured = {}

        def fake_runner_call(config):
            captured['mask_dir'] = config.mask_dir
            return iter([])

        mock_runner = MagicMock()
        mock_runner.side_effect = fake_runner_call

        with patch.dict(sys.modules, self._bv_modules()), \
             patch('generate_masks.get_sulci_for_regions',
                   return_value=set()), \
             patch('generate_masks.MaskRunner.create', return_value=mock_runner):
            gm.run()

        mask_dir = captured.get('mask_dir', '')
        assert 'canonical_25' in mask_dir
        assert '2mm' in mask_dir

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the mask-generation runner strategies in generate_masks.py:
  - _log_invalid_subject
  - MaskRunner.create factory (--buffered / --njobs matrix)
  - SerialRunner.__call__ (skip / ok / invalid foldlabel / failure)
  - BufferedRunner.__call__ (two-phase load + compute, verbose, failures)
  - GenerateMasks.run() result accounting and exit code

BrainVISA (``compute_mask``, ``deep_folding``, ``soma.aims``) is injected as
mocks through sys.modules, and the joblib workers are replaced by in-process
stand-ins, so nothing here needs PyAIMS, a graph database or a cluster.
The ``n_jobs``/soma-workflow contract is exercised, not changed: joblib is
still driven through ``Parallel(n_jobs=...)`` exactly as in production.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline import generate_masks as gm
from champollion_pipeline.generate_masks import (
    RETURN_DICTIONARY,
    BufferedRunner,
    GenerateMasks,
    MaskRunner,
    RunConfig,
    SerialRunner,
    _log_invalid_subject,
    main,
)


def complete_sulci_name(sulcus, side):
    """Stand-in for deep_folding's sulcus-name completion."""
    return f"{sulcus}_{'left' if side == 'L' else 'right'}"


def deep_folding_modules(**extra):
    """sys.modules entries for the deep_folding sub-packages used by runners."""
    sulcus_mod = types.ModuleType("cortical_tiles.brainvisa.utils.sulcus")
    sulcus_mod.complete_sulci_name = complete_sulci_name
    modules = {
        "cortical_tiles": MagicMock(),
        "cortical_tiles.brainvisa": MagicMock(),
        "cortical_tiles.brainvisa.utils": MagicMock(),
        "cortical_tiles.brainvisa.utils.sulcus": sulcus_mod,
    }
    modules.update(extra)
    return modules


def make_config(tmp_path, **overrides):
    """Build a RunConfig pointing at tmp_path."""
    params = {
        "sulci": {"S.C."},
        "sides": ["L"],
        "mask_dir": str(tmp_path / "masks"),
        "labeled_subjects_dir": str(tmp_path / "subjects"),
        "path_to_graph_supervised": "t1mri/t1/default_analysis/folds/3.3/base",
        "nb_subjects": -1,
        "voxel_size": 2.0,
        "force": False,
        "brainvisa_dir": str(tmp_path / "brainvisa"),
        "public_use": False,
    }
    params.update(overrides)
    return RunConfig(**params)


class TestLogInvalidSubject:
    """Test _log_invalid_subject."""

    def test_creates_parent_directory_and_file(self, tmp_path):
        log_path = tmp_path / "nested" / "invalid_subjects.log"
        _log_invalid_subject(str(log_path), "sub-01")
        assert log_path.exists()

    def test_line_is_timestamp_tab_subject(self, tmp_path):
        log_path = tmp_path / "invalid_subjects.log"
        _log_invalid_subject(str(log_path), "sub-01 graph=bad")
        stamp, info = log_path.read_text().rstrip("\n").split("\t")
        assert info == "sub-01 graph=bad"
        assert stamp.startswith("20")

    def test_appends_instead_of_truncating(self, tmp_path):
        log_path = tmp_path / "invalid_subjects.log"
        _log_invalid_subject(str(log_path), "first")
        _log_invalid_subject(str(log_path), "second")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[1].endswith("second")


class TestMaskRunnerFactory:
    """Test the MaskRunner.create dispatch matrix."""

    def test_no_flags_gives_serial_runner(self):
        runner = MaskRunner.create(buffered=False, njobs=None)
        assert isinstance(runner, SerialRunner)

    def test_buffered_without_njobs_gives_single_worker_buffered(self):
        runner = MaskRunner.create(buffered=True, njobs=None)
        assert isinstance(runner, BufferedRunner)
        assert runner.njobs == 1

    def test_njobs_gives_buffered_with_that_worker_count(self):
        runner = MaskRunner.create(buffered=False, njobs=7)
        assert isinstance(runner, BufferedRunner)
        assert runner.njobs == 7

    def test_njobs_wins_over_buffered(self):
        runner = MaskRunner.create(buffered=True, njobs=3)
        assert runner.njobs == 3

    def test_verbose_is_propagated(self):
        assert MaskRunner.create(False, None, verbose=True).verbose is True
        assert MaskRunner.create(True, None, verbose=True).verbose is True
        assert MaskRunner.create(False, 2, verbose=True).verbose is True

    def test_verbose_defaults_to_false(self):
        assert MaskRunner.create(False, None).verbose is False


class TestSerialRunner:
    """Test SerialRunner.__call__."""

    def test_yields_one_entry_per_side(self, tmp_path):
        config = make_config(tmp_path, sides=["L", "R"])
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = MagicMock()
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            out = list(SerialRunner()(config))
        assert [side for side, _ in out] == ["L", "R"]

    def test_successful_sulcus_is_ok(self, tmp_path):
        config = make_config(tmp_path)
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = MagicMock()
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            (_, results), = list(SerialRunner()(config))
        assert results == {"S.C.": RETURN_DICTIONARY["ok"]}

    def test_compute_mask_receives_config_values(self, tmp_path):
        config = make_config(tmp_path, nb_subjects=12, voxel_size=1.0)
        mock_compute = MagicMock()
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = mock_compute
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            list(SerialRunner()(config))
        kwargs = mock_compute.call_args.kwargs
        assert kwargs["src_dir"] == config.labeled_subjects_dir
        assert kwargs["path_to_graph"] == config.path_to_graph_supervised
        assert kwargs["number_subjects"] == 12
        assert kwargs["out_voxel_size"] == 1.0
        assert kwargs["side"] == "L"

    def test_existing_mask_is_skipped(self, tmp_path):
        config = make_config(tmp_path)
        mask_file = Path(config.mask_dir) / "L" / "S.C._left.nii.gz"
        mask_file.parent.mkdir(parents=True)
        mask_file.touch()
        mock_compute = MagicMock()
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = mock_compute
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            (_, results), = list(SerialRunner()(config))
        assert results == {"S.C.": RETURN_DICTIONARY["skipped"]}
        mock_compute.assert_not_called()

    def test_force_recomputes_existing_mask(self, tmp_path):
        config = make_config(tmp_path, force=True)
        mask_file = Path(config.mask_dir) / "L" / "S.C._left.nii.gz"
        mask_file.parent.mkdir(parents=True)
        mask_file.touch()
        mock_compute = MagicMock()
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = mock_compute
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            (_, results), = list(SerialRunner()(config))
        assert results == {"S.C.": RETURN_DICTIONARY["ok"]}
        mock_compute.assert_called_once()

    def test_verbose_prints_compute_mask_parameters(self, tmp_path, capsys):
        config = make_config(tmp_path)
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = MagicMock()
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            list(SerialRunner(verbose=True)(config))
        out = capsys.readouterr().out
        assert "calling compute_mask" in out
        assert "nb_subjects=-1" in out

    def test_generic_failure_is_reported_as_failed(self, tmp_path):
        config = make_config(tmp_path)
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            (_, results), = list(SerialRunner()(config))
        assert results["S.C."].startswith("failed: ")
        assert "boom" in results["S.C."]

    def test_too_many_simple_surfaces_is_invalid_foldlabel(self, tmp_path):
        config = make_config(tmp_path)
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = MagicMock(
            side_effect=RuntimeError("graph has too many simple surfaces")
        )
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            (_, results), = list(SerialRunner()(config))
        assert results["S.C."] == RETURN_DICTIONARY["invalid_foldlabel"]

    def test_invalid_foldlabel_is_logged_to_file(self, tmp_path):
        config = make_config(tmp_path)
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = MagicMock(
            side_effect=RuntimeError("too many simple surfaces")
        )
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            list(SerialRunner()(config))
        log_path = Path(config.mask_dir) / "invalid_subjects.log"
        assert log_path.exists()
        assert "S.C._left" in log_path.read_text()

    def test_invalid_foldlabel_detected_through_exception_cause(self, tmp_path):
        config = make_config(tmp_path)

        def raise_chained(**kwargs):
            try:
                raise ValueError("too many simple surfaces")
            except ValueError as inner:
                raise RuntimeError("wrapper") from inner

        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = raise_chained
        with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
            (_, results), = list(SerialRunner()(config))
        assert results["S.C."] == RETURN_DICTIONARY["invalid_foldlabel"]

    def test_brainvisa_dir_is_added_to_sys_path(self, tmp_path):
        config = make_config(tmp_path)
        compute_mask_mod = types.ModuleType("compute_mask")
        compute_mask_mod.compute_mask = MagicMock()
        original = list(sys.path)
        try:
            with patch.dict(sys.modules, deep_folding_modules(compute_mask=compute_mask_mod)):
                list(SerialRunner()(config))
            assert config.brainvisa_dir in sys.path
        finally:
            sys.path[:] = original


class TestBufferedRunner:
    """Test BufferedRunner.__call__ (two-phase load + compute)."""

    @pytest.fixture
    def subjects_modules(self):
        """deep_folding modules exposing the subject-selection helpers."""
        subjects_mod = types.ModuleType("cortical_tiles.brainvisa.utils.subjects")
        subjects_mod.get_all_subjects_as_dictionary = lambda dirs, patterns, side: [
            {"subject": "sub01", "dir": dirs[0], "graph_file": patterns[0], "side": side},
            {"subject": "sub02", "dir": dirs[0], "graph_file": patterns[0], "side": side},
        ]
        subjects_mod.select_subjects_int_if_list_of_dict = lambda subs, _all, nb: (
            subs if nb == -1 else subs[:nb]
        )
        return deep_folding_modules(**{"cortical_tiles.brainvisa.utils.subjects": subjects_mod})

    def _patch_workers(self, monkeypatch, load=None, compute=None, mask_dir=None):
        """Replace the two module-level joblib workers with in-process fakes."""
        if load is None:

            def load(sub, sulci_full_set, voxel_size_tuple, brainvisa_dir):
                return sub["subject"], {sf: [(0, 0, 0)] for sf in sulci_full_set}

        if compute is None:

            def compute(sf, per_subject_voxels, voxel_size_tuple, md, side, bv_dir, public_use):
                out = Path(md) / side / f"{sf}.nii.gz"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.touch()
                return RETURN_DICTIONARY["ok"]

        monkeypatch.setattr(gm, "_load_and_extract_subject", load)
        monkeypatch.setattr(gm, "_compute_one_sulcus", compute)

    def test_successful_run_marks_every_sulcus_ok(self, tmp_path, monkeypatch, subjects_modules):
        self._patch_workers(monkeypatch)
        config = make_config(tmp_path, sulci={"S.C.", "S.Or."})
        with patch.dict(sys.modules, subjects_modules):
            (side, results), = list(BufferedRunner(1)(config))
        assert side == "L"
        assert results == {"S.C.": RETURN_DICTIONARY["ok"], "S.Or.": RETURN_DICTIONARY["ok"]}

    def test_all_masks_present_message(self, tmp_path, monkeypatch, subjects_modules, capsys):
        self._patch_workers(monkeypatch)
        config = make_config(tmp_path)
        with patch.dict(sys.modules, subjects_modules):
            list(BufferedRunner(1)(config))
        assert "All 1 mask files present." in capsys.readouterr().out

    def test_missing_mask_files_are_reported(self, tmp_path, monkeypatch, subjects_modules, capsys):
        def compute(sf, *args, **kwargs):
            return RETURN_DICTIONARY["ok"]

        self._patch_workers(monkeypatch, compute=compute)
        config = make_config(tmp_path)
        with patch.dict(sys.modules, subjects_modules):
            list(BufferedRunner(1)(config))
        out = capsys.readouterr().out
        assert "1 mask file(s) missing" in out
        assert "MISSING: S.C._left.nii.gz" in out

    def test_subject_without_graph_is_warned_and_dropped(
        self, tmp_path, monkeypatch, subjects_modules, capsys
    ):
        def load(sub, sulci_full_set, voxel_size_tuple, brainvisa_dir):
            if sub["subject"] == "sub02":
                return sub["subject"], None
            return sub["subject"], {sf: [(0, 0, 0)] for sf in sulci_full_set}

        self._patch_workers(monkeypatch, load=load)
        config = make_config(tmp_path)
        with patch.dict(sys.modules, subjects_modules):
            list(BufferedRunner(1)(config))
        assert "no graph for sub02, skipped" in capsys.readouterr().out

    def test_verbose_reports_per_subject_and_per_sulcus_details(
        self, tmp_path, monkeypatch, subjects_modules, capsys
    ):
        self._patch_workers(monkeypatch)
        config = make_config(tmp_path)
        with patch.dict(sys.modules, subjects_modules):
            list(BufferedRunner(1, verbose=True)(config))
        out = capsys.readouterr().out
        assert "loaded sub01: 1 sulci, 1 voxels" in out
        assert "S.C._left: 2 subjects contributing, 2 voxels total" in out

    def test_existing_masks_are_not_recomputed(self, tmp_path, monkeypatch, subjects_modules):
        computed = []

        def compute(sf, *args, **kwargs):
            computed.append(sf)
            return RETURN_DICTIONARY["ok"]

        self._patch_workers(monkeypatch, compute=compute)
        config = make_config(tmp_path)
        mask_file = Path(config.mask_dir) / "L" / "S.C._left.nii.gz"
        mask_file.parent.mkdir(parents=True)
        mask_file.touch()
        with patch.dict(sys.modules, subjects_modules):
            (_, results), = list(BufferedRunner(1)(config))
        assert computed == []
        assert results == {"S.C.": RETURN_DICTIONARY["skipped"]}

    def test_force_recomputes_existing_masks(self, tmp_path, monkeypatch, subjects_modules):
        computed = []

        def compute(sf, per_subject_voxels, voxel_size_tuple, md, side, bv_dir, public_use):
            computed.append(sf)
            out = Path(md) / side / f"{sf}.nii.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.touch()
            return RETURN_DICTIONARY["ok"]

        self._patch_workers(monkeypatch, compute=compute)
        config = make_config(tmp_path, force=True)
        mask_file = Path(config.mask_dir) / "L" / "S.C._left.nii.gz"
        mask_file.parent.mkdir(parents=True)
        mask_file.touch()
        with patch.dict(sys.modules, subjects_modules):
            list(BufferedRunner(1)(config))
        assert computed == ["S.C._left"]

    def test_compute_phase_failure_marks_all_sulci_failed(
        self, tmp_path, monkeypatch, subjects_modules
    ):
        def compute(*args, **kwargs):
            raise RuntimeError("worker pool died")

        self._patch_workers(monkeypatch, compute=compute)
        config = make_config(tmp_path, sulci={"S.C.", "S.Or."})
        with patch.dict(sys.modules, subjects_modules):
            (_, results), = list(BufferedRunner(1)(config))
        assert all(v.startswith("failed: ") for v in results.values())
        assert "worker pool died" in results["S.C."]

    def test_public_use_flag_is_forwarded_to_the_worker(
        self, tmp_path, monkeypatch, subjects_modules
    ):
        seen = {}

        def compute(sf, per_subject_voxels, voxel_size_tuple, md, side, bv_dir, public_use):
            seen["public_use"] = public_use
            return RETURN_DICTIONARY["ok"]

        self._patch_workers(monkeypatch, compute=compute)
        config = make_config(tmp_path, public_use=True)
        with patch.dict(sys.modules, subjects_modules):
            list(BufferedRunner(1)(config))
        assert seen["public_use"] is True

    def test_graph_pattern_includes_path_to_graph_supervised(
        self, tmp_path, monkeypatch, subjects_modules
    ):
        seen = {}

        def load(sub, *args, **kwargs):
            seen["graph_file"] = sub["graph_file"]
            return sub["subject"], {}

        self._patch_workers(monkeypatch, load=load)
        config = make_config(tmp_path)
        with patch.dict(sys.modules, subjects_modules):
            list(BufferedRunner(1)(config))
        assert seen["graph_file"] == (
            "%(subject)s/" + config.path_to_graph_supervised + "/%(side)s%(subject)s*.arg"
        )

    def test_both_sides_are_processed(self, tmp_path, monkeypatch, subjects_modules):
        self._patch_workers(monkeypatch)
        config = make_config(tmp_path, sides=["L", "R"])
        with patch.dict(sys.modules, subjects_modules):
            out = list(BufferedRunner(2)(config))
        assert [side for side, _ in out] == ["L", "R"]


class TestGenerateMasksRunAccounting:
    """Test GenerateMasks.run() summary counters and exit code."""

    def _script(self, tmp_path, extra=None):
        script = GenerateMasks()
        script.parse_args(
            [
                "--labeled_subjects_dir",
                str(tmp_path / "subjects"),
                "--path_to_graph_supervised",
                "folds/3.3/base",
                "--output_dir",
                str(tmp_path / "out"),
            ]
            + (extra or [])
        )
        return script

    def _run(self, script, yielded, sulci=None):
        runner = MagicMock(return_value=iter(yielded))
        with (
            patch.object(gm, "get_sulci_for_regions", return_value=sulci or {"S.C."}),
            patch.object(gm.MaskRunner, "create", return_value=runner),
        ):
            return script.run()

    def test_returns_zero_when_nothing_failed(self, tmp_path):
        script = self._script(tmp_path)
        result = self._run(script, [("L", {"S.C.": RETURN_DICTIONARY["ok"]})])
        assert result == 0

    def test_returns_one_when_a_sulcus_failed(self, tmp_path):
        script = self._script(tmp_path)
        result = self._run(script, [("L", {"S.C.": "failed: boom"})])
        assert result == 1

    def test_summary_counts_each_result_kind(self, tmp_path, capsys):
        script = self._script(tmp_path, ["--sides", "L"])
        yielded = [
            (
                "L",
                {
                    "a": RETURN_DICTIONARY["ok"],
                    "b": RETURN_DICTIONARY["skipped"],
                    "c": RETURN_DICTIONARY["invalid_foldlabel"],
                    "d": "failed: boom",
                },
            )
        ]
        self._run(script, yielded, sulci={"a", "b", "c", "d"})
        out = capsys.readouterr().out
        assert "Success:  1" in out
        assert "Skipped:  1" in out
        assert "Invalid:  1" in out
        assert "Failed:   1" in out
        assert "Total:    4" in out

    def test_per_sulcus_lines_are_printed(self, tmp_path, capsys):
        script = self._script(tmp_path)
        self._run(script, [("L", {"S.C.": RETURN_DICTIONARY["invalid_foldlabel"]})])
        assert "invalid foldlabel" in capsys.readouterr().out

    def test_masks_version_is_inserted_into_output_path(self, tmp_path, capsys):
        script = self._script(tmp_path, ["--masks", "canonical_25"])
        self._run(script, [("L", {})])
        out = capsys.readouterr().out
        assert str(tmp_path / "out" / "canonical_25" / "2mm") in out

    def test_without_masks_version_output_path_is_output_dir(self, tmp_path, capsys):
        script = self._script(tmp_path)
        self._run(script, [("L", {})])
        out = capsys.readouterr().out
        assert str(tmp_path / "out" / "2mm") in out

    def test_default_regions_are_used_when_none_given(self, tmp_path):
        script = self._script(tmp_path)
        runner = MagicMock(return_value=iter([]))
        with (
            patch.object(gm, "get_sulci_for_regions", return_value=set()) as get_sulci,
            patch.object(gm.MaskRunner, "create", return_value=runner),
        ):
            script.run()
        assert get_sulci.call_args.args[0] == gm._REGIONS_DEFAULT

    def test_explicit_regions_are_forwarded(self, tmp_path):
        script = self._script(tmp_path, ["--regions", "S.Or.", "S.T.s."])
        runner = MagicMock(return_value=iter([]))
        with (
            patch.object(gm, "get_sulci_for_regions", return_value=set()) as get_sulci,
            patch.object(gm.MaskRunner, "create", return_value=runner),
        ):
            script.run()
        assert get_sulci.call_args.args[0] == ["S.Or.", "S.T.s."]


class TestMain:
    """Test the main() entry point."""

    def test_main_delegates_to_script_main(self, monkeypatch):
        called = []

        class FakeScript:
            def main(self):
                called.append(True)
                return 0

        monkeypatch.setattr(gm, "GenerateMasks", FakeScript)
        assert main() == 0
        assert called == [True]

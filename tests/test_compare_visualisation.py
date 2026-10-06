#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for ``compare.py --visualisation`` (the Anatomist viewer path).

One test class per backlog task so each can be verified alone with ``-k``:

- TestTask191Ranking            REQ-COMPARE-45  ranking by real magnitude
- TestTask192XorFiles           REQ-COMPARE-46  XOR file source and temp cleanup
- TestTask187ViewerFailure      REQ-COMPARE-47  non-zero viewer exit is reported
- TestTask188DisplayCheck       REQ-COMPARE-49  skip when no usable display
- TestTask193NoQApplication     REQ-COMPARE-50  viewer exits non-zero without a QApplication
- TestTask190FusedLinkedViews   REQ-COMPARE-51  fused, linked Axial/Sagittal/Coronal windows
- TestTask189HelpText           REQ-COMPARE-52  help text and docstring match the code
- TestTask185VisualiseImports   REQ-COMPARE-53  no function-level imports

Contract pinned here: ``visualise_mask_diffs(scores, masks_a, masks_b,
xor_dir=None, top_k=5) -> int`` where ``scores`` maps a mask name to its
ranking score; the viewer is launched with
``subprocess.run([sys.executable, "-c", _VIEWER_SCRIPT, json.dumps(entries)])``
and each entry carries ``name``, ``path_a``, ``path_b`` and ``path_xor``.
``_VIEWER_SCRIPT`` is a module-level constant; it is executed here with fake
``anatomist.direct.api`` and ``soma.qt_gui.qt_backend`` modules, so no display
and no real Anatomist are needed.

Self-contained: the fake PyAIMS below mirrors the one in tests/test_compare.py.
"""

import ast
import json
import math
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

import compare
from compare import Compare, visualise_mask_diffs

pytestmark = pytest.mark.unit  # noqa: V107

SRC_COMPARE = Path(__file__).resolve().parent.parent / "src" / "compare.py"


# ---------------------------------------------------------------------------
# Fake PyAIMS
# ---------------------------------------------------------------------------


class _FakeVolume:
    """Stand-in for an aims Volume carrying a header."""

    def __init__(self, array, header=None):
        self.array = np.asarray(array)
        self._header = header if header is not None else {"voxel_size": [2.0, 2.0, 2.0]}
        self.copied_header = None

    def header(self):
        return self._header

    def copyHeaderFrom(self, header):  # noqa: N802 - mirrors the PyAIMS API
        self.copied_header = header

    def __array__(self, dtype=None, copy=None):
        return self.array.astype(dtype) if dtype is not None else self.array


class _FakeAims:
    """Fake ``soma.aims``: serves registered volumes, records (and touches) writes."""

    def __init__(self, objects=None, fail_read=(), fail_write_after=None):
        self.objects = dict(objects or {})
        self.written = []
        self.fail_read = set(fail_read)
        self.fail_write_after = fail_write_after

    def read(self, path):
        if str(path) in self.fail_read:
            raise RuntimeError(f"fake aims.read failure: {path}")
        return self.objects[str(path)]

    def Volume(self, array):  # noqa: N802 - mirrors the PyAIMS API
        return _FakeVolume(array)

    def write(self, vol, path):
        if self.fail_write_after is not None and len(self.written) >= self.fail_write_after:
            raise RuntimeError(f"fake aims.write failure: {path}")
        self.written.append((vol, str(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()


def _point(index, shape=(4, 4, 4)):
    vol = np.zeros(shape, dtype=np.float64)
    vol[index] = 1.0
    return vol


def _mask_pairs(names):
    """Fake mask paths for ``names``; set A has one voxel at (0,0,0), set B at (1,0,0)."""
    objects, masks_a, masks_b = {}, {}, {}
    for name in names:
        pa, pb = f"/a/{name}", f"/b/{name}"
        objects[pa] = _FakeVolume(_point((0, 0, 0)))
        objects[pb] = _FakeVolume(_point((1, 0, 0)))
        masks_a[name], masks_b[name] = pa, pb
    return objects, masks_a, masks_b


def _make_tree(root: Path, rel_paths) -> Path:
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def _make_script(argv) -> Compare:
    script = Compare()
    script.args = script.parse_args(argv)
    return script


class _RunRecorder:
    """Replacement for ``subprocess.run`` recording each call."""

    def __init__(self, returncode=0, raises=None, on_call=None):
        self.calls = []
        self.returncode = returncode
        self.raises = raises
        self.on_call = on_call

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        if self.on_call is not None:
            self.on_call(cmd)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(args=cmd, returncode=self.returncode)

    def entries(self, call=0):
        return json.loads(self.calls[call][-1])


@pytest.fixture  # noqa: V103
def display_env(monkeypatch):
    """A usable X11 display, so the viewer is not skipped."""
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Redirect every tempfile location into an empty directory that tests can inspect."""
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def run_recorder(monkeypatch):
    def _install(**kwargs):
        rec = _RunRecorder(**kwargs)
        monkeypatch.setattr(subprocess, "run", rec)
        return rec

    return _install


@pytest.fixture
def mask_tree(tmp_path, monkeypatch):
    """Build set_a/set_b trees for ``masks`` mode and install a matching fake aims.

    ``specs`` maps a relative mask path to its (array_a, array_b) pair.
    """

    def _install(specs):
        dir_a = _make_tree(tmp_path / "a", list(specs))
        dir_b = _make_tree(tmp_path / "b", list(specs))
        objects = {}
        for rel, (arr_a, arr_b) in specs.items():
            objects[str((dir_a / rel).resolve())] = _FakeVolume(arr_a)
            objects[str((dir_b / rel).resolve())] = _FakeVolume(arr_b)
        fake = _FakeAims(objects)
        monkeypatch.setattr(compare, "aims", fake)
        return dir_a, dir_b, fake

    return _install


def _capture_visualise(monkeypatch):
    """Replace compare.visualise_mask_diffs with a recorder returning 0."""
    seen = {}

    def _fake(scores, masks_a, masks_b, *args, **kwargs):
        seen["scores"] = dict(scores)
        seen["args"] = args
        seen["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(compare, "visualise_mask_diffs", _fake)
    return seen


# ---------------------------------------------------------------------------
# TASK-191 / REQ-COMPARE-45: ranking by real magnitude
# ---------------------------------------------------------------------------


class TestTask191Ranking:
    @pytest.mark.usefixtures("display_env")
    def test_top_five_by_score_descending(self, monkeypatch, run_recorder):
        names = [f"m{i}.nii.gz" for i in range(7)]
        objects, masks_a, masks_b = _mask_pairs(names)
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        run = run_recorder()

        scores = {name: float(i + 1) for i, name in enumerate(names)}
        assert visualise_mask_diffs(scores, masks_a, masks_b) == 0

        assert len(run.calls) == 1
        assert [e["name"] for e in run.entries()] == ["m6.nii.gz", "m5.nii.gz", "m4.nii.gz", "m3.nii.gz", "m2.nii.gz"]

    @pytest.mark.usefixtures("display_env")
    def test_infinite_score_ranks_first(self, monkeypatch, run_recorder):
        names = ["near.nii.gz", "far.nii.gz", "emptied.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        run = run_recorder()

        scores = {"near.nii.gz": 0.5, "far.nii.gz": 3.25, "emptied.nii.gz": math.inf}
        visualise_mask_diffs(scores, masks_a, masks_b)

        assert [e["name"] for e in run.entries()] == ["emptied.nii.gz", "far.nii.gz", "near.nii.gz"]

    @pytest.mark.usefixtures("display_env")
    def test_zero_scores_excluded(self, monkeypatch, run_recorder):
        names = ["a.nii.gz", "b.nii.gz", "c.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        run = run_recorder()

        visualise_mask_diffs({"a.nii.gz": 0.0, "b.nii.gz": 2.0, "c.nii.gz": 0}, masks_a, masks_b)

        assert [e["name"] for e in run.entries()] == ["b.nii.gz"]

    @pytest.mark.usefixtures("display_env")
    def test_all_zero_scores_skip_viewer(self, monkeypatch, run_recorder, capsys):
        run = run_recorder()
        assert visualise_mask_diffs({"m.nii.gz": 0.0}, {"m.nii.gz": "/a/m"}, {"m.nii.gz": "/b/m"}) == 0
        assert "No changed masks to visualise." in capsys.readouterr().out
        assert run.calls == []

    def test_run_passes_changed_counts_for_diff(self, mask_tree, tmp_path, monkeypatch):
        dir_a, dir_b, _ = mask_tree(
            {
                "L/two.nii.gz": (_point((0, 0, 0)), _point((2, 0, 0))),
                "L/same.nii.gz": (_point((0, 0, 0)), _point((0, 0, 0))),
            }
        )
        seen = _capture_visualise(monkeypatch)
        script = _make_script(
            [
                "masks",
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(tmp_path / "r.json"),
                "--visualisation",
            ]
        )
        assert script.run() == 0
        assert seen["scores"]["L/two.nii.gz"] == 2
        assert seen["scores"].get("L/same.nii.gz", 0) == 0

    def test_run_passes_changed_counts_for_both(self, mask_tree, tmp_path, monkeypatch):
        dir_a, dir_b, _ = mask_tree({"L/two.nii.gz": (_point((0, 0, 0)), _point((3, 0, 0)))})
        seen = _capture_visualise(monkeypatch)
        script = _make_script(
            [
                "masks",
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(tmp_path / "r.json"),
                "--metric",
                "both",
                "--visualisation",
            ]
        )
        assert script.run() == 0
        # Changed-voxel count (2), not the Wasserstein distance (3.0).
        assert seen["scores"] == {"L/two.nii.gz": 2}

    def test_run_passes_distances_for_wasserstein(self, mask_tree, tmp_path, monkeypatch):
        dir_a, dir_b, _ = mask_tree(
            {
                "L/shifted.nii.gz": (_point((0, 0, 0)), _point((3, 0, 0))),
                "L/emptied.nii.gz": (_point((0, 0, 0)), np.zeros((4, 4, 4))),
                "L/near.nii.gz": (_point((0, 0, 0)), _point((1, 0, 0))),
            }
        )
        seen = _capture_visualise(monkeypatch)
        script = _make_script(
            [
                "masks",
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(tmp_path / "r.json"),
                "--metric",
                "wasserstein",
                "--visualisation",
            ]
        )
        assert script.run() == 0
        scores = seen["scores"]
        assert scores["L/shifted.nii.gz"] == pytest.approx(3.0)
        assert scores["L/near.nii.gz"] == pytest.approx(1.0)
        assert math.isinf(scores["L/emptied.nii.gz"])


# ---------------------------------------------------------------------------
# TASK-192 / REQ-COMPARE-46: XOR files come from save_xor_vol or --xor_dir
# ---------------------------------------------------------------------------


class TestTask192XorFiles:
    def _spy_save_xor_vol(self, monkeypatch):
        calls = []
        original = compare.save_xor_vol

        def _spy(*args, **kwargs):
            calls.append(args[3] if len(args) > 3 else kwargs.get("out_path"))
            return original(*args, **kwargs)

        monkeypatch.setattr(compare, "save_xor_vol", _spy)
        return calls

    @pytest.mark.usefixtures("display_env")
    def test_temp_xor_files_come_from_save_xor_vol(self, monkeypatch, run_recorder, tmp_root):
        names = ["x.nii.gz", "y.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        calls = self._spy_save_xor_vol(monkeypatch)
        run = run_recorder()

        visualise_mask_diffs({"x.nii.gz": 2.0, "y.nii.gz": 1.0}, masks_a, masks_b)

        assert sorted(str(c) for c in calls) == sorted(e["path_xor"] for e in run.entries())

    @pytest.mark.usefixtures("display_env")
    def test_temp_xor_files_share_one_directory_removed_after_return(self, monkeypatch, run_recorder, tmp_root):
        names = ["x.nii.gz", "y.nii.gz", "z.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        existed = {}

        def _check(cmd):
            for entry in json.loads(cmd[-1]):
                existed[entry["path_xor"]] = Path(entry["path_xor"]).is_file()

        run = run_recorder(on_call=_check)
        visualise_mask_diffs({n: 1.0 for n in names}, masks_a, masks_b)

        parents = {Path(e["path_xor"]).parent for e in run.entries()}
        assert len(parents) == 1
        (parent,) = parents
        # One dedicated temporary directory, not loose files in the temp root.
        assert parent != tmp_root
        assert tmp_root in parent.parents
        assert all(existed.values()) and len(existed) == 3
        assert list(tmp_root.iterdir()) == []

    @pytest.mark.usefixtures("display_env")
    def test_xor_dir_files_reused_without_temp_files(self, monkeypatch, run_recorder, tmp_root, tmp_path):
        names = ["L/x.nii.gz", "L/y.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        fake = _FakeAims(objects)
        monkeypatch.setattr(compare, "aims", fake)
        calls = self._spy_save_xor_vol(monkeypatch)
        xor_dir = _make_tree(tmp_path / "xors", names)
        temp_seen = []
        run = run_recorder(on_call=lambda _cmd: temp_seen.extend(tmp_root.iterdir()))

        visualise_mask_diffs({"L/x.nii.gz": 2.0, "L/y.nii.gz": 1.0}, masks_a, masks_b, xor_dir=str(xor_dir))

        assert {e["name"]: e["path_xor"] for e in run.entries()} == {
            "L/x.nii.gz": str(xor_dir / "L" / "x.nii.gz"),
            "L/y.nii.gz": str(xor_dir / "L" / "y.nii.gz"),
        }
        assert calls == []
        assert fake.written == []
        assert temp_seen == []
        assert list(tmp_root.iterdir()) == []

    def test_run_passes_xor_dir_to_viewer(self, mask_tree, tmp_path, monkeypatch):
        dir_a, dir_b, _ = mask_tree({"L/two.nii.gz": (_point((0, 0, 0)), _point((2, 0, 0)))})
        seen = _capture_visualise(monkeypatch)
        xor_dir = tmp_path / "xors"
        script = _make_script(
            [
                "masks",
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(tmp_path / "r.json"),
                "--xor_dir",
                str(xor_dir),
                "--visualisation",
            ]
        )
        assert script.run() == 0
        passed = seen["kwargs"].get("xor_dir", seen["args"][0] if seen["args"] else None)
        assert passed is not None
        assert Path(passed) == xor_dir

    @pytest.mark.usefixtures("display_env")
    def test_temp_dir_removed_when_aims_write_raises(self, monkeypatch, run_recorder, tmp_root):
        names = ["x.nii.gz", "y.nii.gz", "z.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        monkeypatch.setattr(compare, "aims", _FakeAims(objects, fail_write_after=1))
        run = run_recorder()

        with pytest.raises(RuntimeError, match="fake aims.write failure"):
            visualise_mask_diffs({"x.nii.gz": 3.0, "y.nii.gz": 2.0, "z.nii.gz": 1.0}, masks_a, masks_b)

        assert run.calls == []
        assert list(tmp_root.iterdir()) == []

    @pytest.mark.usefixtures("display_env")
    def test_temp_dir_removed_when_aims_read_raises(self, monkeypatch, run_recorder, tmp_root):
        names = ["x.nii.gz", "y.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        # The second-ranked mask's set_b file cannot be read.
        monkeypatch.setattr(compare, "aims", _FakeAims(objects, fail_read={masks_b["y.nii.gz"]}))
        run = run_recorder()

        with pytest.raises(RuntimeError, match="fake aims.read failure"):
            visualise_mask_diffs({"x.nii.gz": 2.0, "y.nii.gz": 1.0}, masks_a, masks_b)

        assert run.calls == []
        assert list(tmp_root.iterdir()) == []

    @pytest.mark.usefixtures("display_env")
    def test_temp_dir_removed_when_subprocess_raises(self, monkeypatch, run_recorder, tmp_root):
        names = ["x.nii.gz"]
        objects, masks_a, masks_b = _mask_pairs(names)
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        run_recorder(raises=OSError("cannot spawn"))

        try:
            visualise_mask_diffs({"x.nii.gz": 1.0}, masks_a, masks_b)
        except OSError:
            pass

        assert list(tmp_root.iterdir()) == []


# ---------------------------------------------------------------------------
# TASK-187 / REQ-COMPARE-47: non-zero viewer exit is reported
# ---------------------------------------------------------------------------


class TestTask187ViewerFailure:
    @pytest.mark.usefixtures("display_env")
    def test_nonzero_return_code_reported_on_stderr(self, monkeypatch, run_recorder, tmp_root, capsys):
        objects, masks_a, masks_b = _mask_pairs(["m.nii.gz"])
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        run_recorder(returncode=139)

        assert visualise_mask_diffs({"m.nii.gz": 1.0}, masks_a, masks_b) == 1
        assert "139" in capsys.readouterr().err

    @pytest.mark.usefixtures("display_env")
    def test_returns_zero_when_viewer_succeeds(self, monkeypatch, run_recorder, tmp_root, capsys):
        objects, masks_a, masks_b = _mask_pairs(["m.nii.gz"])
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        run_recorder(returncode=0)

        assert visualise_mask_diffs({"m.nii.gz": 1.0}, masks_a, masks_b) == 0

    @pytest.mark.parametrize(
        ("mode", "rel"),
        [
            ("masks", "L/m.nii.gz"),
            ("cortical_tiles", "REGION/mask/Lmask_skeleton.nii.gz"),
        ],
        ids=["masks", "cortical_tiles"],
    )
    @pytest.mark.usefixtures("display_env")
    def test_run_exits_one_and_writes_report_on_viewer_failure(
        self, mode, rel, mask_tree, tmp_path, run_recorder, tmp_root, capsys
    ):
        dir_a, dir_b, _ = mask_tree({rel: (_point((0, 0, 0)), _point((1, 0, 0)))})
        run = run_recorder(returncode=139)
        out = tmp_path / "report.json"
        script = _make_script(
            [mode, "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--visualisation"]
        )

        assert script.run() == 1
        assert len(run.calls) == 1
        assert "139" in capsys.readouterr().err
        assert json.loads(out.read_text())["mode"] == mode


# ---------------------------------------------------------------------------
# TASK-188 / REQ-COMPARE-49: skip the viewer without a usable display
# ---------------------------------------------------------------------------


class TestTask188DisplayCheck:
    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            ({"DISPLAY": ":0"}, True),
            ({"WAYLAND_DISPLAY": "wayland-0"}, True),
            ({"DISPLAY": ":0", "QT_QPA_PLATFORM": "xcb"}, True),
            ({}, False),
            ({"QT_QPA_PLATFORM": "xcb"}, False),
            ({"DISPLAY": ":0", "QT_QPA_PLATFORM": "offscreen"}, False),
            ({"WAYLAND_DISPLAY": "wayland-0", "QT_QPA_PLATFORM": "offscreen"}, False),
        ],
        ids=["x11", "wayland", "x11_xcb", "none", "xcb_only", "x11_offscreen", "wayland_offscreen"],
    )
    def test_has_usable_display(self, monkeypatch, env, expected):
        for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        assert compare._has_usable_display() is expected

    @pytest.mark.parametrize(
        "env",
        [{}, {"DISPLAY": ":0", "QT_QPA_PLATFORM": "offscreen"}],
        ids=["no_display", "offscreen"],
    )
    def test_skips_viewer_without_usable_display(self, monkeypatch, run_recorder, tmp_root, capsys, env):
        for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        objects, masks_a, masks_b = _mask_pairs(["m.nii.gz"])
        monkeypatch.setattr(compare, "aims", _FakeAims(objects))
        run = run_recorder()

        assert visualise_mask_diffs({"m.nii.gz": 1.0}, masks_a, masks_b) == 0
        assert run.calls == []
        assert "display" in capsys.readouterr().err.lower()

    def test_run_exits_zero_when_display_missing(self, mask_tree, tmp_path, monkeypatch, run_recorder, capsys):
        for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
            monkeypatch.delenv(var, raising=False)
        dir_a, dir_b, _ = mask_tree({"L/m.nii.gz": (_point((0, 0, 0)), _point((1, 0, 0)))})
        run = run_recorder()
        out = tmp_path / "r.json"
        script = _make_script(
            ["masks", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--visualisation"]
        )

        assert script.run() == 0
        assert run.calls == []
        assert "display" in capsys.readouterr().err.lower()
        assert out.is_file()


# ---------------------------------------------------------------------------
# Fake Anatomist / Qt for executing _VIEWER_SCRIPT
# ---------------------------------------------------------------------------


class _AnaObject:
    def __init__(self, label):
        self.label = label

    def __getattr__(self, name):
        return lambda *a, **k: None


class _AnaWindow:
    def __init__(self, wtype):
        self.wtype = wtype
        self.objects = []

    def addObjects(self, objs, *args, **kwargs):  # noqa: N802 - mirrors the Anatomist API
        self.objects.extend(objs if isinstance(objs, (list, tuple)) else [objs])

    def __getattr__(self, name):
        return lambda *a, **k: None


class _FakeAnatomist:
    def __init__(self):
        self.loaded = []
        self.fusions = []
        self.windows = []
        self.links = []

    def loadObject(self, path, *args, **kwargs):  # noqa: N802 - mirrors the Anatomist API
        obj = _AnaObject(path)
        self.loaded.append(obj)
        return obj

    def fusionObjects(self, objects, *args, **kwargs):  # noqa: N802, V105 - Anatomist API, called by the script
        method = kwargs.get("method", args[0] if args else None)
        fusion = _AnaObject("fusion")
        self.fusions.append((list(objects), method, fusion))
        return fusion

    def createWindow(self, wtype, *args, **kwargs):  # noqa: N802 - mirrors the Anatomist API
        win = _AnaWindow(wtype)
        self.windows.append(win)
        return win

    def linkWindows(self, windows, *args, **kwargs):  # noqa: N802, V105 - Anatomist API, called by the script
        self.links.append(list(windows))

    def __getattr__(self, name):
        return lambda *a, **k: _AnaObject(name)


class _FakeApp:
    def __init__(self):
        self.exec_calls = 0

    def exec_(self):  # noqa: V105 - Qt API, called by the script
        self.exec_calls += 1
        return 0

    def exec(self):
        self.exec_calls += 1
        return 0


def _run_viewer_script(monkeypatch, entries, app):
    """Execute compare._VIEWER_SCRIPT with fake Anatomist/Qt; return (anatomist, exit code)."""
    anatomist = _FakeAnatomist()
    api = types.ModuleType("anatomist.direct.api")
    api.Anatomist = lambda *a, **k: anatomist
    direct = types.ModuleType("anatomist.direct")
    direct.api = api
    pkg = types.ModuleType("anatomist")
    pkg.direct = direct
    monkeypatch.setitem(sys.modules, "anatomist", pkg)
    monkeypatch.setitem(sys.modules, "anatomist.direct", direct)
    monkeypatch.setitem(sys.modules, "anatomist.direct.api", api)

    class _QApplication:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def instance():
            return app

    qt_backend = types.ModuleType("soma.qt_gui.qt_backend")
    qt_backend.Qt = types.SimpleNamespace(QApplication=_QApplication)  # noqa: V101
    monkeypatch.setitem(sys.modules, "soma.qt_gui.qt_backend", qt_backend)
    monkeypatch.setattr(sys, "argv", ["-c", json.dumps(entries)])

    code = 0
    try:
        exec(compile(compare._VIEWER_SCRIPT, "<viewer>", "exec"), {"__name__": "__main__"})  # noqa: S102
    except SystemExit as exc:
        code = exc.code
    return anatomist, code


def _entries(n):
    return [
        {"name": f"m{i}", "path_a": f"/a/m{i}.nii.gz", "path_b": f"/b/m{i}.nii.gz", "path_xor": f"/x/m{i}.nii.gz"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TASK-193 / REQ-COMPARE-50: no QApplication -> non-zero exit with a message
# ---------------------------------------------------------------------------


class TestTask193NoQApplication:
    def test_exits_nonzero_with_stderr_message_without_qapplication(self, monkeypatch, capsys):
        _, code = _run_viewer_script(monkeypatch, _entries(1), app=None)

        assert code not in (0, None)
        # sys.exit("message") prints the message on stderr and exits 1 in a real process.
        message = code if isinstance(code, str) else capsys.readouterr().err
        assert message.strip() != ""

    def test_runs_event_loop_when_qapplication_exists(self, monkeypatch):
        app = _FakeApp()
        _, code = _run_viewer_script(monkeypatch, _entries(1), app=app)

        assert code in (0, None)
        assert app.exec_calls == 1


# ---------------------------------------------------------------------------
# TASK-190 / REQ-COMPARE-51: fused, linked Axial/Sagittal/Coronal windows
# ---------------------------------------------------------------------------


class TestTask190FusedLinkedViews:
    def test_fuses_set_a_set_b_and_xor_with_fusion2d(self, monkeypatch):
        entries = _entries(2)
        ana, _ = _run_viewer_script(monkeypatch, entries, app=_FakeApp())

        by_path = {obj.label: obj for obj in ana.loaded}
        expected = [[by_path[e["path_a"]], by_path[e["path_b"]], by_path[e["path_xor"]]] for e in entries]
        assert [objs for objs, _m, _f in ana.fusions] == expected
        assert [m for _o, m, _f in ana.fusions] == ["Fusion2DMethod", "Fusion2DMethod"]

    def test_shows_fusion_in_axial_sagittal_coronal_windows(self, monkeypatch):
        ana, _ = _run_viewer_script(monkeypatch, _entries(2), app=_FakeApp())

        assert len(ana.fusions) == 2
        for _objs, _m, fusion in ana.fusions:
            views = sorted(w.wtype for w in ana.windows if fusion in w.objects)
            assert views == ["Axial", "Coronal", "Sagittal"]

    def test_links_the_three_windows_per_mask(self, monkeypatch):
        ana, _ = _run_viewer_script(monkeypatch, _entries(2), app=_FakeApp())

        assert len(ana.fusions) == 2
        link_sets = [{id(w) for w in group} for group in ana.links]
        for _objs, _m, fusion in ana.fusions:
            mask_windows = {id(w) for w in ana.windows if fusion in w.objects}
            assert len(mask_windows) == 3
            assert mask_windows in link_sets


# ---------------------------------------------------------------------------
# TASK-189 / REQ-COMPARE-52: help text and docstring match the code
# ---------------------------------------------------------------------------


def _visualisation_help(mode: str, capsys) -> str:
    with pytest.raises(SystemExit):
        Compare().parser.parse_args([mode, "--help"])
    out = capsys.readouterr().out
    # The option's own help entry (argparse indents option lines by two spaces;
    # wrapped help lines are indented further), not its mention in the usage line.
    start = re.search(r"\n {2}--visualisation\b", out).start()
    nxt = re.search(r"\n {2}-", out[start + 1 :])
    section = out[start : start + 1 + nxt.start()] if nxt else out[start:]
    return " ".join(section.split()).lower()


def _assert_describes_viewer(text: str) -> None:
    text = " ".join(text.split()).lower()
    assert re.search(r"\b5 most[- ]changed\b", text), "top-5 most-changed limit"
    assert "changed voxel" in text, "ranking by changed voxels"
    assert "wasserstein" in text, "ranking by Wasserstein distance"
    assert "grey" in text and "violet" in text and re.search(r"\bred\b", text), "palettes"
    assert "white" not in text
    assert "all changed" not in text
    for view in ("axial", "sagittal", "coronal"):
        assert view in text, view
    assert "linked" in text, "linked windows"
    assert "fus" in text, "fusion"
    assert "anatomist" in text and "display" in text, "prerequisites"


class TestTask189HelpText:
    @pytest.mark.parametrize("mode", ["masks", "cortical_tiles"])
    def test_help_text_describes_viewer(self, mode, capsys):
        _assert_describes_viewer(_visualisation_help(mode, capsys))

    def test_docstring_describes_viewer(self):
        _assert_describes_viewer(compare.visualise_mask_diffs.__doc__ or "")


# ---------------------------------------------------------------------------
# TASK-185 (partial) / REQ-COMPARE-53: no import inside visualise_mask_diffs
# ---------------------------------------------------------------------------


class TestTask185VisualiseImports:
    def test_visualise_mask_diffs_has_no_import_statement(self):
        tree = ast.parse(SRC_COMPARE.read_text())
        func = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "visualise_mask_diffs"
        )
        imports = [
            (node.lineno, ast.unparse(node))
            for node in ast.walk(func)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert imports == []

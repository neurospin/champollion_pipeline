"""Listing-order rule of the pipeline (TASK-READORDER-BDRABCZUK-EFCBDE20C624).

User rule: files must always be read in the same order, for reproducibility. A lint-style test:
it parses main.py and every Python file under src/ and scripts/ and fails on a listing call (glob,
iglob, rglob, listdir, scandir, walk, iterdir) that is not inside a sorted() call and has no entry
in tests/listing_order_exemptions.py. Each exemption entry carries the reason its order never
reaches a result, and must still name a call the rule reports.

A call counts as sorted when it lies in a positional argument of a sorted() call, at any depth of
that argument expression (so sorted(p for p in d.iterdir() if p.is_dir()) is sorted). A walk call
heading a for loop also counts as sorted when the loop body sorts the directory list in place
(dirs.sort() or dirs[:] = sorted(...)) and reads the file list only inside sorted() or after
files.sort(), so the walk itself visits folders and files in a fixed order.

Mirrors the cortical_tiles rule (REQ-READORDER-BDRABCZUK-5A6652ECC6B8 and -04E14C889178).

* REQ-READORDER-BDRABCZUK-AFD194435759 - the test fails on an unsorted listing call not exempted.
* REQ-READORDER-BDRABCZUK-A2DE66459BD8 - the test fails on an exemption entry without a reason.
* REQ-READORDER-BDRABCZUK-F7541C9A3B01 - the test fails on a stale exemption entry.
* REQ-READORDER-BDRABCZUK-0934E4E06AF8 - a walk loop sorting its folders and files counts as sorted.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.listing_order_exemptions import EXEMPTIONS

pytestmark = pytest.mark.unit  # noqa: V107

LISTING_NAMES = frozenset({"glob", "iglob", "rglob", "listdir", "scandir", "walk", "iterdir"})
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("src", "scripts")
SCANNED_FILES = ("main.py",)
MODULE_SCOPE = "<module>"
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class ListingCall:
    """One listing call outside sorted(), with the key the exemption list uses"""

    path: str
    scope: str
    call: str
    line: int

    @property
    def key(self):
        return (self.path, self.scope, self.call)


def _callee_name(call):
    """Returns the called name of call: f for f(...), attr for x.attr(...), else None"""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_sorted_call(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sorted"


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _in_place_sorts(nodes, name):
    """Returns the nodes of nodes sorting the list called name in place: name.sort() or name[:] = sorted(...)"""
    return [node for node in nodes if _is_sort_method_call(node, name) or _is_sorted_slice_assign(node, name)]


def _is_sort_method_call(node, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sort"
        and _is_name(node.func.value, name)
    )


def _is_sorted_slice_assign(node, name):
    return (
        isinstance(node, ast.Assign)
        and _is_sorted_call(node.value)
        and any(isinstance(t, ast.Subscript) and _is_name(t.value, name) for t in node.targets)
    )


def _loads_outside_sorted(statements, name):
    """Returns the reads of name in statements that lie outside every sorted() positional argument"""
    loads = []

    def visit(node, in_sorted):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load) and not in_sorted:
            loads.append(node)
        if _is_sorted_call(node):
            for arg in node.args:
                visit(arg, True)
            for keyword in node.keywords:
                visit(keyword, in_sorted)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, in_sorted)

    for statement in statements:
        visit(statement, False)
    return loads


def _walk_loop_is_sorted(loop):
    """True when the for loop over a walk call sorts its folder list in place and reads its files sorted"""
    target = loop.target
    if not (isinstance(target, ast.Tuple) and len(target.elts) == 3):
        return False
    dirs, files = target.elts[1], target.elts[2]
    if not (isinstance(dirs, ast.Name) and isinstance(files, ast.Name)):
        return False
    body_nodes = [node for statement in loop.body for node in ast.walk(statement)]
    if not _in_place_sorts(body_nodes, dirs.id):
        return False
    file_sorts = _in_place_sorts(body_nodes, files.id)
    sort_receivers = {id(sort.func.value) for sort in file_sorts if isinstance(sort, ast.Call)}
    first_sort_end = min(((sort.end_lineno, sort.end_col_offset) for sort in file_sorts), default=None)
    for load in _loads_outside_sorted(loop.body, files.id):
        if id(load) in sort_receivers:
            continue
        if first_sort_end is None or (load.lineno, load.col_offset) < first_sort_end:
            return False
    return True


def _is_listing_call(node):
    return isinstance(node, ast.Call) and _callee_name(node) in LISTING_NAMES


def find_unsorted_listing_calls(source, rel_path):
    """Returns the listing calls of source that lie outside every sorted() argument"""
    found = []
    sorted_walks = set()

    def visit(node, scope, in_sorted):
        if (
            isinstance(node, (ast.For, ast.AsyncFor))
            and _is_listing_call(node.iter)
            and _callee_name(node.iter) == "walk"
            and _walk_loop_is_sorted(node)
        ):
            sorted_walks.add(id(node.iter))
        if _is_listing_call(node) and not in_sorted and id(node) not in sorted_walks:
            found.append(ListingCall(rel_path, ".".join(scope) or MODULE_SCOPE, ast.unparse(node), node.lineno))
        if isinstance(node, SCOPE_NODES):
            scope = (*scope, node.name)
        if _is_sorted_call(node):
            visit(node.func, scope, in_sorted)
            for arg in node.args:
                visit(arg, scope, True)
            for keyword in node.keywords:
                visit(keyword, scope, in_sorted)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, scope, in_sorted)

    visit(ast.parse(source), (), False)
    return found


def unexempted(calls, exemptions):
    """Returns the calls whose key has no entry in exemptions"""
    return [call for call in calls if call.key not in exemptions]


def empty_reason_keys(exemptions):
    """Returns the exemption keys whose reason is empty after whitespace stripping"""
    return [key for key, reason in exemptions.items() if not str(reason).strip()]


def stale_keys(calls, exemptions):
    """Returns the exemption keys that name no reported call"""
    reported = {call.key for call in calls}
    return [key for key in exemptions if key not in reported]


def scanned_files():
    """Returns main.py and the Python files under src/ and scripts/, relative to the pipeline root"""
    files = [PIPELINE_ROOT / name for name in SCANNED_FILES]
    for folder in SCANNED_DIRS:
        files.extend(sorted((PIPELINE_ROOT / folder).rglob("*.py")))
    return [path.relative_to(PIPELINE_ROOT).as_posix() for path in files if path.is_file()]


def scan_pipeline():
    """Returns the unsorted listing calls of the scanned files"""
    calls = []
    for rel_path in scanned_files():
        calls.extend(find_unsorted_listing_calls((PIPELINE_ROOT / rel_path).read_text(encoding="utf-8"), rel_path))
    return calls


# --- REQ-READORDER-BDRABCZUK-AFD194435759: unsorted listing call not exempted -------------------

UNSORTED_CASES = [
    ("import glob\nfiles = glob.glob('d/*')\n", MODULE_SCOPE, "glob.glob('d/*')"),
    ("from glob import glob\nfiles = glob('d/*')\n", MODULE_SCOPE, "glob('d/*')"),
    ("import glob\nfiles = list(glob.iglob('d/*'))\n", MODULE_SCOPE, "glob.iglob('d/*')"),
    (
        "from pathlib import Path\ndef f():\n    return list(Path('d').rglob('*.nii.gz'))\n",
        "f",
        "Path('d').rglob('*.nii.gz')",
    ),
    ("import os\nclass C:\n    def m(self):\n        return os.listdir('d')\n", "C.m", "os.listdir('d')"),
    ("import os\nentries = [e.name for e in os.scandir('d')]\n", MODULE_SCOPE, "os.scandir('d')"),
    ("import os\nwalked = list(os.walk('d'))\n", MODULE_SCOPE, "os.walk('d')"),
    ("from pathlib import Path\nnames = [p.name for p in Path('d').iterdir()]\n", MODULE_SCOPE, "Path('d').iterdir()"),
    ("import os\nfiles = min(os.listdir('d'))\n", MODULE_SCOPE, "os.listdir('d')"),
    ("import os\nfiles = sorted(['a'], key=lambda p: len(os.listdir(p)))\n", MODULE_SCOPE, "os.listdir(p)"),
    ("import os\nfiles = os.listdir('d')\nfiles.sort()\n", MODULE_SCOPE, "os.listdir('d')"),
]


@pytest.mark.parametrize(("source", "scope", "call"), UNSORTED_CASES)
def test_unsorted_listing_call_is_reported(source, scope, call):
    calls = find_unsorted_listing_calls(source, "src/m.py")
    assert [c.key for c in unexempted(calls, {})] == [("src/m.py", scope, call)]


SORTED_CASES = [
    "import glob\nfiles = sorted(glob.glob('d/*'))\n",
    "import os\nfiles = sorted(os.listdir('d'))\n",
    "from pathlib import Path\ndirs = sorted(p for p in Path('d').iterdir() if p.is_dir())\n",
    "import os\ndef f():\n    return [x for x in sorted(os.listdir('d'))]\n",
    "import glob\nfiles = sorted(glob.glob('a/*') + glob.glob('b/*'))\n",
    "import glob\nname = glob.escape('d')\n",
]


@pytest.mark.parametrize("source", SORTED_CASES)
def test_sorted_listing_call_is_not_reported(source):
    assert find_unsorted_listing_calls(source, "src/m.py") == []


def test_exempt_listing_call_is_not_reported():
    source = "import os\ndef f(d):\n    if os.listdir(d) == []:\n        return os.listdir(d + '/x')\n"
    calls = find_unsorted_listing_calls(source, "src/m.py")
    exemptions = {("src/m.py", "f", "os.listdir(d)"): "emptiness test, order unused"}
    assert [c.key for c in unexempted(calls, exemptions)] == [("src/m.py", "f", "os.listdir(d + '/x')")]


def test_scanned_files_cover_src_scripts_and_main():
    files = scanned_files()
    assert "main.py" in files
    assert any(path.startswith("src/champollion_pipeline/") for path in files)
    assert any(path.startswith("scripts/") for path in files)
    assert not any(path.startswith("tests/") for path in files)


def test_pipeline_has_no_unexempted_unsorted_listing_call():
    missing = unexempted(scan_pipeline(), EXEMPTIONS)
    report = "\n".join(f"    {c.key!r}: '',  # line {c.line}" for c in missing)
    assert not missing, (
        f"{len(missing)} listing calls of main.py, src/ and scripts/ are neither inside sorted() nor in "
        f"tests/listing_order_exemptions.py:\n{report}"
    )


# --- REQ-READORDER-BDRABCZUK-A2DE66459BD8: exemption entry without a reason ---------------------

KEY = ("src/m.py", "f", "os.listdir(d)")


@pytest.mark.parametrize(("reason", "reported"), [("", True), ("   \n\t", True), ("emptiness test", False)])
def test_empty_exemption_reason_is_reported(reason, reported):
    assert empty_reason_keys({KEY: reason}) == ([KEY] if reported else [])


def test_exemption_reasons_are_not_empty():
    assert empty_reason_keys(EXEMPTIONS) == []


# --- REQ-READORDER-BDRABCZUK-F7541C9A3B01: stale exemption entry --------------------------------


def test_stale_exemption_is_reported():
    source = "import os\ndef f(d):\n    return sorted(os.listdir(d)), os.listdir(d + '/x')\n"
    calls = find_unsorted_listing_calls(source, "src/m.py")
    live = ("src/m.py", "f", "os.listdir(d + '/x')")
    exemptions = {
        KEY: "the call is sorted now",
        ("src/m.py", "g", "os.listdir(d + '/x')"): "no such function",
        ("src/gone.py", "f", "os.listdir(d)"): "no such file",
        live: "emptiness test",
    }
    assert stale_keys(calls, exemptions) == [k for k in exemptions if k != live]


def test_exemptions_match_a_reported_listing_call():
    stale = stale_keys(scan_pipeline(), EXEMPTIONS)
    assert stale == [], f"exemption entries naming no reported listing call: {stale}"


# --- REQ-READORDER-BDRABCZUK-0934E4E06AF8: walk loop sorting its folders and files --------------

WALK_SORTED_CASES = [
    "import os\nfor root, dirs, files in os.walk('d'):\n    dirs.sort()\n"
    "    for f in sorted(files):\n        print(f)\n",
    "import os\nfor root, dirs, files in os.walk('d'):\n    dirs[:] = sorted(dirs)\n    names = sorted(files)\n",
    "import os\nfor root, dirs, files in os.walk('d'):\n    dirs.sort()\n    files.sort()\n    print(files)\n",
    "import os\nfor root, dirs, _ in os.walk('d'):\n    dirs.sort()\n    print(root)\n",
]


@pytest.mark.parametrize("source", WALK_SORTED_CASES)
def test_walk_loop_sorting_dirs_and_files_is_not_reported(source):
    assert find_unsorted_listing_calls(source, "src/m.py") == []


WALK_UNSORTED_CASES = [
    "import os\nfor root, dirs, files in os.walk('d'):\n    for f in sorted(files):\n        print(f)\n",
    "import os\nfor root, dirs, files in os.walk('d'):\n    dirs.sort()\n    for f in files:\n        print(f)\n",
    "import os\nfor root, dirs, files in os.walk('d'):\n    dirs.sort()\n    print(files)\n    files.sort()\n",
    "import os\nfor entry in os.walk('d'):\n    print(entry)\n",
    "import os\nnames = [f for root, dirs, files in os.walk('d') for f in sorted(files)]\n",
]


@pytest.mark.parametrize("source", WALK_UNSORTED_CASES)
def test_walk_loop_not_sorting_is_reported(source):
    calls = find_unsorted_listing_calls(source, "src/m.py")
    assert [c.key for c in calls] == [("src/m.py", MODULE_SCOPE, "os.walk('d')")]

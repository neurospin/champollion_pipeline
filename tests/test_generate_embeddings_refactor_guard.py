"""
Guard for the move of cohesive blocks out of src/champollion_pipeline/generate_embeddings.py into modules under
models/ (TASK-CODE-BDRABCZUK-43093B756C69, refactor, behaviour-preserving).

The reference is src/champollion_pipeline/generate_embeddings.py at commit 1d3f723 (599 lines). The differential
tests load that version with git show as a separate module and compare the current module against it; they pass on
the reference itself. Only the line-count test fails at the reference.

Requirements (area EMBSPLIT, recorded by requirement:red for this Task):
- generate_embeddings.py at most 500 lines (budget generate_embeddings_source_lines);
- each module-level name defined at 1d3f723 importable from champollion_pipeline.generate_embeddings;
- each method of GenerateEmbeddings at 1d3f723 still a callable attribute of the class;
- parser help text equal to the reference;
- parse_args outcome (parsed values, exit status, error message) equal to the reference;
- a failed evaluate.py run leaves no full_embeddings.csv (pin, uncovered before this Task);
- --subjects emits its FutureWarning (pin, uncovered before this Task).
"""

import ast
import contextlib
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from champollion_utils.script_builder import ScriptBuilder
from hypothesis import given, settings
from hypothesis import strategies as st

import champollion_pipeline.generate_embeddings as current
from champollion_pipeline.utils.region_masks import REGION_MODEL_NAMES
from tests.embeddings_parse_helpers import MASKS_ARGS
from tests.region_model_dirs import make_region_logs

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = "src/champollion_pipeline/generate_embeddings.py"
REFERENCE_COMMIT = "1d3f7232de90e4b652712778a7eb26cb41a404e9"
MAX_SOURCE_LINES = 500
MEASURE = "generate_embeddings_source_lines"

# Names defined (not imported) at module level in generate_embeddings.py at REFERENCE_COMMIT, in file order.
REFERENCE_NAMES = (
    "PROVENANCE_FILE_NAME",
    "INPUT_QC_FILENAME",
    "FAILED_STATUS",
    "_PROFILING_DEPRECATION",
    "logger",
    "_CheckpointEpochAction",
    "GenerateEmbeddings",
    "main",
)
N_REFERENCE_METHODS = 24

# Command-line fragments drawn after the two positional arguments; each is a whole option with its values.
ARGV_FRAGMENTS = (
    ("--overwrite",),
    ("--cpu",),
    ("--profiling",),
    ("--run-cka",),
    ("--no-cache",),
    ("--per-model-folders",),
    ("--legacy",),
    ("--allow-version-mismatch",),
    ("--checkpoint-epoch", "80"),
    ("--checkpoint-epoch", "3"),
    ("--checkpoint-epoch", "a/b=5"),
    ("--checkpoint-epoch", "x"),
    ("--checkpoint-epoch", "-1"),
    ("--checkpoint-epoch", "=2"),
    ("--cortical_version", "cortical_tiles-2027"),
    ("--model-version", "canonical_25"),
    ("--model-version", "canonical_corrected_26_1"),
    ("--model-version", "unknown_version"),
    ("--masks-version", "canonical_25"),
    ("--masks", "canonical_25"),
    ("--masks", "canonical_corrected_26_1"),
    ("--masks", "other_masks"),
    ("--region-masks", f"{REGION_MODEL_NAMES[0]}=canonical_25"),
    ("--region-masks", f"{REGION_MODEL_NAMES[1]}=canonical_corrected_26_1", f"{REGION_MODEL_NAMES[2]}=canonical_25"),
    ("--region-masks", "no_such_region=canonical_25"),
    ("--regions", REGION_MODEL_NAMES[0], REGION_MODEL_NAMES[1]),
    ("--output", "/out"),
    ("--subjects", "/subjects.csv"),
    ("--models-cache", "relative/cache"),
)
# models_path values that are not local directories, so the parser reads no models tree.
MODELS_PATHS = ("org/repo", "/no/such/dir/canonical_25", "/no/such/dir/models")


def _read_reference_source() -> str:
    shown = subprocess.run(
        ["git", "show", f"{REFERENCE_COMMIT}:{SOURCE_PATH}"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert shown.returncode == 0, f"reference {REFERENCE_COMMIT} not readable: {shown.stderr}"
    return shown.stdout


@pytest.fixture(scope="module")
def reference_source() -> str:
    """generate_embeddings.py at REFERENCE_COMMIT, as text."""
    return _read_reference_source()


@pytest.fixture(scope="module")
def reference(tmp_path_factory, reference_source):
    """generate_embeddings.py at REFERENCE_COMMIT, loaded as a separate module."""
    path = tmp_path_factory.mktemp("reference") / "generate_embeddings_reference.py"
    path.write_text(reference_source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("generate_embeddings_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_level_names(source: str) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return tuple(names)


def _class_method_names(source: str, class_name: str) -> tuple[str, ...]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return tuple(item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
    raise AssertionError(f"class {class_name} not found in the reference source")


def _parse_outcome(module, argv: list[str]):
    """("ok", vars of the namespace) or ("exit", SystemExit code, stderr text) of module's parser on argv."""
    stderr = io.StringIO()
    script = module.GenerateEmbeddings()
    try:
        with contextlib.redirect_stderr(stderr):
            parsed = script.parse_args(argv)
    except SystemExit as exc:
        return ("exit", exc.code, stderr.getvalue())
    return ("ok", vars(parsed))


def test_generate_embeddings_source_at_most_500_lines():
    """generate_embeddings.py has at most 500 lines; the count goes to $ALM_MEASURES."""
    lines = len((REPO_ROOT / SOURCE_PATH).read_text(encoding="utf-8").splitlines())
    measures = os.environ.get("ALM_MEASURES")
    if measures:
        with open(measures, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({MEASURE: lines}) + "\n")
    assert lines <= MAX_SOURCE_LINES, f"{SOURCE_PATH} has {lines} lines, over {MAX_SOURCE_LINES}"


def test_reference_module_names_importable(reference_source):
    """Each 1d3f723 module-level name still resolves on champollion_pipeline.generate_embeddings."""
    assert _module_level_names(reference_source) == REFERENCE_NAMES
    missing = [name for name in REFERENCE_NAMES if not hasattr(current, name)]
    assert missing == [], f"names no longer importable from champollion_pipeline.generate_embeddings: {missing}"


def test_reference_methods_on_generate_embeddings(reference_source):
    """Each method of the 1d3f723 GenerateEmbeddings is still a callable attribute of the class, not ScriptBuilder's."""
    methods = _class_method_names(reference_source, "GenerateEmbeddings")
    assert len(methods) == N_REFERENCE_METHODS
    cls = current.GenerateEmbeddings
    missing = [name for name in methods if not callable(getattr(cls, name, None))]
    assert missing == [], f"GenerateEmbeddings no longer has: {missing}"
    inherited = [name for name in methods if getattr(cls, name) is getattr(ScriptBuilder, name, None)]
    assert inherited == [], f"GenerateEmbeddings no longer overrides: {inherited}"


def test_help_text_matches_reference(reference):
    """The parser help text is the 1d3f723 one, character for character."""
    expected = reference.GenerateEmbeddings().parser.format_help()
    assert current.GenerateEmbeddings().parser.format_help() == expected


@settings(max_examples=150, deadline=None)
@given(
    models_path=st.sampled_from(MODELS_PATHS),
    fragments=st.lists(st.sampled_from(ARGV_FRAGMENTS), max_size=6),
)
def test_parse_args_matches_reference(reference, models_path, fragments):
    """parse_args gives the 1d3f723 parsed values, or the 1d3f723 exit status and error message."""
    argv = [models_path, "/datasets", *(token for fragment in fragments for token in fragment)]
    assert _parse_outcome(current, argv) == _parse_outcome(reference, argv)


def test_failed_evaluation_removes_partial_embeddings(tmp_path):
    """A non-zero evaluate.py exit deletes the full_embeddings.csv it wrote for that model."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    make_region_logs(models_dir / "SC-sylv_left")
    script = current.GenerateEmbeddings()
    script.args = script.parse_args([str(models_dir), str(tmp_path), *MASKS_ARGS])
    written: list[str] = []

    def failing_evaluate(cmd, **_kwargs):
        saving_path = cmd[cmd.index("-s") + 1]
        Path(saving_path).write_text("ID,dim1\nsub-01,0.5\n", encoding="utf-8")
        written.append(saving_path)
        return 1

    with patch.object(script, "execute_command", side_effect=failing_evaluate):
        script._run_per_region(
            evaluate_script="/eval.py", crops_2mm_dir=str(tmp_path / "crops"), output_base=str(tmp_path / "out")
        )

    assert len(written) == 1
    assert not os.path.exists(written[0])


def test_subjects_option_warns_deprecated(tmp_path):
    """A run given --subjects emits the FutureWarning naming the option deprecated and ignored."""
    (tmp_path / "SC-sylv_left").mkdir()
    script = current.GenerateEmbeddings()
    script.parse_args([str(tmp_path), str(tmp_path), *MASKS_ARGS, "--subjects", str(tmp_path / "subjects.csv")])

    with (
        patch.object(script, "fetch_models", return_value=str(tmp_path)),
        patch.object(script, "_run_per_region", return_value=0),
        pytest.warns(FutureWarning, match=r"^--subjects is deprecated and ignored"),
    ):
        script.run()

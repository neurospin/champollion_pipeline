"""
Split of src/champollion_pipeline/cli.py: the six PipelineStage classes and their _compute_* path helpers move to
src/champollion_pipeline/pipeline_stages.py (TASK-CODE-BDRABCZUK-D457E2861B14, refactor, pure move).

The reference is cli.py at WIP commit 5911c7a (merge-scope run RUN-CLICMD-BDRABCZUK-FBF1AE9F06A1 at that tree).
The differential tests read that file with ``git show``, load it under a private module name and compare each moved
stage class with the reference class of the same name on Hypothesis-generated PipelineConfig values; a raised
exception counts as its type.

Requirements:
REQ-CLISPLIT-BDRABCZUK-1BEA41BADEF3 (cli defines no PipelineStage subclass),
REQ-CLISPLIT-BDRABCZUK-2A9646866292 (cli.py and pipeline_stages.py at most 1000 lines; budget cli_split_file_lines_max),
REQ-CLISPLIT-BDRABCZUK-38095C750F40 (importing pipeline_stages adds no toolchain module),
REQ-CLISPLIT-BDRABCZUK-7ED02864B6B7 (build_args equals the 5911c7a class's when dataset.hf_enabled is true or
  models_path is non-empty; supersedes A6FA35DCCAA2, narrowed by TASK-MODELSRC-BDRABCZUK-EFFEE1006BFE),
REQ-CLISPLIT-BDRABCZUK-AFFE25E4DDE2 (output_dir equals the 5911c7a class's).
"""

import copy
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from champollion_pipeline._stage_base import PipelineStage
from champollion_pipeline.config import DatasetConfig, PipelineConfig
from tests.test_lazy_import_checkpoint_epoch import _toolchain_modules_added_by_import

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "src" / "champollion_pipeline"
CLI_MODULE = "champollion_pipeline.cli"
NEW_MODULE = "champollion_pipeline.pipeline_stages"
SPLIT_FILES = (PACKAGE_DIR / "cli.py", PACKAGE_DIR / "pipeline_stages.py")
MAX_FILE_LINES = 1000
MEASURE = "cli_split_file_lines_max"
REFERENCE_COMMIT = "5911c7a"
REFERENCE_PATH = "src/champollion_pipeline/cli.py"
REFERENCE_MODULE = "_cli_reference_5911c7a"
STAGE_COUNT = 6
LOGGER = logging.getLogger("test_pipeline_stages_split")

# Path values chosen so find_dataset_folder sometimes finds the dataset name ("DS") and sometimes does not.
_PATHS = st.sampled_from(
    [
        "",
        "/data",
        "/data/DS",
        "/data/DS/derivatives/cortical_tiles/crops/2mm",
        "/data/DS/runs/DS/derivatives/x",
        "/models",
        "relative/DS/crops",
    ]
)
_NAMES = st.sampled_from(["", "DS", "other"])
_MASKS = st.sampled_from(["canonical_25", "canonical_corrected_26_1"])
_WORDS = st.text(alphabet="abc._-0", max_size=6)
_REGIONS = st.lists(st.sampled_from(["S.C.-sylv.", "F.I.P.-F.I.P.Po.C.inf.", "Lobule_parietal_sup."]), max_size=3)


@st.composite
def pipeline_configs(draw):
    """PipelineConfig values over every field the six stages' build_args and output_dir read."""
    dataset = DatasetConfig(
        name=draw(_NAMES),
        datasets_root=draw(_PATHS),
        input_path=draw(_PATHS),
        morphologist_graphs=draw(_PATHS),
        cortical_tiles_output=draw(_PATHS),
        crops_path=draw(_PATHS),
        embeddings_path=draw(_PATHS),
        njobs=draw(st.integers(min_value=1, max_value=64)),
        path_to_graph=draw(_WORDS),
        path_sk_with_hull=draw(_WORDS),
        sk_qc_path=draw(_PATHS),
        masks_version=draw(_MASKS),
        skel_seed=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=9))),
        labelling_session=draw(_WORDS),
        model=draw(_WORDS),
        regions=draw(_REGIONS),
        overwrite=draw(st.booleans()),
        checkpoint_epoch=draw(st.integers(min_value=0, max_value=200)),
        checkpoint_epochs=draw(st.lists(_WORDS, max_size=2)),
        per_model_folders=draw(st.booleans()),
        models_cache=draw(st.one_of(st.none(), _PATHS)),
        hf_enabled=draw(st.booleans()),
        hf_repo_id=draw(st.one_of(st.none(), _WORDS)),
        cpu=draw(st.booleans()),
        bids=draw(st.booleans()),
        allow_prefix_ids=draw(st.booleans()),
        snapshots_path=draw(_PATHS),
        reference_data_path=draw(_PATHS),
        champollion_data_root=draw(st.one_of(st.none(), _PATHS)),
    )
    return PipelineConfig(
        models_path=draw(_PATHS),
        outputs_path=draw(_PATHS),
        parallel=draw(st.booleans()),
        dataset=dataset,
    )


@pytest.fixture(scope="module")
def reference_cli(tmp_path_factory):
    """cli.py of commit 5911c7a, loaded under a private module name."""
    if shutil.which("git") is None:
        pytest.skip("git not available: the reference cli.py cannot be read")
    shown = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{REFERENCE_COMMIT}:{REFERENCE_PATH}"],
        capture_output=True,
        text=True,
    )
    if shown.returncode != 0:
        pytest.skip(f"reference commit {REFERENCE_COMMIT} not in this clone: {shown.stderr.strip()}")
    source = tmp_path_factory.mktemp("reference") / "cli_5911c7a.py"
    # 5911c7a imports AmbiguousGraphFolders, removed with the graph-folder rule (TASK-ARGDISC-BDRABCZUK-1601F53514C5);
    # no stage class compared here uses it.
    source.write_text(
        shown.stdout.replace("AmbiguousGraphFolders, DatasetRootNotInferable", "DatasetRootNotInferable"),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(REFERENCE_MODULE, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_pairs(reference, new_module):
    """(stage name, reference class, moved class) for each stage of the reference STAGE_REGISTRY."""
    registry = reference.PipelineOrchestrator.STAGE_REGISTRY
    assert len(registry) == STAGE_COUNT, sorted(registry)
    return [(name, cls, getattr(new_module, cls.__name__)) for name, cls in sorted(registry.items())]


def _outcome(stage_cls, name, config, method):
    """What ``method`` of a fresh stage returns, or the type of the exception it raises."""
    stage = stage_cls(name, copy.deepcopy(config), LOGGER)
    try:
        return ("returned", getattr(stage, method)())
    except Exception as error:
        return ("raised", type(error).__module__, type(error).__qualname__)


def _explicit_models_source(config):
    """True when the stage 4 models source is explicit: Hugging Face enabled, or a non-empty models_path.

    Otherwise the source follows the fallback cascade of REQ-MODELSRC-BDRABCZUK-74BDF84B5789, which 5911c7a lacks.
    """
    return bool(config.dataset.hf_enabled or config.models_path)


def _assert_same_as_reference(reference, method, configs=None):
    new_module = importlib.import_module(NEW_MODULE)
    pairs = _stage_pairs(reference, new_module)

    @settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(config=configs if configs is not None else pipeline_configs())
    def check(config):
        for name, reference_cls, moved_cls in pairs:
            expected = _outcome(reference_cls, name, config, method)
            assert _outcome(moved_cls, name, config, method) == expected, (name, method)

    check()


def test_cli_defines_no_pipeline_stage_subclass():
    """REQ-CLISPLIT-BDRABCZUK-1BEA41BADEF3: no class whose __module__ is cli subclasses PipelineStage."""
    cli = importlib.import_module(CLI_MODULE)
    defined = sorted(
        name
        for name, value in vars(cli).items()
        if isinstance(value, type)
        and issubclass(value, PipelineStage)
        and value is not PipelineStage
        and value.__module__ == CLI_MODULE
    )
    assert defined == []


def test_cli_split_files_at_most_1000_lines():
    """REQ-CLISPLIT-BDRABCZUK-2A9646866292: each of the two files exists and holds at most 1000 lines."""
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in SPLIT_FILES if path.exists()}
    measures = os.environ.get("ALM_MEASURES")
    if measures and counts:
        with open(measures, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({MEASURE: max(counts.values())}) + "\n")
    assert sorted(counts) == sorted(path.name for path in SPLIT_FILES), f"missing file(s); found {sorted(counts)}"
    assert max(counts.values()) <= MAX_FILE_LINES, counts


def test_pipeline_stages_import_adds_no_toolchain_module():
    """REQ-CLISPLIT-BDRABCZUK-38095C750F40: importing pipeline_stages alone adds no soma/cortical_tiles/champollion."""
    assert _toolchain_modules_added_by_import(NEW_MODULE) == []


def test_stage_build_args_equal_reference(reference_cli):
    """REQ-CLISPLIT-BDRABCZUK-7ED02864B6B7: each moved stage class builds the argv of its 5911c7a class.

    Only configs whose models source is explicit (hf_enabled true or models_path non-empty) are compared.
    """
    _assert_same_as_reference(reference_cli, "build_args", pipeline_configs().filter(_explicit_models_source))


def test_stage_output_dir_equal_reference(reference_cli):
    """REQ-CLISPLIT-BDRABCZUK-AFFE25E4DDE2: each moved stage class reports the output folder of its 5911c7a class."""
    _assert_same_as_reference(reference_cli, "output_dir")

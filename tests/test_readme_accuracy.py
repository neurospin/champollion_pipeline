#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""README accuracy guards for REQ-DOCS-05, REQ-DOCS-06, REQ-DOCS-09, and
REQ-DOCS-10.

Joël asked (FEEDBACK-JOEL-G1) for ``README.md`` to reflect the pipeline's
current architecture and naming. Two independent, mechanically checkable
claims come out of that:

* **REQ-DOCS-05** — the ``Project Structure`` tree is the README's only
  architectural map of the installable package, so it must list every module
  that actually ships in ``src/champollion_pipeline/``.
* **REQ-DOCS-06** — the ``champollion-*`` console scripts declared in
  ``pyproject.toml``'s ``[project.scripts]`` are the pipeline's public command
  names, so every one of them must be named in the README.

Julien separately flagged (FEEDBACK-JULIEN-G1) that the README's "Project
website" link is dead (404) while the paper is under journal editorial
review, so an editor may click through from GitHub right now:

* **REQ-DOCS-09** — README.md must not contain the dead project-website URL.

A 2026-09-23 staleness audit found README.md documenting CLI arguments,
output paths, and step framing that no longer match the current code across
stages 2-6 (cortical_tiles, config, embeddings, combine, snapshots):

* **REQ-DOCS-10** — README.md's documented command-line arguments, output
  paths, and step descriptions for those stages must match the actual
  current CLI/behavior of ``run_cortical_tiles.py``,
  ``generate_champollion_config.py``, ``generate_embeddings.py``,
  ``put_together_embeddings.py``, and ``generate_snapshots.py``.

Both artifacts are read as plain text: no import of the package, no Sphinx
build, no network, and no checked-out ``external/`` submodule is required —
except for REQ-DOCS-10's argparse-introspection tests, which instantiate the
real ``ScriptBuilder`` subclasses (no network, no BrainVISA runtime calls;
only their ``argparse.ArgumentParser`` is inspected).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
PKG_DIR = PROJECT_ROOT / "src" / "champollion_pipeline"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _shipped_modules() -> list[str]:
    """Every ``*.py`` module in the installable package except ``__init__.py``.

    Sub-packages (e.g. ``utils/``) are deliberately excluded: REQ-DOCS-05
    constrains the top-level module files only.
    """
    return sorted(p.name for p in PKG_DIR.glob("*.py") if p.name != "__init__.py")


def _declared_console_scripts() -> list[str]:
    """Keys of ``[project.scripts]`` in ``pyproject.toml``.

    Parsed as text rather than via ``tomllib`` so the test behaves identically
    on every interpreter the project supports.
    """
    text = _read(PYPROJECT)
    section = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
    assert section, "no [project.scripts] table found in pyproject.toml"
    names = re.findall(r"^([A-Za-z0-9_.-]+)\s*=", section.group(1), re.MULTILINE)
    assert names, "[project.scripts] declares no entry points"
    return sorted(names)


def _project_structure_block() -> str:
    """The fenced code block that follows the ``## Project Structure`` heading."""
    text = _read(README)
    after = re.search(r"^##\s+Project Structure\s*$(.*?)(?=^##\s)", text, re.MULTILINE | re.DOTALL)
    assert after, "README.md has no '## Project Structure' section"
    block = re.search(r"```[^\n]*\n(.*?)```", after.group(1), re.DOTALL)
    assert block, "the 'Project Structure' section contains no fenced tree block"
    return block.group(1)


@pytest.mark.smoke
class TestProjectStructureTree:
    """REQ-DOCS-05: the tree lists every shipped pipeline module."""

    def test_tree_block_is_present(self):
        """The README still has a fenced tree under '## Project Structure'."""
        assert _project_structure_block().strip(), "the Project Structure tree block is empty"

    @pytest.mark.parametrize("module", _shipped_modules())
    def test_module_is_listed_in_tree(self, module):
        """Each ``src/champollion_pipeline/<module>`` appears as a tree entry."""
        lines = [ln for ln in _project_structure_block().splitlines() if module in ln]
        assert lines, (
            f"{module} ships in src/champollion_pipeline/ but is not listed in the README's Project Structure tree"
        )


@pytest.mark.smoke
class TestConsoleScriptNames:
    """REQ-DOCS-06: the README names every declared console script."""

    @pytest.mark.parametrize("script", _declared_console_scripts())
    def test_console_script_is_named_in_readme(self, script):
        """Each ``[project.scripts]`` key occurs verbatim somewhere in the README."""
        assert script in _read(README), (
            f"{script} is declared in pyproject.toml's [project.scripts] but is never named in README.md"
        )


@pytest.mark.smoke
class TestProjectWebsiteLink:
    """REQ-DOCS-09: the README does not link to the dead project website."""

    def test_readme_does_not_contain_dead_project_website_url(self):
        """``neurospin.fr/champollion_pipeline`` (404) must not appear in the README."""
        assert "neurospin.fr/champollion_pipeline" not in _read(README), (
            "README.md still contains the dead project-website URL "
            "'https://www.neurospin.fr/champollion_pipeline' (404)"
        )


# ---------------------------------------------------------------------------
# REQ-DOCS-10 helpers
# ---------------------------------------------------------------------------


def _quickstart_block() -> str:
    """The single fenced ``bash`` block under the top-level ``## Quick start`` heading."""
    text = _read(README)
    after = re.search(r"^##\s+Quick start\s*$(.*?)(?=^##\s)", text, re.MULTILINE | re.DOTALL)
    assert after, "README.md has no '## Quick start' section"
    block = re.search(r"```bash\n(.*?)```", after.group(1), re.DOTALL)
    assert block, "the 'Quick start' section contains no fenced bash block"
    return block.group(1)


def _quickstart_step(n: int) -> str:
    """The text of quickstart step ``n`` (a ``# n. ...`` comment through the next step marker).

    Quickstart steps are numbered 1-6 in their own inline-comment sequence
    (``# 1. ...`` through ``# 6. ...``), which is offset by one from the full
    numbered sections below it (``## 2.`` through ``## 7.``).
    """
    block = _quickstart_block()
    pattern = rf"^# {n}\.(.*?)(?=^# {n + 1}\.|\Z)"
    m = re.search(pattern, block, re.MULTILINE | re.DOTALL)
    assert m, f"quickstart step {n} not found in the Quick start block"
    return m.group(0)


def _section_block(heading_regex: str) -> str:
    """Full text of a top-level ``## <heading_regex>`` section, up to the next ``##`` heading."""
    text = _read(README)
    m = re.search(rf"^##\s+{heading_regex}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"README.md has no section matching '## {heading_regex}'"
    return m.group(1)


def _flags_in_text(text: str) -> set[str]:
    """Every ``--flag``-shaped token appearing in ``text``."""
    return set(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", text))


def _real_option_strings(builder_cls) -> set[str]:
    """Every option string (``--foo``) declared on a ``ScriptBuilder`` subclass's parser.

    Instantiates the class to build its ``argparse.ArgumentParser`` — this only
    registers arguments (see ``ScriptBuilder.__init__``); it never parses
    arguments, touches the filesystem, or calls into BrainVISA/soma-workflow.
    """
    instance = builder_cls()
    return {opt for action in instance.parser._actions for opt in action.option_strings}


@pytest.mark.smoke
class TestEmbeddingsCLIMatchesReadme:
    """REQ-DOCS-10: README's embeddings docs must match generate_embeddings.py's real argparse."""

    def _documented_flags(self) -> set[str]:
        quickstart = _quickstart_step(4)  # "# 4. Generate embeddings ..."
        section = _section_block(r"5\. Generate Embeddings")
        return _flags_in_text(quickstart) | _flags_in_text(section)

    def test_documented_flags_exist_in_real_parser(self):
        """Every ``--flag`` shown for the embeddings step must be a real generate_embeddings.py argument."""
        documented = self._documented_flags()
        real = _real_option_strings(GenerateEmbeddings)
        bogus = sorted(documented - real)
        assert not bogus, (
            "README documents embeddings flags that do not exist in generate_embeddings.py's "
            f"argparse: {bogus}"
        )

    def test_output_path_does_not_use_stale_split_named_subfolder(self):
        """generate_embeddings.py writes {output_base}/{region}/full_embeddings.csv, no {run}_{split}_embeddings/."""
        section = _section_block(r"5\. Generate Embeddings")
        assert "_random_embeddings" not in section, (
            "the real per-region output path is {output_base}/{region}/full_embeddings.csv "
            "(GenerateEmbeddings._run_per_region's saving_path); README's embeddings section "
            "still documents a '..._random_embeddings/full_embeddings.csv' subfolder that "
            "generate_embeddings.py does not write"
        )

    def test_legacy_flag_is_not_documented_as_rewriting_config_yaml_paths(self):
        """--legacy only changes the derivatives folder read for crops; it never rewrites config YAMLs."""
        text = _read(README)
        assert "Rewrite config YAML paths" not in text, (
            "GenerateEmbeddings._patch_config_paths (which would rewrite YAML paths) is dead code — "
            "it is defined but never called from anywhere, including --legacy's actual code path "
            "(_get_derivatives_folder, which only changes which derivatives folder crops are read "
            "from); README must not document --legacy as rewriting config YAMLs"
        )


@pytest.mark.smoke
class TestCombineCLIMatchesReadme:
    """REQ-DOCS-10: README's combine docs must match put_together_embeddings.py's real argparse."""

    def _documented_flags(self) -> set[str]:
        quickstart = _quickstart_step(5)  # "# 5. Combine embeddings"
        section = _section_block(r"6\. Combine Embeddings")
        return _flags_in_text(quickstart) | _flags_in_text(section)

    def test_documented_flags_exist_in_real_parser(self):
        """Every ``--flag`` shown for the combine step must be a real put_together_embeddings.py argument."""
        documented = self._documented_flags()
        real = _real_option_strings(PutTogetherEmbeddings)
        bogus = sorted(documented - real)
        assert not bogus, (
            "README documents combine flags that do not exist in put_together_embeddings.py's "
            f"argparse: {bogus}"
        )

    def test_embeddings_subpath_pattern_not_referenced(self):
        """put_together_embeddings.py takes embeddings_source + --output_path, with no subpath templating."""
        section = _section_block(r"6\. Combine Embeddings")
        assert "{short_name}_{split}" not in section, (
            "put_together_embeddings.py copies {embeddings_source}/{region}/full_embeddings.csv "
            "to {output_path}/{region}_embeddings.csv directly; there is no --embeddings_subpath "
            "argument and no '{short_name}_{split}' path templating in the real CLI"
        )


@pytest.mark.smoke
class TestCorticalTilesQuickstartMatchesReadme:
    """REQ-DOCS-10: README's cortical_tiles docs must match run_cortical_tiles.py's real argparse/output."""

    def test_quickstart_step_includes_required_flags(self):
        """As written, the quickstart's step 2 command is missing run_cortical_tiles.py's required flags."""
        step = _quickstart_step(2)  # "# 2. Extract sulcal regions"
        real_required = {
            action.option_strings[0]
            for action in RunCorticalTiles().parser._actions
            if getattr(action, "required", False) and action.option_strings
        }
        assert real_required, "expected run_cortical_tiles.py to declare at least one required flag"
        missing = sorted(flag for flag in real_required if flag not in step)
        assert not missing, (
            f"quickstart step 2 omits required run_cortical_tiles.py flag(s) {missing}; as written, "
            "running this exact command would exit with an argparse 'the following arguments are "
            "required' error"
        )

    def test_crops_output_path_includes_masks_version_segment(self):
        """Crops land at {output}/cortical_tiles-{VERSION}/crops/{masks_version}/2mm/, not crops/2mm/."""
        section = _section_block(r"3\. Generate Sulcal Region Crops")
        assert "crops/canonical_25/2mm" in section, (
            "CorticalTilesConfigFactory.from_args()/versioned_crops_exist() write and look for crops "
            "under {derivatives}/crops/{masks_version}/2mm/ (default masks_version='canonical_25'); "
            "README's documented output path is missing the mask-version path segment"
        )

    def test_region_file_flag_documented_as_inactive(self):
        """--region-file is accepted by argparse but the code that would forward it is commented out."""
        source = _read(PKG_DIR / "run_cortical_tiles.py")
        active_references = [
            line
            for line in source.splitlines()
            if ("region_file" in line or "region-file" in line) and not line.strip().startswith("#")
        ]
        # Exactly one active reference is expected: the add_optional_argument("--region-file", ...)
        # declaration itself. Anything beyond that would mean it actually gets forwarded.
        assert len(active_references) == 1, (
            "expected --region-file to be referenced nowhere outside its own argparse declaration "
            f"(the forwarding code is commented out); found active reference(s): {active_references}"
        )

        section = _section_block(r"3\. Generate Sulcal Region Crops")
        row = next((line for line in section.splitlines() if "--region-file" in line), None)
        assert row is not None, "README no longer documents --region-file at all"
        assert any(
            phrase in row.lower() for phrase in ("no effect", "inactive", "ignored", "not passed", "not forwarded")
        ), (
            "--region-file is accepted by argparse but never forwarded to generate_sulcal_regions.py "
            "(the cmd.extend(['--region-file', ...]) call is commented out); README documents it as "
            "a normal working option with no caveat"
        )


@pytest.mark.smoke
class TestConfigStepFramingMatchesReadme:
    """REQ-DOCS-10: the config step must not be framed as a prerequisite generate_embeddings.py reads."""

    def test_generate_embeddings_has_no_config_path_argument(self):
        """generate_embeddings.py never imports or reads the YAML config step's output."""
        real = _real_option_strings(GenerateEmbeddings)
        assert "--config_path" not in real, (
            "generate_champollion_config.py's YAML output feeds train_champollion.py's Hydra config "
            "groups (training), not inference; generate_embeddings.py declares no --config_path "
            "argument (the only 'config_path' in its source is a local parameter name on the dead, "
            "never-called _patch_config_paths method) and never reads the config step's output, so "
            "README must not frame step 4 (config) as a required predecessor to step 5's embeddings "
            "generation, and must not document a --config_path flag"
        )


@pytest.mark.smoke
class TestReferenceDataClaimMatchesReadme:
    """REQ-DOCS-10: reference_data/ is gitignored and does not ship in the repository."""

    def test_reference_data_dir_not_claimed_to_ship_in_repo(self):
        gitignore = _read(PROJECT_ROOT / ".gitignore")
        assert "reference_data/" in gitignore, "sanity check: reference_data/ is expected to be gitignored"
        assert not (PROJECT_ROOT / "reference_data").exists(), (
            "sanity check: reference_data/ is expected to not be checked out locally either"
        )
        text = _read(README)
        assert "(in repo)" not in text, (
            "reference_data/ is gitignored and is not checked into the repository; README's "
            "'--reference_data_dir reference_data/ # (in repo)' comment is stale"
        )


@pytest.mark.smoke
class TestMorphologistStepMatchesReadme:
    """REQ-DOCS-10: step 1 should use the safe wrapper, and the Pixi claim must be accurate."""

    def test_quickstart_uses_champollion_morphologist_wrapper(self):
        """champollion-morphologist wraps morphologist-cli with explicit --if/--of to avoid BIDS auto-detection."""
        step = _quickstart_step(1)  # "# 1. Generate Morphologist sulcal graphs ..."
        assert "champollion-morphologist" in step, (
            "champollion-morphologist (generate_morphologist_graphs.py) sets --if/--of explicitly to "
            "route around the BIDS-filename auto-detection failure mode that broke Ivan's run; the "
            "quickstart demonstrates raw morphologist-cli instead of this safer wrapper"
        )

    def test_brainvisa_pixi_claim_is_accurate(self):
        """Morphologist/BrainVISA is installed via pixi.toml's own 'brainvisa' feature, part of the default env."""
        pixi_toml = _read(PROJECT_ROOT / "pixi.toml")
        assert "[feature.brainvisa" in pixi_toml, "sanity check: expected a [feature.brainvisa...] table in pixi.toml"
        assert re.search(r'default\s*=\s*\{[^}]*"brainvisa"', pixi_toml), (
            "sanity check: expected the 'brainvisa' feature to be part of pixi.toml's default environment"
        )
        text = _read(README)
        assert "not through Pixi" not in text, (
            "BrainVISA/Morphologist is installed via pixi.toml's own [feature.brainvisa] table, which "
            "is part of the default environment (`default = { features = [\"brainvisa\", ...] }`); "
            "README's '...not through Pixi' claim in the Morphologist step is stale"
        )

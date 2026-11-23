#!/usr/bin/env python3
"""README accuracy guards for REQ-DOCS-05, REQ-DOCS-06, REQ-DOCS-09,
REQ-DOCS-10, REQ-DOCS-11, and REQ-DOCS-12.

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

Following TASK-068 (``--labelling_session``) and REQ-WHOLEBRAIN-04, section 3
had fallen behind ``run_cortical_tiles.py`` again:

* **REQ-DOCS-11** — section "3. Generate Sulcal Region Crops" must name the
  whole-brain output directory ``{output}/cortical_tiles-2026/whole_brain/F/``,
  and its "All options" table must carry a ``--labelling_session`` row stating
  default ``deepcnn_session_auto`` and an ``--overwrite`` row.

The interactive install wizard (``scripts/setup_wizard.py``) was never
mentioned in README.md:

* **REQ-DOCS-12** — section "1. Installation" must contain ``./install.sh`` or
  ``pixi run setup`` as the interactive install wizard entry point.

The README's collapsed Advanced block (TASK-DOCS-BDRABCZUK-7954511DD1EC) shows the six stage commands:

* **REQ-DOCS-BDRABCZUK-92F8CCADA55C** — each per-stage ``champollion-*`` command line in that
  ``<details>`` element is accepted by the argument parser of the matching console script
  (``TestQuickStartDetailsCommandsParse``; the parser is called on the documented tokens, nothing runs).

README.md sections 3, 5 and 6 moved to the Stage 2, Stage 4 and Stage 5 sections of ``docs/usage.md``
(TASK-DOCS-BDRABCZUK-84BE8AB6FC73): the REQ-DOCS-10 and REQ-DOCS-11 checks of those sections read
``docs/usage.md`` since.

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

from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.run_cortical_tiles import LABELLING_SESSION_DEFAULT, RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.docs_test_helpers import first_section, refused_command_lines, stage_details

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"
USAGE = PROJECT_ROOT / "docs" / "usage.md"
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
    """The fenced ``bash`` block of the collapsed Advanced ``<details>`` element of README's ``## Quick start``."""
    details = stage_details(first_section(_read(README)))
    block = re.search(r"```bash\n(.*?)```", details.group("body"), re.DOTALL)
    assert block, "the Advanced <details> element of 'Quick start' contains no fenced bash block"
    return block.group(1)


def _quickstart_command(command: str) -> str:
    """The ``pixi run <command>`` line of the Advanced block, with its backslash-continued lines."""
    lines = _quickstart_block().splitlines()
    start = next((i for i, ln in enumerate(lines) if re.search(rf"\bpixi run {re.escape(command)}(?![\w-])", ln)), None)
    assert start is not None, f"no 'pixi run {command}' line in the Quick start Advanced block"
    end = start
    while CONTINUATION_RE.search(lines[end]) and end + 1 < len(lines):
        end += 1
    return "\n".join(lines[start : end + 1])


def _section_block(heading_regex: str) -> str:
    """Full text of a top-level ``## <heading_regex>`` section, up to the next ``##`` heading."""
    text = _read(README)
    m = re.search(rf"^##\s+{heading_regex}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"README.md has no section matching '## {heading_regex}'"
    return m.group(1)


def _usage_stage_block(number: int) -> str:
    """Full text of the ``## Stage <number> ...`` section of docs/usage.md, up to the next ``##`` heading."""
    text = _read(USAGE)
    m = re.search(rf"^##\s+Stage {number}\b[^\n]*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"docs/usage.md has no section matching '## Stage {number}'"
    return m.group(1)


def _flags_in_text(text: str) -> set[str]:
    """Every ``--flag``-shaped token appearing in ``text``."""
    return set(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", text))


EMBEDDINGS_COMMAND_RE = re.compile(r"(?<![\w.-])(?:champollion-embeddings|generate_embeddings\.py)(?![\w.-])")
# A shell continuation as written in README: a trailing backslash, optionally followed by an inline comment.
CONTINUATION_RE = re.compile(r"\\\s*(?:#.*)?$")
SHELL_OPERATOR_RE = re.compile(r"\s(?:&&|\|\||\||;)\s")


def _shell_code(line: str) -> str:
    """``line`` without its shell comment (a whole comment line, or a ``#`` tail after whitespace)."""
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"(?:^|\s)#.*$", "", line)


def _embeddings_invocation_flags(text: str, *, is_code: bool = False) -> set[str]:
    """``--flag`` tokens of every embeddings-command invocation inside the fenced code blocks of ``text``.

    An invocation starts on a fenced-code line whose shell code (comments dropped) names
    ``champollion-embeddings`` (bare or via ``pixi run``) or ``generate_embeddings.py``; it runs
    through the backslash-continued lines that follow, and stops at a shell operator
    (``&&``, ``||``, ``|``, ``;``). Commands that are not the embeddings script (``rsync``,
    ``tar``, ...) and prose are ignored, so their own options are never collected.
    ``is_code=True`` treats the whole of ``text`` as code (an already-extracted fenced block body).
    """
    flags: set[str] = set()
    in_fence = is_code
    continuing = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continuing = False
            continue
        if not in_fence:
            continue
        code = _shell_code(line)
        if continuing:
            part = code
        else:
            m = EMBEDDINGS_COMMAND_RE.search(code)
            if not m:
                continue
            part = code[m.end() :]
        op = SHELL_OPERATOR_RE.search(part)
        if op:
            part = part[: op.start()]
        flags |= _flags_in_text(part)
        continuing = op is None and bool(CONTINUATION_RE.search(line))
    return flags


def _options_table_flags(section: str) -> set[str]:
    """``--flag`` tokens of the rows of every Markdown table in ``section`` whose header's first cell is ``Option``."""
    flags: set[str] = set()
    in_table = False
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        if re.match(r"^\|\s*Option\s*\|", stripped):
            in_table = True
            continue
        if in_table:
            flags |= _flags_in_text(stripped)
    return flags


def _documented_embeddings_flags(quickstart_step: str, section: str) -> set[str]:
    """Embeddings flags README documents: embeddings-command invocations plus the section's options table."""
    return (
        _embeddings_invocation_flags(quickstart_step, is_code=True)
        | _embeddings_invocation_flags(section)
        | _options_table_flags(section)
    )


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
        quickstart = _quickstart_command("champollion-embeddings")
        section = _usage_stage_block(4)
        return _documented_embeddings_flags(quickstart, section)

    def test_documented_flags_exist_in_real_parser(self):
        """Every ``--flag`` passed to the embeddings command, or listed in the usage.md Stage 4 options table,
        must be a real generate_embeddings.py argument.

        Only embeddings-command invocations (``champollion-embeddings`` / ``generate_embeddings.py``,
        with their backslash-continued lines) and the options table are scanned: other commands shown
        in that section (e.g. ``rsync``/``tar --exclude`` in the models_cache note) are not embeddings flags.
        """
        documented = self._documented_flags()
        assert documented, (
            "no embeddings flags found in the README quickstart or the usage.md Stage 4 section (scope helper broken?)"
        )
        real = _real_option_strings(GenerateEmbeddings)
        bogus = sorted(documented - real)
        assert not bogus, (
            "README or docs/usage.md documents embeddings flags that do not exist in "
            f"generate_embeddings.py's argparse: {bogus}"
        )

    def test_flag_scope_catches_misspelled_embeddings_flag_and_ignores_other_commands(self):
        """Self-check of the flag scope on a synthetic section: a misspelled embeddings flag (in an
        invocation, a continued line, or the options table) is collected; rsync/tar options and prose are not."""
        section = (
            "Prose mentioning `--prose-only` and `rsync --exclude`.\n"
            "```bash\n"
            "pixi run champollion-embeddings \\\n"
            "    neurospin/Champollion_V1 \\      # model source; --comment-only\n"
            "    --maks canonical_25\n"
            "python generate_embeddings.py src root --cpuu && rsync -a --exclude models_cache a/ b/\n"
            "rsync -a --exclude 'derivatives/champollion_V1/models_cache/' /data/myproject/ backup:/x/\n"
            "tar --exclude='*/models_cache' -czf ds.tar.gz /data/myproject\n"
            "```\n"
            "| Option | Description |\n"
            "|--------|-------------|\n"
            "| `--overwirte` | Recompute; see also `--masks` |\n"
            "\n"
            "| Source | Example value |\n"
            "|--------|---------------|\n"
            "| Archive | `--not-a-flag-table` |\n"
        )
        documented = _documented_embeddings_flags("", section)
        assert documented == {"--maks", "--cpuu", "--overwirte", "--masks"}, documented
        quickstart_step = (
            "# 4. Generate embeddings\n"
            "pixi run champollion-embeddings \\\n"
            "    neurospin/Champollion_V1 \\      # model source\n"
            "    --regons S.C.-sylv._left\n"
            "\n"
            "# 5. Combine embeddings\n"
            "pixi run champollion-combine dir/ --output_path out/\n"
        )
        assert _documented_embeddings_flags(quickstart_step, "") == {"--regons"}
        real = _real_option_strings(GenerateEmbeddings)
        assert {"--maks", "--cpuu", "--overwirte", "--regons"}.isdisjoint(real)

    def test_output_path_does_not_use_stale_split_named_subfolder(self):
        """generate_embeddings.py writes {output_base}/{region}/full_embeddings.csv, no {run}_{split}_embeddings/."""
        section = _usage_stage_block(4)
        assert "_random_embeddings" not in section, (
            "the real per-region output path is {output_base}/{region}/full_embeddings.csv "
            "(GenerateEmbeddings._run_per_region's saving_path); the usage.md embeddings section "
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
        quickstart = _quickstart_command("champollion-combine")
        section = _usage_stage_block(5)
        return _flags_in_text(quickstart) | _flags_in_text(section)

    def test_documented_flags_exist_in_real_parser(self):
        """Every ``--flag`` shown for the combine step must be a real put_together_embeddings.py argument."""
        documented = self._documented_flags()
        real = _real_option_strings(PutTogetherEmbeddings)
        bogus = sorted(documented - real)
        assert not bogus, (
            "README or docs/usage.md documents combine flags that do not exist in "
            f"put_together_embeddings.py's argparse: {bogus}"
        )

    def test_embeddings_subpath_pattern_not_referenced(self):
        """put_together_embeddings.py takes embeddings_source + --output_path, with no subpath templating."""
        section = _usage_stage_block(5)
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
        step = _quickstart_command("champollion-cortical-tiles")
        real_required = {
            action.option_strings[0]
            for action in RunCorticalTiles().parser._actions
            if getattr(action, "required", False) and action.option_strings
        }
        assert real_required, "expected run_cortical_tiles.py to declare at least one required flag"
        missing = sorted(flag for flag in real_required if flag not in step)
        assert not missing, (
            f"the quickstart champollion-cortical-tiles line omits required run_cortical_tiles.py flag(s) {missing}; "
            "as written, "
            "running this exact command would exit with an argparse 'the following arguments are "
            "required' error"
        )

    def test_crops_output_path_includes_masks_version_segment(self):
        """Crops land at {output}/cortical_tiles-{VERSION}/crops/{masks_version}/2mm/, not crops/2mm/."""
        section = _usage_stage_block(2)
        assert "crops/canonical_25/2mm" in section, (
            "CorticalTilesConfigFactory.from_args()/versioned_crops_exist() write and look for crops "
            "under {derivatives}/crops/{masks_version}/2mm/ (default masks_version='canonical_25'); "
            "docs/usage.md's documented output path is missing the mask-version path segment"
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

        section = _usage_stage_block(2)
        row = next((line for line in section.splitlines() if "--region-file" in line), None)
        assert row is not None, "docs/usage.md Stage 2 no longer documents --region-file at all"
        assert any(
            phrase in row.lower() for phrase in ("no effect", "inactive", "ignored", "not passed", "not forwarded")
        ), (
            "--region-file is accepted by argparse but never forwarded to generate_sulcal_regions.py "
            "(the cmd.extend(['--region-file', ...]) call is commented out); docs/usage.md documents it as "
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
        step = _quickstart_block()
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
            'is part of the default environment (`default = { features = ["brainvisa", ...] }`); '
            "README's '...not through Pixi' claim in the Morphologist step is stale"
        )


def _cortical_tiles_options_rows() -> dict[str, str]:
    """Map each ``--flag`` to its row in the usage.md Stage 2 ``All options`` table.

    Only rows whose first cell is a backticked ``--flag`` are collected; the
    table lives inside the ``<details><summary>All options</summary>`` block.
    """
    section = _usage_stage_block(2)
    m = re.search(r"<summary>\s*All options\s*</summary>(.*?)</details>", section, re.DOTALL)
    assert m, "usage.md Stage 2 has no '<summary>All options</summary>' ... '</details>' block"
    rows: dict[str, str] = {}
    for line in m.group(1).splitlines():
        cell = re.match(r"^\|\s*`(--[A-Za-z][A-Za-z0-9_-]*)`\s*\|", line)
        if cell:
            rows[cell.group(1)] = line
    assert rows, "sanity check: expected at least one `--flag` row in the usage.md Stage 2 'All options' table"
    return rows


@pytest.mark.smoke
class TestCorticalTilesWholeBrainAndOptionsMatchReadme:
    """REQ-DOCS-11: usage.md Stage 2 (README section 3 before the move) documents the whole-brain output.

    Its 'All options' table lists --labelling_session and --overwrite.
    """

    def test_whole_brain_output_directory_is_documented(self):
        """run() fuses L+R and strips the ventricle into {output}/cortical_tiles-2026/whole_brain/F/."""

        expected = f"{{output}}/{DERIVATIVES_FOLDER}/whole_brain/F/"
        section = _usage_stage_block(2)
        assert expected in section, (
            f"run_cortical_tiles.py writes the fused whole-brain volume to '{expected}' (remove_ventricle "
            "output_dir=<derivatives>/whole_brain, side 'F'), alongside the per-region crops; the Stage 2 section "
            "of docs/usage.md does not document this output directory"
        )

    def test_labelling_session_row_states_default(self):
        """--labelling_session (REQ-LABELSESSION-01) must appear in 'All options' with its real default."""

        assert "--labelling_session" in _real_option_strings(RunCorticalTiles), (
            "sanity check: expected run_cortical_tiles.py to declare --labelling_session"
        )
        rows = _cortical_tiles_options_rows()
        assert "--labelling_session" in rows, (
            "run_cortical_tiles.py declares --labelling_session but the usage.md Stage 2 'All options' table "
            "has no row for it"
        )
        assert LABELLING_SESSION_DEFAULT in rows["--labelling_session"], (
            f"the --labelling_session row does not state its default '{LABELLING_SESSION_DEFAULT}': "
            f"{rows['--labelling_session']!r}"
        )

    def test_overwrite_row_is_listed(self):
        """--overwrite exists in run_cortical_tiles.py's parser and must appear in 'All options'."""
        assert "--overwrite" in _real_option_strings(RunCorticalTiles), (
            "sanity check: expected run_cortical_tiles.py to declare --overwrite"
        )
        rows = _cortical_tiles_options_rows()
        assert "--overwrite" in rows, (
            "run_cortical_tiles.py declares --overwrite but the usage.md Stage 2 'All options' table has no row for it"
        )


INSTALL_SH = PROJECT_ROOT / "install.sh"
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
SETUP_WIZARD = PROJECT_ROOT / "scripts" / "setup_wizard.py"
WIZARD_ENTRY_POINTS = ("./install.sh", "pixi run setup")


@pytest.mark.smoke
class TestInstallWizardMatchesReadme:
    """REQ-DOCS-12: section 1 names the interactive install wizard entry point."""

    def test_installation_section_names_wizard_entry_point(self):
        """Section '1. Installation' must mention `./install.sh` or `pixi run setup`."""
        section = _section_block(r"1\. Installation")
        assert any(cmd in section for cmd in WIZARD_ENTRY_POINTS), (
            "README.md's '1. Installation' section never names the interactive install wizard "
            f"(scripts/setup_wizard.py); expected one of {WIZARD_ENTRY_POINTS} to appear"
        )

    def test_wizard_entry_points_exist(self):
        """Guard: the documented entry points really exist in source."""
        assert SETUP_WIZARD.is_file(), f"expected wizard script at {SETUP_WIZARD}"
        assert "pixi run setup" in _read(INSTALL_SH), "expected install.sh to invoke 'pixi run setup'"
        assert re.search(r'^setup\s*=\s*"python scripts/setup_wizard\.py"\s*$', _read(PIXI_TOML), re.MULTILINE), (
            'expected pixi.toml to declare task: setup = "python scripts/setup_wizard.py"'
        )

    def test_installation_section_still_documents_install_all(self):
        """Regression guard: the manual/non-interactive `pixi run install-all` path stays documented."""
        section = _section_block(r"1\. Installation")
        assert "pixi run install-all" in section, (
            "README.md's '1. Installation' section must keep documenting 'pixi run install-all' "
            "(non-interactive/CI path) alongside the wizard"
        )


STAGE_BUILDERS = {
    "champollion-morphologist": GenerateMorphologistGraphs,
    "champollion-cortical-tiles": RunCorticalTiles,
    "champollion-config": GenerateChampollionConfig,
    "champollion-embeddings": GenerateEmbeddings,
    "champollion-combine": PutTogetherEmbeddings,
    "champollion-snapshots": GenerateSnapshots,
}


@pytest.mark.smoke
class TestQuickStartDetailsCommandsParse:
    """REQ-DOCS-BDRABCZUK-92F8CCADA55C: the Advanced block's stage command lines parse with the real parsers."""

    def test_stage_command_lines_parse(self):
        body = stage_details(first_section(_read(README))).group("body")
        seen, refused = refused_command_lines(body, STAGE_BUILDERS)
        assert seen, "no 'pixi run champollion-<stage>' line in the Advanced <details> element"
        assert not refused, "README Advanced command lines refused by their argument parser:\n" + "\n".join(refused)

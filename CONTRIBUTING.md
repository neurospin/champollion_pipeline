# Contributing to Champollion Pipeline

Champollion Pipeline turns raw T1 MRI scans into compact, comparable
representations of sulcal morphology through a six-stage pipeline (see
`README.md` for the full architecture and usage instructions). This document
covers how to set up a development environment, the conventions we expect in
a contribution, and a couple of rules specific to this repository's
structure.

## Prerequisites

- Linux (`linux-64`) — this is the only platform currently supported.
- [Git](https://git-scm.com/)
- [Pixi](https://pixi.sh/) — this project's package/environment manager.
  Never use bare `pip` or `conda` to install dependencies here; always go
  through `pixi`.

## Getting the code

Clone the repository together with its submodules:

```bash
git clone --recurse-submodules https://github.com/neurospin/champollion_pipeline.git
cd champollion_pipeline
```

If you already cloned without `--recurse-submodules`, fetch them separately:

```bash
git submodule update --init
```

Then install all dependencies and verify the install:

```bash
pixi run install-all
pixi run check-install
```

## Development workflow

1. Create a branch for your change.
2. Make your change.
3. Before committing, run, in order:

   ```bash
   pixi run format      # auto-format with ruff
   pixi run lint         # check with ruff, or `pixi run lint-fix` to auto-fix
   pixi run test          # run the full test suite
   ```

   `pixi run lint-fix` applies ruff's auto-fixable findings; re-run
   `pixi run lint` afterward to confirm a clean result. See `README.md`'s
   `## Testing` section for the narrower `test-unit` / `test-integration` /
   `test-smoke` / `test-cov` / `test-fast` variants of `pixi run test`.

## Commit conventions

- **No AI attribution.** Commit messages must not include any AI attribution (no `Co-Authored-By` AI lines, no "Generated with ..." lines, no Anthropic/Claude.ai mentions anywhere in the message).
- **One logical change per commit.** Refactoring, documentation, and features are separate commits — don't bundle unrelated changes into one commit.
- **Imperative mood.** Write commit messages as commands: "Add mask version override", not "Added mask version override".

## Submodules

This repository depends on two upstream git submodules, checked out under
`external/`:

- `external/champollion_V1` — the self-supervised embedding models
  (upstream: [neurospin/champollion](https://github.com/neurospin/champollion.git)).
- `external/cortical_tiles` — the graph-to-region-crop extraction code
  (upstream: [neurospin/cortical_tiles](https://github.com/neurospin/cortical_tiles.git)).

These must never be edited directly from this repository. If you find a bug or want to change behavior in either one, open your fix against its own upstream repository first, get it merged there, then come back here and bump the submodule pointer to the new upstream commit in its own dedicated commit.

## Submitting changes

Push your branch and open a pull request. Make sure `pixi run test` and
`pixi run lint` are both green before requesting review.

## License

By contributing, you agree that your contributions will be licensed under
the same [CeCILL-B](LICENSE) license that covers the rest of this project.

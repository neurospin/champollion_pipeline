# Installation

## Prerequisites

- [Pixi](https://pixi.sh/) package manager
- Git
- [BrainVISA / Morphologist](https://brainvisa.info) — required for step 1 (sulcal graph extraction) only. If you already have Morphologist `.arg` graphs, you can skip that step.

## Setup

Clone the repository and install all dependencies:

```bash
mkdir Champollion && cd Champollion
git clone https://github.com/neurospin/champollion_pipeline.git
cd champollion_pipeline
pixi run install-all
```

`install-all` initializes the git submodules (`champollion_V1` and `cortical_tiles`), installs them in editable mode, clones `champollion_utils`, and creates the `data/` directory.

To enter the managed environment interactively:

```bash
pixi shell
```

## Uninstallation

```bash
pixi run uninstall        # Remove installed packages and cloned repos
pixi run uninstall-all    # Also remove Pixi-managed dependencies
```

> **Warning:** `uninstall-all` removes the `data/` folder as well. Back up any data stored there first.

## Updating an existing install

```bash
pixi run update
```

If you hit issues (submodule URL mismatch, missing `hatchling`, stale pixi tasks, etc.),
see the migration manual for symptoms and fixes.

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

## Virtual-environment installation (pip)

For users who manage their own virtualenv, venv, or uv environment and prefer a standard pip workflow.

> **Note:** BrainVISA/Morphologist (pipeline step 1 — sulcal graph extraction) requires conda and **cannot be pip-installed**. If you need that step, use the pixi route above. The pip route covers steps 2–6 (cortical tiles, config, embeddings, combine, snapshots).

Steps:

1. Clone the repository and initialize submodules.
2. Run `pip install -e .` to install the pipeline. `champollion-utils` is resolved automatically from GitHub — no separate clone needed.
3. Install `external/cortical_tiles` in editable mode (needs the deprecated sklearn compatibility shim).
4. Optionally install `external/champollion_V1` in editable mode.

```bash
git clone https://github.com/neurospin/champollion_pipeline.git
cd champollion_pipeline
git submodule update --init

pip install -e .
pip install -e external/champollion_V1 --no-deps --no-build-isolation
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
    pip install -e external/cortical_tiles --no-deps --no-build-isolation
```

## Diagnosing install problems

Run the health check at any time — no pixi env required:

```bash
pixi run check-install          # report current state
pixi run check-install-fix      # auto-fix stale editable installs
pixi run pre-update             # preview remote changes before pulling
```

Or with bare Python (useful when pixi itself fails):

```bash
python3 scripts/install_health.py --report
```

The script checks every editable install, every submodule, and prints a `CHAMPOLLION INSTALL REPORT` you can paste into a support ticket.

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

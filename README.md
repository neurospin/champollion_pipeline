# Champollion Pipeline

[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pixi](https://img.shields.io/badge/pixi-package%20manager-yellow?logo=prefix&logoColor=white)](https://pixi.sh/)
[![License: CeCILL-B](https://img.shields.io/badge/license-CeCILL--B-blue)](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-Champollion__V1-orange)](https://huggingface.co/neurospin/Champollion_V1)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue)](https://huggingface.co/spaces/neurospin/Champollion_demo)

Cortical sulci — the folds of the brain surface — vary in shape across individuals and are linked to cognitive function, development, and neurological conditions. The **Champollion pipeline** turns T1 MRI scans into compact, comparable representations of sulcal morphology using self-supervised contrastive learning. It processes MRIs through the [BrainVISA/Morphologist](https://brainvisa.info) toolchain to extract sulcal graphs, uses *cortical_tiles* to crop standardized 3D patches around 28 sulcal regions per hemisphere, and then runs pre-trained Champollion encoders to produce 32-dimensional embeddings per region. These embeddings can be projected onto pre-trained UMAP reference maps for visualization and compared across cohorts. The pipeline is designed for researchers who want to apply Champollion to their own neuroimaging datasets without retraining.

> **Try it online:** A [live demo is available on Hugging Face Spaces](https://huggingface.co/spaces/neurospin/Champollion_demo). It runs on 2 CPU cores and is suited for quick testing with a single subject. For batch processing or production use, install this pipeline locally where it can leverage all available CPUs and GPUs.

## Quick start

Here is the minimal happy path to generate embeddings from a pre-trained model — replace placeholders '/data/myproject' with your actual paths:

```bash
# 1. Generate Morphologist sulcal graphs (requires BrainVISA — skip if you already have .arg files)
#    # This takes as inputs the list of T1 MRI (here: sub-001.nii.gz sub-002.nii.gz)
#    # and outputs the morphologist graphs into the folder /data/myproject/derivatives/morphologist-6.0/subjects
pixi run champollion-morphologist sub-001.nii.gz sub-002.nii.gz /data/myproject/ \
    -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0

# 2. Extract sulcal regions
#    # Input:  morphologist subjects directory (read-only, never written to)
#    # Output: /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm/
pixi run champollion-cortical-tiles \
    /data/myproject/derivatives/morphologist-6.0/subjects \  # morphologist output dir
    /data/myproject/derivatives/ \                            # derivatives root (crops land here)
    --path_to_graph "t1mri/default_acquisition/*/folds/3.1" \      # required: glob pattern to .arg graph files
    --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation"  # required: skeleton directory

# 3. Generate Champollion configuration (optional — only needed for training, skip for inference)
#    # Links your crop paths to the Champollion model config
pixi run champollion-config \
    /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm \  # path to 2mm crops (step 2 output)
    --dataset myproject \                                          # dataset name used throughout the pipeline
    --output /data/myproject/derivatives/champollion_V1/configs   # where to write YAML config files

# 4. Generate embeddings (downloads pre-trained models from Hugging Face)
pixi run champollion-embeddings \
    neurospin/Champollion_V1 \      # model source: HF repo ID, local path, or archive
    /data/myproject/ \               # dataset root (parent of derivatives/)
    --masks canonical_25            # crops mask version (must match step 2)

# 5. Combine embeddings
#    # Collects the per-region CSVs from {dataset}embeddings/ into a single output directory
pixi run champollion-combine \
    /data/myprojectembeddings \  # {dataset}embeddings/ dir written by step 4
    --output_path /data/myproject/derivatives/champollion_V1/embeddings/ # destination for combined CSVs

# 6. Generate visualization snapshots
pixi run champollion-snapshots \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/embeddings/ \  # step 5 output
    --reference_data_dir /data/myproject/reference_data/ \                     # pre-trained UMAP models (not shipped, see below)
    --output_dir /data/myproject/derivatives/champollion_V1/snapshots/         # where to write images
```

The sections below explain each step in detail.

---

## 1. Installation

### Prerequisites

- [Pixi](https://pixi.sh/) package manager
- Git
- [BrainVISA / Morphologist](https://brainvisa.info) — required for step 2 only (sulcal graph extraction). If you already have Morphologist `.arg` graphs, you can skip step 2 and start from step 3.

### Setup

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

### Uninstallation

```bash
pixi run uninstall        # Remove installed packages and cloned repos
pixi run uninstall-all    # Also remove Pixi-managed dependencies
```

> ⚠️ `uninstall-all` removes the `data/` folder as well. Back up any data stored there first.

### Updating an existing install

```bash
pixi run update
```

If you hit issues (submodule URL mismatch, missing `hatchling`, stale pixi tasks, etc.),
see [migration_manual.md](migration_manual.md) for symptoms and fixes.

---

## 2. Generate Morphologist Graphs

> **Skip this step** if you already have Morphologist sulcal graph files (`.arg` format) for your subjects.

Morphologist is a tool from the [BrainVISA](https://brainvisa.info) neuroimaging suite that segments T1 MRI images and produces a graph representation of each subject's sulcal folds. It runs inside Pixi's `brainvisa` feature (part of the default environment).

Use the `champollion-morphologist` wrapper rather than calling `morphologist-cli` directly — it always sets `--if`/`--of` explicitly to avoid BIDS filenames being auto-detected as a different input format:

```bash
pixi run champollion-morphologist sub-001.nii.gz sub-002.nii.gz /data/myproject/ \
    -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0
```

This writes one folder per subject under `/data/myproject/derivatives/morphologist-6.0/subjects/`.

### Optional: parallel processing on HPC (soma-workflow)

For large cohorts, [soma-workflow](https://brainvisa.info/web/soma-workflow.html) (BrainVISA's HPC job scheduler) can parallelize graph generation across CPU cores or cluster nodes. This is entirely optional — serial processing works fine for small cohorts.

First configure soma-workflow:

```bash
soma_workflow_gui   # Set max CPUs under "Computing resources"
```

Then add `--swf`:

```bash
pixi run champollion-morphologist sub-001.nii.gz ... /data/myproject/ \
    -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0 --swf
```

---

## 3. Generate Sulcal Region Crops

Extract standardized 3D patches around 28 sulcal regions per hemisphere using *cortical_tiles*:

```bash
pixi run champollion-cortical-tiles \
    /data/myproject/derivatives/morphologist-6.0/subjects \  # morphologist subjects dir (read-only)
    /data/myproject/derivatives/ \                            # derivatives root; crops land at {root}/cortical_tiles-2026/crops/{masks_version}/2mm/
    --path_to_graph "t1mri/default_acquisition/*/folds/3.1" \
    --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation"
```

- **First argument** — directory containing one folder per subject (Morphologist's `subjects/` output). This path may be read-only (e.g. a shared NFS database); the script never writes to it.
- **Second argument** — derivatives parent directory. Crops are written to `{output}/cortical_tiles-2026/crops/{masks_version}/2mm/` (`{masks_version}` defaults to `canonical_25`, see [Mask versions](#mask-versions)).
- **`--path_to_graph` / `--path_sk_with_hull`** — required; there is no default, since the sub-path under a subject's Morphologist output depends on your acquisition/analysis naming.

`--path_to_graph` supports wildcards (`*`) for variable path segments (e.g. an acquisition or analysis label that differs per subject).

**Verify** that 28 sulcal region folders were created:

```bash
ls /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm
```

### QC File

To skip subjects with failing quality control, pass a tab-separated file with `participant_id` and `qc` columns (1 = keep, 0 = skip):

```
participant_id	qc	comments
sub-001         1
sub-002         0   Motion artefact
```

```bash
pixi run champollion-cortical-tiles ... --sk_qc_path /path/to/qc.tsv
```

### Mask versions

| Version | Description |
|---------|-------------|
| `canonical_25` | Original masks, used for Champollion V1. |
| `canonical_corrected_26_1` | Revised labeling with reduced artifacts. |

Pass `--masks canonical_25` (or another version) to override the default.

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--path_to_graph` | **Required.** Glob pattern to `.arg` graph files, relative to the subjects dir. |
| `--path_sk_with_hull` | **Required.** Skeleton-with-hull sub-path, relative to the subjects dir. |
| `--sk_qc_path` | Path to QC TSV file |
| `--njobs` | Number of CPU cores (default: auto) |
| `--region-file` | Accepted but currently has no effect (dead code upstream) — do not rely on it. |
| `--input-types` | Input types to generate (e.g. `skeleton foldlabel extremities`). Default: all. |
| `--skip-distbottom` | Skip distbottom generation (saves time; not needed for inference) |
| `--masks` | Mask version tag |
| `--regions` | Restrict to specific sulcal regions (space-separated) |

</details>

---

## 4. Generate Champollion Configuration

> **Only needed for training** a new encoder. `champollion-embeddings` (next step) reads crop paths directly from `models_path`/`datasets_root` and never reads these YAML files — skip this step if you're only generating embeddings from a pre-trained model.

Create the YAML configuration files that link your dataset's crop paths to the Champollion model:

```bash
pixi run champollion-config \
    /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm \  # path to 2mm crops (step 2 output)
    --dataset myproject \                                          # dataset name used in YAML and paths
    --output /data/myproject/derivatives/champollion_V1/configs   # config root; YAMLs land at {output}/dataset/{dataset}/
```

This writes region YAML files to `configs/dataset/myproject/` and sets `dataset_folder` in `dataset_localization/local.yaml` so the model knows where your data lives. This config directory is consumed by `champollion-train` (see [Console scripts](#console-scripts)), not by `champollion-embeddings`.

### Non-standard derivatives layout

If your crops live outside `derivatives/cortical_tiles-2026/` (e.g. a legacy dataset under `deep_folding-2025/`), add `--external_crops`:

```bash
pixi run champollion-config \
    /data/myproject/derivatives/deep_folding-2025/crops/2mm \  # legacy crop path
    --dataset myproject \
    --external_crops \   # use the exact crop path instead of the standard derivatives layout
    --output /data/myproject/derivatives/champollion_V1/configs
```

### Read-only containers (Apptainer / Docker)

When the pipeline directory is read-only, write `local.yaml` to a writable path with `--external-config`:

```bash
pixi run champollion-config \
    /path/to/crops/2mm \
    --dataset myproject \
    --output /writable/path/configs \
    --external-config /writable/path/configs/dataset_localization/local.yaml  # write local.yaml here instead of inside the pipeline dir
```

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--champollion_loc` | Path to Champollion binaries (default: `external/champollion_V1`) |
| `--output` | Configs root directory. Region YAMLs land at `{output}/dataset/{dataset}/`. |
| `--external_crops` | Use the exact crop path instead of assuming the standard derivatives layout. |
| `--external-config` | For read-only containers: write `local.yaml` to a writable path. |

</details>

---

## 5. Generate Embeddings

Run the pre-trained Champollion encoders across every region fold found under `models_path` (28 regions × 2 hemispheres for the default model set). Models are downloaded automatically from Hugging Face on first run and cached locally.

```bash
pixi run champollion-embeddings \
    neurospin/Champollion_V1 \      # model source: HF repo ID, local path, or archive
    /data/myproject/ \               # dataset root (parent of derivatives/); subjects list defaults to {datasets_root}/participants.tsv
    --masks canonical_25            # crops mask version (must match step 2)
```

Each region fold writes a `full_embeddings.csv` (one row per subject, columns = embedding dimensions) under:
```
{parent_of_datasets_root}/{dataset_name}embeddings/{region}/full_embeddings.csv
```
Override this location with `--output`.

To re-run on an existing dataset, add `--overwrite`.

### Model sources

You can point to different model sources:

| Source | Example value |
|--------|---------------|
| Hugging Face repo | `neurospin/Champollion_V1` |
| Cached local directory | `/data/myproject/derivatives/champollion_V1/models_cache/Champollion_V1` |
| Local archive | `/path/to/models.tar.gz` |
| Remote archive URL | `https://example.com/models.tar.gz` |

When using Hugging Face, models are cached in `data/{dataset}/derivatives/champollion_V1/models_cache/`. Pass the cached path directly on subsequent runs to avoid network checks. Use `--no-cache` to force a full re-download.

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--masks-version` | Mask version subfolder to download from Hugging Face (e.g. `canonical_25`). Ignored when `models_path` is a local directory. |
| `--masks` | Cortical tiles mask version used as the crops subdirectory (default: `canonical_25`) |
| `--regions` | Restrict to specific region names (space-separated). Default: all regions found in `models_path`. |
| `--output` | Output base directory override. Default: `{parent_of_datasets_root}/{dataset_name}embeddings/`. |
| `--subjects` | Path to subjects CSV with a `Subject` column. Default: `{datasets_root}/participants.tsv`. |
| `--cpu` | Force CPU usage (disable CUDA) |
| `--overwrite` | Recompute embeddings that already exist on disk |
| `--no-cache` | Force re-extraction of the archive (ignore cache) |
| `--run-cka` | Run CKA coherence test after embeddings |
| `--cortical_version` | Override the cortical tiles derivatives folder name (default: `cortical_tiles-2026`) |
| `--legacy` | Shorthand for `--cortical_version deep_folding-2025` (for older datasets) |
| `--profiling` | Enable Python profiling (cProfile) |

</details>

---

## 6. Combine Embeddings

Collect the per-region embedding CSVs into a single output directory:

```bash
pixi run champollion-combine \
    /data/myprojectembeddings \  # {dataset}embeddings/ dir written by step 5
    --output_path /data/myproject/derivatives/champollion_V1/embeddings/  # destination for the combined CSVs
```

This copies each region's `full_embeddings.csv` to `{output_path}/{region}_embeddings.csv`.

**Verify** the CSV files were created (one per region):

```bash
ls /data/myproject/derivatives/champollion_V1/embeddings/*.csv | wc -l
```

---

## 7. Generate Visualization Snapshots

Generate visualizations: sulcal graph meshes, cortical tile masks, and UMAP scatter plots projecting your subjects onto a pre-trained reference embedding space.

```bash
pixi run champollion-snapshots \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/embeddings/ \  # step 6 output
    --reference_data_dir /data/myproject/reference_data/ \                     # pre-trained UMAP models (not shipped — see below)
    --output_dir /data/myproject/derivatives/champollion_V1/snapshots/          # where to write images
```

Add `--morphologist_dir` and `--cortical_tiles_dir` to also generate mesh and mask snapshots:

```bash
pixi run champollion-snapshots \
    --morphologist_dir /data/myproject/derivatives/morphologist-6.0/ \          # for sulcal graph mesh snapshots
    --cortical_tiles_dir /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm/ \  # for tile mask snapshots
    --embeddings_dir /data/myproject/derivatives/champollion_V1/embeddings/ \
    --reference_data_dir /data/myproject/reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/snapshots/
```

Use `--sulcal-only`, `--tiles-only`, or `--umap-only` to generate only one snapshot type.

### UMAP Visualization

UMAP scatter plots project each subject's sulcal embeddings onto pre-trained 2D maps (one per region and hemisphere). Each plot shows a reference cloud of embeddings from a large pre-trained cohort, with the new subject highlighted. `--reference_data_dir` points at a directory of pre-trained UMAP artifacts (`umap_{region}_{hemi}.pkl` / `_coords.npy`) that contain no subject identifiers; this directory is **not shipped in the repository** (it's gitignored) — obtain it separately and pass its path explicitly.

By default all regions with an available embedding CSV and a pre-trained model are plotted. Restrict to specific regions with `--umap_region`:

```bash
pixi run champollion-snapshots \
    --embeddings_dir /path/to/embeddings/ \
    --reference_data_dir /path/to/reference_data/ \
    --output_dir /path/to/snapshots/ \
    --umap-only --umap_region FColl-SRh  # restrict to a single region
```

### Multiple acquisitions per subject

If a subject has several Morphologist acquisitions (e.g. two time points), the script warns and uses the first one found. Specify the acquisition explicitly to avoid ambiguity:

```bash
pixi run champollion-snapshots \
    --morphologist_dir /path/to/subjects/ \
    --subject sub-001 --acquisition wk40 \  # acquisition label to disambiguate multiple time points
    --output_dir /path/to/snapshots/ --sulcal-only
```

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--morphologist_dir` | Path to Morphologist output (for sulcal graph snapshots) |
| `--subject` | Subject folder name (default: first subject found) |
| `--acquisition` | Acquisition label when a subject has multiple segmentations |
| `--cortical_tiles_dir` | Path to `crops/{masks_version}/2mm/` (for tiles mask snapshots) |
| `--embeddings_dir` | Path to combined embeddings (for UMAP scatter plots) |
| `--reference_data_dir` | Path to pre-trained UMAP models and reference coordinates (not shipped in the repo, gitignored — obtain separately) |
| `--umap_region` | Comma-separated region name(s) to plot |
| `--output_dir` | Directory to save snapshot images |
| `--sulcal-only` / `--tiles-only` / `--umap-only` | Generate only one snapshot type |
| `--width` / `--height` | Snapshot dimensions (default: 800×600) |
| `--tiles_level` | Cortical tiles level to visualize |

</details>

---

## Project Structure

```
champollion_pipeline/
├── external/
│   ├── champollion_V1/         # Self-supervised encoder submodule (contrastive learning)
│   └── cortical_tiles/         # Sulcal region crop extraction submodule
├── reference_data/             # NOT shipped (gitignored) — pre-trained UMAP models go here, obtained separately
├── src/
│   └── champollion_pipeline/   # Installable Python package (entry points: champollion-*)
│       ├── generate_morphologist_graphs.py
│       ├── run_cortical_tiles.py
│       ├── generate_champollion_config.py
│       ├── generate_embeddings.py
│       ├── put_together_embeddings.py
│       ├── generate_snapshots.py
│       ├── train_champollion.py
│       ├── generate_masks.py
│       ├── prune_failed_subjects.py
│       └── purge_subject.py
├── data/                       # Created by install-all; not committed
└── pixi.toml
```

### Console scripts

Installing the package provides these entry points, declared in
`pyproject.toml`'s `[project.scripts]`:

| Command | Module | Purpose |
|---|---|---|
| `champollion-morphologist` | `generate_morphologist_graphs.py` | Step 1 — run Morphologist over raw T1 MRI to produce sulcal graphs |
| `champollion-cortical-tiles` | `run_cortical_tiles.py` | Step 2 — extract sulcal region crops from Morphologist graphs |
| `champollion-config` | `generate_champollion_config.py` | Step 3 — write the Champollion YAML dataset configuration |
| `champollion-embeddings` | `generate_embeddings.py` | Step 4 — compute per-region embeddings from a pre-trained model |
| `champollion-combine` | `put_together_embeddings.py` | Step 5 — collect the per-region CSVs into one output directory |
| `champollion-snapshots` | `generate_snapshots.py` | Step 6 — render UMAP visualization snapshots |
| `champollion-train` | `train_champollion.py` | Optional — train a `champollion_V1` encoder for one sulcal region |
| `champollion-prune` | `prune_failed_subjects.py` | Maintenance — delete cortical_tiles outputs for QC-failing subjects |
| `champollion-purge` | `purge_subject.py` | Maintenance — delete all cortical_tiles derivatives for one subject |

`generate_masks.py` also ships in the package but declares no console script.

## Testing

```bash
pixi run test               # All tests
pixi run test-unit          # Unit tests only
pixi run test-integration   # Integration tests only
pixi run test-smoke         # Smoke tests only
pixi run test-cov           # Tests with coverage report
pixi run test-fast          # Stop on first failure
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'champollion_pipeline'`

The Python package is installed by `pixi run install-all` (first-time setup) or `pixi run install-embeddings` (embeddings env only). If you reinstalled Pixi, changed environments, or ran `pixi run uninstall`, run the install step again before any pipeline task:

```bash
pixi run install-embeddings   # re-installs champollion_pipeline, champollion_V1, and champollion_utils
```

This must be run from the repository root (the directory containing `pixi.toml`).

### `pixi run champollion-embeddings` task not found

The `champollion-*` entry-point tasks were added in a recent release. If you are on an older checkout, update first:

```bash
pixi run update-pipeline   # pulls latest code and pixi.lock from origin
pixi run install-embeddings
```

### `pixi run update-submodules` fails with a rebase conflict

If a submodule has local commits that conflict with the remote tip, the update aborts. These submodules are upstream dependencies — local commits inside them are not expected. Run:

```bash
git submodule update --init --remote --force external/champollion_V1
git submodule update --init --remote --force external/cortical_tiles
```

This resets each submodule to the remote tip, discarding any local commits inside it.

---

## Dependencies

All dependencies are managed through `pixi.toml`. Core requirements:

- Python ≥ 3.10
- PyTorch
- BrainVISA / Morphologist (for step 2 only — see [brainvisa.info](https://brainvisa.info))
- huggingface-hub

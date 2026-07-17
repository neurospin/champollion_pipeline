# Champollion Pipeline

[![Python](https://img.shields.io/badge/python-%E2%89%A53.8-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pixi](https://img.shields.io/badge/pixi-package%20manager-yellow?logo=prefix&logoColor=white)](https://pixi.sh/)
[![License: CeCILL-B](https://img.shields.io/badge/license-CeCILL--B-blue)](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-Champollion__V1-orange)](https://huggingface.co/neurospin/Champollion_V1)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue)](https://huggingface.co/spaces/neurospin/Champollion_demo)

Cortical sulci — the folds of the brain surface — vary in shape across individuals and are linked to cognitive function, development, and neurological conditions. The **Champollion pipeline** turns T1 MRI scans into compact, comparable representations of sulcal morphology using self-supervised contrastive learning. It processes MRIs through the [BrainVISA/Morphologist](https://brainvisa.info) toolchain to extract sulcal graphs, uses *cortical_tiles* to crop standardized 3D patches around 28 sulcal regions per hemisphere, and then runs pre-trained Champollion encoders to produce 32-dimensional embeddings per region. These embeddings can be projected onto pre-trained UMAP reference maps for visualization and compared across cohorts. The pipeline is designed for researchers who want to apply Champollion to their own neuroimaging datasets without retraining.

> **Project website:** [https://www.neurospin.fr/champollion_pipeline](https://www.neurospin.fr/champollion_pipeline)

> **Try it online:** A [live demo is available on Hugging Face Spaces](https://huggingface.co/spaces/neurospin/Champollion_demo). It runs on 2 CPU cores and is suited for quick testing with a single subject. For batch processing or production use, install this pipeline locally where it can leverage all available CPUs and GPUs.

## Quick start

The pipeline runs in 6 steps. Here is the minimal happy path — replace placeholders '/data/myproject' with your actual paths:

```bash
# 1. Generate Morphologist sulcal graphs (requires BrainVISA — skip if you already have .arg files)
#    This takes as inputs the list of T1 MRI (here: sub-001.nii.gz sub-002.nii.gz)
#    and outputs the morphologist graphs into the folder /data/myproject/derivatives/morphologist-6.0/subjects
morphologist-cli sub-001.nii.gz sub-002.nii.gz /data/myproject/ \
    -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0

# 2. Extract sulcal region crops
pixi run python3 src/run_cortical_tiles.py \
    /data/myproject/derivatives/morphologist-6.0/subjects \
    /data/myproject/derivatives/

# 3. Generate Champollion configuration
pixi run python3 src/generate_champollion_config.py \
    /data/myproject/derivatives/cortical_tiles-2026/crops/2mm \
    --dataset myproject \
    --output /data/myproject/derivatives/champollion_V1/configs

# 4. Generate embeddings (downloads pre-trained models from Hugging Face)
pixi run python3 src/generate_embeddings.py \
    neurospin/Champollion_V1 local myproject my_run \
    --embeddings_only \
    --config_path /data/myproject/derivatives/champollion_V1/configs/dataset/myproject

# 5. Combine embeddings
pixi run python3 src/put_together_embeddings.py \
    --path_models /data/myproject/derivatives/champollion_V1/models_cache/Champollion_V1/ \
    --embeddings_subpath my_run_random_embeddings/full_embeddings.csv \
    --output_path /data/myproject/derivatives/champollion_V1/embeddings/

# 6. Generate visualization snapshots
pixi run python3 src/generate_snapshots.py \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/snapshots/
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

---

## 2. Generate Morphologist Graphs

> **Skip this step** if you already have Morphologist sulcal graph files (`.arg` format) for your subjects.

Morphologist is a tool from the [BrainVISA](https://brainvisa.info) neuroimaging suite that segments T1 MRI images and produces a graph representation of each subject's sulcal folds. It runs in its own BrainVISA environment, not through Pixi.

```bash
morphologist-cli sub-001.nii.gz sub-002.nii.gz /data/myproject/ \
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
morphologist-cli sub-001.nii.gz ... /data/myproject/ \
    -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0 --swf
```

---

## 3. Generate Sulcal Region Crops

Extract standardized 3D patches around 28 sulcal regions per hemisphere using *cortical_tiles*:

```bash
pixi run python3 src/run_cortical_tiles.py \
    /data/myproject/derivatives/morphologist-6.0/subjects \
    /data/myproject/derivatives/
```

- **First argument** — directory containing one folder per subject (Morphologist's `subjects/` output). This path may be read-only (e.g. a shared NFS database); the script never writes to it.
- **Second argument** — derivatives parent directory. Crops are written to `{output}/cortical_tiles-2026/crops/2mm/`.

If your Morphologist graphs are stored under a non-default sub-path, override it:

```bash
pixi run python3 src/run_cortical_tiles.py \
    /data/myproject/derivatives/morphologist-6.0/subjects \
    /data/myproject/derivatives/ \
    --path_to_graph "t1mri/default_acquisition/*/folds/3.1" \
    --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation"
```

The `--path_to_graph` option supports wildcards (`*`) for variable path segments.

**Verify** that 28 sulcal region folders were created:

```bash
ls /data/myproject/derivatives/cortical_tiles-2026/crops/2mm
```

### QC File

To skip subjects with failing quality control, pass a tab-separated file with `participant_id` and `qc` columns (1 = keep, 0 = skip):

```
participant_id	qc	comments
sub-001         1
sub-002         0   Motion artefact
```

```bash
pixi run python3 src/run_cortical_tiles.py ... --sk_qc_path /path/to/qc.tsv
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
| `--sk_qc_path` | Path to QC TSV file |
| `--njobs` | Number of CPU cores (default: auto) |
| `--region-file` | Custom sulcal region configuration file |
| `--input-types` | Input types to generate (e.g. `skeleton foldlabel extremities`). Default: all. |
| `--skip-distbottom` | Skip distbottom generation (saves time; not needed for inference) |
| `--masks` | Mask version tag |
| `--regions` | Restrict to specific sulcal regions (space-separated) |

</details>

---

## 4. Generate Champollion Configuration

Create the YAML configuration files that link your dataset's crop paths to the Champollion model:

```bash
pixi run python3 src/generate_champollion_config.py \
    /data/myproject/derivatives/cortical_tiles-2026/crops/2mm \
    --dataset myproject \
    --output /data/myproject/derivatives/champollion_V1/configs
```

This writes region YAML files to `configs/dataset/myproject/` and sets `dataset_folder` in `dataset_localization/local.yaml` so the model knows where your data lives. Pass the same `--output` path as `--config_path` in the next step (plus `/dataset/myproject`).

### Non-standard derivatives layout

If your crops live outside `derivatives/cortical_tiles-2026/` (e.g. a legacy dataset under `deep_folding-2025/`), add `--external_crops`:

```bash
pixi run python3 src/generate_champollion_config.py \
    /data/myproject/derivatives/deep_folding-2025/crops/2mm \
    --dataset myproject \
    --external_crops \
    --output /data/myproject/derivatives/champollion_V1/configs
```

### Read-only containers (Apptainer / Docker)

When the pipeline directory is read-only, write `local.yaml` to a writable path with `--external-config`:

```bash
pixi run python3 src/generate_champollion_config.py \
    /path/to/crops/2mm \
    --dataset myproject \
    --output /writable/path/configs \
    --external-config /writable/path/configs/dataset_localization/local.yaml
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

Run the pre-trained Champollion encoders across all 56 model folds (28 regions × 2 hemispheres). Models are downloaded automatically from Hugging Face on first run and cached locally.

```bash
pixi run python3 src/generate_embeddings.py \
    neurospin/Champollion_V1 \          # model source (HF repo ID)
    local \                              # localization preset
    myproject \                          # dataset name
    my_run \                             # label for this run
    --embeddings_only \
    --config_path /data/myproject/derivatives/champollion_V1/configs/dataset/myproject
```

Each of the 56 folds writes a `full_embeddings.csv` (one row per subject, columns = embedding dimensions) under:
```
models_cache/Champollion_V1/{region}/
    my_run_random_embeddings/full_embeddings.csv
```

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
| `--config_path` | Path to dataset config directory (generated in step 4) |
| `--embeddings_only` | Only compute embeddings (skip classifier training) |
| `--cpu` | Force CPU usage (disable CUDA) |
| `--overwrite` | Overwrite existing embeddings |
| `--no-cache` | Force re-extraction of archive |
| `--run-cka` | Run CKA coherence test after embeddings |
| `--split` | Splitting strategy: `random` or `custom` (default: `random`) |
| `--nb_jobs` | Number of CPU workers for the DataLoader |
| `--cortical_version` | Override the cortical tiles folder name (e.g. `cortical_tiles-2025`) |
| `--legacy` | Rewrite config YAML paths to use `deep_folding-2025` (for older datasets) |
| `--labels` | Labels for classifiers (default: `['Sex']`) |
| `--classifier_name` | Classifier type (default: `svm`) |

</details>

---

## 6. Combine Embeddings

Collect the 56 per-region embedding CSVs into a single output directory:

```bash
pixi run python3 src/put_together_embeddings.py \
    --path_models /data/myproject/derivatives/champollion_V1/models_cache/Champollion_V1/ \
    --embeddings_subpath my_run_random_embeddings/full_embeddings.csv \
    --output_path /data/myproject/derivatives/champollion_V1/embeddings/
```

`--embeddings_subpath` is `{short_name}_{split}_embeddings/full_embeddings.csv`, using the `my_run` label and `random` split from step 5.

**Verify** 56 CSV files were created:

```bash
ls /data/myproject/derivatives/champollion_V1/embeddings/*.csv | wc -l
```

---

## 7. Generate Visualization Snapshots

Generate visualizations: sulcal graph meshes, cortical tile masks, and UMAP scatter plots projecting your subjects onto a pre-trained reference embedding space.

```bash
pixi run python3 src/generate_snapshots.py \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/snapshots/
```

Add `--morphologist_dir` and `--cortical_tiles_dir` to also generate mesh and mask snapshots:

```bash
pixi run python3 src/generate_snapshots.py \
    --morphologist_dir /data/myproject/derivatives/morphologist-6.0/ \
    --cortical_tiles_dir /data/myproject/derivatives/cortical_tiles-2026/crops/2mm/ \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/snapshots/
```

Use `--sulcal-only`, `--tiles-only`, or `--umap-only` to generate only one snapshot type.

### UMAP Visualization

UMAP scatter plots project each subject's sulcal embeddings onto pre-trained 2D maps (one per region and hemisphere). Each plot shows a reference cloud of embeddings from a large pre-trained cohort, with the new subject highlighted. Pre-trained UMAP artifacts are stored in `reference_data/` and contain no subject identifiers.

By default all regions with an available embedding CSV and a pre-trained model are plotted. Restrict to specific regions with `--umap_region`:

```bash
pixi run python3 src/generate_snapshots.py \
    --embeddings_dir /path/to/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /path/to/snapshots/ \
    --umap-only --umap_region FColl-SRh
```

### Multiple acquisitions per subject

If a subject has several Morphologist acquisitions (e.g. two time points), the script warns and uses the first one found. Specify the acquisition explicitly to avoid ambiguity:

```bash
pixi run python3 src/generate_snapshots.py \
    --morphologist_dir /path/to/subjects/ \
    --subject sub-001 --acquisition wk40 \
    --output_dir /path/to/snapshots/ --sulcal-only
```

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--morphologist_dir` | Path to Morphologist output (for sulcal graph snapshots) |
| `--subject` | Subject folder name (default: first subject found) |
| `--acquisition` | Acquisition label when a subject has multiple segmentations |
| `--cortical_tiles_dir` | Path to crops/2mm/ (for tiles mask snapshots) |
| `--embeddings_dir` | Path to combined embeddings (for UMAP scatter plots) |
| `--reference_data_dir` | Path to pre-trained UMAP models and reference coordinates |
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
│   ├── champollion_V1/     # Self-supervised encoder submodule (contrastive learning)
│   └── cortical_tiles/     # Sulcal region crop extraction submodule
├── reference_data/         # Pre-trained UMAP models and anonymous reference coordinates
├── src/
│   ├── generate_morphologist_graphs.py
│   ├── run_cortical_tiles.py
│   ├── generate_champollion_config.py
│   ├── generate_embeddings.py
│   ├── put_together_embeddings.py
│   ├── generate_snapshots.py
│   └── train_champollion.py
├── data/                   # Created by install-all; not committed
└── pixi.toml
```

## Testing

```bash
pixi run test               # All tests
pixi run test-unit          # Unit tests only
pixi run test-integration   # Integration tests only
pixi run test-smoke         # Smoke tests only
pixi run test-cov           # Tests with coverage report
pixi run test-fast          # Stop on first failure
```

## Dependencies

All dependencies are managed through `pixi.toml`. Core requirements:

- Python ≥ 3.8
- PyTorch
- BrainVISA / Morphologist (for step 2 only — see [brainvisa.info](https://brainvisa.info))
- huggingface-hub

# Champollion Pipeline

[![Python](https://img.shields.io/badge/python-%E2%89%A53.8-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pixi](https://img.shields.io/badge/pixi-package%20manager-yellow?logo=prefix&logoColor=white)](https://pixi.sh/)
[![License: CeCILL-B](https://img.shields.io/badge/license-CeCILL--B-blue)](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-Champollion__V1-orange)](https://huggingface.co/neurospin/Champollion_V1)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue)](https://huggingface.co/spaces/neurospin/Champollion_demo)

This pipeline generates Champollion embeddings from T1 MRI images. It processes MRIs through Morphologist to extract sulcal graphs, then uses cortical_tiles to create sulcal regions, and finally generates embeddings using pre-trained Champollion models.

> **Try it online:** A [live demo is available on Hugging Face Spaces](https://huggingface.co/spaces/neurospin/Champollion_demo). It runs on 2 CPU cores and is suited for quick testing with a single subject. For batch processing or production use, install this pipeline locally where it can leverage all available CPUs and GPUs.

## 1. Installation

### Prerequisites

- [Pixi](https://pixi.sh/) package manager
- Git

### Setup

If pixi is not setup in your environment yet, please follow the official instruction on:

https://pixi.prefix.dev/latest/installation/

Clone the repository and set up the environment:

```bash
mkdir Champollion
cd Champollion
git clone https://github.com/neurospin/champollion_pipeline.git
cd champollion_pipeline
```

Install all dependencies using Pixi:

```bash
pixi run install-all
```

This command will:
- Initialize git submodules (champollion_V1 and cortical_tiles)
- Install cortical_tiles in editable mode
- Install champollion_V1 in editable mode
- Clone and install champollion_utils
- Create the data directory

To enter the Pixi environment:

```bash
pixi shell
```

### Uninstallation

To remove installed packages and cloned repositories:

```bash
pixi run uninstall
```

To completely remove everything including Pixi-managed dependencies:

```bash
pixi run uninstall-all
```

## 2. Generate Morphologist Graphs

Generate sulcal graphs from T1 MRI images using morphologist-cli.

### Serial Processing

```bash
LIST_MRI_FILES="/path/to/data/sub-0001.nii.gz /path/to/data/sub-0002.nii.gz"
OUTPUT_PATH="/path/to/data/TESTXX/"
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0
```

### Parallel Processing with soma-workflow

First, configure soma-workflow:

```bash
soma_workflow_gui
```

Set the maximum number of CPUs (e.g., 24) in the "Computing resources" subwindow.

Then run with the `--swf` flag:

```bash
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0 --swf
```


## 3. Generate Sulcal Regions

Use cortical_tiles to extract sulcal regions from Morphologist's graphs:

```bash
pixi run python3 src/run_cortical_tiles.py \
    /path/to/data/TESTXX/derivatives/morphologist-6.0/subjects \
    /path/to/data/TESTXX/derivatives/ \
    --path_to_graph "t1mri/default_acquisition/default_analysis/folds/3.1" \
    --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation"
```

- **`input`** — directory containing one folder per subject (e.g., morphologist's `subjects/` output). This path may be read-only (e.g., an NFS database): the script never writes to it.
- **`output`** — derivatives parent directory. The script writes `pipeline_loop_2mm.json` here and produces sulcal region crops at `{output}/cortical_tiles-2026/crops/2mm/`.

The `--path_to_graph` supports wildcards (`*`) for variable path segments, e.g.:

```bash
--path_to_graph "t1mri/default_acquisition/*/folds/3.1"
```

### Options

| Option | Description |
|--------|-------------|
| `--sk_qc_path` | Path to QC file (optional) |
| `--njobs` | Number of CPU cores to use (default: auto) |
| `--region-file` | Custom sulcal region configuration file |
| `--input-types` | Input types to generate (e.g. `skeleton foldlabel extremities`). Default: all types |
| `--skip-distbottom` | Skip distbottom generation (unused during inference, saves time) |

### QC File Format

If you have a QC file, it should be tab-separated with columns `participant_id` and `qc`:

```
participant_id	qc	comments
bvdb            0   Right graph does not exist
sub-1000021     1
```

### Verification

Check that 28 sulcal region folders were created:

```bash
ls /path/to/data/TESTXX/derivatives/cortical_tiles-2026/crops/2mm
```

## 4. Generate Champollion Configuration

Create dataset configuration files for Champollion.

> **Recommended:** always pass `--external-config` to keep the `local.yaml` outside the pipeline directory. This is required in read-only containers (Apptainer/Docker) and avoids accidentally committing paths specific to your machine.

```bash
pixi run python3 src/generate_champollion_config.py \
    /path/to/data/TESTXX/derivatives/cortical_tiles-2026/crops/2mm \
    --dataset TESTXX \
    --external-config /path/to/data/TESTXX/derivatives/champollion_V1/configs/local.yaml
```

### Options

| Option | Description |
|--------|-------------|
| `--champollion_loc` | Path to Champollion binaries (default: external/champollion_V1) |
| `--output` | Custom output path for config files |
| `--external-config` | Path for `local.yaml` outside the pipeline directory (recommended) |

## 5. Generate Embeddings

Generate embeddings using pre-trained Champollion models. Each of the 56 model folds (28 regions x 2 hemispheres) produces a `full_embeddings.csv` file containing one embedding vector per subject.

### Basic Usage

```bash
pixi run python3 src/generate_embeddings.py \
    <models_path> \
    <dataset_localization> \
    <datasets_root> \
    <short_name> \
    --embeddings_only \
    --config_path /path/to/data/TESTXX/derivatives/champollion_V1/configs/dataset
```

To re-run on an existing dataset, add `--overwrite` to replace previously generated embeddings.

| Positional Argument | Description |
|---------------------|-------------|
| `models_path` | Path to models: local directory, local archive, HuggingFace repo ID, or URL |
| `dataset_localization` | Key for dataset localization (use `local` for local datasets) |
| `datasets_root` | Name of the dataset directory under `data/` (e.g., `TESTXX`) |
| `short_name` | Name for the output embeddings directory (e.g., `my_run`) |

### Model Sources

The script supports multiple model sources:

| Source | Example |
|--------|---------|
| Local directory | `/path/to/models/` |
| Local archive | `/path/to/models.tar.gz` |
| Hugging Face repo ID | `neurospin/Champollion_V1` |
| Hugging Face URL | `https://huggingface.co/neurospin/Champollion_V1` |
| Remote archive URL | `https://example.com/models.tar.gz` |

### Examples

Download models from Hugging Face and generate embeddings only:

```bash
pixi run python3 src/generate_embeddings.py \
    https://huggingface.co/neurospin/Champollion_V1 \
    local \
    TESTXX \
    my_run \
    --embeddings_only \
    --config_path /path/to/data/TESTXX/derivatives/champollion_V1/configs/dataset
```

Reuse previously cached models with CPU-only mode:

```bash
pixi run python3 src/generate_embeddings.py \
    /path/to/data/TESTXX/derivatives/champollion_V1/models_cache/Champollion_V1 \
    local \
    TESTXX \
    my_run \
    --embeddings_only \
    --cpu \
    --config_path /path/to/data/TESTXX/derivatives/champollion_V1/configs/dataset
```

Generate embeddings and train classifiers (requires a `subject_labels_file` in the dataset configs):

```bash
pixi run python3 src/generate_embeddings.py \
    neurospin/Champollion_V1 \
    local \
    TESTXX \
    my_run \
    --config_path /path/to/data/TESTXX/derivatives/champollion_V1/configs/dataset
```

### Options

| Option | Description |
|--------|-------------|
| `--config_path` | Path to dataset config directory (generated in step 4) |
| `--embeddings_only` | Only compute embeddings (skip classifier training) |
| `--cpu` | Force CPU usage (disable CUDA) |
| `--overwrite` | Overwrite existing embeddings |
| `--run-cka` | Run CKA coherence test after embeddings |
| `--no-cache` | Force re-extraction of archive (ignore cache) |
| `--split` | Splitting strategy: `random` or `custom` (default: `random`) |
| `--use_best_model` | Use the best model saved during training |
| `--profiling` | Enable Python profiling (cProfile) |
| `--labels` | List of labels for classifiers (default: `['Sex']`) |
| `--classifier_name` | Classifier name (default: `svm`) |

### Archive Caching

When using HuggingFace or archive sources, models are cached in:
```
data/{datasets_root}/derivatives/champollion_V1/models_cache/
```

On subsequent runs with the same HuggingFace repo, the script will still contact HuggingFace to check for updates (but won't re-download unchanged files). To skip this entirely, pass the cached local path directly as `models_path`.

Use `--no-cache` to force a full re-download.

### Output Structure

Embeddings are saved inside each model folder with the naming pattern `{short_name}_{split}_embeddings`:

```
models_cache/Champollion_V1/
    SC-sylv_left/name07-58-00_111/
        my_run_random_embeddings/
            full_embeddings.csv       # One row per subject, columns are embedding dimensions
            train_embeddings.csv
            val_embeddings.csv
            test_embeddings.csv
    SC-sylv_right/name06-17-02_84/
        my_run_random_embeddings/
            full_embeddings.csv
            ...
    ...  (56 model folds total)
```

To avoid overwriting previous embeddings, use different `short_name` values for each run.

## 6. Combine Embeddings

Combine the per-region embeddings from all 56 model folds into a single output directory:

```bash
pixi run python3 src/put_together_embeddings.py \
    --path_models /path/to/data/TESTXX/derivatives/champollion_V1/models_cache/Champollion_V1/ \
    --embeddings_subpath my_run_random_embeddings/full_embeddings.csv \
    --output_path /path/to/data/TESTXX/derivatives/champollion_V1/embeddings/
```

| Argument | Description |
|----------|-------------|
| `--path_models` | Path to the directory containing model fold directories |
| `--embeddings_subpath` | Relative path to the embeddings CSV **including the filename** (e.g., `my_run_random_embeddings/full_embeddings.csv`) |
| `--output_path` | Directory where all 56 CSV files will be copied |

The subpath is constructed from `{short_name}_{split}_embeddings/full_embeddings.csv`, matching the `short_name` and `split` used in step 5.

### Verification

Check that 56 CSV files were created:

```bash
ls /path/to/data/TESTXX/derivatives/champollion_V1/embeddings/*.csv | wc -l
```

## 7. Generate Visualization Snapshots

Generate visualizations of the pipeline outputs: sulcal graph meshes, cortical tiles masks, and UMAP scatter plots.

### All snapshots at once

```bash
pixi run python3 src/generate_snapshots.py \
    --morphologist_dir /path/to/data/TESTXX/derivatives/morphologist-6.0/ \
    --cortical_tiles_dir /path/to/data/TESTXX/derivatives/cortical_tiles-2026/crops/2mm/ \
    --embeddings_dir /path/to/data/TESTXX/derivatives/champollion_V1/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /path/to/data/TESTXX/derivatives/champollion_V1/snapshots/
```

### Single snapshot type

Use `--sulcal-only`, `--tiles-only`, or `--umap-only` to generate only one type:

```bash
pixi run python3 src/generate_snapshots.py \
    --embeddings_dir /path/to/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /path/to/snapshots/ \
    --umap-only
```

### Options

| Option | Description |
|--------|-------------|
| `--morphologist_dir` | Path to Morphologist output (for sulcal graph snapshots) |
| `--subject` | Subject folder name to visualize (e.g. `sub_0001`). When omitted the first subject found is used. |
| `--acquisition` | Acquisition folder to use (e.g. `wk30`, `wk40`). Required when a subject has multiple segmentations. |
| `--cortical_tiles_dir` | Path to crops/2mm/ directory (for tiles mask snapshots) |
| `--embeddings_dir` | Path to combined embeddings (for UMAP scatter plots) |
| `--reference_data_dir` | Path to pre-trained UMAP models and reference coordinates |
| `--umap_region` | Comma-separated region name(s) to plot (e.g. `FColl-SRh,S.Or.`). Defaults to all regions with available models. |
| `--output_dir` | Directory to save snapshot images |
| `--sulcal-only` | Only generate sulcal graph snapshots |
| `--tiles-only` | Only generate cortical tiles snapshots |
| `--umap-only` | Only generate UMAP scatter plots |
| `--width` / `--height` | Snapshot dimensions (default: 800x600) |

### Disambiguating multiple segmentations

If a subject has several Morphologist acquisitions (e.g. two timepoints `wk30` and `wk40`), the script warns and uses the first one found. Specify the acquisition explicitly to avoid ambiguity:

```bash
pixi run python3 src/generate_snapshots.py \
    --morphologist_dir /path/to/subjects/ \
    --subject sub_0001 --acquisition wk40 \
    --output_dir /path/to/snapshots/ --sulcal-only
```

### UMAP Visualization

The UMAP scatter plots project a new subject's sulcal region embeddings onto pre-trained 2D maps, one per region and hemisphere. Each plot shows a blue reference cloud (42,433 UKBioBank subjects) with the new subject highlighted in red.

By default all regions for which both an embedding CSV and a pre-trained model exist in `reference_data/` are plotted. Use `--umap_region` to restrict the output:

```bash
pixi run python3 src/generate_snapshots.py \
    --embeddings_dir /path/to/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /path/to/snapshots/ \
    --umap-only --umap_region FColl-SRh
```

Pre-trained UMAP artifacts are stored in `reference_data/` and contain no subject identifiers (only anonymous 2D coordinates and fitted model parameters).

## Project Structure

```
champollion_pipeline/
    external/
        champollion_V1/     # Champollion v1 submodule
        cortical_tiles/     # Cortical tiles submodule
    reference_data/         # Pre-trained UMAP models and anonymous reference coords
    src/
        generate_champollion_config.py
        generate_embeddings.py
        generate_morphologist_graphs.py
        generate_snapshots.py
        put_together_embeddings.py
        run_cortical_tiles.py
        train_model.py
    data/                   # Data directory (created by install-all)
    pixi.toml               # Pixi configuration
```

## Testing

Run all tests:

```bash
pixi run test
```

Run specific test categories:

```bash
pixi run test-unit          # Unit tests only
pixi run test-integration   # Integration tests only
pixi run test-smoke         # Smoke tests only
pixi run test-cov           # Tests with coverage report
pixi run test-fast          # Fast tests (stop on first failure)
```

## Dependencies

The pipeline requires:
- Python >= 3.8
- soma-env (BrainVISA environment)
- morphologist
- anatomist
- PyTorch
- huggingface-hub
- transformers

All dependencies are managed through the `pixi.toml` configuration.


# Champollion Pipeline

This pipeline generates Champollion embeddings from T1 MRI images. It processes MRIs through Morphologist to extract sulcal graphs, then uses cortical_tiles to create sulcal regions, and finally generates embeddings using pre-trained Champollion models.

## 1. Installation

### Prerequisites

- [Pixi](https://pixi.sh/) package manager
- Git

### Setup

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
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0
```

### Parallel Processing with soma-workflow

First, configure soma-workflow:

```bash
soma_workflow_gui
```

Set the maximum number of CPUs (e.g., 24) in the "Computing resources" subwindow.

Then run with the `--swf` flag:

```bash
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0 --swf
```

## 3. Generate Sulcal Regions

Use cortical_tiles to extract sulcal regions from Morphologist's graphs:

```bash
pixi run python3 src/run_cortical_tiles.py \
    /path/to/data/TESTXX/ \
    /path/to/data/TESTXX/derivatives/ \
    --path_to_graph "t1mri/default_acquisition/default_analysis/folds/3.1" \
    --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation"
```

### Options

| Option | Description |
|--------|-------------|
| `--sk_qc_path` | Path to QC file (optional) |
| `--njobs` | Number of CPU cores to use (default: auto) |
| `--region-file` | Custom sulcal region configuration file |

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
ls /path/to/data/TESTXX/derivatives/deep_folding/crops/2mm
```

## 4. Generate Champollion Configuration

Create dataset configuration files for Champollion:

```bash
pixi run python3 src/generate_champollion_config.py \
    /path/to/data/TESTXX/derivatives/deep_folding/crops/2mm \
    --dataset TESTXX
```

### Options

| Option | Description |
|--------|-------------|
| `--champollion_loc` | Path to Champollion binaries (default: external/champollion_V1) |
| `--output` | Custom output path for config files |
| `--external-config` | External path for local.yaml (for read-only containers) |

### Read-only Container Support (Apptainer)

When running in a read-only container environment:

```bash
pixi run python3 src/generate_champollion_config.py \
    /path/to/crops \
    --dataset TESTXX \
    --external-config /writable/path/local.yaml
```

## 5. Generate Embeddings

Generate embeddings using pre-trained Champollion models.

### Basic Usage

```bash
pixi run python3 src/generate_embeddings.py \
    /path/to/models \
    dataset_localization \
    datasets_root \
    short_name
```

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

Using Hugging Face models:

```bash
pixi run python3 src/generate_embeddings.py \
    neurospin/Champollion_V1 \
    local \
    TESTXX \
    my_embeddings \
    --embeddings_only \
    --use_best_model
```

Using a local archive with CPU-only mode:

```bash
pixi run python3 src/generate_embeddings.py \
    /path/to/models.tar.gz \
    local \
    TESTXX \
    my_embeddings \
    --cpu \
    --embeddings_only
```

### Options

| Option | Description |
|--------|-------------|
| `--datasets` | List of dataset names (default: ['toto']) |
| `--labels` | List of labels (default: ['Sex']) |
| `--classifier_name` | Classifier name (default: 'svm') |
| `--overwrite` | Overwrite existing embeddings |
| `--embeddings_only` | Only compute embeddings (skip classifiers) |
| `--use_best_model` | Use the best model saved during training |
| `--subsets` | Subsets of data to train on (default: ['full']) |
| `--epochs` | List of epochs to evaluate (default: [None]) |
| `--split` | Splitting strategy: 'random' or 'custom' (default: 'random') |
| `--cv` | Number of cross-validation folds (default: 5) |
| `--cpu` | Force CPU usage (disable CUDA) |
| `--skip-cka` | Skip CKA coherence test after embeddings |
| `--no-cache` | Force re-extraction of archive (ignore cache) |
| `--profiling` | Enable Python profiling (cProfile) |

### Archive Caching

When using archive sources (local or remote), extracted files are cached in:
```
data/{datasets_root}/derivatives/champollion_V1/models_cache/
```

Use `--no-cache` to force re-extraction.

### Embedding Naming

Embedding folders are named using the pattern: `{short_name}_{split}_embeddings`

To avoid overwriting previous embeddings, use different `short_name` values for each run.

## 6. Combine Embeddings

Combine embeddings from all 56 sulcal regions into a single output:

```bash
pixi run python3 src/put_together_embeddings.py \
    /path/to/models \
    short_name \
    /path/to/output
```

Verify that 56 CSV files were created in the output directory.

## Project Structure

```
champollion_pipeline/
    external/
        champollion_V1/     # Champollion v1 submodule
        cortical_tiles/     # Cortical tiles submodule
    src/
        generate_champollion_config.py
        generate_embeddings.py
        generate_morphologist_graphs.py
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


# Usage

The pipeline runs in 6 stages: T1 MRI → sulcal graphs → region crops → config → embeddings → combined embeddings → snapshots.

The default Mask version is `canonical_corrected_26_1` (a breaking change: it was `canonical_25`). Pass `--masks canonical_25` to work on an old dataset or to rerun it with the old masks; `champollion-purge` and `champollion-prune` need `--masks canonical_25` for an old tree. Stage 2 stops with an error when a `canonical_corrected_26_1` mask is missing. UMAP snapshots (stage 6) stay opt-in through `--reference_data_dir` and need a locally fitted reference of the same Mask version. See [Mask versions](#mask-versions).

## Stage 1 — Generate Morphologist Graphs

> **Skip this step** if you already have Morphologist sulcal graph files (`.arg` format) for your subjects.

Morphologist is a tool from the [BrainVISA](https://brainvisa.info) neuroimaging suite that segments T1 MRI images and produces a graph representation of each subject's sulcal folds. It runs inside Pixi's `brainvisa` feature (part of the default environment).

Use the `champollion-morphologist` wrapper rather than calling `morphologist-cli` directly — it always sets `--if`/`--of` explicitly to avoid BIDS filenames being auto-detected as a different input format:

```bash
#    /data/myproject/sourcedata: folder of T1 MRI files (nested folders included)
#    /data/myproject/: dataset root; the Morphologist outputs are written below its derivatives/ folder
pixi run champollion-morphologist /data/myproject/sourcedata /data/myproject/
```

This writes one folder per subject directly under `/data/myproject/derivatives/morphologist-6.0/`: stage 1 writes no `subjects/` level. A Morphologist tree that already exists may have one (`.../morphologist-6.0/subjects/`); see [Morphologist trees](#morphologist-trees).

### Optional: parallel processing on HPC (soma-workflow)

For large cohorts, [soma-workflow](https://brainvisa.info/web/soma-workflow.html) (BrainVISA's HPC job scheduler) can parallelize graph generation across CPU cores or cluster nodes. This is entirely optional — serial processing works fine for small cohorts.

First configure soma-workflow:

```bash
soma_workflow_gui   # Set max CPUs under "Computing resources"
```

Then add `--parallel`, which passes `--swf` to Morphologist:

```bash
pixi run champollion-morphologist /data/myproject/sourcedata /data/myproject/ --parallel
```

## Stage 2 — Extract Sulcal Region Crops

Extract standardized 3D patches around 28 sulcal regions per hemisphere using *cortical_tiles*:

```bash
#    /data/myproject/derivatives/morphologist-6.0: morphologist subjects dir (read-only); pass the subjects/ folder of an existing tree that has one
#    /data/myproject/derivatives/: derivatives root; crops land at {root}/cortical_tiles-2026/crops/{masks_version}/2mm/
pixi run champollion-cortical-tiles \
    /data/myproject/derivatives/morphologist-6.0 \
    /data/myproject/derivatives/ \
    --path_to_graph "t1mri/default_acquisition/*/folds/3.1" \
    --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation"
```

- **First argument** — directory containing one folder per subject (the folder stage 1 writes, `derivatives/morphologist-6.0/`, or the `subjects/` folder of an existing tree that has one). This path may be read-only (e.g. a shared NFS database); the script never writes to it.
- **Second argument** — derivatives parent directory. Crops are written to `{output}/cortical_tiles-2026/crops/{masks_version}/2mm/` (`{masks_version}` defaults to `canonical_corrected_26_1`, see [Mask versions](#mask-versions)).
- **`--path_to_graph` / `--path_sk_with_hull`** — required; there is no default, since the sub-path under a subject's Morphologist output depends on your acquisition/analysis naming.

`--path_to_graph` supports wildcards (`*`) for variable path segments (e.g. an acquisition or analysis label that differs per subject).

**Verify** that 28 sulcal region folders were created:

```bash
ls /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm
```

### Whole-brain volume

After the per-region crops, the same run fuses the left and right hemisphere skeletons into a single whole-brain volume and strips the ventricle from it. The result is written to `{output}/cortical_tiles-2026/whole_brain/F/` (one volume per subject). The ventricle is located from each subject's labelled Morphologist graph. By default the session is chosen per subject: the first of `deepcnn_session_auto` and `default_session_auto` that holds a labelled graph; `--labelling_session` forces one session for all subjects. Subjects with no labelled graph in the session(s) searched (sulcal recognition is required: typically Morphologist was run without it) are left out of the whole-brain output with one warning naming them; their crops are unaffected and the stage still exits 0.

If crop generation fails (the cortical_tiles subprocess exits with a non-zero code), the command returns that code immediately: mask `.npy` generation and whole-brain fusion are skipped for that run, so no whole-brain volume is built from incomplete hemisphere data.

### QC File

To skip subjects with failing quality control, pass a tab-separated file with `participant_id` and `qc` columns (1 = keep, 0 = skip):

```
participant_id	qc	comments
sub-001         1
sub-002         0   Motion artefact
```

```bash
pixi run champollion-cortical-tiles \
    /data/myproject/derivatives/morphologist-6.0 \
    /data/myproject/derivatives/ \
    --path_to_graph "t1mri/default_acquisition/*/folds/3.1" \
    --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation" \
    --sk_qc_path /path/to/qc.tsv
```

### Skeleton seed

Crop generation re-skeletonizes each subject at 2 mm, and that step is random. Its seed comes from the `skel_seed` key of `{output}/pipeline_loop_2mm.json`; the shipped template sets it to 42 and is copied to `{output}/` only when no copy exists there. Two runs with the same input and the same skeleton seed produce identical crops.

Datasets whose crops were built before the `skel_seed` key existed used clock-seeded skeletons: their crops, and the embeddings computed from them, differ from those of a seeded run. Regenerate them (with `--overwrite`) before comparing them with new runs.

### Mask versions

| Version | Description |
|---------|-------------|
| `canonical_25` | Original masks, used for Champollion V1. |
| `canonical_corrected_26_1` | Revised labeling with reduced artifacts. **Default.** |

**Breaking change:** the default Mask version is now `canonical_corrected_26_1` (it was `canonical_25`). Every stage that takes `--masks`, and the `masks_version` field of the pipeline config, uses it when nothing else is given.

- A rerun of a dataset built with the old default writes a new versioned tree (`crops/canonical_corrected_26_1/`, `champollion_V1/canonical_corrected_26_1/`) next to the old one; the old tree is not modified.
- `canonical_25` stays selectable: pass `--masks canonical_25` to work on an old dataset or to rerun it as before.
- `champollion-purge` and `champollion-prune` act on `crops/<masks>/2mm` for `--masks` (default `canonical_corrected_26_1`) and leave other `crops/<version>/` trees untouched. For a tree built with `canonical_25`, pass `--masks canonical_25`.
- Stage 2 (cortical tiles) stops with an error when a `canonical_corrected_26_1` mask is missing; it writes no crops in that case.
- In stage 4 (embeddings), `--masks` follows `--model-version` or the local models folder name, so a `canonical_25` model set keeps reading `canonical_25` crops; the default applies only when neither names a version.
- UMAP snapshots stay opt-in through `--reference_data_dir` and need a locally fitted UMAP reference of the same Mask version as the embeddings; no reference is shipped.

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--path_to_graph` | **Required.** Glob pattern to `.arg` graph files, relative to the subjects dir. |
| `--path_sk_with_hull` | **Required.** Skeleton-with-hull sub-path, relative to the subjects dir. |
| `--sk_qc_path` | Path to QC TSV file |
| `--sk_qc_input_only` | Limit the input QC to the subjects of the input: an id of the `--sk_qc_path` file with no input subject folder gets no `input_qc.tsv` row. For a run on part of the cohort the file covers. |
| `--njobs` | Number of CPU cores (default: auto) |
| `--region-file` | Accepted but currently has no effect (dead code upstream) — do not rely on it. |
| `--input-types` | Input types to generate: `skeleton` and/or `foldlabel`. Default: `skeleton foldlabel`. |
| `--masks` | Mask version tag; crops are read from `<datasets_root>/derivatives/<cortical_version>/crops/<masks>/2mm`. See the `--masks` row in the embeddings options table for the default |
| `--regions` | Restrict to specific sulcal regions (space-separated) |
| `--skel-seed` | Skeleton seed for the 2 mm re-skeletonization. Without the option, the `skel_seed` key of `{output}/pipeline_loop_2mm.json` sets the seed. Pipeline config: `dataset.skel_seed`; null (the default) leaves the `pipeline_loop_2mm.json` value in effect. |
| `--labelling_session` | Morphologist labelling session whose labelled graphs locate the ventricle for whole-brain removal (e.g. `0_auto`). Default: per subject, the first of `deepcnn_session_auto` and `default_session_auto` that holds a labelled graph. |
| `--overwrite` | Regenerate crops even if crops for the requested mask version already exist; without it the run keeps crops holding `crops_complete.marker` and regenerates crops without it, with a warning. Also removes legacy unversioned `crops/2mm/` crops when they conflict with versioned ones. |

</details>

## Stage 3 — Generate Champollion Configuration

> **Only needed for training** a new encoder. `champollion-embeddings` (next step) reads crop paths directly from `models_path`/`datasets_root` and never reads these YAML files — skip this step if you're only generating embeddings from a pre-trained model.

Create the YAML configuration files that link your dataset's crop paths to the Champollion model:

```bash
#    /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm: path to 2mm crops (stage 2 output)
#    --dataset myproject: dataset name used in YAML and paths
pixi run champollion-config \
    /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm \
    --dataset myproject
```

By default, YAMLs are written to `/data/myproject/derivatives/champollion_V1/configs/` — the dataset's own derivatives tree. Pass `--output <dir>` to choose a different configs root; region YAMLs then land at `{output}/dataset/{dataset}/`. Use `--config-dir <dir>` in `champollion-train` when the configs root is not the default.

This writes region YAML files to `configs/dataset/myproject/` and sets `dataset_folder` in `dataset_localization/local.yaml` so the model knows where your data lives. This config directory is consumed by `champollion-train` (see "Console scripts" in the README), not by `champollion-embeddings`.

### Non-standard derivatives layout

If your crops live outside `derivatives/cortical_tiles-2026/` (e.g. a legacy dataset under `deep_folding-2025/`), add `--external_crops`:

```bash
#    /data/myproject/derivatives/deep_folding-2025/crops/2mm: legacy crop path
#    --external_crops: use the exact crop path instead of the standard derivatives layout
pixi run champollion-config \
    /data/myproject/derivatives/deep_folding-2025/crops/2mm \
    --dataset myproject \
    --external_crops \
    --output /data/myproject/derivatives/champollion_V1/configs
```

### Read-only containers (Apptainer / Docker)

Because configs are written to the dataset derivatives tree by default, no special flag is needed for read-only pipeline directories. If you still need `local.yaml` in a custom location, use `--external-config`:

```bash
pixi run champollion-config \
    /path/to/crops/2mm \
    --dataset myproject \
    --external-config /writable/path/configs/dataset_localization/local.yaml  # write local.yaml to an explicit path
```

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--champollion_loc` | Path to champollion_V1 checkout (default: `external/champollion_V1`). Read only by default — never written to unless `--output` / `--external-config` point inside it. |
| `--output` | Configs root; region YAMLs land at `{output}/dataset/{dataset}/`. Default: `<D>/<dataset>/derivatives/champollion_V1/configs`. |
| `--external_crops` | Use the exact crop path instead of assuming the standard derivatives layout. |
| `--external-config` | Where to write `local.yaml`: a directory (file lands at `{dir}/dataset_localization/{localization}.yaml`) or a file path. Default: `{configs root}/dataset_localization/`. |

</details>

## Stage 4 — Generate Embeddings

Run the pre-trained Champollion encoders across every region fold found under `models_path` (28 regions × 2 hemispheres for the default model set). Models are downloaded automatically from Hugging Face on first run and cached locally.

```bash
#    neurospin/Champollion_V1: model source: HF repo ID, local path, or archive
#    /data/myproject/: dataset root (parent of derivatives/); subjects list defaults to {datasets_root}/participants.tsv
#    --masks canonical_25: crops mask version (must match stage 2)
pixi run champollion-embeddings \
    neurospin/Champollion_V1 \
    /data/myproject/ \
    --masks canonical_25
```

Each region fold writes a `full_embeddings.csv` (one row per subject, columns = embedding dimensions) under:
```
{datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/{region}/full_embeddings.csv
```
Override this location with `--output`.
Earlier versions wrote this output to a sibling `*embeddings` directory next to the dataset root; those
outputs are not found automatically, so pass that directory explicitly to `champollion-combine` or re-run this step.

To re-run on an existing dataset, add `--overwrite`.

A models path holding two or more models of one region (model folders below the region folder, see [Models of one region](#models-of-one-region)) is embedded in one run, with one output per model id: `<output>/<region>/<model id>/full_embeddings.csv`. `--per-model-folders` gives that folder to a single model too, and `--checkpoint-epoch <id>=N` sets the epoch of one model id:

```bash
#    neurospin/Champollion_V1: models path; use a local folder to embed several models of one region
pixi run champollion-embeddings \
    neurospin/Champollion_V1 \
    /data/myproject/ \
    --masks canonical_25 \
    --per-model-folders \
    --checkpoint-epoch 80 \
    --checkpoint-epoch run_b=60
```

In a pipeline Config file the same settings are the keys `dataset.per_model_folders` (boolean) and `dataset.checkpoint_epochs` (a list of `<id>=N` strings); `dataset.model` restricts the combine stage to one model id. `champollion-pipeline` passes them on to stages 4 and 5.

### Checkpoint selection

Each region is evaluated with the Lightning checkpoint `logs/lightning_logs/version_*/checkpoints/epoch=N-step=*.ckpt` of its model directory, N being `--checkpoint-epoch` (default `80`, 0-based; pipeline config `dataset.checkpoint_epoch`). Every `version_*` directory is searched; when several checkpoints match, the highest step wins.

As an interim measure, `logs/best_model_weights.pt` stands in for a region lacking the epoch-N checkpoint. The embeddings stage prints one WARNING line for each region evaluated with `best_model_weights.pt`.

A region holding neither `logs/best_model_weights.pt` nor the epoch-N checkpoint stops the embeddings stage before any region is evaluated; the error names every such region.

The `--use_last_checkpoint` option has been removed: `champollion-embeddings` exits with status 2 when given it. In an existing pipeline configuration file, the keys `dataset.use_last_checkpoint` and `dataset.use_best_model` are ignored with a warning.

### Models of one region

A models path can hold the folders of two or more trained models of one region, for example several training runs of `SC-sylv_left`. Each model gets a **model id**: the folder path between the region folder and the model folder. The region folder is the nearest folder at or above the model folder, below `models_path`, whose name ends in `_left` or `_right`. A model folder that is itself a region folder, or that has no such folder above it, has an empty model id.

```
<models_path>/<Region>_<side>/<name>/logs/...     model id = <name>
<models_path>/<Region>_<side>/<a>/<b>/logs/...    model id = <a>/<b>
<models_path>/<Region>_<side>/logs/...            model id empty (one model per region)
```

The alternative layout keeps one model per region folder and uses one models_path per variant: run the stage once for each models_path, with a distinct `--output` for each.

Two models of one region with the same model id are refused: the run stops with an error naming both model folders.

One run embeds every model and writes one file per model. A region holding two or more models, or any run with `--per-model-folders`, writes each file with a model id to:

```
<output>/<region>/<model id>/full_embeddings.csv
```

A region holding one model, without `--per-model-folders`, keeps the flat `<output>/<region>/full_embeddings.csv`. `--regions` and `--overwrite` apply to each model. Pass `--checkpoint-epoch <id>=N` to set the epoch of one model id; the other models keep the default epoch, and an id matching no model stops the run with an error listing the known model ids.

### Model sources

You can point to different model sources:

| Source | Example value |
|--------|---------------|
| Hugging Face repo | `neurospin/Champollion_V1` |
| Cached local directory | `/data/myproject/derivatives/champollion_V1/models_cache/Champollion_V1` |
| Local archive | `/path/to/models.tar.gz` |
| Remote archive URL | `https://example.com/models.tar.gz` |

Downloaded and extracted models are cached in `{datasets_root}/derivatives/champollion_V1/models_cache/` by default (override with `--models-cache`). Pass the cached path directly on subsequent runs to avoid network checks. Use `--no-cache` to force a full re-download.
In a pipeline Config file the cache directory is the key `dataset.models_cache`, which `champollion-pipeline` passes on to this stage.

<details>
<summary>All options</summary>

| Option | Description |
|--------|-------------|
| `--model-version` | Model set to use: the subfolder downloaded from Hugging Face (e.g. `canonical_25`). With a local `models_path` nothing is downloaded, but the value still sets the `--masks` default and is checked against `--masks`. |
| `--masks-version` | Deprecated alias of `--model-version` (same value); prints a FutureWarning. Giving both with different values exits with status 2. |
| `--masks` | Cortical tiles mask version used as the crops subdirectory: crops are read from `<datasets_root>/derivatives/<cortical_version>/crops/<masks>/2mm`. Default: the `--model-version` value when given; else, when `models_path` is a local folder, that folder's name if it is a known Mask version (otherwise exit 2 with one ERROR, pass `--masks`); `canonical_corrected_26_1` only when `models_path` is not a local directory. |
| `--allow-version-mismatch` | Run when `--masks` differs from `--model-version`, or from the `masks` of a local `model_set.json`, with a warning instead of exiting with status 2. |
| `--regions` | Restrict to specific region names (space-separated). Default: all regions found in `models_path`. |
| `--output` | Output base directory override. Default: `{datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/`. |
| `--subjects` | Path to subjects CSV with a `Subject` column. Default: `{datasets_root}/participants.tsv`. |
| `--cpu` | Force CPU usage (disable CUDA) |
| `--checkpoint-epoch` | Training epoch N (0-based) of the Lightning checkpoint evaluated for each region (default: `80`). Repeatable: a bare `N` sets the default epoch (at most one), `<id>=N` sets the epoch of the models with that model id (see "Models of one region"). Pipeline config: `dataset.checkpoint_epoch`, `dataset.checkpoint_epochs`. |
| `--per-model-folders` | Write the embeddings of each model with a model id to the folder `<region>/<model id>/`, even a single model of its region (by default only a region holding two or more models does). Pipeline config: `dataset.per_model_folders`. |
| `--overwrite` | Recompute embeddings that already exist on disk |
| `--no-cache` | Force re-extraction of the archive (ignore cache) |
| `--models-cache` | Directory for downloaded and extracted models. Default: `{datasets_root}/derivatives/champollion_V1/models_cache/`. A relative path resolves against the current working directory. Pipeline config: `dataset.models_cache`. |
| `--run-cka` | Run CKA coherence test after embeddings |
| `--cortical_version` | Override the cortical tiles derivatives folder name (default: `cortical_tiles-2026`); crops are read from `<datasets_root>/derivatives/<cortical_version>/crops/<masks>/2mm` |
| `--legacy` | Shorthand for `--cortical_version deep_folding-2025` (for older datasets) |
| `--profiling` | Enable Python profiling (cProfile) |

</details>

### Model set marker: `model_set.json`

A model set folder can carry a `model_set.json` marker that says which crops its models were trained on:

```json
{
  "model_version": "canonical_25",
  "masks": "canonical_25",
  "checkpoint_epoch": 80
}
```

`model_version` and `masks` are required; `checkpoint_epoch` is optional.

- **Local model folder:** when `models_path` is a directory with a `model_set.json`, its `masks` must equal `--masks`; otherwise the run stops with exit status 2, unless `--allow-version-mismatch` is given (then a warning is printed and the run goes on). An unreadable marker always stops the run. A local folder without the marker runs with a warning that it cannot be checked.
- **Hugging Face download:** after downloading a model set subfolder, the pipeline writes a `model_set.json` into the cached folder (`model_version` and `masks` set to the subfolder) unless the repository already ships one, so the cached folder can later be checked as a local model folder.

### Run record: `provenance.json`

Each run writes `provenance.json` next to the per-region outputs (in the `--output` directory). It records:

- `masks`: the crops mask version used;
- `model_version`: from the model set's `model_set.json` when present, else `--model-version`;
- `checkpoint_epoch`: the `--checkpoint-epoch` default epoch used;
- `models_source`: where the models came from (local path, Hugging Face repo and subfolder, or archive URL);
- `models`: written only when the run holds two or more models of one region or uses `--per-model-folders`; a list with one object per model, with the keys `region`, `model_id`, `checkpoint_epoch` (the epoch of that model) and `weights_source` (the weights file used, `null` when the model has none);
- `pipeline_commit`: the HEAD commit of the repository rooted at the pipeline code directory, else the commit in that directory's `COMMIT` file, else `null`; never the commit of an enclosing repository.

### Excluding `models_cache/` from dataset sync and archives

Downloaded and extracted models are cached inside the dataset tree, under `{datasets_root}/derivatives/champollion_V1/models_cache/`. The cache is multi-GB and regenerable (the next embeddings run re-creates it), so leave it out when you sync or archive the dataset:

```bash
rsync -a --exclude='/derivatives/champollion_V1/models_cache/' /data/myproject/ backup-host:/data/myproject/
tar --exclude='derivatives/champollion_V1/models_cache' -czf myproject.tar.gz -C /data myproject
```

For a read-only dataset, pass `--models-cache PATH` to `champollion-embeddings` to keep the cache outside the dataset tree.

## Stage 5 — Combine Embeddings

Collect the per-region embedding CSVs into a single output directory:

```bash
#    /data/myproject/derivatives/champollion_V1/canonical_25/region_embeddings/: region_embeddings/ dir written by stage 4
#    --output_path /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/: destination for the combined CSVs
pixi run champollion-combine \
    /data/myproject/derivatives/champollion_V1/canonical_25/region_embeddings/ \
    --output_path /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/
```

This copies each region's `full_embeddings.csv` to `{output_path}/{region}_embeddings.csv`.

When the embeddings step output holds model ids (see [Models of one region](#models-of-one-region)), each model id is combined to `<output_path>/<model id>/<region>_embeddings.csv`, and a region holding one model is copied into the folder of each model id. Add `--model <id>` to combine only the named model id, which writes only `<output_path>/<id>/`; an id matching no model id stops the run with an error. A region folder holding `full_embeddings.csv` beside model id folders is refused.

```bash
pixi run champollion-combine \
    /data/myproject/derivatives/champollion_V1/canonical_25/region_embeddings/ \
    --output_path /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --model run_b
```

**Verify** the CSV files were created (one per region):

```bash
ls /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/*.csv | wc -l
```

## Stage 6 — Generate Visualization Snapshots

Generate visualizations: sulcal graph meshes, cortical tile masks, and UMAP scatter plots projecting your subjects onto a pre-trained reference embedding space.

```bash
#    --embeddings_dir /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/: stage 5 output
#    --reference_data_dir /data/myproject/reference_data/: pre-trained UMAP models (not shipped — see below)
#    --output_dir /data/myproject/derivatives/champollion_V1/canonical_25/snapshots/: where to write images
pixi run champollion-snapshots \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --reference_data_dir /data/myproject/reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/canonical_25/snapshots/
```

Add `--morphologist_dir` and `--cortical_tiles_dir` to also generate mesh and mask snapshots:

```bash
#    --morphologist_dir /data/myproject/derivatives/morphologist-6.0/: for sulcal graph mesh snapshots
#    --cortical_tiles_dir /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm/: for tile mask snapshots
pixi run champollion-snapshots \
    --morphologist_dir /data/myproject/derivatives/morphologist-6.0/ \
    --cortical_tiles_dir /data/myproject/derivatives/cortical_tiles-2026/crops/canonical_25/2mm/ \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --reference_data_dir /data/myproject/reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/canonical_25/snapshots/
```

Use `--sulcal-only`, `--tiles-only`, or `--umap-only` to generate only one snapshot type.

When `--embeddings_dir` holds one folder per model id (the stage 5 output for several models), add `--model <id>` to choose the model id folder `<embeddings_dir>/<id>` that the UMAP snapshots render. Without `--model`, a flat folder or the only model id folder is used; two or more model id folders are refused with an error naming each one.

```bash
pixi run champollion-snapshots \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --reference_data_dir /data/myproject/reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/canonical_25/snapshots/ \
    --model run_b
```

### UMAP Visualization

UMAP scatter plots project each subject's sulcal embeddings onto pre-trained 2D maps (one per region and hemisphere). Each plot shows a reference cloud of embeddings from a large pre-trained cohort, with the new subject highlighted. `--reference_data_dir` points at a directory of pre-trained UMAP artifacts (`umap_{region}_{hemi}.pkl` / `_coords.npy`) that contain no subject identifiers; this directory is **not shipped in the repository** (it's gitignored) — obtain it separately and pass its path explicitly. UMAP snapshots need a locally fitted UMAP reference of the same Mask version as the embeddings; without `--reference_data_dir` the UMAP step is skipped with one warning.

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
#    --subject sub-001 --acquisition wk40: acquisition label to disambiguate multiple time points
pixi run champollion-snapshots \
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
| `--cortical_tiles_dir` | Path to `crops/{masks_version}/2mm/` (for tiles mask snapshots) |
| `--embeddings_dir` | Path to combined embeddings (for UMAP scatter plots) |
| `--reference_data_dir` | Path to pre-trained UMAP models and reference coordinates (not shipped in the repo, gitignored — obtain separately) |
| `--umap_region` | Comma-separated region name(s) to plot |
| `--output_dir` | Directory to save snapshot images |
| `--sulcal-only` / `--tiles-only` / `--umap-only` | Generate only one snapshot type |
| `--width` / `--height` | Snapshot dimensions (default: 800×600) |
| `--tiles_level` | Cortical tiles level to visualize |
| `--champollion_data_root` | Directory containing `mask/2mm/regions/meshes/` for tile snapshots (default: cortical_tiles config, overridable via `CHAMPOLLION_DATA_ROOT`). Pipeline config: `dataset.champollion_data_root`. |
| `--labelling_session` | Morphologist labelling session whose labelled graph the sulcal snapshot uses (e.g. `deepcnn_session_auto`). Default: per hemisphere, the first of `deepcnn_session_auto` and `default_session_auto` that holds one of its graphs. A hemisphere with no graph in the given session keeps all its graphs. Pipeline config: `dataset.labelling_session`. |
| `--model` | Model id folder of `--embeddings_dir` that the UMAP snapshots render; required when it holds two or more model id folders. |

</details>

## Run logs

Add `--log=PATH` to any stage script to also write its whole output (child processes included) to a file:

```bash
pixi run champollion-embeddings \
    neurospin/Champollion_V1 \
    /data/myproject/ \
    --masks canonical_25 \
    --log=/data/myproject/logs/embeddings.log
```

A bare `--log` writes `<stage>_<YYYYmmdd-HHMMSS>.log` in the stage's default folder:

- `champollion-morphologist`: `<output>/derivatives/morphologist-6.0/.log`
- `champollion-cortical-tiles`: `<output>/cortical_tiles-2026/.log`
- `champollion-config`: `<dataset>/derivatives/champollion_V1/<masks>/.log`
- `champollion-embeddings`: `<datasets_root>/derivatives/champollion_V1/<masks>/.log`
- `champollion-combine`: `<--output_path>/.log`
- `champollion-snapshots`: `<--output_dir>/.log`

A bare `--log` placed just before a positional argument takes that argument as its PATH: use `--log=PATH`, or put a
bare `--log` last.

## Running champollion-pipeline without a Config file

`champollion-pipeline <input> [<output>]` is the nominal run. It needs no Config file: `<input>` (or `--input-path`) is the one
required path, and every other path of the PipelineConfig is inferred from it. A sequential run with an input path and no `--config` enables every
stage unless `--stages` names some.

What the input path holds decides the run. `champollion-pipeline` reads it up to its first T1 file:

- T1 files: `.nii` or `.nii.gz` files, at any depth, so nested T1 folders work. Folders named `derivatives`, `qc` and
  `t1mri` are not searched for T1 files, and below a `sub-*` folder a `sub-*` file must end in `_T1w`. The Dataset root
  (`dataset.datasets_root`) is the input path itself, and stage 1 writes the Morphologist outputs below it.
- Morphologist outputs: a tree holding `.arg` graphs under `t1mri` folders. Stage 1 is skipped (one INFO line), the
  subjects folder is found in the tree (see [Morphologist trees](#morphologist-trees)), and the Dataset root is the
  parent of the nearest `derivatives` folder above that subjects folder.
- Neither kind (or not a folder): the run logs one error and exits with code 1.
- Both kinds, or T1 files below a folder named `sourcedata`: at a terminal `champollion-pipeline` asks which kind to use; without
  one it exits with code 1 and names `--input-kind t1` and `--input-kind morphologist`. Pass `--input-kind` to
  answer in advance (the replay config keeps it).
- Several subjects folders in one Morphologist input: the same prompt, or an error naming the folders; pass the one
  to use as `--input-path` or `--morphologist-graphs`.

`<output>` (or `--output-path`, or `--datasets-root`) sets the Dataset root, `dataset.datasets_root`, and wins over
the inferred one. Two forms that name different paths stop the run with code 2. A Morphologist input whose Dataset root
cannot be inferred (a single subject, or a subjects folder below no `derivatives` folder) gives a warning that names
`--datasets-root`.

From the Dataset root and the input, `champollion-pipeline` fills the fields that are still unset, among them
`dataset.morphologist_graphs`, `cortical_tiles_output`, `crops_path`, `embeddings_path`, `snapshots_path`,
`outputs_path`, `path_to_graph` and `path_sk_with_hull`, and it names the Dataset after the root folder. `models_path`
and `sk_qc_path` are never inferred.

### An argument for every config field

Every PipelineConfig and Dataset field has an optional argument, named after the field with dashes: `--njobs`,
`--masks-version`, `--embeddings-only`, `--reference-data-path`, `--mode`, `--n-workers` and so on (`champollion-pipeline
--help` lists them all). Exceptions: `name` is `--dataset-name`, `profile_dir` is `--profile`, and the Dataset
`input_path` and `datasets_root` also have the `INPUT` and `OUTPUT` forms above. A flag is `--x` or `--no-x`, a list
takes several values, and `--stages` replaces the whole selection. A field given on the command line wins over the
`--config` file and the inference.

Four fields have no argument, as they are secrets or machine paths, and `--env [PATH]` sets them from a `.env` file
(bare `--env`: `./.env`): `HF_TOKEN`, `CHAMPOLLION_ROOT_PATH`, `CHAMPOLLION_CONFIG_PATH` and
`CHAMPOLLION_DEPENDENCIES` (a JSON object of stage names to lists of stage names). `CHAMPOLLION_DATA_ROOT` sets
`dataset.champollion_data_root`, whose `--champollion-data-root` argument wins over it. The same variables in the
process environment win over the file; both win over the `--config` file and lose to the command line. A missing
file or a bad value exits with code 2.

Stage 6 renders the UMAP snapshots only when `--reference-data-path` is given; without it the run logs a warning, skips
the UMAP snapshots, and still renders the sulcal graph and cortical tiles snapshots.

### Morphologist trees

Two layouts of a Morphologist tree are read:

```text
<root>/derivatives/morphologist-6.0/            # written by stage 1: no subjects/ level
    sub-001/t1mri/default_acquisition/...
    sub-002/t1mri/default_acquisition/...
    qc/                                         # not a subject unless it holds a t1mri folder
<root>/derivatives/morphologist-6.0/subjects/   # an existing tree: with a subjects/ level
    sub-001/t1mri/default_acquisition/...
    qc/
```

The subjects folder is the folder whose children are the subject folders: `morphologist-6.0` in the first layout,
`subjects` in the second. A subject folder holds `t1mri`, or `ses-*/t1mri` with sessions. A folder named `qc` beside
the subject folders is read as a subject only when it holds a `t1mri` folder, and a folder starting with a dot never is.
Give `<input>` as the tree root, the subjects folder or the `morphologist-6.0` folder: `champollion-pipeline` finds the subjects
folder, or takes `--morphologist-graphs`. When `path_to_graph` is unset, it is read from the subjects' graph folders
(`t1mri/<acquisition>/<analysis>/folds/<version>`); graphs in several such folders stop the inference with a warning
that names `--path-to-graph`.

## Replay config

A Config file is an option, not a requirement (see [Running champollion-pipeline without a Config file](#running-champollion-pipeline-without-a-config-file)).

Every `champollion-pipeline` run writes its effective PipelineConfig to `replay_<YYYYmmdd-HHMMSS>.yaml` in the run's logs folder
(`log_dir` when set, else `<outputs_path>/logs`). The run is a sequential run, a streaming run or a `--dry-run`.
The file merges the command line, the inference and the `--config` YAML, with path fields made absolute.

To run the same configuration again, pass the replay file as the only argument:

```bash
champollion-pipeline --config <logs folder>/replay_<YYYYmmdd-HHMMSS>.yaml
```

The replay file leaves out four fields: `root_path`, `dependencies`, `dataset.config_path` and `dataset.hf_token`. The
Hugging Face token is a secret and is never written to a file. A run that read them through `--env` needs `--env`
again (see the `HF_TOKEN` and `CHAMPOLLION_*` variables of the `.env` file).

If the replay file cannot be written (a read-only logs folder, for example), the run logs one WARNING that names the
path and the error, and the run goes on.

## Streaming mode

`champollion-pipeline --mode streaming` runs inference scan by scan instead of stage by stage: each scan's crops are extracted
while earlier scans are already being embedded. It only computes embeddings with existing models; training stays a
batch run.

```bash
champollion-pipeline --mode streaming --config my_pipeline.yaml --n-workers 8 --log
```

The configuration must set `dataset.embeddings_only: true`; without it the run logs an error and exits with code 1
before any scan is read.

A non-empty `dataset.embeddings_path` is refused: the run exits with code 1, since streaming always writes the
versioned default embeddings folder.

A Config file is not needed: `champollion-pipeline <input> [<output>] --mode streaming --embeddings-only` infers the paths
from a Morphologist input as in [Running champollion-pipeline without a Config file](#running-champollion-pipeline-without-a-config-file), and
the replay config of the run is written as for any run.

The subjects directory read is `dataset.morphologist_graphs`, else the folder found in `dataset.input_path`, else
the one stage 1 writes below `dataset.datasets_root`. It must already hold
Morphologist graphs in the `<subject>/t1mri/<acquisition>` layout: a streaming run executes no Morphologist stage,
it starts at cortical_tiles.

Options:

- `--n-workers N`: number of producer worker processes; default one per CPU.
- `--worker-timeout SECONDS`: a producer worker running longer than this (default 7200 s) ends FAILED.
- `--dry-run`: lists the scans that would run and processes nothing; the dataset config is not written.

Logs. The log directory is chosen in this order:

- with `--log DIR`, DIR is the streaming log directory;
- a bare `--log` gives `./.log/champollion_pipeline_<YYYYmmdd-HHMMSS>/`, next to the run log of the same
  name;
- without `--log`, the YAML `log` value, else `log_dir`, else `<outputs_path>/logs`.

Inside it, each scan's producer worker writes `<log directory>/<subject>/t1mri/<acquisition>/worker.log`, and the
inference consumer writes `<log directory>/consumer.log`. Each FAILED scan is logged with the path of its log.

Resume. Rerunning the same command picks up where the last run stopped:

- the worker of a scan is skipped when its tiles folder holds a `tiles_complete.json` completion marker recording the
  run's masks version;
- the embedding of a scan is skipped when each region's `full_embeddings.csv` already holds a row for it.

Exit code. The run exits with code 1 when at least one scan ends FAILED (worker or consumer), when the inputs are
refused, or when the combine fails; it exits with code 0 when no scan ends FAILED and the combine succeeds.

Models. The resident engine embeds each model of the models path, as in stage 4, and writes one output per model id (`full_embeddings.csv` in `<region>/<model id>/` for a region holding two or more models, else in `<region>/`); the combine step then writes one combined folder per model id.

Limits: inference only (no training); no Morphologist stage; the consumer gives up after 7200 s without a newly ready
scan.

The profiling of the embedding path behind this mode is in [Streaming profile](streaming_profile.md).

## Running Tests

```bash
pixi run test               # All tests
pixi run test-unit          # Unit tests only
pixi run test-smoke         # Smoke tests only
pixi run test-cov           # Tests with coverage report
```

# Usage

The pipeline runs in 6 stages: T1 MRI → sulcal graphs → region crops → config → embeddings → combined embeddings → snapshots.

The default Mask version is `canonical_corrected_26_1` (a breaking change: it was `canonical_25`). Pass `--masks canonical_25` to work on an old dataset or to rerun it with the old masks; `champollion-purge` and `champollion-prune` need `--masks canonical_25` for an old tree. Stage 2 stops with an error when a `canonical_corrected_26_1` mask is missing. UMAP snapshots (stage 6) stay opt-in through `--reference_data_dir` and need a locally fitted reference of the same Mask version. See "Mask versions" in the README.

## champollion-pipeline reference

`champollion-pipeline` runs the six stages in one command, from T1 files or from Morphologist outputs.
`champollion-pipeline --help` prints the options below.

### Examples

From T1 files (stage 1 runs Morphologist and writes below the input folder, which is also the Dataset root):

```bash
champollion-pipeline /data/myproject/rawdata --input-kind t1 --njobs 8 --log
```

From Morphologist outputs (stage 1 is skipped; the Dataset root is the parent of the `derivatives` folder):

```bash
champollion-pipeline /data/myproject/derivatives/morphologist-6.0 --input-kind morphologist \
    --masks-version canonical_25 --reference-data-path reference_data/
```

`--input-kind` is optional: without it the kind is detected from the input (see
[Running champollion-pipeline without a Config file](#running-champollion-pipeline-without-a-config-file)).

### Options

Each option sets the Config file key of its row; a key written `dataset.<field>` sits in the `dataset:` block of the
Config file, and a dash means the option sets no key. The command line wins over the `.env` values and the
`--config` file. The Default column gives the value in force when the option is absent: first the Config file
default (a Config file without the key holds it), then, where it differs, what a run without `--config` uses
(`<checkout>` is the champollion_pipeline folder, the `root_path` of the run). A flag has an on form and an off form,
both in the Option column; a list option takes one or more values.

| Option | Argument | Default | Config key | Effect |
|---|---|---|---|---|
| `INPUT` | `INPUT` | none: no input path | `dataset.input_path` | Input to process: a folder of T1 files or a Morphologist tree; same as `--input-path`. Without `--config`, every other path is inferred from it and every stage is enabled unless `--stages` names some |
| `OUTPUT` | `OUTPUT` | inferred from `INPUT` | `dataset.datasets_root` | Dataset root that holds `derivatives/`; same as `--output-path` and `--datasets-root`. Two forms naming different paths exit with status 2 |
| `-h`, `--help` | - | off | - | Print the help and exit with status 0 |
| `--config` | `CONFIG` | none: the default PipelineConfig | - | Read the PipelineConfig from this YAML Config file (a replay file works too); the other options override its values |
| `--enable-all-stages` | - | off: the stages of `stages` | `stages` | Turn on all six stages |
| `--generate-config` | `OUTPUT_PATH` | none: run the pipeline | - | Write a Config file template, with the other options applied, to `OUTPUT_PATH` and exit with status 0 without running a stage |
| `--dry-run` | - | off | - | Streaming mode only: list the scans that would run and process none |
| `--data-path` | `DATA_PATH` | empty; without `--config`: `<checkout>/data` | `data_path` | Data folder; without `--config` and without a Dataset root, the Dataset root is `<data_path>/<dataset name>` |
| `--models-path` | `MODELS_PATH` | empty; without `--config`: `<checkout>/models` | `models_path` | Local models folder, archive or Hugging Face repo id that stage 4 and streaming mode embed with, unless `dataset.hf_enabled` is true |
| `--outputs-path` | `OUTPUTS_PATH` | empty; without `--config`: `<checkout>/outputs`; config-less input run: `<Dataset root>/derivatives/champollion_V1` | `outputs_path` | Pipeline output folder: its `logs` folder holds `pipeline.log` and the replay file when `log_dir` is empty; streaming mode writes there |
| `--champollion-v1-path` | `CHAMPOLLION_V1_PATH` | empty; without `--config`: `<checkout>/external/champollion_V1` | `champollion_v1_path` | Path of the champollion_V1 checkout; kept in the Config file, no stage reads it |
| `--stages` | `STAGE [STAGE ...]`, each one of `generate_morphologist_graphs`, `run_cortical_tiles`, `generate_champollion_config`, `generate_embeddings`, `put_together_embeddings`, `generate_snapshots` | `generate_embeddings` only; config-less input run: all six | `stages` | Run only the named stages, in dependency order; the stages it does not name are turned off |
| `--mode` | `{sequential,streaming}` | `sequential` | `mode` | `sequential` runs stage by stage; `streaming` runs inference scan by scan (see [Streaming mode](#streaming-mode)) |
| `--n-workers` | `N` | `0`: one worker per CPU | `n_workers` | Streaming mode only: number of producer worker processes |
| `--worker-timeout` | `SECONDS` | `7200` | `worker_timeout` | Streaming mode only: a producer worker running longer than this on one scan ends FAILED |
| `--stop-on-error`, `--no-stop-on-error` | - | `true` | `stop_on_error` | On: stop at the first failed stage. Off: run the remaining stages |
| `--verbose`, `--no-verbose` | - | `false` | `verbose` | On: verbose output; without `--log-level` the log level becomes `DEBUG` |
| `--parallel`, `--no-parallel` | - | `false` | `parallel` | On: stage 1 schedules Morphologist through soma-workflow |
| `--log-level` | `{DEBUG,INFO,WARNING,ERROR,CRITICAL}` | `INFO` (`DEBUG` with `--verbose`) | `log_level` | Level of the pipeline logger |
| `--log-dir` | `LOG_DIR` | empty: `<outputs_path>/logs` | `log_dir` | Folder of `pipeline.log` and of the replay file |
| `--log-to-file`, `--no-log-to-file` | - | `true` | `log_to_file` | On: write `pipeline.log` in the log folder |
| `--log-to-console`, `--no-log-to-console` | - | `true` | `log_to_console` | On: show the pipeline log on the console |
| `--log` | `[DIR]` | none: no run log files | `log` | Write a run log per stage, `DIR/<stage>_<YYYYmmdd-HHMMSS>.log`; bare: each stage's default log path plus this run's log `./.log/champollion_pipeline_<YYYYmmdd-HHMMSS>.log` |
| `--dataset-name` | `DATASET_NAME` | `example_dataset`; config-less input run: the Dataset root's folder name | `dataset.name` | Dataset name, given to stage 3 |
| `--dataset-localization` | `DATASET_LOCALIZATION` | `local` | `dataset.dataset_localization` | Training setting kept in the Config file; no stage of `champollion-pipeline` reads it |
| `--datasets-root` | `DATASETS_ROOT` | empty; without `--config`: `<data_path>/<dataset name>`; config-less input run: inferred from `INPUT` | `dataset.datasets_root` | Dataset root that holds `derivatives/`; every inferred output path sits below it |
| `--datasets` | `DATASETS [DATASETS ...]` | `example` | `dataset.datasets` | Training setting kept in the Config file; no stage reads it |
| `--labels` | `LABELS [LABELS ...]` | `Sex` | `dataset.labels` | Training setting kept in the Config file; no stage reads it |
| `--input-path` | `INPUT_PATH` | empty: no input path | `dataset.input_path` | Same as `INPUT` |
| `--input-kind` | `INPUT_KIND`: `t1` or `morphologist` | none: detected from the input | `dataset.input_kind` | Answer the input kind question in advance: `t1` (T1 files) or `morphologist` (Morphologist outputs, stage 1 skipped); any other value exits with status 1 |
| `--morphologist-graphs` | `MORPHOLOGIST_GRAPHS` | empty: inferred (the subjects folder of the input, or the one stage 1 writes) | `dataset.morphologist_graphs` | Subjects folder of the Morphologist graphs that stage 2 and streaming mode read |
| `--cortical-tiles-output` | `CORTICAL_TILES_OUTPUT` | empty; config-less input run: `<Dataset root>/derivatives` | `dataset.cortical_tiles_output` | Output root of stage 2 |
| `--crops-path` | `CROPS_PATH` | empty; config-less input run: `<Dataset root>/derivatives/cortical_tiles-<year>/crops/<masks version>/2mm` | `dataset.crops_path` | Crops folder that stages 3 and 6 read |
| `--embeddings-path` | `EMBEDDINGS_PATH` | empty: `<Dataset root>/derivatives/champollion_V1/<masks version>/region_embeddings` | `dataset.embeddings_path` | Per-region embeddings folder, written by stage 4 and read by stage 5; streaming mode refuses a non-empty value |
| `--njobs` | `NJOBS` | `22` | `dataset.njobs` | Number of parallel jobs of stage 2 |
| `--path-to-graph` | `PATH_TO_GRAPH` | `t1mri/default_acquisition/default_analysis/folds/3.3/base`; config-less input run: the folder stage 1 writes (T1 input) or the one read from the tree (Morphologist input) | `dataset.path_to_graph` | Graph folder below each subject, relative to the subject folder |
| `--path-sk-with-hull` | `PATH_SK_WITH_HULL` | `t1mri/default_acquisition/default_analysis/segmentation`; config-less input run: the `segmentation` folder beside the graph folder | `dataset.path_sk_with_hull` | Folder of the skeleton-with-hull volumes below each subject, given to stage 2 |
| `--sk-qc-path` | `SK_QC_PATH` | empty: no QC filter | `dataset.sk_qc_path` | QC TSV file (`participant_id`, `qc`) that stage 2 filters subjects with |
| `--masks-version` | `MASKS_VERSION` | `canonical_corrected_26_1` | `dataset.masks_version` | Mask version of stages 2, 3 and 4; names the versioned output folders |
| `--skel-seed` | `SKEL_SEED` | none: the `skel_seed` recorded in `<derivatives>/pipeline_loop_2mm.json` | `dataset.skel_seed` | Skeleton seed given to stage 2 |
| `--labelling-session` | `LABELLING_SESSION` | empty: stage 2 picks the session per subject | `dataset.labelling_session` | Morphologist labelling session that stages 2 and 6 read |
| `--model` | `MODEL` | empty: every model id | `dataset.model` | Model id that stages 5 and 6 keep |
| `--regions` | `REGIONS [REGIONS ...]` | empty: all 56 regions | `dataset.regions` | Regions that stages 2, 4 and 5 process |
| `--classifier-name` | `CLASSIFIER_NAME` | `svm` | `dataset.classifier_name` | Training setting kept in the Config file; no stage reads it |
| `--overwrite`, `--no-overwrite` | - | `false` | `dataset.overwrite` | On: stage 4 overwrites existing embeddings |
| `--embeddings-only`, `--no-embeddings-only` | - | `false` | `dataset.embeddings_only` | On: inference only, no training; streaming mode requires it |
| `--checkpoint-epoch` | `CHECKPOINT_EPOCH` | `80` | `dataset.checkpoint_epoch` | Checkpoint epoch of the models in stage 4 |
| `--checkpoint-epochs` | `CHECKPOINT_EPOCHS [CHECKPOINT_EPOCHS ...]` | empty | `dataset.checkpoint_epochs` | One `<id>=N` entry per model id: the checkpoint epoch of that model id in stage 4 |
| `--per-model-folders`, `--no-per-model-folders` | - | `false` | `dataset.per_model_folders` | On: stage 4 writes one output folder per model id, even for a single model |
| `--models-cache` | `MODELS_CACHE` | none: `<Dataset root>/derivatives/champollion_V1/models_cache` | `dataset.models_cache` | Cache folder of the downloaded and extracted models of stage 4 |
| `--subsets` | `SUBSETS [SUBSETS ...]` | `full` | `dataset.subsets` | Training setting kept in the Config file; no stage reads it |
| `--epochs` | `EPOCHS [EPOCHS ...]` | one unset epoch | `dataset.epochs` | Training setting kept in the Config file; no stage reads it |
| `--split` | `SPLIT` | `random` | `dataset.split` | Training setting kept in the Config file; no stage reads it |
| `--cv` | `CV` | `5` | `dataset.cv` | Training setting kept in the Config file; no stage reads it |
| `--splits-basedir` | `SPLITS_BASEDIR` | none | `dataset.splits_basedir` | Training setting kept in the Config file; no stage reads it |
| `--idx-region-evaluation` | `IDX_REGION_EVALUATION` | none | `dataset.idx_region_evaluation` | Training setting kept in the Config file; no stage reads it |
| `--short-name` | `SHORT_NAME` | `eval` | `dataset.short_name` | Training setting kept in the Config file; no stage reads it |
| `--hf-enabled`, `--no-hf-enabled` | - | `false` | `dataset.hf_enabled` | On: stage 4 takes its models from the Hugging Face repo `dataset.hf_repo_id` (token `HF_TOKEN`, see `--env`) instead of `models_path` |
| `--hf-repo-id` | `HF_REPO_ID` | none | `dataset.hf_repo_id` | Hugging Face repo id of the models; required when `dataset.hf_enabled` is true |
| `--cpu`, `--no-cpu` | - | `false` | `dataset.cpu` | On: stage 4 runs on the CPU, CUDA off |
| `--bids`, `--no-bids` | - | `false` | `dataset.bids` | On: the input follows the BIDS layout (`subjects/sub-*/ses-*/...`): BIDS-aware scan listing, passed on to stages 1 and 2 |
| `--allow-prefix-ids`, `--no-allow-prefix-ids` | - | `false` | `dataset.allow_prefix_ids` | Temporary. On: stage 2 runs when a subject id is a strict prefix of another (`sub-1` and `sub-10`); its transforms may then be wrong |
| `--snapshots-path` | `SNAPSHOTS_PATH` | empty: `<Dataset root>/derivatives/champollion_V1/<masks version>/snapshots` | `dataset.snapshots_path` | Output folder of stage 6 |
| `--reference-data-path` | `REFERENCE_DATA_PATH` | empty: no UMAP snapshots | `dataset.reference_data_path` | UMAP reference folder of stage 6; without it stage 6 skips the UMAP snapshots with a warning |
| `--champollion-data-root` | `CHAMPOLLION_DATA_ROOT` | none: `CHAMPOLLION_DATA_ROOT` of the `.env` file or the environment, else unset | `dataset.champollion_data_root` | Champollion data folder given to stage 6; wins over `CHAMPOLLION_DATA_ROOT` |
| `--output-path` | `DIR` | inferred from `INPUT` | `dataset.datasets_root` | Same as `OUTPUT` |
| `--env` | `[PATH]` | none: no `.env` file, the process environment only | - | Read `HF_TOKEN`, `CHAMPOLLION_ROOT_PATH`, `CHAMPOLLION_CONFIG_PATH`, `CHAMPOLLION_DEPENDENCIES` and `CHAMPOLLION_DATA_ROOT` from the `.env` file `PATH` (bare: `./.env`); the process environment wins over the file; a missing file or a bad value exits with status 2 |
| `--profile` | `DIR` | none: no profiling unless `CHAMPOLLION_PROFILE` is set | `profile_dir` | Profile this run and its Python children into `DIR` |
| `--profile-mode` | `{cpu,light}` | `cpu` | - | Profiling mode; requires `--profile` |

Every run writes its effective configuration to a replay file in the log folder (see [Replay config](#replay-config));
`champollion-pipeline --config <replay file>` runs it again.

### Exit status

| Exit status | Meaning |
|---|---|
| `0` | Success: every enabled stage completed, or no stage was enabled, or `--generate-config` wrote its template, or `--help` printed the help |
| `1` | Failure: a stage failed, the input was refused (neither kind, an unanswered kind question, a bad `dataset.input_kind`), a streaming guard stopped the run, or a streaming scan or combine failed |
| `2` | Usage error: an unknown option or a bad value, two forms of one path naming different paths, or a `.env` file that is missing or holds a bad value |
| `3` | Partial run, sequential mode only: no stage failed and at least one stage completed partially (some subjects skipped) |

Streaming mode exits with status 0 or 1 only.

## Stage 1 — Generate Morphologist Graphs

> Skip this step if you already have Morphologist `.arg` graph files.

```bash
morphologist-cli sub-001.nii.gz sub-002.nii.gz /data/myproject/ \
    -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0
```

## Stage 2 — Extract Sulcal Region Crops

```bash
pixi run champollion-cortical-tiles \
    /data/myproject/derivatives/morphologist-6.0 \
    /data/myproject/derivatives/
```

## Stage 3 — Generate Champollion Configuration

```bash
pixi run champollion-config \
    /data/myproject/derivatives/cortical_tiles-2026/crops/2mm \
    --dataset myproject
```

YAMLs are written to `/data/myproject/derivatives/champollion_V1/configs/` by default.
Pass `--output <dir>` to choose a different configs root.

## Stage 4 — Generate Embeddings

```bash
pixi run champollion-embeddings \
    neurospin/Champollion_V1 \
    /data/myproject/ \
    --masks canonical_25
```

A models path holding two or more models of one region (model folders below the region folder, as in "Models of one region" in the README) is embedded in one run, with one output per model id: `<output>/<region>/<model id>/full_embeddings.csv`. `--per-model-folders` gives that folder to a single model too, and `--checkpoint-epoch <id>=N` sets the epoch of one model id:

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

Downloaded and extracted models are cached in `{datasets_root}/derivatives/champollion_V1/models_cache/` by default.
Pass `--models-cache PATH` to keep the cache elsewhere, for instance outside a read-only dataset tree; a relative
`PATH` resolves against the current working directory. In a pipeline Config file the same setting is the key
`dataset.models_cache`, which `champollion-pipeline` passes on to this stage. `--no-cache` forces a fresh download.

## Stage 5 — Combine Embeddings

```bash
pixi run champollion-combine \
    /data/myproject/derivatives/champollion_V1/canonical_25/region_embeddings/ \
    --output_path /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/
```

With model ids in the source, each model id is combined to `<output_path>/<model id>/`; `--model <id>` combines only that model id:

```bash
pixi run champollion-combine \
    /data/myproject/derivatives/champollion_V1/canonical_25/region_embeddings/ \
    --output_path /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --model run_b
```

## Stage 6 — Generate Visualization Snapshots

```bash
pixi run champollion-snapshots \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/canonical_25/snapshots/
```

With one folder per model id in `--embeddings_dir`, `--model <id>` chooses the model id folder to render; without it two or more model id folders are refused:

```bash
pixi run champollion-snapshots \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/canonical_25/snapshots/ \
    --model run_b
```

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

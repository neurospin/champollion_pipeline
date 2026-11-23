# Usage

The pipeline runs in 6 stages: T1 MRI → sulcal graphs → region crops → config → embeddings → combined embeddings → snapshots.

The default Mask version is `canonical_corrected_26_1` (a breaking change: it was `canonical_25`). Pass `--masks canonical_25` to work on an old dataset or to rerun it with the old masks; `champollion-purge` and `champollion-prune` need `--masks canonical_25` for an old tree. Stage 2 stops with an error when a `canonical_corrected_26_1` mask is missing. UMAP snapshots (stage 6) stay opt-in through `--reference_data_dir` and need a locally fitted reference of the same Mask version. See "Mask versions" in the README.

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

## Stage 6 — Generate Visualization Snapshots

```bash
pixi run champollion-snapshots \
    --embeddings_dir /data/myproject/derivatives/champollion_V1/canonical_25/embeddings/ \
    --reference_data_dir reference_data/ \
    --output_dir /data/myproject/derivatives/champollion_V1/canonical_25/snapshots/
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

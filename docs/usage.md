# Usage

The pipeline runs in 6 stages: T1 MRI → sulcal graphs → region crops → config → embeddings → combined embeddings → snapshots.

## Stage 1 — Generate Morphologist Graphs

> Skip this step if you already have Morphologist `.arg` graph files.

```bash
morphologist-cli sub-001.nii.gz sub-002.nii.gz /data/myproject/ \
    -- --of morphologist-auto-nonoverlap-1.0 --if morphologist-auto-nonoverlap-1.0
```

## Stage 2 — Extract Sulcal Region Crops

```bash
pixi run champollion-cortical-tiles \
    /data/myproject/derivatives/morphologist-6.0/subjects \
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
    local \
    myproject \
    my_run \
    --embeddings_only \
    --config_path /data/myproject/derivatives/champollion_V1/configs/dataset/myproject
```

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
    local \
    myproject \
    my_run \
    --embeddings_only \
    --config_path /data/myproject/derivatives/champollion_V1/configs/dataset/myproject \
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

## Streaming mode

`main.py --mode streaming` runs inference scan by scan instead of stage by stage: each scan's crops are extracted
while earlier scans are already being embedded. It only computes embeddings with existing models; training stays a
batch run.

```bash
python main.py --mode streaming --config my_pipeline.yaml --n-workers 8 --log
```

The configuration must set `dataset.embeddings_only: true`; without it the run logs an error and exits with code 1
before any scan is read.

A non-empty `dataset.embeddings_path` is refused: the run exits with code 1, since streaming always writes the
versioned default embeddings folder.

The subjects directory read is `dataset.input_path`, else `dataset.morphologist_graphs`. It must already hold
Morphologist graphs in the `<subject>/t1mri/<acquisition>` layout: a streaming run executes no Morphologist stage,
it starts at cortical_tiles.

Options:

- `--n-workers N`: number of producer worker processes; default one per CPU.
- `--worker-timeout SECONDS`: a producer worker running longer than this (default 7200 s) ends FAILED.
- `--dry-run`: lists the scans that would run and processes nothing; the dataset config is not written.

Logs. The log directory is chosen in this order:

- with `--log DIR`, DIR is the streaming log directory;
- a bare `--log` gives `./.log/champollion_pipeline_<YYYYmmdd-HHMMSS>/`, next to main.py's own run log of the same
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

## Running Tests

```bash
pixi run test               # All tests
pixi run test-unit          # Unit tests only
pixi run test-smoke         # Smoke tests only
pixi run test-cov           # Tests with coverage report
```

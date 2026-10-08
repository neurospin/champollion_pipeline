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

## Running Tests

```bash
pixi run test               # All tests
pixi run test-unit          # Unit tests only
pixi run test-smoke         # Smoke tests only
pixi run test-cov           # Tests with coverage report
```

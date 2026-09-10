# Internals

Architecture and implementation details for the Champollion pipeline codebase.

## ScriptBuilder pattern

All `src/` scripts subclass `ScriptBuilder` from `champollion_utils.script_builder`, which provides:

- Fluent `argparse` builder API
- Automatic `check_for_updates` on startup
- Consistent `--njobs` and QC-TSV wiring across stages

## Module map

| Module | Stage | Entry point |
|--------|-------|-------------|
| `generate_morphologist_graphs` | 1 | `pixi run morphologist` |
| `run_cortical_tiles` | 2 | `pixi run cortical-tiles` |
| `generate_champollion_config` | 3 | `pixi run champollion-config` |
| `generate_embeddings` | 4 | `pixi run embeddings` |
| `put_together_embeddings` | 5 | `pixi run combine` |
| `generate_snapshots` | 6 | `pixi run snapshots` |
| `train_champollion` | — | `pixi run train` |

## Path conventions

- `src/champollion_pipeline/utils/lib.py` — `DERIVATIVES_FOLDER`, `find_dataset_folder`
- `reference_data/` — pre-trained UMAP models (`.pkl`) and 2D reference coords (`.npy`); no subject identifiers
- `sulci_regions_champollion_V1.json` — 28 sulcal region names used throughout

See {doc}`workflow` for the end-to-end data flow.

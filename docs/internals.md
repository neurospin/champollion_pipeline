# Internals

Architecture and implementation details for the Champollion pipeline codebase.

## ScriptBuilder pattern

All `src/` scripts subclass `ScriptBuilder` from `champollion_utils.script_builder`, which provides:

- Fluent `argparse` builder API
- Automatic `check_for_updates` on startup
- Consistent `--njobs` and QC-TSV wiring across stages

## Module map

Each entry point is a console script declared under `[project.scripts]` in
`pyproject.toml`; it calls the `main` function of the module on its row.

| Module | Stage | Entry point |
|--------|-------|-------------|
| `cli` | 1–6 (launcher) | `champollion-pipeline` |
| `generate_morphologist_graphs` | 1 | `champollion-morphologist` |
| `run_cortical_tiles` | 2 | `champollion-cortical-tiles` |
| `generate_champollion_config` | 3 | `champollion-config` |
| `generate_embeddings` | 4 | `champollion-embeddings` |
| `put_together_embeddings` | 5 | `champollion-combine` |
| `generate_snapshots` | 6 | `champollion-snapshots` |
| `train_champollion` | — | `champollion-train` |
| `generate_labelling_qc` | — (tool) | `champollion-labelling-qc` |
| `prune_failed_subjects` | — (tool) | `champollion-prune` |
| `purge_subject` | — (tool) | `champollion-purge` |

## Path conventions

- `src/champollion_pipeline/utils/lib.py` — `DERIVATIVES_FOLDER`, `find_dataset_folder`
- `reference_data/` — pre-trained UMAP models (`.pkl`) and 2D reference coords (`.npy`); no subject identifiers
- `src/champollion_pipeline/resources/sulci_regions_champollion_V1.json` — 28 sulcal region names used throughout

See {doc}`workflow` for the end-to-end data flow.

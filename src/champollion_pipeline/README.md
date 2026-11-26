# champollion_pipeline

## Purpose

The top-level package of the pipeline. It takes raw T1 MRI scans, or Morphologist outputs, to sulcal embeddings in
six stages: Morphologist graphs, cortical_tiles region crops, champollion_V1 configs, embeddings, combine and
snapshots. Each stage is a script module with its own console script. `cli.py` chains the stages: it reads the
command line, an optional `--config` YAML and an optional `--env` file, infers the missing paths from the input and
runs the enabled stages in order. The other modules hold the configuration, the path layout of the derivatives and
the maintenance scripts (prune, purge, labelling QC, masks, training, profiling).

Subpackages, each with its own README:

- [embedding_engine](embedding_engine/README.md) - in-process embedding engine that keeps the region encoders loaded.
- [file_indexer](file_indexer/README.md) - in-memory indexes of a directory's files, to check a stage's inputs and outputs.
- [models](models/README.md) - model sets for stage 4: fetch, region models, checkpoints and embeddings QC.
- [streaming](streaming/README.md) - streaming mode: producers, barrier and consumer.
- [snapshot_types](snapshot_types/README.md) - the snapshot kinds of stage 6 (sulcal mesh, tiles, UMAP) and their QC.
- [utils](utils/README.md) - shared helpers (input kind detection, QC reports, subject names, resource copies, cortical_tiles config).
- [resources](resources/README.md) - the JSON and YAML data files the stage scripts copy or read.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker and version; exports the stage script classes lazily, so importing the package loads no BrainVISA, cortical_tiles or champollion_V1 code. |
| `_stage_base.py` | `PipelineStage`, the template-method base of the stages `cli.py` runs: log the start, build the argv, run the script, log the result. |
| `cli.py` | The orchestrator and command line of `champollion-pipeline` (and of `python main.py`): argument parsing, config loading, path inference, streaming mode and the stage loop. |
| `cli_options.py` | One optional command-line argument per `PipelineConfig` and `DatasetConfig` field, built from the dataclass fields, and the input and output path aliases. |
| `config.py` | `DatasetConfig`, `PipelineConfig` and `ConfigLoader`, which loads and saves the pipeline YAML config. |
| `derivatives_layout.py` | Where the champollion_V1 outputs live in a dataset's derivatives tree (configs, crops, embeddings, snapshots, models cache, run logs). Pure path computation. |
| `env_file.py` | Reads the HF token, root path, config path, stage dependencies and data root from a `--env` file and from the process environment. |
| `generate_champollion_config.py` | Stage 3 script: writes the champollion_V1 dataset YAML configs for each region of the crops. |
| `generate_embeddings.py` | Stage 4 script: runs champollion_V1's `evaluate.py` for each region model to write per-region embeddings. |
| `generate_labelling_qc.py` | Writes a labelling QC TSV: per subject, whether both labelled hemisphere graphs that cortical_tiles needs exist. |
| `generate_masks.py` | Builds sulcal mask files from manually labelled graphs, using the region list of `resources/sulci_regions_champollion_V1.json`. |
| `generate_morphologist_graphs.py` | Stage 1 script: runs BrainVISA Morphologist on raw T1 scans to make the sulcal graphs. |
| `generate_snapshots.py` | Stage 6 script: renders the sulcal mesh, the region map on the ICBM152 template and the UMAP scatter plot (headless Anatomist). |
| `input_kind_choice.py` | Decides how the input path is read (T1 files or Morphologist outputs): detection, the `input_kind` field, a terminal prompt or a stop. |
| `mask_version_checks.py` | The run's Mask versions checked against the local model set before stage 1 (no single Mask version for a processed region model name; a given Mask version or map differing from the set's). Pure core. |
| `path_inference.py` | Fills every unset path field of a config from the one input path and its kind, through `derivatives_layout.py`. |
| `pipeline_stages.py` | The six stage classes `cli.py` runs, one per stage script, and the paths each stage passes to its script. |
| `process_setup.py` | Per-process logging setup: removes the root handlers champollion_V1 and cortical_tiles install, then installs the champollion_utils handlers. |
| `profile_report.py` | Aggregates a `--profile` run directory (sidecars, cProfile dumps, tree samples) into `report.json` and merged profiles. |
| `prune_failed_subjects.py` | Deletes the cortical_tiles outputs of subjects that failed QC or are absent from the QC file. |
| `purge_subject.py` | Deletes every cortical_tiles output of one subject: crop files, per-subject folders and its rows in the aggregated arrays. |
| `put_together_embeddings.py` | Stage 5 script: checks each region's `full_embeddings.csv` and collects them into one output folder. |
| `replay_config.py` | Writes the effective config of a run to `replay_<timestamp>.yaml` in the logs folder, so `--config` can run it again. |
| `run_cortical_tiles.py` | Stage 2 script: runs cortical_tiles on the Morphologist graphs to write the sulcal region crops. |
| `run_cortical_tiles_graphs.py` | Stage 2 graph handoff: without `--path_to_graph`, the graphs of each subject folder found by graph discovery (labelled set first, then the first in natural order) as the `--graphs` map cortical_tiles reads. |
| `run_cortical_tiles_inputs.py` | Stage 2 input side: subject listing, `--path_to_graph` resolution, the `--sk_qc_path` file, `input_qc.tsv` rows and the end status; re-exported by `run_cortical_tiles.py`. |
| `run_cortical_tiles_whole_brain.py` | Stage 2 whole-brain volume: L+R skeleton fusion, labelling pre-check (`labelling_qc.tsv`) and ventricle removal; re-exported by `run_cortical_tiles.py`. |
| `run_masks.py` | The run's per-region Mask version map, resolved once before stage 1 (given map, given Mask version, or the map read from a full local model set), and its run label naming the champollion_V1 output folders; refuses a mixed given map on a Hugging Face models source and a local model set missing a model the run needs. Pure core. |
| `train_champollion.py` | Trains a champollion_V1 encoder for one sulcal region through champollion_V1's `train.py` with Hydra overrides. |

## Entry points

Console scripts, declared in `[project.scripts]` of `pyproject.toml`:

- `champollion-pipeline` - `cli.main`: the whole pipeline (`python main.py` runs the same function).
- `champollion-morphologist` - `generate_morphologist_graphs.main` (stage 1).
- `champollion-cortical-tiles` - `run_cortical_tiles.main` (stage 2).
- `champollion-config` - `generate_champollion_config.main` (stage 3).
- `champollion-embeddings` - `generate_embeddings.main` (stage 4).
- `champollion-combine` - `put_together_embeddings.main` (stage 5).
- `champollion-snapshots` - `generate_snapshots.main` (stage 6).
- `champollion-train` - `train_champollion.main`.
- `champollion-prune` - `prune_failed_subjects.main`.
- `champollion-purge` - `purge_subject.main`.
- `champollion-labelling-qc` - `generate_labelling_qc.main`.

Modules that also run with `python -m`:

```bash
python -m champollion_pipeline.cli
python -m champollion_pipeline.generate_morphologist_graphs
python -m champollion_pipeline.run_cortical_tiles
python -m champollion_pipeline.generate_champollion_config
python -m champollion_pipeline.generate_embeddings
python -m champollion_pipeline.put_together_embeddings
python -m champollion_pipeline.generate_snapshots
python -m champollion_pipeline.train_champollion
python -m champollion_pipeline.prune_failed_subjects
python -m champollion_pipeline.purge_subject
python -m champollion_pipeline.generate_labelling_qc
python -m champollion_pipeline.generate_masks
python -m champollion_pipeline.profile_report
```

`generate_masks` and `profile_report` have no console script. Each command takes `--help`.

## See also

- [Usage](../../docs/usage.md) - options of each stage, config-less runs, replay config and streaming mode.
- [Workflow](../../docs/workflow.md) - what cortical_tiles and champollion_V1 do inside stages 2 to 4.
- [Internals](../../docs/internals.md) - module map and path conventions.
- [Module reference](../../docs/api.md) - class diagrams and the API of each stage script.
- [Troubleshooting](../../docs/troubleshooting.md) - known failures and their fixes.

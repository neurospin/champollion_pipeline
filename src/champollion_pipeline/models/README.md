# models

## Purpose

Model sets for the embeddings stage (stage 4) and its QC. This package fetches a model set, reads and writes
its `model_set.json` marker, finds the region models and the checkpoint each one uses, checks the model
version options, and checks the embeddings that stages 4 and 5 write. Only `checkpoint.py` imports torch, so
the argument parser and the config can import the other modules cheaply.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker. |
| `cache.py` | The models cache directory and the `model_set.json` marker of a model set: read, validate and write it atomically. |
| `checkpoint.py` | Finds the region models of a model set and their weights: the epoch-N checkpoint, else the best-weights stand-in, and the temporary mirror given to `evaluate.py`. |
| `checkpoint_defaults.py` | The default checkpoint epoch of the embeddings stage; no imports, so no torch. |
| `defaults.py` | Model set defaults that need no torch; re-exports the default checkpoint epoch for the config. |
| `embeddings_options.py` | The option declarations of the stage 4 parser, without torch: the `--checkpoint-epoch` action and `add_embeddings_arguments`, which declares every embeddings stage option in order. |
| `embeddings_qc.py` | QC of stages 4 and 5: crop rows of a run's region models, embedding widths, and the rows of each embeddings CSV file. |
| `epoch_spec.py` | Parses the `--checkpoint-epoch` values and gives each model id its epoch; pure functions, no I/O. |
| `fetch.py` | The model set fetch strategies (local path, Hugging Face, remote archive, interactive fallback) and the order they are tried in. |
| `model_run.py` | The models of one stage 4 run: discovery, the epoch of each model, output folders and provenance entries. |
| `region_crops.py` | The crops folder of each region model of a stage 4 run under `--region-masks`: the crop directory of a region model name, the map checks (`--legacy`, a local models folder not named after a Mask version, a region model the map leaves out), the crops directory of each region, the input QC rows and the `region_mask_versions` provenance entry. |
| `version_options.py` | Parse-time checks of the model version options (`--model-version`, `--masks`) and the mask version of each region model. |

## Entry points

- `fetch.fetch_model_set`: fetches the model set with the first strategy that works.
- `model_run.resolve_model_run`: finds the models under a models path and gives each its epoch.
- `checkpoint.find_region_models`: lists the region models of a model set.
- `cache.read_model_set` and `cache.write_model_set`: read and write the model set marker.
- `epoch_spec.parse_epoch_specs`: parses the checkpoint epoch values of the command line.
- `version_options.check_local_model_set`: checks a local model set against the requested masks.
- `embeddings_qc.fetch_model_files_qc_rows`: QC rows of the embeddings files of a run.

## See also

- [Usage, stage 4](../../../docs/usage.md) - the embeddings stage options and the models path.
- [Workflow](../../../docs/workflow.md) - champollion_V1 inference and the data flow.
- [Internals](../../../docs/internals.md) - module map of the package.

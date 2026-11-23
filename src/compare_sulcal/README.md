# compare_sulcal

## Purpose

Implementation of `src/compare.py`, the tool that compares two outputs of the pipeline or of its
inputs: two sets of sulcal masks, two cortical_tiles outputs, two crop sets, two graph labelling
campaigns or two embedding folders. It sits outside the `champollion_pipeline` package and is not
installed with it; each module holds one comparison responsibility.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker. |
| `brainvisa.py` | Optional BrainVISA imports (PyAIMS, cortical_tiles subjects lookup) shared by the compare modes. |
| `cli.py` | The command-line front end: the `Compare` script and its subcommands, `run_subcommand` and `main`. |
| `crops.py` | The `crops` mode: aligns two crop sets on a common grid and compares them subject by subject. |
| `databases.py` | The `databases` mode: per-subject sulcal voxel counts and mask statistics. |
| `embeddings.py` | The `embeddings` mode: CKA, k-NN overlap, Procrustes alignment and UMAP comparison of two embedding sets. |
| `masks.py` | Mask loading and per-mask metrics (voxel difference, Wasserstein distance, change buckets) of the mask modes. |
| `viewer.py` | Anatomist viewer of the most changed masks (`--visualisation` of the `masks` and `cortical_tiles` modes). |

## Entry points

`src/compare.py` is the command (`python src/compare.py <subcommand> ...`); it calls `cli.main`. Its subcommands:

- `masks`: compares two sets of sulcal NIfTI masks.
- `cortical_tiles`: compares two cortical_tiles outputs.
- `crops`: compares two cortical_tiles `crops/2mm` crop sets per subject, region and side.
- `databases`: compares the sulcal labelling of two graph annotation campaigns.
- `embeddings`: compares two champollion_V1 embedding folders per subject, region and side.

`cli.run_subcommand` runs one subcommand from Python; `src/compare_masks.py` and `src/compare_cortical_tiles.py`
use it. The `compare.py` docstring lists every option and output.

## See also

- [Usage, stage 2](../../docs/usage.md) - the crops that the `cortical_tiles` and `crops` modes compare.
- [Workflow](../../docs/workflow.md) - how cortical_tiles builds the masks and crops.
- [Compare script](../compare.py) - the options, outputs and skip reasons of each subcommand.

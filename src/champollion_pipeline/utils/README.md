# utils

## Purpose

Helpers shared by the pipeline stages: input classification, path and folder rules, the shared QC report
format, input checks and small command-line options. The stage scripts, `cli.py` and the other packages
import them; no module here runs on its own. Several modules use the standard library only, so that
`main.py` and every pixi environment can import them without a toolchain package.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker. |
| `archive.py` | Archive extraction and the extraction cache, used by the model fetch strategies. |
| `cortical_tiles_config.py` | The cortical tiles config (Mask version, voxel size) and the versioned crops folder path. |
| `graph_item.py` | The item of graph discovery (a relative path plus its attributes) and the error for a graph outside any subject folder. |
| `input_kind.py` | Reads one input path as T1 files, Morphologist outputs, neither or both, and finds the subjects folder. |
| `lib.py` | The derivatives folder name, the Morphologist labelling sessions searched, and `find_dataset_folder`. |
| `nifti_check.py` | Header-level QC of one raw NIfTI input file, standard library only. |
| `pipeline_commit.py` | The commit of the running code, for the `pipeline_commit` key of `provenance.json`. |
| `qc_report.py` | The shared QC report format: writes and reads the per-subject and per-run QC TSV files. |
| `region_masks.py` | The `--region-masks` option: one Mask version per region model name, parsed and checked. |
| `resource_copy.py` | A writable working copy of a packaged resource file. |
| `skeleton_qc.py` | The skeleton check of the stage 2 input QC and the `crop_qc.tsv` file that keeps its exclusions out of the crop step. |
| `subject_folders.py` | Which entries of a Morphologist subjects folder are subjects. |
| `subject_name.py` | Validates a subject name used to build per-subject paths. |
| `transform_qc.py` | Transform QC rows of the crop stage (temporary, goes with the prefix-ID guard of `run_cortical_tiles.py`). |

## Entry points

The package defines no console script and no module runs as a script. Its main functions:

- `input_kind.detect_input_kind` and `input_kind.resolve_subjects_dir`: what an input path holds, and its subjects folder.
- `qc_report.write_qc_tsv` and `qc_report.read_qc_tsv`: write and read a per-subject QC report.
- `cortical_tiles_config.compute_versioned_crops_dir`: the crops folder of a Mask version.
- `lib.find_dataset_folder`: the folder that holds a dataset, found from a path.
- `nifti_check.check_nifti_input`: the header checks of one NIfTI input.

## See also

- [Internals](../../../docs/internals.md) - path conventions and the module map.
- [Module reference](../../../docs/api.md) - the cortical tiles config class diagram.
- [Usage](../../../docs/usage.md) - the stage options these helpers serve.

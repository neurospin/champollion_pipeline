# resources

## Purpose

Data files shipped with the package. The stage scripts copy them into a dataset's derivatives tree, or read them in
place, to configure cortical_tiles (stage 2), the champollion_V1 dataset configs (stage 3) and the mask builder. They
are templates and fixed tables: the scripts never edit them here, only their copies.

## Files

| File | What it holds |
|------|---------------|
| `pipeline_loop_2mm.json` | Template of the cortical_tiles run config (2 mm voxel size, input type, mask and cropping settings, QC path). Stage 2 copies it into the derivatives folder and fills in the run's paths. |
| `reference.yaml` | Template of a champollion_V1 dataset config, with `REPLACE_*` and `TESTXX` placeholders. Stage 3 copies it once per dataset and writes the real crop paths in. |
| `sulci_regions_champollion_V1.json` | The sulcal regions of champollion_V1: 56 entries, one per region and side, each with the sulci it holds. |

## Entry points

This directory defines no console script and no Python module. Its files are read by these modules of the package:

- `pipeline_loop_2mm.json` - `run_cortical_tiles.py` copies it to `<output>/pipeline_loop_2mm.json`.
- `reference.yaml` - `generate_champollion_config.py` copies it into each dataset config folder.
- `sulci_regions_champollion_V1.json` - `run_cortical_tiles.py` passes its path to cortical_tiles as
  `brain_regions_json`; `generate_masks.py` reads it to list the sulci of each region.

## See also

- [Workflow](../../../docs/workflow.md) - the cortical_tiles config file and the champollion_V1 Hydra configs.
- [Usage](../../../docs/usage.md) - stages 2 and 3, which use these files.
- [Package README](../README.md) - the modules that read these files.

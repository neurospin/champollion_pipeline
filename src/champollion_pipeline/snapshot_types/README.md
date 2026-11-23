# snapshot_types

## Purpose

Snapshot generators of the snapshots stage (stage 6). One module per snapshot type renders it: the
sulcal graphs and white meshes of a subject, the cortical tiles region graphs, and the UMAP scatter
plots of the embeddings. Two QC modules check the inputs before any render and the expected files
after it. `generate_snapshots` drives the whole stage and imports every module here.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker. |
| `brainvisa_imports.py` | Guarded BrainVISA imports (Anatomist, cortical_tiles config, PyAIMS) and `require`, the one patch point for tests. |
| `input_qc.py` | Input QC of stage 6: per-subject checks of the snapshot inputs, written to `input_qc.tsv`, and warnings for region-level gaps. |
| `output_qc.py` | Output QC of stage 6: the snapshot files a run expects, checked on the disk for `output_qc.tsv`. |
| `sulcal.py` | Sulcal graph snapshots: finds the subjects, their graphs and white meshes, and renders them with Anatomist. |
| `tiles.py` | Cortical tiles snapshots: finds the completed regions and renders their region graphs with Anatomist. |
| `umap.py` | UMAP snapshots: finds the embedding CSV files and their UMAP references, and draws the scatter plots with matplotlib. |

## Entry points

- `champollion-snapshots` (console script, `champollion_pipeline.generate_snapshots:main`): the command that runs
  the package's snapshot generators. No module here runs on its own.
- `sulcal.generate_sulcal_graph_snapshot`, `tiles.generate_tiles_snapshot` and `umap.generate_umap_snapshot`:
  render one snapshot of each type.
- `input_qc.SnapshotsInputQC` and `output_qc.SnapshotsOutputQC`: the input and output QC of a run.

## See also

- [Usage, stage 6](../../../docs/usage.md) - the snapshots stage options and the UMAP reference data.
- [Module reference](../../../docs/api.md) - `generate_snapshots` and the pipeline data flow.
- [Internals](../../../docs/internals.md) - module map of the pipeline.

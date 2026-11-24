# Changelog

## [Unreleased]

## [1.0.0] - 2026-11-24

1.0.0 turns the pipeline into one installed command, `champollion-pipeline <input> [<output>]`, that takes a folder of
T1 MRI files or of Morphologist outputs to sulcal embeddings in six stages, with no Config file needed. The default
Mask version changes to `canonical_corrected_26_1`. Every stage now checks its inputs and its own outputs, and it
reports skipped subjects instead of failing silently.

### Upgrading from 0.2.x

- The default Mask version is now `canonical_corrected_26_1` (it was `canonical_25`) for every stage that takes
  `--masks`, for `masks_version` in the pipeline config, and for `champollion-prune` / `champollion-purge`. To keep
  working on an existing `canonical_25` dataset, pass `--masks canonical_25` (or `--masks-version canonical_25` to
  `champollion-pipeline`). A rerun without it writes a new `canonical_corrected_26_1` tree next to the old one and
  leaves the old tree unchanged.
- `champollion-pipeline` now exits with status 2, before any stage runs, when the given Mask version contradicts the
  local model set. Pass `--allow-version-mismatch` to run anyway (one warning per region model).
- Exit statuses have new meanings: `2` is a usage error (bad option or value, conflicting paths, bad `.env` file), and
  `3` means a partial sequential run (no stage failed, but some subjects were skipped). Scripts that treat any
  non-zero status as a failure must now handle `3`.
- Stage 2 no longer produces `distbottom` crops: `--skip-distbottom` is refused, and `--input-types` accepts only
  `skeleton` and `foldlabel`.
- Stage 2 refuses to run when one subject ID is a strict prefix of another (for example `sub-1` and `sub-10`),
  because cortical_tiles could give one subject the other's transform. `--allow-prefix-ids` runs anyway, with a
  warning.
- Use `champollion-pipeline` instead of `python main.py`. `main.py` still works as a shim.

### Added

- **One command, no Config file.** `champollion-pipeline <input> [<output>]` works out whether the input holds T1
  files (searched at any depth) or Morphologist outputs. From that it infers the dataset root and every
  derivatives path, then runs the enabled stages in order. `--input-kind t1|morphologist` settles an ambiguous input,
  and `--stages` selects the stages to run.
- **A command-line option for every config field.** Examples: `--njobs`, `--masks-version`, `--embeddings-only`,
  `--reference-data-path`, `--checkpoint-epoch`. An option given on the command line wins over `--config`. The
  `--env [PATH]` option reads `HF_TOKEN` and the `CHAMPOLLION_*` root paths from a `.env` file, so secrets stay off
  the command line.
- **Replay files.** Every run writes `replay_<timestamp>.yaml` to its logs folder. `champollion-pipeline --config
  <that file>` runs the same configuration again. The Hugging Face token is never written to the file.
- **Run logs.** `--log [PATH]` on `champollion-pipeline` and on every stage script copies the whole run output, child
  processes included, into a log file. Each log has a header (command, start time) and a footer (exit code, elapsed
  time).
- **Mixed Mask versions per region.** `--region-masks NAME=VERSION ...` crops and embeds each region model with its
  own Mask version. With a local model set, the Mask version of each region is read from the set's `model_set.json`
  or `PROVENANCE.csv`.
- **Several models of one region.** A models folder can hold several trained models per region. They are all
  embedded in one run, each with its own output folder. `--checkpoint-epoch <id>=N` sets one model's epoch,
  `--per-model-folders` gives a single model its own folder, and `--model <id>` limits the combine and snapshot
  stages to one model.
- **Whole-brain volume.** Stage 2 also writes one whole-brain volume per subject, with left and right hemispheres
  fused and the ventricle removed. The labelling session is chosen per subject unless `--labelling_session` is given.
  Subjects with no labelled graph are left out of this volume, with one warning that names them.
- **Profiling.** `--profile DIR` profiles a run and its Python child processes. `pixi run profile-report` then
  combines the results into a report of CPU time, memory and GPU use per stage.

### Changed

- **Input and output checks at every stage.** Each stage checks its inputs before it starts and its outputs when it
  ends. It writes QC tables of the subjects it kept or excluded, with the reasons. A subject that fails is excluded
  with one error naming it, and the run continues with the others.
- **Model source order.** Stage 4 takes models from the first source available: `--models-path`, then the
  checkout's `models/` folder, then the models cache, then a Hugging Face download. The source used is logged. Every
  run records the Mask version, model version, model source and pipeline commit in `provenance.json`.
- **Morphologist inputs.** `champollion-morphologist` now accepts a folder of T1 files, including nested folders, and
  reads Morphologist trees with or without a `subjects/` level. A `qc/` folder or a hidden folder is not mistaken for a
  subject.
- **Fixed read order.** File and folder listings are now sorted, so subjects and regions are processed in the same
  order on every run.
- **Consistent logging.** Pipeline messages go through one logger with levels (`--log-level`, `--verbose`)
  instead of bare prints, and raw debug dumps are no longer printed on startup.
- **Read-only installs.** Stages 2 and 3 now work from a read-only install (Apptainer or Docker), because the shipped
  templates are copied into writable working copies.

### Fixed

- Stage 2 (cortical_tiles) is more robust:
  - It now regenerates crops left incomplete by a crashed run.
  - One subject's failed fold-label step no longer stops every region.
  - Subject IDs with leading zeros are kept.
  - Subjects whose IDs prefix one another no longer get each other's transforms or distance maps.
  - A failed whole-brain fusion now returns a non-zero status.
- `--bids` inputs now chain from stage 1 to the later stages. The participant IDs of stages 1 and 2 now match, and
  `champollion-prune` no longer deletes crops of scans that passed QC.
- `champollion-prune` and `champollion-purge` now find the versioned `crops/<masks>/2mm` folders that stage 2 writes.
- A clean run no longer ends with status 3 because the Morphologist `qc/` folder was counted as a subject.
- A local model set given without a Mask version no longer runs silently on `canonical_25` crops that do not match
  its models.
- `provenance.json` now records the commit of the code that actually ran.
- Stage 6 no longer skips a sulcal snapshot without warning when a subject has both an unlabelled and a labelled
  graph.

Full requirement-level detail: `alm report release --since v0.2.36` in the project ledger.

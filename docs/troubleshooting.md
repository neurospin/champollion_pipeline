# Troubleshooting

## `ModuleNotFoundError: No module named 'champollion_pipeline'`

The Python package is installed by `pixi run install-all` (first-time setup) or `pixi run install-embeddings` (embeddings env only). If you reinstalled Pixi, changed environments, or ran `pixi run uninstall`, run the install step again before any pipeline task:

```bash
pixi run install-embeddings   # re-installs champollion_pipeline, champollion_V1, and champollion_utils
```

This must be run from the repository root (the directory containing `pixi.toml`).

## `pixi run champollion-embeddings` task not found

The `champollion-*` entry-point tasks were added in a recent release. If you are on an older checkout, update first:

```bash
pixi run update-pipeline   # pulls latest code and pixi.lock from origin
pixi run install-embeddings
```

## Morphologist crash: `RuntimeError: the parameter input is not readable or does not exist`

This happens when filenames contain BIDS entities (`_acq-`, `_run-`, `_ses-`, etc.) and you call `morphologist-cli` directly. `morphologist-cli` auto-detects the BIDS format from the filename, silently overrides the `--if morphologist-auto-nonoverlap-1.0` flag, switches to `--if morphologist-bids-2.0`, and then tries to reconstruct file paths using a BIDS directory layout that may not match your actual data location.

**Use the champollion wrapper instead of raw `morphologist-cli`:**

```bash
champollion-morphologist <input_dir> <output_dir>
# or via pixi:
pixi run morphologist <input_dir> <output_dir>
```

The wrapper passes both `--if` and `--of` explicitly, preventing the format override.

**If you must call `morphologist-cli` directly**, pass both flags explicitly:

```bash
morphologist-cli <files...> <output_dir> -- \
    --if morphologist-auto-nonoverlap-1.0 \
    --of morphologist-auto-nonoverlap-1.0
```

**Alternative**: rename your NIfTI files to remove BIDS entities before processing (e.g. `sub-001_T1w.nii.gz` instead of `sub-001_acq-iso08_T1w.nii.gz`).

## `pixi run update-submodules` fails with a rebase conflict

If a submodule has local commits that conflict with the remote tip, the update aborts. These submodules are upstream dependencies — local commits inside them are not expected. Run:

```bash
git submodule update --init --remote --force external/champollion_V1
git submodule update --init --remote --force external/cortical_tiles
```

This resets each submodule to the remote tip, discarding any local commits inside it.

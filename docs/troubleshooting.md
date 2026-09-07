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

## `pixi run update-submodules` fails with a rebase conflict

If a submodule has local commits that conflict with the remote tip, the update aborts. These submodules are upstream dependencies — local commits inside them are not expected. Run:

```bash
git submodule update --init --remote --force external/champollion_V1
git submodule update --init --remote --force external/cortical_tiles
```

This resets each submodule to the remote tip, discarding any local commits inside it.

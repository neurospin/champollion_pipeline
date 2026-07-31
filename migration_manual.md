# Migration Manual — Updating an Existing Champollion Install

This document covers known issues when updating `champollion_pipeline` from an older install.
The recommended update command is always:

```bash
pixi run update
```

If that fails, work through the issues below in order.

---

## Issue 1 — Submodule URL mismatch

**Symptom:** `pixi run update` (or `git submodule update`) fails or hangs on the
`champollion_V1` submodule. Your `.git/config` still references the old URL:

```
url = git@github.com:neurospin/champollion_V1.git   ← old, wrong
```

instead of the current one:

```
url = git@github.com:neurospin/champollion.git       ← correct
```

**Fix:**

```bash
git submodule sync
git submodule update --init --remote --rebase external/champollion_V1
```

`git submodule sync` re-reads the canonical URL from `.gitmodules` and patches
`.git/config`. Only needs to be done once per clone.

---

## Issue 2 — `hatchling` missing from the pixi environment

**Symptom:** During `pixi run install-all` or `pixi run update`, a pip install step
fails with:

```
ERROR: Could not build wheels ... No module named 'hatchling'
```

This happens because install tasks use `--no-build-isolation`, which expects the
build backend (`hatchling`) to already be present in the environment.

**Fix for existing environments (one-time):**

```bash
pixi add hatchling
```

**Permanent fix:** included in the current version of `pixi.toml` — `pixi run update`
will add it automatically once your repo is up to date.

---

## Issue 3 — Stale pixi task paths after script relocation

**Symptom:** `pixi run champollion-config` (or `pixi run embeddings`, `pixi run combine`)
fails with:

```
python3: can't open file '.../src/generate_champollion_config.py': [Errno 2] No such file or directory
```

**Cause:** A refactor moved pipeline scripts from `src/*.py` to
`src/champollion_pipeline/*.py`, but the pixi tasks were not updated at the same time.

**Fix:** `pixi run update` (patched in commit `c2846ce`).

**Workaround while unpatched:** call the console script directly inside `pixi shell`
(bypasses the stale pixi task):

```bash
pixi shell
champollion-config <cortical_tiles_path> --dataset <name>
```

---

## Issue 4 — `snapshot_download` crash with `--masks-version`

**Symptom:** Running embeddings with `--masks-version` fails with:

```
HuggingFace strategy failed: snapshot_download() got an unexpected keyword argument 'subfolder'
```

**Cause:** `snapshot_download()` does not accept a `subfolder` argument
(only `hf_hub_download()` does). The `--masks-version` feature incorrectly passed it.

**Fix:** `pixi run update` (patched in commit `31287de`).

---

## Quick reference — full reset for a broken environment

If `pixi run update` itself is broken, run these steps manually:

```bash
# 1. Sync submodule URLs
git submodule sync

# 2. Pull latest pipeline code
git fetch && git checkout -- pixi.lock && git merge --no-edit -X theirs FETCH_HEAD

# 3. Update submodule
git submodule update --init --remote --rebase external/champollion_V1

# 4. Ensure build backend is present
pixi add hatchling

# 5. Reinstall packages
pixi run reinstall-packages
```

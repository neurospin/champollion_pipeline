#!/usr/bin/env bash
# check_update_history.sh — guard against unrelated-history merges.
#
# Exit codes:
#   0 — HEAD and FETCH_HEAD share a common ancestor (related histories); safe to merge.
#   1 — no common ancestor (unrelated histories); prints recovery steps to stderr.
#   2 — no common ancestor but the clone is shallow; prints a `git fetch --unshallow` hint.
#   other — git returned an unexpected error; prints a one-line error to stderr and
#           exits with that status.

set -u
export GIT_TERMINAL_PROMPT=0

readonly EXIT_UNRELATED=1
readonly EXIT_SHALLOW=2
readonly MERGE_BASE_NOT_FOUND=1
readonly UPSTREAM_URL="https://github.com/neurospin/champollion_pipeline.git"

print_recovery_steps() {
    cat >&2 <<EOF
ERROR: unrelated histories. The commit fetched from origin shares no history
with your local HEAD: the upstream branch was rewritten (force-pushed), so it
cannot be merged. Nothing has been changed.

To recover:
  1. Back up any local changes you want to keep, for example:
       git stash push --include-untracked
     or
       git diff HEAD > ~/champollion_pipeline_local_changes.patch
  2. Then either reset this clone onto the rewritten upstream:
       git fetch origin && git reset --hard origin/main
       git submodule sync --recursive && git submodule update --init --force
     or start from a fresh clone:
       git clone ${UPSTREAM_URL}
  3. Re-run: pixi run install-all
EOF
}

print_shallow_hint() {
    cat >&2 <<'HINT'
ERROR: no common ancestor found, but this clone is shallow (created with
--depth), so older commits are missing locally and the shared history may
simply be out of view. Nothing has been changed.

To fix it, fetch the full history, then re-run the update:
  git fetch --unshallow
HINT
}

# Succeeds only for a shallow clone. Git older than 2.15 does not know
# --is-shallow-repository and echoes it back verbatim, so anything other than
# "true" (including a failure) means "not known to be shallow".
is_shallow_clone() {
    [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]
}

git merge-base HEAD FETCH_HEAD >/dev/null 2>&1
status=$?

if [ "$status" -eq 0 ]; then
    exit 0
elif [ "$status" -eq "$MERGE_BASE_NOT_FOUND" ] && is_shallow_clone; then
    print_shallow_hint
    exit "$EXIT_SHALLOW"
elif [ "$status" -eq "$MERGE_BASE_NOT_FOUND" ]; then
    print_recovery_steps
    exit "$EXIT_UNRELATED"
else
    echo "ERROR: could not compare HEAD with FETCH_HEAD (git merge-base exited ${status}). Run 'git fetch' and retry." >&2
    exit "$status"
fi

#!/usr/bin/env bash
# check_update_history.sh — guard against unrelated-history merges.
#
# Exit codes:
#   0 — HEAD and FETCH_HEAD share a common ancestor (related histories); safe to merge.
#   1 — no common ancestor (unrelated histories); prints recovery steps to stderr.
#   other — git returned an unexpected error; prints a one-line error to stderr and
#           exits with that status.

set -u
export GIT_TERMINAL_PROMPT=0

readonly EXIT_UNRELATED=1
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

git merge-base HEAD FETCH_HEAD >/dev/null 2>&1
status=$?

if [ "$status" -eq 0 ]; then
    exit 0
elif [ "$status" -eq $EXIT_UNRELATED ]; then
    print_recovery_steps
    exit 1
else
    echo "ERROR: could not compare HEAD with FETCH_HEAD (git merge-base exited ${status}). Run 'git fetch' and retry." >&2
    exit "$status"
fi

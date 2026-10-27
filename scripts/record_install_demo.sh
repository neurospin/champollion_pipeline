#!/usr/bin/env bash
# record_install_demo.sh — record the README install steps as an asciinema cast
# and, when agg is available, convert it to a GIF for the README.
#
# Usage: scripts/record_install_demo.sh [CAST_FILE] [GIF_FILE]
#   CAST_FILE  default: docs/_static/install-demo.cast
#   GIF_FILE   default: CAST_FILE with .gif instead of .cast
#
# Environment:
#   DEMO_REPO_URL  repository to clone (default: the public GitHub repository)
#   DEMO_SPEED     agg playback speed-up (default: 10, keeps the README GIF under 3 MB)
#
# The demo runs non-interactively in a fresh temporary directory with a neutral
# "$ " prompt and a minimal environment, so no home path, hostname or data path
# reaches the recording. The cast is always written to a named local file:
# asciinema without a filename offers to upload the session publicly.

set -euo pipefail

readonly DEFAULT_REPO_URL="https://github.com/neurospin/champollion_pipeline.git"
readonly IDLE_TIME_LIMIT=2
readonly COLS=100
readonly ROWS=30
readonly TYPING_DELAY=0.04
readonly GIF_IDLE_TIME_LIMIT=1
readonly GIF_FPS_CAP=8

cast_file="${1:-docs/_static/install-demo.cast}"
gif_file="${2:-${cast_file%.cast}.gif}"
repo_url="${DEMO_REPO_URL:-$DEFAULT_REPO_URL}"
speed="${DEMO_SPEED:-10}"

if ! command -v asciinema >/dev/null 2>&1; then
    echo "ERROR: asciinema not found; install it (e.g. apt install asciinema)." >&2
    exit 1
fi
if ! command -v pixi >/dev/null 2>&1; then
    echo "ERROR: pixi not found; see https://pixi.sh." >&2
    exit 1
fi

mkdir -p "$(dirname "$cast_file")"
cast_file="$(cd "$(dirname "$cast_file")" && pwd)/$(basename "$cast_file")"
gif_file="$(mkdir -p "$(dirname "$gif_file")" && cd "$(dirname "$gif_file")" && pwd)/$(basename "$gif_file")"

work_dir="$(mktemp -d /tmp/champollion-demo.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

# The scripted session: each command is "typed" after a neutral prompt, then run.
cat >"$work_dir/demo.sh" <<EOF
set -e
cd "$work_dir"
type_and_run() {
    printf '\$ '
    local cmd="\$1" i
    for (( i = 0; i < \${#cmd}; i++ )); do
        printf '%s' "\${cmd:i:1}"
        sleep $TYPING_DELAY
    done
    printf '\n'
    eval "\$cmd"
}
type_and_run "mkdir Champollion && cd Champollion"
type_and_run "git clone $repo_url"
type_and_run "cd champollion_pipeline"
type_and_run "pixi run install-all"
printf '\$ \n'
sleep 1
EOF

(
    cd "$work_dir"
    env -i HOME="$HOME" PATH="$PATH" TERM=xterm-256color LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        asciinema rec "$cast_file" --overwrite \
        --command "bash --noprofile --norc $work_dir/demo.sh" \
        --idle-time-limit "$IDLE_TIME_LIMIT" --cols "$COLS" --rows "$ROWS" \
        --title "Champollion pipeline installation"
)

echo "Cast written to $cast_file"

if command -v agg >/dev/null 2>&1; then
    agg --speed "$speed" --idle-time-limit "$GIF_IDLE_TIME_LIMIT" --fps-cap "$GIF_FPS_CAP" \
        --last-frame-duration 5 --theme asciinema "$cast_file" "$gif_file"
    echo "GIF written to $gif_file"
else
    echo "agg not found; to build the GIF, install it with" >&2
    echo "  cargo install --git https://github.com/asciinema/agg" >&2
    echo "then run: agg --speed $speed --idle-time-limit $GIF_IDLE_TIME_LIMIT --fps-cap $GIF_FPS_CAP $cast_file $gif_file" >&2
fi

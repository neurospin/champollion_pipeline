#!/usr/bin/env bash
set -euo pipefail

if ! command -v pixi &>/dev/null; then
    echo "pixi not found. Install it from https://pixi.sh" >&2
    exit 1
fi

cd "$(dirname "$0")"
pixi run setup

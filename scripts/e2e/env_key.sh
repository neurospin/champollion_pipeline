#!/usr/bin/env bash
# Print the key of the e2e env SIF (the pixi environment image, no pipeline code).
#
# Usage: scripts/e2e/env_key.sh [--sif-name] [ROOT]
#
# The key is the first 12 hex characters of a SHA-256 over the seven inputs that
# shape the environment, in this order: pixi.lock, E2E_PIXI_VERSION (from
# scripts/e2e/e2e.env), the champollion-utils specifier of pyproject.toml, the
# champollion_V1 and cortical_tiles setup.py, cortical_tiles/info.py and
# .github/Dockerfile.base. Each input contributes `label NUL sha256(content) LF`.
# ROOT defaults to this script's repository. --sif-name prints E2E_ENV_SIF_NAME
# with {key12} filled in. Exit 0 key printed, 1 absent input or missing tool,
# 2 usage. Nothing is written anywhere.
set -euo pipefail

usage() {
    echo "usage: $0 [--sif-name] [ROOT]" >&2
    exit 2
}

SIF_NAME=0
ROOT=""
for arg in "$@"; do
    case "$arg" in
        --sif-name) SIF_NAME=1 ;;
        -*) usage ;;
        *)
            [ -z "$ROOT" ] || usage
            ROOT="$arg"
            ;;
    esac
done
if [ -z "$ROOT" ]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi

if ! command -v python3 >/dev/null 2>&1 \
    || ! python3 -I -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "env_key.sh: python3 >= 3.11 required" >&2
    exit 1
fi

ENV_FILE="$ROOT/scripts/e2e/e2e.env"
FILE_INPUTS=(
    external/champollion_V1/setup.py
    external/cortical_tiles/setup.py
    external/cortical_tiles/cortical_tiles/info.py
    .github/Dockerfile.base
)

# Value of one e2e.env setting, read in a subshell so nothing leaks into this shell.
read_setting() {
    (
        set +eu
        set -a
        # shellcheck source=/dev/null
        source "$ENV_FILE" >/dev/null 2>&1
        printf '%s' "${!1-}"
    )
}

# The one champollion-utils entry of [project].dependencies, stripped; empty when none or several.
utils_specifier() {
    python3 -I - "$ROOT/pyproject.toml" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    dependencies = tomllib.load(handle).get("project", {}).get("dependencies", [])
matches = []
for entry in dependencies:
    name = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", entry)
    if name and re.sub(r"[-_.]+", "-", name.group(1)).lower() == "champollion-utils":
        matches.append(entry.strip())
if len(matches) == 1:
    sys.stdout.write(matches[0])
PY
}

missing=()
[ -f "$ROOT/pixi.lock" ] || missing+=("pixi.lock")
PIXI_VERSION=""
if [ -f "$ENV_FILE" ]; then
    PIXI_VERSION="$(read_setting E2E_PIXI_VERSION)"
    [ -n "$PIXI_VERSION" ] || missing+=("E2E_PIXI_VERSION")
else
    missing+=("scripts/e2e/e2e.env")
fi
UTILS_PIN=""
if [ -f "$ROOT/pyproject.toml" ]; then
    UTILS_PIN="$(utils_specifier)" || UTILS_PIN=""
    [ -n "$UTILS_PIN" ] || missing+=("champollion-utils")
else
    missing+=("pyproject.toml")
fi
for rel in "${FILE_INPUTS[@]}"; do
    [ -f "$ROOT/$rel" ] || missing+=("$rel")
done
SIF_TEMPLATE=""
if [ "$SIF_NAME" = 1 ] && [ -f "$ENV_FILE" ]; then
    SIF_TEMPLATE="$(read_setting E2E_ENV_SIF_NAME)"
    [ -n "$SIF_TEMPLATE" ] || missing+=("E2E_ENV_SIF_NAME")
fi
if [ "${#missing[@]}" -ne 0 ]; then
    for label in "${missing[@]}"; do
        echo "env_key.sh: missing input: $label" >&2
    done
    exit 1
fi

digest_file() { sha256sum < "$1" | cut -d' ' -f1; }
digest_text() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }

KEY="$(
    {
        printf '%s\0%s\n' pixi.lock "$(digest_file "$ROOT/pixi.lock")"
        printf '%s\0%s\n' E2E_PIXI_VERSION "$(digest_text "$PIXI_VERSION")"
        printf '%s\0%s\n' champollion-utils "$(digest_text "$UTILS_PIN")"
        for rel in "${FILE_INPUTS[@]}"; do
            printf '%s\0%s\n' "$rel" "$(digest_file "$ROOT/$rel")"
        done
    } | sha256sum | cut -c1-12
)"

if [ "$SIF_NAME" = 1 ]; then
    echo "${SIF_TEMPLATE//\{key12\}/$KEY}"
else
    echo "$KEY"
fi

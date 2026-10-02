#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_FILE="$SCRIPT_DIR/install_report_$(date +%Y%m%d_%H%M%S).txt"

readonly PIXI_MANIFEST="$SCRIPT_DIR/pixi.toml"
readonly PIXI_UPGRADE_COMMAND="pixi self-update"
# Same pattern as REQUIRES_PIXI_FLOOR in scripts/setup_wizard.py: line-anchored, first ">=" bound.
readonly REQUIRES_PIXI_FLOOR_PATTERN='^requires-pixi[[:space:]]*=[[:space:]]*"[^"]*>=[[:space:]]*v?([0-9]+(\.[0-9]+)*)'
readonly VERSION_NUMBER_PATTERN='([0-9]+(\.[0-9]+)*)'

# Prints the ">=" bound of pixi.toml's requires-pixi, or nothing if absent. O(n) in manifest lines.
find_minimum_pixi_version() {
    local line
    [ -r "$PIXI_MANIFEST" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ $line =~ $REQUIRES_PIXI_FLOOR_PATTERN ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi
    done < "$PIXI_MANIFEST"
}

# Prints the X.Y.Z from `pixi --version` ("pixi X.Y.Z"), or nothing if unparsable. O(1).
find_installed_pixi_version() {
    local output
    output="$(pixi --version 2>/dev/null)" || return 0
    if [[ $output =~ $VERSION_NUMBER_PATTERN ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    fi
}

# Succeeds when dotted version $1 is numerically lower than $2; missing components count as 0. O(k) components.
is_version_lower() {
    local -a have want
    local i count
    IFS=. read -r -a have <<< "$1"
    IFS=. read -r -a want <<< "$2"
    count=$(( ${#have[@]} > ${#want[@]} ? ${#have[@]} : ${#want[@]} ))
    for (( i = 0; i < count; i++ )); do
        if (( 10#${have[i]:-0} < 10#${want[i]:-0} )); then return 0; fi
        if (( 10#${have[i]:-0} > 10#${want[i]:-0} )); then return 1; fi
    done
    return 1
}

# Exits 1 with the floor and upgrade command when pixi is older than requires-pixi (REQ-PIXIVER-04/05).
require_minimum_pixi_version() {
    local minimum installed
    minimum="$(find_minimum_pixi_version)"
    installed="$(find_installed_pixi_version)"
    if [ -z "$minimum" ] || [ -z "$installed" ]; then
        echo "Note: could not read the pixi version or the requires-pixi floor; skipping the pixi version check." >&2
        return 0
    fi
    is_version_lower "$installed" "$minimum" || return 0
    echo "pixi $installed is too old: this pipeline needs pixi >= $minimum (requires-pixi in pixi.toml)." >&2
    echo "Upgrade with: $PIXI_UPGRADE_COMMAND" >&2
    exit 1
}

_on_error() {
    echo ""
    echo "Installation encountered an error. Generating diagnostic report..."
    python3 "$SCRIPT_DIR/scripts/install_health.py" --report --output "$REPORT_FILE" || true
    echo ""
    echo "Report saved: $REPORT_FILE"
    echo "Send this file to Barthelemy or open a GitHub issue with it."
}
trap '_on_error' ERR

if ! command -v pixi &>/dev/null; then
    echo "pixi not found. Install it from https://pixi.sh" >&2
    exit 1
fi

require_minimum_pixi_version

cd "$SCRIPT_DIR"

# Resolve the conda environment first: the wizard itself runs inside it and
# needs rich, so 'pixi run setup' cannot be the first command.
pixi install -e default

if [ -t 0 ]; then
    # Interactive terminal: let the wizard ask the three questions and run the
    # commands that match the answers (REQ-WIZARD-03).
    pixi run setup
else
    # No TTY (CI, docker build, piped installer): the wizard cannot prompt, so
    # fall back to the full default installation. 'install-all' ends with
    # 'check-install', which reports the resulting health status.
    echo "No interactive terminal detected — running the full default installation."
    pixi run install-all
fi

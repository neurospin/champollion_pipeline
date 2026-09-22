#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_FILE="$SCRIPT_DIR/install_report_$(date +%Y%m%d_%H%M%S).txt"

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

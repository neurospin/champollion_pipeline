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
pixi install -e default
pixi run install-all
pixi run check-install

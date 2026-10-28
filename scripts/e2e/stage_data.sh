#!/usr/bin/env bash
# Stage the read-only QUICK tier data directory: the raw IXI T1 of each
# E2E_TIER_QUICK_SUBJECTS subject, in both input layouts the pipeline accepts
# (flat files: bids/<source name> and classic/<subject>.nii.gz), plus an
# ATTRIBUTION file and a sha256 MANIFEST.
#
# Usage: scripts/e2e/stage_data.sh   (no arguments; every setting comes from e2e.env)
#
# Every source passes `e2e_env.py check-path` (raw path and resolved path)
# before anything is written. The data are staged in a temporary directory and
# renamed into place, so a failed run leaves E2E_DATA_DIR as it was. A re-run
# on an existing tier_quick copies nothing: it verifies the MANIFEST.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIER_DIRNAME=tier_quick

if [ "$#" -ne 0 ]; then
    echo "usage: $0   (no arguments; settings come from $HERE/e2e.env)" >&2
    exit 2
fi

set -a
# shellcheck source=/dev/null
source "$HERE/e2e.env"
set +a

DEST="$E2E_DATA_DIR/$TIER_DIRNAME"

check_path() {
    python3 "$HERE/e2e_env.py" check-path "$1"
}

verify_tier() {
    local manifest="$DEST/MANIFEST" listed actual rel
    if [ ! -f "$manifest" ]; then
        echo "stage_data: $manifest missing" >&2
        return 1
    fi
    if ! (cd "$DEST" && sha256sum --check --strict --quiet MANIFEST) >&2; then
        echo "stage_data: $DEST does not match its MANIFEST" >&2
        return 1
    fi
    listed="$(cut -c67- "$manifest" | LC_ALL=C sort)"
    actual="$(cd "$DEST" && find . -type f ! -name MANIFEST -printf '%P\n' | LC_ALL=C sort)"
    if [ "$listed" != "$actual" ]; then
        while IFS= read -r rel; do
            [ -n "$rel" ] && echo "stage_data: $DEST/$rel is not listed in MANIFEST" >&2
        done < <(LC_ALL=C comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$actual"))
        return 1
    fi
    echo "stage_data: $DEST verified against its MANIFEST"
}

if [ -e "$DEST" ]; then
    verify_tier
    exit $?
fi

# Select one T1 per subject and check every source before writing anything.
shopt -s nullglob
declare -a SUBJECTS SOURCES
for subject in $E2E_TIER_QUICK_SUBJECTS; do
    matches=()
    # shellcheck disable=SC2206
    for path in $E2E_IXI_RAWDATA/$E2E_IXI_T1_GLOB; do
        rel="${path#"$E2E_IXI_RAWDATA"/}"
        [ "${rel%%/*}" = "$subject" ] && matches+=("$path")
    done
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "stage_data: subject $subject: expected 1 T1 under $E2E_IXI_RAWDATA, found ${#matches[@]}" >&2
        exit 1
    fi
    src="${matches[0]}"
    check_path "$src"
    check_path "$(realpath "$src")"
    SUBJECTS+=("$subject")
    SOURCES+=("$src")
done
shopt -u nullglob

CREATED_DATA_DIR=0
if [ ! -d "$E2E_DATA_DIR" ]; then
    mkdir -p "$E2E_DATA_DIR"
    CREATED_DATA_DIR=1
fi
TMP="$(mktemp -d "$E2E_DATA_DIR/.$TIER_DIRNAME.XXXXXX")"

cleanup() {
    if [ -n "${TMP:-}" ] && [ -d "$TMP" ]; then
        chmod -R u+w "$TMP"
        rm -rf "$TMP"
    fi
    if [ "$CREATED_DATA_DIR" = 1 ] && [ ! -e "$DEST" ]; then
        rmdir "$E2E_DATA_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir "$TMP/bids" "$TMP/classic"
{
    echo "IXI dataset, https://brain-development.org/ixi-dataset/"
    echo "Licence: $E2E_DATA_LICENSE, https://creativecommons.org/licenses/by-sa/3.0/"
    echo "Staged files (staged path <- source path):"
} > "$TMP/ATTRIBUTION"
for i in "${!SUBJECTS[@]}"; do
    src="${SOURCES[$i]}"
    bids_rel="bids/$(basename "$src")"
    classic_rel="classic/${SUBJECTS[$i]}.nii.gz"
    cp -- "$src" "$TMP/$bids_rel"
    cp -- "$src" "$TMP/$classic_rel"
    echo "$bids_rel <- $src" >> "$TMP/ATTRIBUTION"
    echo "$classic_rel <- $src" >> "$TMP/ATTRIBUTION"
done
(cd "$TMP" && find . -type f ! -name MANIFEST.part -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$TMP/MANIFEST.part"
mv "$TMP/MANIFEST.part" "$TMP/MANIFEST"
find "$TMP" -type f -exec chmod 0444 {} +
find "$TMP" -type d -exec chmod 0555 {} +
mv -T "$TMP" "$DEST"
TMP=""
echo "stage_data: staged ${#SUBJECTS[@]} subjects into $DEST"

#!/usr/bin/env bash
# Stage the read-only COHORT view: one symbolic link per IXI subject to its raw T1, in both input
# layouts the pipeline accepts (bids/<source name> and classic/<subject>.nii.gz), plus an
# ATTRIBUTION file and a MANIFEST of "<link> -> <target>" lines.
#
# Usage: scripts/e2e/stage_cohort_view.sh   (no arguments; every setting comes from e2e.env)
#
# Every source passes `e2e_env.py check-path` (raw path and resolved path) before anything is
# written. The view is built in a temporary directory and renamed into place, so a failed run
# leaves E2E_DATA_DIR as it was. A re-run on an existing tier_cohort writes nothing: it checks
# the links against the MANIFEST. The IXI files themselves are never copied or modified.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIER_DIRNAME=tier_cohort
SEP=' -> '

if [ "$#" -ne 0 ]; then
    echo "usage: $0   (no arguments; settings come from $HERE/e2e.env)" >&2
    exit 2
fi

while IFS= read -r name; do
    unset "$name"
done < <(compgen -v E2E_ || true)
set -a
# shellcheck source=/dev/null
source "$HERE/e2e.env"
set +a

DEST="$E2E_DATA_DIR/$TIER_DIRNAME"

check_path() {
    python3 "$HERE/e2e_env.py" check-path "$1"
}

# One "<link> -> <target>" line per symbolic link under $1, C-sorted by link path.
list_links() {
    (cd "$1" && find . -type l -printf "%P$SEP%l\n" | LC_ALL=C sort)
}

verify_view() {
    local manifest="$DEST/MANIFEST" line link target findings=0
    local -A listed=() actual=()
    if [ ! -f "$manifest" ] || [ -L "$manifest" ]; then
        echo "stage_cohort_view: $manifest missing" >&2
        return 1
    fi
    while IFS= read -r line; do
        [ -n "$line" ] && listed["${line%%"$SEP"*}"]="${line#*"$SEP"}"
    done < "$manifest"
    while IFS= read -r line; do
        [ -n "$line" ] && actual["${line%%"$SEP"*}"]="${line#*"$SEP"}"
    done < <(list_links "$DEST")
    while IFS= read -r link; do
        [ -z "$link" ] && continue
        if [ -z "${actual[$link]+set}" ]; then
            echo "stage_cohort_view: $link: missing (listed in MANIFEST)" >&2
            findings=1
        elif [ "${actual[$link]}" != "${listed[$link]}" ]; then
            echo "stage_cohort_view: $link: targets ${actual[$link]}, MANIFEST lists ${listed[$link]}" >&2
            findings=1
        fi
    done < <(printf '%s\n' "${!listed[@]}" | LC_ALL=C sort)
    while IFS= read -r link; do
        [ -z "$link" ] && continue
        if [ -z "${listed[$link]+set}" ]; then
            echo "stage_cohort_view: $link: not listed in MANIFEST" >&2
            findings=1
        fi
    done < <(printf '%s\n' "${!actual[@]}" | LC_ALL=C sort)
    if [ "$findings" -ne 0 ]; then
        return 1
    fi
    echo "stage_cohort_view: $DEST verified against its MANIFEST"
}

if [ -e "$DEST" ] || [ -L "$DEST" ]; then
    status=0
    verify_view || status=$?
    exit "$status"
fi

# Scan the raw tree (read-only) and check every source before writing anything.
case "$E2E_IXI_RAWDATA" in
    /*) ;;
    *)
        echo "stage_cohort_view: E2E_IXI_RAWDATA must be an absolute path: $E2E_IXI_RAWDATA" >&2
        exit 1
        ;;
esac
shopt -s nullglob
# shellcheck disable=SC2206
CANDIDATES=("$E2E_IXI_RAWDATA"/$E2E_IXI_T1_GLOB)
shopt -u nullglob
if [ "${#CANDIDATES[@]}" -eq 0 ]; then
    echo "stage_cohort_view: no T1 under $E2E_IXI_RAWDATA matches $E2E_IXI_T1_GLOB" >&2
    exit 1
fi

declare -A COUNT=() SOURCE=()
bad=0
for src in "${CANDIDATES[@]}"; do
    if [[ "$src" == *$'\n'* || "$src" == *"$SEP"* ]]; then
        echo "stage_cohort_view: unsupported characters in source path: $src" >&2
        bad=1
        continue
    fi
    rel="${src#"$E2E_IXI_RAWDATA"/}"
    subject="${rel%%/*}"
    COUNT["$subject"]=$(( ${COUNT["$subject"]:-0} + 1 ))
    SOURCE["$subject"]="$src"
done
mapfile -t SUBJECTS < <(printf '%s\n' "${!COUNT[@]}" | LC_ALL=C sort)
for subject in "${SUBJECTS[@]}"; do
    if [ "${COUNT[$subject]}" -ne 1 ]; then
        echo "stage_cohort_view: subject $subject: expected 1 T1 under $E2E_IXI_RAWDATA, found ${COUNT[$subject]}" >&2
        bad=1
    fi
done
if [ "$bad" -ne 0 ]; then
    exit 1
fi
for subject in "${SUBJECTS[@]}"; do
    src="${SOURCE[$subject]}"
    check_path "$src"
    check_path "$(realpath -- "$src")"
done

CREATED_DATA_DIR=0
if [ ! -d "$E2E_DATA_DIR" ]; then
    mkdir -p "$E2E_DATA_DIR"
    CREATED_DATA_DIR=1
fi
TMP=""

cleanup() {
    if [ -n "${TMP:-}" ] && [ -d "$TMP" ]; then
        find "$TMP" -type d -exec chmod u+w {} +
        rm -rf -- "$TMP"
    fi
    if [ "$CREATED_DATA_DIR" = 1 ] && [ ! -e "$DEST" ]; then
        rmdir "$E2E_DATA_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT
TMP="$(mktemp -d "$E2E_DATA_DIR/.$TIER_DIRNAME.XXXXXX")"

mkdir "$TMP/bids" "$TMP/classic"
for subject in "${SUBJECTS[@]}"; do
    src="${SOURCE[$subject]}"
    ln -s -- "$src" "$TMP/bids/$(basename -- "$src")"
    ln -s -- "$src" "$TMP/classic/$subject.nii.gz"
done
{
    echo "IXI dataset, https://brain-development.org/ixi-dataset/"
    echo "Licence: $E2E_DATA_LICENSE, https://creativecommons.org/licenses/by-sa/3.0/"
    echo "Linked files: see MANIFEST (link -> source path)"
} > "$TMP/ATTRIBUTION"
list_links "$TMP" > "$TMP/MANIFEST.part"
mv "$TMP/MANIFEST.part" "$TMP/MANIFEST"
# Never chmod a symbolic link: chmod follows it and would change the IXI file.
find "$TMP" -type f -exec chmod 0444 {} +
find "$TMP" -type d -exec chmod 0555 {} +
mv -T "$TMP" "$DEST"
TMP=""
echo "stage_cohort_view: linked ${#SUBJECTS[@]} subjects into $DEST"

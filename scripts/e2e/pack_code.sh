#!/usr/bin/env bash
# Pack the code directory of one commit for the e2e runs: the repository files
# tracked at <commit>, external/champollion_V1 and external/cortical_tiles at
# their recorded gitlinks, champollion_utils at its pyproject pin, a stub git
# repository at external/cortical_tiles, a COMMIT file and a sha256 MANIFEST.
#
# Usage: scripts/e2e/pack_code.sh <commit>
#
# Reads only the local repositories (no network, no fetch): an unpushed commit
# can be packed. Writes $E2E_CODE_DIR/<E2E_CODE_DIR_NAME> through a temporary
# directory renamed into place, and prints its path as the last stdout line.
# No CSV, notebook, .git (except the stub), cortical_tiles fixture data or path
# matching E2E_BANNED_DATA_PATTERNS is packed. A re-run on an existing code
# directory writes nothing: it verifies the MANIFEST.
# Exit 0 packed or verified, 1 refused (reason on stderr), 2 usage.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$HERE/../.." && pwd -P)"
UTILS="$(dirname "$ROOT")/champollion_utils"
SUBMODULES=(external/champollion_V1 external/cortical_tiles)

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <commit>   (settings come from $HERE/e2e.env)" >&2
    exit 2
fi
REQUESTED="$1"

die() {
    echo "pack_code: $*" >&2
    exit 1
}

if ! command -v python3 >/dev/null 2>&1 \
    || ! python3 -I -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    die "python3 >= 3.11 required"
fi

set -a
# shellcheck source=/dev/null
source "$HERE/e2e.env"
set +a

for setting in E2E_CODE_DIR E2E_CODE_DIR_NAME E2E_BANNED_DATA_PATTERNS; do
    [ -n "${!setting:-}" ] || die "$setting is unset or empty in $HERE/e2e.env"
done

COMMIT40="$(git -C "$ROOT" rev-parse --verify --quiet --end-of-options "$REQUESTED^{commit}")" \
    || die "unknown commit: $REQUESTED"
NAME="${E2E_CODE_DIR_NAME//\{commit12\}/${COMMIT40:0:12}}"
case "$NAME" in
    "" | */* | .*) die "E2E_CODE_DIR_NAME gives an unusable directory name: '$NAME'" ;;
esac
CODE="${E2E_CODE_DIR%/}/$NAME"

verify_code() {
    local manifest="$CODE/MANIFEST" listed actual rel
    [ -f "$manifest" ] || die "$manifest missing"
    if ! (cd "$CODE" && sha256sum --check --strict --quiet MANIFEST) >&2; then
        die "$CODE does not match its MANIFEST"
    fi
    listed="$(cut -c67- "$manifest" | LC_ALL=C sort)"
    actual="$(cd "$CODE" && find . -type f ! -path ./MANIFEST -printf '%P\n' | LC_ALL=C sort)"
    if [ "$listed" != "$actual" ]; then
        while IFS= read -r rel; do
            [ -n "$rel" ] && echo "pack_code: $CODE/$rel is not listed in MANIFEST" >&2
        done < <(LC_ALL=C comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$actual"))
        exit 1
    fi
    echo "pack_code: $CODE verified against its MANIFEST"
    echo "$CODE"
}

if [ -e "$CODE" ]; then
    verify_code
    exit 0
fi

# Gitlinks of the two submodules, and their commits in the local submodule repositories.
declare -A GITLINK
for sub in "${SUBMODULES[@]}"; do
    entry="$(git -C "$ROOT" ls-tree "$COMMIT40" -- "$sub")"
    case "$entry" in
        "160000 commit "*) GITLINK[$sub]="$(printf '%s' "$entry" | cut -d' ' -f3 | cut -f1)" ;;
        *) die "$sub: no gitlink recorded at $COMMIT40" ;;
    esac
    [ -e "$ROOT/$sub/.git" ] || die "$sub: no submodule repository at $ROOT/$sub"
    git -C "$ROOT/$sub" cat-file -e "${GITLINK[$sub]}^{commit}" 2>/dev/null \
        || die "$sub: commit ${GITLINK[$sub]} not in the local submodule repository"
done

# champollion_utils at the reference after "@" in the pyproject pin.
PYPROJECT="$(git -C "$ROOT" show "$COMMIT40:pyproject.toml" 2>/dev/null)" \
    || die "champollion-utils: pyproject.toml absent at $COMMIT40"
UTILS_REF="$(printf '%s' "$PYPROJECT" | python3 -I -c '
import re, sys, tomllib
dependencies = tomllib.loads(sys.stdin.read()).get("project", {}).get("dependencies", [])
pins = []
for entry in dependencies:
    name = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", entry)
    if name and re.sub(r"[-_.]+", "-", name.group(1)).lower() == "champollion-utils":
        pins.append(entry)
if len(pins) != 1:
    sys.exit(f"{len(pins)} champollion-utils entries in [project].dependencies, expected 1")
url = re.match(r"[^@]*@\s*(\S+)", pins[0])
if not url:
    sys.exit(f"pin {pins[0].strip()!r} is not of the form name @ URL")
ref = re.search(r"@([^/@]+)$", url.group(1).split("#", 1)[0])
if not ref:
    sys.exit(f"pin URL {url.group(1)!r} names no reference after @")
print(ref.group(1))
' 2>&1)" || die "champollion-utils: $UTILS_REF"
git -C "$UTILS" rev-parse --git-dir >/dev/null 2>&1 \
    || die "champollion-utils: $UTILS is not a git repository"
SHA_UTILS="$(git -C "$UTILS" rev-parse --verify --quiet --end-of-options "$UTILS_REF^{commit}")" \
    || die "champollion-utils: reference $UTILS_REF not found in $UTILS"

# Selection of every packed file, checked (gitlinks, names, banned patterns) before any write;
# with a destination directory, also writes the blobs there as raw bytes.
select_files() {
    python3 -I - "$@" <<'PY'
import os
import re
import subprocess
import sys

root, commit, sha_v1, sha_ct, utils, sha_utils, patterns, dest = sys.argv[1:9]
SUBMODULES = ("external/champollion_V1", "external/cortical_tiles")
CT_FIXTURE_DIRS = ("data/source/", "data/reference/", "data/target/")
EXCLUDED_SUFFIXES = (".csv", ".ipynb")
trees = [
    (root, commit, "", ()),
    (os.path.join(root, SUBMODULES[0]), sha_v1, SUBMODULES[0] + "/", ()),
    (os.path.join(root, SUBMODULES[1]), sha_ct, SUBMODULES[1] + "/", CT_FIXTURE_DIRS),
    (utils, sha_utils, "champollion_utils/", ()),
]
banned = [re.compile(p, re.IGNORECASE) for p in patterns.split()]
errors, plan = [], []
for repo, sha, prefix, excluded_dirs in trees:
    listing = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "-z", "--full-tree", sha], check=True, capture_output=True
    ).stdout
    for record in filter(None, listing.split(b"\0")):
        meta, path = record.decode().split("\t", 1)
        mode, _kind, blob = meta.split()
        if mode == "160000":
            if not (prefix == "" and path in SUBMODULES):
                errors.append(f"unexpected gitlink: {prefix}{path}")
            continue
        if path.lower().endswith(EXCLUDED_SUFFIXES) or path.startswith(excluded_dirs):
            continue
        rel = prefix + path
        if "\n" in rel or "\\" in rel:
            errors.append(f"unsupported path name: {rel!r}")
            continue
        hits = [p.pattern for p in banned if p.search(rel)]
        errors.extend(f"banned data path: {rel} (pattern {p})" for p in hits)
        plan.append((repo, blob, mode, rel))
if errors:
    for error in errors:
        print(f"pack_code: {error}", file=sys.stderr)
    sys.exit(1)
if dest == "-":
    sys.exit(0)
by_repo = {}
for repo, blob, mode, rel in plan:
    by_repo.setdefault(repo, []).append((blob, mode, rel))
for repo, entries in by_repo.items():
    batch = subprocess.Popen(
        ["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    for blob, mode, rel in entries:
        batch.stdin.write(blob.encode() + b"\n")
        batch.stdin.flush()
        size = int(batch.stdout.readline().split()[2])
        data = batch.stdout.read(size)
        batch.stdout.read(1)
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if mode == "120000":
            os.symlink(data, target)
            continue
        with open(target, "wb") as handle:
            handle.write(data)
        os.chmod(target, 0o555 if mode == "100755" else 0o444)
    batch.stdin.close()
    if batch.wait() != 0:
        sys.exit(f"pack_code: git cat-file failed in {repo}")
PY
}

SELECT_ARGS=("$ROOT" "$COMMIT40" "${GITLINK[external/champollion_V1]}" "${GITLINK[external/cortical_tiles]}" \
    "$UTILS" "$SHA_UTILS" "$E2E_BANNED_DATA_PATTERNS")
select_files "${SELECT_ARGS[@]}" - || exit 1

CREATED_CODE_ROOT=0
if [ ! -d "$E2E_CODE_DIR" ]; then
    mkdir -p "$E2E_CODE_DIR"
    CREATED_CODE_ROOT=1
fi
TMP="$(mktemp -d "${E2E_CODE_DIR%/}/.$NAME.XXXXXX")"

cleanup() {
    if [ -n "${TMP:-}" ] && [ -d "$TMP" ]; then
        chmod -R u+w "$TMP"
        rm -rf "$TMP"
    fi
    if [ "$CREATED_CODE_ROOT" = 1 ] && [ ! -e "$CODE" ]; then
        rmdir "$E2E_CODE_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

select_files "${SELECT_ARGS[@]}" "$TMP"

# Stub repository: cortical_tiles asks gitpython for a HEAD. One empty commit, deterministic per commit.
STUB="$TMP/external/cortical_tiles/.git"
mkdir -p "$TMP/external/cortical_tiles" "$TMP/.template"
git -c init.defaultBranch=main init -q --template="$TMP/.template" "$TMP/external/cortical_tiles" >/dev/null
rmdir "$TMP/.template"
COMMIT_DATE="$(git -C "$ROOT" show -s --format=%cd --date=raw "$COMMIT40")"
EMPTY_TREE="$(git --git-dir="$STUB" hash-object -t tree -w --stdin </dev/null)"
STUB_COMMIT="$(
    GIT_AUTHOR_NAME=pack_code GIT_AUTHOR_EMAIL=pack_code@localhost GIT_AUTHOR_DATE="$COMMIT_DATE" \
        GIT_COMMITTER_NAME=pack_code GIT_COMMITTER_EMAIL=pack_code@localhost GIT_COMMITTER_DATE="$COMMIT_DATE" \
        git --git-dir="$STUB" commit-tree "$EMPTY_TREE" \
        -m "cortical_tiles ${GITLINK[external/cortical_tiles]} packed for $COMMIT40"
)"
git --git-dir="$STUB" update-ref refs/heads/main "$STUB_COMMIT"

echo "$COMMIT40" > "$TMP/COMMIT"
MANIFEST_TEXT="$(cd "$TMP" && find . -type f ! -path ./MANIFEST -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum)"
printf '%s\n' "$MANIFEST_TEXT" > "$TMP/MANIFEST"
find "$TMP" -type f ! -perm -u+x -exec chmod 0444 {} +
find "$TMP" -type f -perm -u+x -exec chmod 0555 {} +
find "$TMP" -type d -exec chmod 0555 {} +
mv -T "$TMP" "$CODE" || die "$CODE could not be created (a concurrent run may have written it)"
TMP=""
echo "pack_code: packed $COMMIT40 into $CODE"
echo "$CODE"

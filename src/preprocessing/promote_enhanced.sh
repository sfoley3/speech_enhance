#!/usr/bin/env bash
#
# promote_enhanced.sh
#
# For each "USC_LSS_*" subdirectory found under ENHANCED_DIR:
#   MOVE the .wav files from the PARENT directory INTO the USC_LSS_* subdir,
#   overwriting any same-named files already in the subdir.
#
# End state: the USC_LSS_* subdir holds the outer wavs; the parent no longer
# has loose .wav files.
#
# Example layout this handles:
#   .../raw/PASE/usc_s1_1.wav             <- moved into the subdir
#   .../raw/PASE/USC_LSS_pase/usc_s1_1.wav   <- overwritten by the outer file
#
# Works for all conditions automatically (USC_LSS_pase / USC_LSS_meta /
# USC_LSS_nvidia) and any group/method nesting, since it globs every
# USC_LSS_* dir.
#
# Usage:
#   ./promote_enhanced.sh [ENHANCED_DIR]            # actually move/overwrite
#   DRYRUN=1 ./promote_enhanced.sh [ENHANCED_DIR]   # show what would happen
#
DEFAULT_DIR="/project2/shrikann_35/sfoley/data/enhanced_audio/raw"

set -euo pipefail

ENHANCED_DIR="${1:-$DEFAULT_DIR}"

if [[ ! -d "$ENHANCED_DIR" ]]; then
    echo "ERROR: directory not found: $ENHANCED_DIR" >&2
    exit 1
fi

echo "Scanning: $ENHANCED_DIR"
[[ "${DRYRUN:-0}" == "1" ]] && echo "(DRY RUN — no files will be changed)"
echo

found=0
while IFS= read -r -d '' subdir; do
    parent="$(dirname "$subdir")"
    found=$((found+1))
    echo ">>> $parent"
    echo "    -> $subdir"

    shopt -s nullglob
    src=("$parent"/*.wav)   # outer wavs (direct children of parent only)
    shopt -u nullglob

    if (( ${#src[@]} == 0 )); then
        echo "    (no loose .wav files in parent; skipping)"
        echo
        continue
    fi

    for f in "${src[@]}"; do
        base="$(basename "$f")"
        if [[ "${DRYRUN:-0}" == "1" ]]; then
            echo "    mv $base"
        else
            mv -f "$f" "$subdir/$base"
        fi
    done
    echo "    ${#src[@]} file(s) moved into $(basename "$subdir") (overwriting)."
    echo
done < <(find "$ENHANCED_DIR" -type d -name 'USC_LSS_*' -print0)

if (( found == 0 )); then
    echo "No USC_LSS_* subdirectories found under $ENHANCED_DIR"
    exit 1
fi

echo "Done. Processed $found USC_LSS_* directory/directories."
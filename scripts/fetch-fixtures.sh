#!/usr/bin/env bash
# Populate tests/fixtures/ with the real clips the fixture tests expect.
#
#   MLV_FIXTURE_DIR=/path/with/clips scripts/fetch-fixtures.sh   # symlink from a local dir
#   MLV_FIXTURE_URL=https://host/path  scripts/fetch-fixtures.sh   # download <URL>/<name>
#
# Expected names (see README "Testing"):
#   lj92_14bit.MLV                   1600x800  14-bit LJ92        814 frames
#   uncompressed_14bit_dualiso.MLV   1600x900  14-bit uncompressed 237 frames
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
dest="$here/tests/fixtures"
mkdir -p "$dest"

names=(
  "lj92_14bit.MLV:M06-1218.MLV"
  "uncompressed_14bit_dualiso.MLV:M23-0728_DualISO.MLV"
)

for entry in "${names[@]}"; do
  name=${entry%%:*}
  orig=${entry##*:}
  target="$dest/$name"
  if [[ -e "$target" ]]; then
    echo "exists  $name"
    continue
  fi
  if [[ -n "${MLV_FIXTURE_DIR:-}" ]]; then
    for cand in "$MLV_FIXTURE_DIR/$name" "$MLV_FIXTURE_DIR/$orig"; do
      if [[ -f "$cand" ]]; then
        ln -s "$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")" "$target"
        echo "linked  $name -> $cand"
        continue 2
      fi
    done
  fi
  if [[ -n "${MLV_FIXTURE_URL:-}" ]]; then
    echo "fetch   $name"
    curl -fL --progress-bar -o "$target.part" "$MLV_FIXTURE_URL/$name"
    mv "$target.part" "$target"
    continue
  fi
  echo "missing $name (set MLV_FIXTURE_DIR or MLV_FIXTURE_URL); its test will be skipped"
done

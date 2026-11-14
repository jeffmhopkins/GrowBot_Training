#!/usr/bin/env bash
# Keep a run's result in the repo: copies the policy JSON, the config that made it and
# its evaluation into policies/<name>/, ready to commit. Takes a run folder or the
# .tar.gz the JBrain launcher hands back.
#   scripts/keep_policy.sh runs/20261114-0930
#   scripts/keep_policy.sh ~/Downloads/growbot_heading_walk_artifacts.tar.gz my-first-walk
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${1:?usage: scripts/keep_policy.sh RUN_DIR_OR_TARBALL [NAME]}
TMP=""
if [ -f "$SRC" ]; then
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  tar xzf "$SRC" -C "$TMP"
  SRC=$(find "$TMP" -name config.json -printf '%h\n' | head -n 1)
  [ -n "$SRC" ] || { echo "no run folder inside the tarball"; exit 1; }
fi
NAME=${2:-$(basename "$SRC")}
DEST="policies/$NAME"
mkdir -p "$DEST"
cp "$SRC"/policy_*.json "$SRC/config.json" "$DEST/"
for f in eval.json eval.txt export_verify.log replay.html; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DEST/"
done
echo "kept $DEST:"
ls -l "$DEST"

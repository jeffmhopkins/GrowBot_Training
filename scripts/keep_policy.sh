#!/usr/bin/env bash
# Keep a run's result in the repo: copies the policy JSON, the config that made it and
# its evaluation into policies/<name>/, ready to commit, and prints the policies/index.toml
# entry that offers it in JBrain's Body tab. Takes a run folder or the .tar.gz the JBrain
# launcher hands back.
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
  CFG=$(find "$TMP" -name config.json | head -n 1)
  [ -n "$CFG" ] || { echo "no run folder inside the tarball"; exit 1; }
  SRC=$(dirname "$CFG")
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
# The Body tab offers it once it's listed in policies/index.toml (and merged to main).
POLICY=$(cd "$DEST" && ls policy_*.json | head -n 1)
SLUG=$(echo "$NAME" | tr 'A-Z_ ' 'a-z--' | tr -cd 'a-z0-9-' | cut -c1-48)
[ "$SLUG" = "$NAME" ] || echo "note: the folder is $NAME but a slug must be a-z, 0-9 and -; rename it to $SLUG to list it"
JOB=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('name', ''))" "$DEST/config.json" 2>/dev/null || true)
cat <<ENTRY

To offer it in JBrain's Body tab, add this to policies/index.toml (fix the title and job):

[[policy]]
slug = "$SLUG"
title = "$NAME"
job = "growbot_${JOB:-heading_walk}"
file = "$DEST/$POLICY"
ENTRY

#!/usr/bin/env bash
# Train, export, evaluate and pack one run:
#   ./run.sh                     -> runs/<date-time>/ and runs/<date-time>.tar.gz
#   ./run.sh my_try --steps 5000000 --set reward.speed_cap_m_s=0.04
# The first argument names the run; everything after it goes to the trainer.
set -euo pipefail
cd "$(dirname "$0")"

NAME=${1:-$(date +%Y%m%d-%H%M%S)}
shift || true
RUN="runs/$NAME"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "run ./install.sh first"; exit 1; }

"$PY" -m growbot_train.train --out "$RUN" "$@"
"$PY" -m growbot_train.evaluate "$RUN"/policy_*.json --sweep --json "$RUN/eval.json" \
  | tee "$RUN/eval.txt"
# A 3D replay page that opens in any browser, phone included (needs no graphics drivers).
"$PY" -m growbot_train.replay "$RUN"/policy_*.json --out "$RUN/replay.html"
tar czf "runs/$NAME.tar.gz" -C runs "$NAME"
echo "== packed runs/$NAME.tar.gz =="

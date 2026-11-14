#!/usr/bin/env bash
# One-time setup: a private Python environment in .venv with everything training needs.
# CPU-only PyTorch on purpose: the networks are tiny and the simulator is the bottleneck,
# so a GPU build would add gigabytes and no speed.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "need Python 3.11 or newer (found $("$PY" --version))"; exit 1; }

"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -e ".[dev]"
.venv/bin/python -c 'import mujoco, stable_baselines3, torch; print("ready: mujoco", mujoco.__version__, "| sb3", stable_baselines3.__version__, "| torch", torch.__version__)'

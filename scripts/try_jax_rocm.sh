#!/usr/bin/env bash
# Optional experiment: can JAX see the Radeon iGPU? Uses its own .venv-jax so it never
# disturbs the CPU training environment. ROCm support for gfx1151 (Strix Halo) is
# experimental; a CPU-only answer here is normal and not a problem.
set -uo pipefail
cd "$(dirname "$0")/.."

# gfx1151 isn't on every ROCm build's list; presenting it as 11.5.1 is what the JBrain
# box's other ROCm containers do.
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.5.1}
python3 -m venv .venv-jax
.venv-jax/bin/python -m pip install --quiet --upgrade pip
if ! .venv-jax/bin/python -m pip install --quiet "jax[rocm]"; then
  echo "no ROCm build of JAX installed for this Python/platform; stay on the CPU path"
  exit 0
fi
.venv-jax/bin/python - <<'PY'
import jax
devs = jax.devices()
print("JAX", jax.__version__, "devices:", devs)
if all(d.platform == "cpu" for d in devs):
    print("JAX can't see the GPU here; stay on the CPU path (./run.sh).")
else:
    print("JAX sees the GPU. See README 'Optional: JAX on the Radeon iGPU' for next steps.")
PY

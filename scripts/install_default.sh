#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/install_default.sh

Default installation steps for the DBPO stage-1 repository:
  1. pip install -e .
  2. install mjrl from the validated GitHub zipball source
  3. install d4rl from the validated GitHub zipball source
  4. install local third_party packages:
     - third_party/metaworld
     - third_party/mj_envs
     - third_party/adroit_metaworld_support
  5. install runtime Python deps that frequently go missing in reused envs:
     - scikit-image
     - pybullet
  6. run a minimal import smoke test
EOF
}

if [[ "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

cd "${REPO_ROOT}"

python -m pip install -e .
python -m pip install "mjrl @ https://api.github.com/repos/aravindr93/mjrl/zipball/master"
python -m pip install --no-deps "d4rl @ https://api.github.com/repos/Farama-Foundation/d4rl/zipball/master"
python -m pip uninstall -y metaworld >/dev/null 2>&1 || true
python -m pip install -e third_party/metaworld
python -m pip install -e third_party/mj_envs
python -m pip install -e third_party/adroit_metaworld_support
python -m pip install scikit-image==0.22.0
python -m pip install pybullet==3.2.6

python - <<'PY'
import gym
import metaworld
import mjrl
import d4rl
import mujoco_py
import mj_envs
import adroit_metaworld_runtime
import skimage
import pybullet
import warnings

print("gym:", gym.__version__)
print("metaworld: ok")
print("mjrl: ok")
print("d4rl: ok")
print("mujoco_py: ok")
print("mj_envs: ok")
print("adroit_metaworld_runtime: ok")
print("skimage:", skimage.__version__)
print("pybullet: ok")

try:
    import mujoco
    print("mujoco:", mujoco.__version__)
except Exception as exc:
    warnings.warn(
        "mujoco import smoke skipped due to runtime backend issue: "
        f"{type(exc).__name__}: {exc}"
    )
PY

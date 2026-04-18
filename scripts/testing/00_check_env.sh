#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/00_check_env.sh [--run|--print-only]

Run stage-1 environment checks and write reports/env_check.txt.
EOF
    common_usage_note
}

SELECTED_FAMILY=""
SELECTED_TASK=""
TASKS_FILE=""
MODE="run"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) MODE="run"; shift ;;
        --print-only) MODE="print-only"; shift ;;
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        --tasks-file) TASKS_FILE="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

REPORT="${DBPO_TEST_REPORT_DIR}/env_check.txt"
COMMANDS=(
    "python --version"
    "python ${REPO_ROOT}/scripts/check_environment.py --group native_suite"
    "python ${REPO_ROOT}/scripts/check_environment.py --group adroit_metaworld_suite"
    "python ${REPO_ROOT}/scripts/check_environment.py --group gym_suite"
)

if [[ "${MODE}" == "print-only" ]]; then
    {
        echo "REPORT=${REPORT}"
        echo "DBPO_DATA_DIR=${DBPO_DATA_DIR}"
        echo "DBPO_LOG_DIR=${DBPO_LOG_DIR}"
        echo "DBPO_TEST_REPORT_DIR=${DBPO_TEST_REPORT_DIR}"
        echo "DBPO_NUM_GPUS=${DBPO_NUM_GPUS}"
        echo "DBPO_NUM_CPUS=${DBPO_NUM_CPUS}"
        echo "COMMANDS:"
        printf '  %s\n' "${COMMANDS[@]}"
    }
    exit 0
fi

mkdir -p "${DBPO_TEST_REPORT_DIR}"
{
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "DBPO_DATA_DIR=${DBPO_DATA_DIR}"
    echo "DBPO_LOG_DIR=${DBPO_LOG_DIR}"
    echo "DBPO_TEST_REPORT_DIR=${DBPO_TEST_REPORT_DIR}"
    echo "DBPO_NUM_GPUS=${DBPO_NUM_GPUS}"
    echo "DBPO_NUM_CPUS=${DBPO_NUM_CPUS}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo
    command -v python || true
    command -v nvidia-smi || true
    command -v conda || true
    echo
    python --version
    python - <<'PY'
import importlib
for name in ("torch", "hydra", "omegaconf"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: MISSING ({type(exc).__name__}: {exc})")
PY
    echo
    REPO_ROOT_ENV="${REPO_ROOT}" python - <<'PY'
import os
from pathlib import Path
repo_root = Path(os.environ["REPO_ROOT_ENV"])
pretrain = repo_root / "pretrain.py"
text = pretrain.read_text(encoding="utf-8")
print(f"pretrain_recursive_false={'hydra.utils.instantiate(cfg, _recursive_=False)' in text}")
for task in ("adroit_hammer", "metaworld_assembly"):
    cfg = (repo_root / "configs" / "task" / f"{task}.yaml").read_text(encoding="utf-8")
    print(f"{task}_point_channels_6={'point_channels: 6' in cfg}")
PY
    echo
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L || true
    fi
    echo
    if command -v conda >/dev/null 2>&1; then
        conda info --envs || true
    fi
    echo
    python "${REPO_ROOT}/scripts/check_environment.py" --group native_suite || true
    python "${REPO_ROOT}/scripts/check_environment.py" --group adroit_metaworld_suite || true
    python "${REPO_ROOT}/scripts/check_environment.py" --group gym_suite || true
} > "${REPORT}" 2>&1

echo "Wrote ${REPORT}"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=("$@")
if [[ $# -ge 1 && "${1}" != --* ]]; then
    ARGS=(--task "$1" "${@:2}")
fi

python "${SCRIPT_DIR}/prepare_pointcloud.py" --suite adroit "${ARGS[@]}"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/download_datasets.sh --benchmark <robomimic|pusht|blockpush>" >&2
    exit 1
fi

BENCHMARK=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)
            BENCHMARK="${2:-}"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${BENCHMARK}" ]]; then
    echo "Missing --benchmark" >&2
    exit 1
fi

bash "${SCRIPT_DIR}/prepare_datasets.sh" "${BENCHMARK}" "${ARGS[@]}"

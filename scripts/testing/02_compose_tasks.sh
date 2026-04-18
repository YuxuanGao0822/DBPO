#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/02_compose_tasks.sh [--run|--print-only]
      [--family native|adroit_metaworld|d4rl_gym|all]
      [--task <task> | --tasks-file <file>]

Compose released tasks with Hydra and write compose_<family>.txt reports.
EOF
    common_usage_note
}

SELECTED_FAMILY="all"
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

tasks="$(resolve_tasks "${SELECTED_FAMILY}")"
validate_task_list <<< "${tasks}"

for family in native adroit_metaworld d4rl_gym; do
    family_tasks="$(printf '%s\n' "${tasks}" | collect_family_tasks "${family}")"
    [[ -z "${family_tasks}" ]] && continue
    report="${DBPO_TEST_REPORT_DIR}/compose_${family}.txt"
    if [[ "${MODE}" == "run" ]]; then
        : > "${report}"
        while IFS= read -r task; do
            [[ -z "${task}" ]] && continue
            {
                echo "===== ${task} ====="
                python "${REPO_ROOT}/scripts/testing/task_driver.py" compose --task "${task}" --report -
                echo
            } >> "${report}" 2>&1
        done <<< "${family_tasks}"
        echo "Wrote ${report}"
    else
        while IFS= read -r task; do
            [[ -z "${task}" ]] && continue
            echo "python ${REPO_ROOT}/scripts/testing/task_driver.py compose --task ${task} --report - >> ${report}"
        done <<< "${family_tasks}"
    fi
done

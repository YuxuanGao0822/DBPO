#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/03_instantiate_smoke.sh [--run|--print-only]
      [--family native|adroit_metaworld|d4rl_gym|all]
      [--task <task> | --tasks-file <file>] [--device <device>]
      [--batch-size-override <n>]

Instantiate dataset, policy, and workspace; run one forward/loss pass without rollout runner construction.
EOF
    common_usage_note
}

SELECTED_FAMILY="all"
SELECTED_TASK=""
TASKS_FILE=""
MODE="run"
DEVICE="cuda:0"
BATCH_SIZE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) MODE="run"; shift ;;
        --print-only) MODE="print-only"; shift ;;
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        --tasks-file) TASKS_FILE="${2:-}"; shift 2 ;;
        --device) DEVICE="${2:-}"; shift 2 ;;
        --batch-size-override) BATCH_SIZE_OVERRIDE="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

tasks="$(resolve_tasks "${SELECTED_FAMILY}")"
validate_task_list <<< "${tasks}"

while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    report="${DBPO_TEST_REPORT_DIR}/instantiate_${task}.txt"
    status_report="${DBPO_TEST_REPORT_DIR}/dataset_status_${task}.txt"
    if [[ "${MODE}" == "run" ]]; then
        if ! python "${REPO_ROOT}/scripts/testing/task_driver.py" dataset-status --task "${task}" --report "${status_report}"; then
            cp "${status_report}" "${report}"
            echo "Skipped ${task} due to missing dataset"
            continue
        fi
    fi
    cmd=(python "${REPO_ROOT}/scripts/testing/task_driver.py" instantiate --task "${task}" --report "${report}" --device "${DEVICE}" --skip-env-runner)
    if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
        cmd+=(--batch-size-override "${BATCH_SIZE_OVERRIDE}")
    fi
    if [[ "${MODE}" == "run" ]]; then
        "${cmd[@]}"
        echo "Wrote ${report}"
    else
        printf '%s\n' "$(quote_cmd "${cmd[@]}")"
    fi
done <<< "${tasks}"

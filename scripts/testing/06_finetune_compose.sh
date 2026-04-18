#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/06_finetune_compose.sh [--family d4rl_gym|all] [--task <task>]

Compose and resolve public stage-2 finetune configs for released gym tasks.
EOF
}

SELECTED_FAMILY="d4rl_gym"
SELECTED_TASK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

resolve_finetune_tasks() {
    if [[ -n "${SELECTED_TASK}" ]]; then
        printf '%s\n' "${SELECTED_TASK}"
        return
    fi
    cat "${MANIFEST_DIR}/gym_finetune_released.txt"
}

tasks="$(resolve_finetune_tasks)"
validate_task_list <<< "${tasks}"

while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    report="${DBPO_TEST_REPORT_DIR}/compose_finetune_${task}.txt"
    python "${REPO_ROOT}/scripts/testing/finetune_driver.py" compose \
        --task "${task}" \
        --actor-policy-path "/tmp/stage1_pretrain_checkpoint.pt" \
        --report "${report}"
    echo "Wrote ${report}"
done <<< "${tasks}"

#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/07_finetune_instantiate_smoke.sh [--family d4rl_gym|all]
      [--task <task>] [--device <device>]

Instantiate public stage-2 gym finetune workspaces and one-step env interaction.
EOF
}

SELECTED_FAMILY="d4rl_gym"
SELECTED_TASK=""
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        --device) DEVICE="${2:-}"; shift 2 ;;
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

pretrain_checkpoint_for_task() {
    local task="$1"
    printf '%s\n' "${DBPO_PRETRAIN_DIR}/${task}/checkpoint/last.pt"
}

tasks="$(resolve_finetune_tasks)"
validate_task_list <<< "${tasks}"

while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    report="${DBPO_TEST_REPORT_DIR}/instantiate_finetune_${task}.txt"
    checkpoint="$(pretrain_checkpoint_for_task "${task}")"
    if [[ ! -f "${checkpoint}" ]]; then
        printf '{\n  "task": "%s",\n  "status": "SKIPPED",\n  "reason": "missing_pretrain_checkpoint",\n  "actor_policy_path": "%s"\n}\n' \
            "${task}" "${checkpoint}" > "${report}"
        echo "Wrote ${report}"
        continue
    fi
    python "${REPO_ROOT}/scripts/testing/task_driver.py" verify-checkpoint \
        --checkpoint "${checkpoint}" \
        --task "${task}" > /dev/null 2>&1
    python "${REPO_ROOT}/scripts/testing/finetune_driver.py" instantiate \
        --task "${task}" \
        --actor-policy-path "${checkpoint}" \
        --report "${report}" \
        --device "${DEVICE}" \
        --n-envs-override 2 \
        --n-steps-override 2 \
        --batch-size-override 4 \
        --n-train-itr-override 1
    printf '\nPRETRAIN_CHECKPOINT=%s\nPRETRAIN_VERIFY_EXIT_CODE=0\n' "${checkpoint}" >> "${report}"
    echo "Wrote ${report}"
done <<< "${tasks}"

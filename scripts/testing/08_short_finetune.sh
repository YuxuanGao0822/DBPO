#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/08_short_finetune.sh [--task <task> | --family d4rl_gym]
      [--n-train-itr-override <n>] [--n-steps-override <n>]
      [--n-envs-override <n>] [--batch-size-override <n>]
      [--update-epochs-override <n>]

Run a short public stage-2 gym finetune smoke and verify the stage-2 checkpoint.
EOF
}

SELECTED_FAMILY="d4rl_gym"
SELECTED_TASK=""
N_TRAIN_ITR_OVERRIDE="${N_TRAIN_ITR_OVERRIDE:-2}"
N_STEPS_OVERRIDE="${N_STEPS_OVERRIDE:-4}"
N_ENVS_OVERRIDE="${N_ENVS_OVERRIDE:-2}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-8}"
UPDATE_EPOCHS_OVERRIDE="${UPDATE_EPOCHS_OVERRIDE:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        --n-train-itr-override) N_TRAIN_ITR_OVERRIDE="${2:-}"; shift 2 ;;
        --n-steps-override) N_STEPS_OVERRIDE="${2:-}"; shift 2 ;;
        --n-envs-override) N_ENVS_OVERRIDE="${2:-}"; shift 2 ;;
        --batch-size-override) BATCH_SIZE_OVERRIDE="${2:-}"; shift 2 ;;
        --update-epochs-override) UPDATE_EPOCHS_OVERRIDE="${2:-}"; shift 2 ;;
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

index=0
while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    report="${DBPO_TEST_REPORT_DIR}/short_finetune_${task}.txt"
    checkpoint="$(pretrain_checkpoint_for_task "${task}")"
    if [[ ! -f "${checkpoint}" ]]; then
        printf 'TASK=%s\nSTATUS=SKIPPED\nREASON=missing_pretrain_checkpoint\nPRETRAIN_CHECKPOINT=%s\n' \
            "${task}" "${checkpoint}" > "${report}"
        echo "Wrote ${report}"
        index=$((index + 1))
        continue
    fi

    pretrain_verify_status=0
    if ! python "${REPO_ROOT}/scripts/testing/task_driver.py" verify-checkpoint \
        --checkpoint "${checkpoint}" \
        --task "${task}" > /dev/null 2>&1; then
        pretrain_verify_status=$?
    fi
    if [[ "${pretrain_verify_status}" -ne 0 ]]; then
        printf 'TASK=%s\nSTATUS=SKIPPED\nREASON=invalid_stage1_checkpoint\nPRETRAIN_CHECKPOINT=%s\nPRETRAIN_VERIFY_EXIT_CODE=%s\n' \
            "${task}" "${checkpoint}" "${pretrain_verify_status}" >> "${report}"
        echo "Wrote ${report}"
        index=$((index + 1))
        continue
    fi

    gpu="$(gpu_for_index "${index}")"
    output_dir="${DBPO_LOG_DIR}/server_finetune/${task}"
    finetune_checkpoint="${output_dir}/checkpoint/last.pt"
    cmd=(
        python "${REPO_ROOT}/finetune.py"
        "task=${task}"
        "actor_policy_path=${checkpoint}"
        "device=cuda:0"
        "logdir=${output_dir}"
        "hydra.run.dir=${output_dir}"
        "wandb.offline_mode=true"
        "task.env_cfg.n_envs=${N_ENVS_OVERRIDE}"
        "train.n_train_itr=${N_TRAIN_ITR_OVERRIDE}"
        "train.n_steps=${N_STEPS_OVERRIDE}"
        "train.batch_size=${BATCH_SIZE_OVERRIDE}"
        "train.update_epochs=${UPDATE_EPOCHS_OVERRIDE}"
        "train.n_critic_warmup_itr=0"
        "train.val_freq=1"
        "train.save_model_freq=1"
    )
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" HYDRA_FULL_ERROR=1 "${cmd[@]}" > "${report}" 2>&1
    status=$?
    set -e
    if [[ "${status}" -eq 0 ]]; then
        set +e
        python "${REPO_ROOT}/scripts/testing/finetune_driver.py" verify-checkpoint \
            --checkpoint "${finetune_checkpoint}" \
            --task "${task}" >> "${report}" 2>&1
        verify_status=$?
        set -e
    else
        verify_status=99
        printf 'CHECKPOINT_VERIFY_SKIPPED=1\n' >> "${report}"
    fi
    printf 'TASK=%s\nGPU=%s\nPRETRAIN_CHECKPOINT=%s\nPRETRAIN_VERIFY_EXIT_CODE=%s\nOUTPUT_DIR=%s\nCHECKPOINT_DIR=%s\nEXIT_CODE=%s\nVERIFY_EXIT_CODE=%s\n' \
        "${task}" "${gpu}" "${checkpoint}" "${pretrain_verify_status}" "${output_dir}" "${output_dir}/checkpoint" "${status}" "${verify_status}" >> "${report}"
    if [[ "${status}" -ne 0 || "${verify_status}" -ne 0 ]]; then
        exit 1
    fi
    echo "Wrote ${report}"
    index=$((index + 1))
done <<< "${tasks}"

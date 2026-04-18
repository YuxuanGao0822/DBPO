#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/09_finetune_matrix.sh [--run|--print-only]
      [--family d4rl_gym|all] [--task <task>]
      [--n-train-itr-override <n>] [--n-steps-override <n>]
      [--n-envs-override <n>] [--batch-size-override <n>]
      [--update-epochs-override <n>]

Generate or execute public stage-2 gym finetune command matrices.
EOF
}

SELECTED_FAMILY="d4rl_gym"
SELECTED_TASK=""
MODE="print-only"
N_TRAIN_ITR_OVERRIDE=""
N_STEPS_OVERRIDE=""
N_ENVS_OVERRIDE=""
BATCH_SIZE_OVERRIDE=""
UPDATE_EPOCHS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) MODE="run"; shift ;;
        --print-only) MODE="print-only"; shift ;;
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

matrix="${DBPO_TEST_REPORT_DIR}/finetune_matrix_d4rl_gym.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
} > "${matrix}"

index=0
while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    report="${DBPO_TEST_REPORT_DIR}/full_finetune_${task}.txt"
    checkpoint="$(pretrain_checkpoint_for_task "${task}")"
    if [[ "${MODE}" == "run" && ! -f "${checkpoint}" ]]; then
        echo "Skipping ${task}: missing stage-1 checkpoint ${checkpoint}"
        continue
    fi
    gpu="$(gpu_for_index "${index}")"
    output_dir="${DBPO_LOG_DIR}/server_finetune_full/${task}"
    finetune_checkpoint="${output_dir}/checkpoint/last.pt"
    cmd="CUDA_VISIBLE_DEVICES=${gpu} HYDRA_FULL_ERROR=1 python ${REPO_ROOT}/finetune.py task=${task} actor_policy_path=${checkpoint} device=cuda:0 logdir=${output_dir} hydra.run.dir=${output_dir} wandb.offline_mode=true"
    if [[ -n "${N_ENVS_OVERRIDE}" ]]; then
        cmd="${cmd} task.env_cfg.n_envs=${N_ENVS_OVERRIDE}"
    fi
    if [[ -n "${N_TRAIN_ITR_OVERRIDE}" ]]; then
        cmd="${cmd} train.n_train_itr=${N_TRAIN_ITR_OVERRIDE}"
    fi
    if [[ -n "${N_STEPS_OVERRIDE}" ]]; then
        cmd="${cmd} train.n_steps=${N_STEPS_OVERRIDE}"
    fi
    if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
        cmd="${cmd} train.batch_size=${BATCH_SIZE_OVERRIDE}"
    fi
    if [[ -n "${UPDATE_EPOCHS_OVERRIDE}" ]]; then
        cmd="${cmd} train.update_epochs=${UPDATE_EPOCHS_OVERRIDE}"
    fi
    cmd="${cmd} > ${report} 2>&1; status=\$?; if [ \$status -eq 0 ]; then python ${REPO_ROOT}/scripts/testing/finetune_driver.py verify-checkpoint --checkpoint ${finetune_checkpoint} --task ${task} >> ${report} 2>&1; verify_status=\$?; else verify_status=99; printf 'CHECKPOINT_VERIFY_SKIPPED=1\n' >> ${report}; fi; printf 'TASK=%s\nGPU=%s\nPRETRAIN_CHECKPOINT=%s\nOUTPUT_DIR=%s\nCHECKPOINT_DIR=%s\nEXIT_CODE=%s\nVERIFY_EXIT_CODE=%s\n' ${task} ${gpu} ${checkpoint} ${output_dir} ${output_dir}/checkpoint \$status \$verify_status >> ${report}; test \$status -eq 0 && test \$verify_status -eq 0"
    echo "${cmd}" >> "${matrix}"
    index=$((index + 1))
done <<< "${tasks}"
chmod +x "${matrix}"
echo "Wrote ${matrix}"

readme_path="${DBPO_TEST_REPORT_DIR}/finetune_README.md"
cat > "${readme_path}" <<EOF
# Finetune Matrix

Generated with:

- family: \`${SELECTED_FAMILY}\`
- n_train_itr override: $( [[ -n "${N_TRAIN_ITR_OVERRIDE}" ]] && printf '\`%s\`' "${N_TRAIN_ITR_OVERRIDE}" || printf '\`default\`' )
- n_steps override: $( [[ -n "${N_STEPS_OVERRIDE}" ]] && printf '\`%s\`' "${N_STEPS_OVERRIDE}" || printf '\`default\`' )
- n_envs override: $( [[ -n "${N_ENVS_OVERRIDE}" ]] && printf '\`%s\`' "${N_ENVS_OVERRIDE}" || printf '\`default\`' )
- batch size override: $( [[ -n "${BATCH_SIZE_OVERRIDE}" ]] && printf '\`%s\`' "${BATCH_SIZE_OVERRIDE}" || printf '\`default\`' )
- update epochs override: $( [[ -n "${UPDATE_EPOCHS_OVERRIDE}" ]] && printf '\`%s\`' "${UPDATE_EPOCHS_OVERRIDE}" || printf '\`default\`' )

Semantics:

- this matrix covers the default public stage-2 gym finetune set
- each run consumes a stage-1 pretrain checkpoint from \`${DBPO_PRETRAIN_DIR}\`
- each successful run verifies the resulting stage-2 checkpoint
EOF
echo "Wrote ${readme_path}"

if [[ "${MODE}" == "print-only" ]]; then
    cat "${matrix}"
    exit 0
fi

awk 'NR > 2 && NF' "${matrix}" | xargs -d '\n' -I{} -P "$(gpu_count)" bash -lc "{}"

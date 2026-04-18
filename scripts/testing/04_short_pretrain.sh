#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/04_short_pretrain.sh [--run|--print-only]
      [--family native|adroit_metaworld|d4rl_gym|all]
      [--task <task> | --tasks-file <file>] [--batch-size-override <n>]

Generate or execute short DBP pretrain smoke runs without rollout evaluation.
EOF
    common_usage_note
}

SELECTED_FAMILY="smoke"
SELECTED_TASK=""
TASKS_FILE=""
MODE="run"
BATCH_SIZE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) MODE="run"; shift ;;
        --print-only) MODE="print-only"; shift ;;
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        --tasks-file) TASKS_FILE="${2:-}"; shift 2 ;;
        --batch-size-override) BATCH_SIZE_OVERRIDE="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "${SELECTED_FAMILY}" == "smoke" && -z "${SELECTED_TASK}" && -z "${TASKS_FILE}" ]]; then
    tasks="$(read_tasks_from_manifest "$(manifest_for_family smoke)")"
else
    tasks="$(resolve_tasks "${SELECTED_FAMILY}")"
fi
validate_task_list <<< "${tasks}"

matrix="${DBPO_TEST_REPORT_DIR}/short_pretrain_matrix.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
} > "${matrix}"

index=0
while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    report="${DBPO_TEST_REPORT_DIR}/short_pretrain_${task}.txt"
    if [[ "${MODE}" == "run" ]]; then
        if ! python "${REPO_ROOT}/scripts/testing/task_driver.py" dataset-status --task "${task}" --report "${report}"; then
            echo "Skipping ${task}: dataset not prepared"
            continue
        fi
    fi
    gpu="$(gpu_for_index "${index}")"
    output_dir="${DBPO_LOG_DIR}/server_smoke/${task}"
    checkpoint_path="${output_dir}/checkpoint/last.pt"
    cmd="CUDA_VISIBLE_DEVICES=${gpu} HYDRA_FULL_ERROR=1 python ${REPO_ROOT}/pretrain.py task=${task} device=cuda:0 logdir=${output_dir} hydra.run.dir=${output_dir} env_runner=null train.n_epochs=2 train.save_model_freq=1 train.val_freq=999999 train.num_workers=2 wandb.offline_mode=true"
    if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
        cmd="${cmd} train.batch_size=${BATCH_SIZE_OVERRIDE}"
    fi
    cmd="${cmd} > ${report} 2>&1; status=\$?; if [ \$status -eq 0 ]; then python ${REPO_ROOT}/scripts/testing/task_driver.py verify-checkpoint --checkpoint ${checkpoint_path} --task ${task} >> ${report} 2>&1; verify_status=\$?; else verify_status=99; printf 'CHECKPOINT_VERIFY_SKIPPED=1\n' >> ${report}; fi; printf 'TASK=%s\nGPU=%s\nOUTPUT_DIR=%s\nCHECKPOINT_DIR=%s\nEXIT_CODE=%s\nVERIFY_EXIT_CODE=%s\n' ${task} ${gpu} ${output_dir} ${output_dir}/checkpoint \$status \$verify_status >> ${report}; test \$status -eq 0 && test \$verify_status -eq 0"
    echo "${cmd}" >> "${matrix}"
    index=$((index + 1))
done <<< "${tasks}"

chmod +x "${matrix}"
echo "Wrote ${matrix}"

if [[ "${MODE}" == "print-only" ]]; then
    cat "${matrix}"
    exit 0
fi

awk 'NR > 2 && NF' "${matrix}" | xargs -d '\n' -I{} -P "$(gpu_count)" bash -lc "{}"

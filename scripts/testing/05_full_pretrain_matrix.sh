#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/05_full_pretrain_matrix.sh [--run|--print-only]
      [--family native|adroit_metaworld|d4rl_gym|all]
      [--task <task> | --tasks-file <file>] [--batch-size-override <n>]
      [--epochs-override <n>]
      [--scope validated|all-configured] [--with-eval]
      [--gen-per-label-override <n>]

Generate or execute full DBP pretrain command matrices.
EOF
    common_usage_note
}

SELECTED_FAMILY="all"
SELECTED_TASK=""
TASKS_FILE=""
MODE="print-only"
BATCH_SIZE_OVERRIDE=""
EPOCHS_OVERRIDE=""
GEN_PER_LABEL_OVERRIDE=""
SCOPE="all-configured"
WITH_EVAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) MODE="run"; shift ;;
        --print-only) MODE="print-only"; shift ;;
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        --tasks-file) TASKS_FILE="${2:-}"; shift 2 ;;
        --batch-size-override) BATCH_SIZE_OVERRIDE="${2:-}"; shift 2 ;;
        --epochs-override) EPOCHS_OVERRIDE="${2:-}"; shift 2 ;;
        --gen-per-label-override) GEN_PER_LABEL_OVERRIDE="${2:-}"; shift 2 ;;
        --scope) SCOPE="${2:-}"; shift 2 ;;
        --with-eval) WITH_EVAL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

case "${SCOPE}" in
    validated|all-configured) ;;
    *)
        echo "Unsupported --scope: ${SCOPE}" >&2
        usage
        exit 1
        ;;
esac

manifest_for_scope_family() {
    local scope="$1"
    local family="$2"
    case "${scope}:${family}" in
        validated:native)
            printf '%s\n' "${MANIFEST_DIR}/native_validated.txt"
            ;;
        validated:adroit_metaworld)
            printf '%s\n' "${MANIFEST_DIR}/adroit_metaworld_validated.txt"
            ;;
        validated:d4rl_gym)
            printf '%s\n' "${MANIFEST_DIR}/gym_validated.txt"
            ;;
        all-configured:native)
            printf '%s\n' "${MANIFEST_DIR}/native_all.txt"
            ;;
        all-configured:adroit_metaworld)
            printf '%s\n' "${MANIFEST_DIR}/adroit_metaworld_all.txt"
            ;;
        all-configured:d4rl_gym)
            printf '%s\n' "${MANIFEST_DIR}/gym_all.txt"
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_scope_tasks() {
    local default_family="$1"
    local family="${SELECTED_FAMILY:-${default_family}}"

    if [[ -n "${SELECTED_TASK:-}" ]]; then
        printf '%s\n' "${SELECTED_TASK}"
        return
    fi
    if [[ -n "${TASKS_FILE:-}" ]]; then
        grep -v '^[[:space:]]*$' "${TASKS_FILE}"
        return
    fi
    if [[ "${family}" == "all" ]]; then
        cat \
            "$(manifest_for_scope_family "${SCOPE}" native)" \
            "$(manifest_for_scope_family "${SCOPE}" adroit_metaworld)" \
            "$(manifest_for_scope_family "${SCOPE}" d4rl_gym)"
        return
    fi
    read_tasks_from_manifest "$(manifest_for_scope_family "${SCOPE}" "${family}")"
}

tasks="$(resolve_scope_tasks "${SELECTED_FAMILY}")"
validate_task_list <<< "${tasks}"
generated_families=()

for family in native adroit_metaworld d4rl_gym; do
    family_tasks="$(printf '%s\n' "${tasks}" | collect_family_tasks "${family}")"
    [[ -z "${family_tasks}" ]] && continue
    generated_families+=("${family}")
    matrix="${DBPO_TEST_REPORT_DIR}/full_pretrain_matrix_${family}.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
    } > "${matrix}"
    index=0
    while IFS= read -r task; do
        [[ -z "${task}" ]] && continue
        report="${DBPO_TEST_REPORT_DIR}/full_pretrain_${task}.txt"
        if [[ "${MODE}" == "run" ]]; then
            if ! python "${REPO_ROOT}/scripts/testing/task_driver.py" dataset-status --task "${task}" --report "${report}"; then
                echo "Skipping ${task}: dataset not prepared"
                continue
            fi
        fi
        gpu="$(gpu_for_index "${index}")"
        output_dir="${DBPO_LOG_DIR}/server_full/${task}"
        checkpoint_path="${output_dir}/checkpoint/last.pt"
        cmd="CUDA_VISIBLE_DEVICES=${gpu} HYDRA_FULL_ERROR=1 python ${REPO_ROOT}/pretrain.py task=${task} device=cuda:0 logdir=${output_dir} hydra.run.dir=${output_dir}"
        if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
            cmd="${cmd} train.batch_size=${BATCH_SIZE_OVERRIDE}"
        fi
        if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
            cmd="${cmd} train.n_epochs=${EPOCHS_OVERRIDE}"
        fi
        if [[ -n "${GEN_PER_LABEL_OVERRIDE}" ]]; then
            cmd="${cmd} model.gen_per_label=${GEN_PER_LABEL_OVERRIDE}"
        fi
        if [[ "${WITH_EVAL}" -eq 0 ]]; then
            cmd="${cmd} env_runner=null train.val_freq=999999"
        fi
        cmd="${cmd} > ${report} 2>&1; status=\$?; if [ \$status -eq 0 ]; then python ${REPO_ROOT}/scripts/testing/task_driver.py verify-checkpoint --checkpoint ${checkpoint_path} --task ${task} >> ${report} 2>&1; verify_status=\$?; else verify_status=99; printf 'CHECKPOINT_VERIFY_SKIPPED=1\n' >> ${report}; fi; printf 'TASK=%s\nGPU=%s\nOUTPUT_DIR=%s\nCHECKPOINT_DIR=%s\nEXIT_CODE=%s\nVERIFY_EXIT_CODE=%s\n' ${task} ${gpu} ${output_dir} ${output_dir}/checkpoint \$status \$verify_status >> ${report}; test \$status -eq 0 && test \$verify_status -eq 0"
        echo "${cmd}" >> "${matrix}"
        index=$((index + 1))
    done <<< "${family_tasks}"
    chmod +x "${matrix}"
    echo "Wrote ${matrix}"
done

readme_path="${DBPO_TEST_REPORT_DIR}/full_pretrain_README.md"
cat > "${readme_path}" <<EOF
# Full Pretrain Matrix

Generated with:

- family: \`${SELECTED_FAMILY}\`
- scope: \`${SCOPE}\`
- eval mode: $( [[ "${WITH_EVAL}" -eq 1 ]] && printf '\`with-eval\`' || printf '\`train-only\`' )
- batch size override: $( [[ -n "${BATCH_SIZE_OVERRIDE}" ]] && printf '\`%s\`' "${BATCH_SIZE_OVERRIDE}" || printf '\`default\`' )
- epochs override: $( [[ -n "${EPOCHS_OVERRIDE}" ]] && printf '\`%s\`' "${EPOCHS_OVERRIDE}" || printf '\`default\`' )
- gen_per_label override: $( [[ -n "${GEN_PER_LABEL_OVERRIDE}" ]] && printf '\`%s\`' "${GEN_PER_LABEL_OVERRIDE}" || printf '\`default\`' )

Semantics:

- \`all-configured\`: default public stage-1 boundary from all configured task manifests
- \`validated\`: fast-smoke / example subset already verified on the server through short pretrain
- default generated commands are \`train-only\` and append \`env_runner=null train.val_freq=999999\`
- pass \`--with-eval\` only if rollout backends are intentionally part of the full run

Recommended order:

1. 先跑 \`00_check_env.sh\`
2. 再跑 \`01_prepare_data.sh\`
3. 再跑 \`02_compose_tasks.sh\`
4. 再跑 \`03_instantiate_smoke.sh\`
5. 再跑 \`04_short_pretrain.sh\`
6. 最后根据 \`full_pretrain_matrix_*.sh\` 手动调度正式训练
EOF
echo "Wrote ${readme_path}"

if [[ "${MODE}" == "print-only" ]]; then
    for family in "${generated_families[@]}"; do
        matrix="${DBPO_TEST_REPORT_DIR}/full_pretrain_matrix_${family}.sh"
        [[ -f "${matrix}" ]] && cat "${matrix}"
    done
    exit 0
fi

for family in "${generated_families[@]}"; do
    matrix="${DBPO_TEST_REPORT_DIR}/full_pretrain_matrix_${family}.sh"
    [[ -f "${matrix}" ]] || continue
    awk 'NR > 2 && NF' "${matrix}" | xargs -d '\n' -I{} -P "$(gpu_count)" bash -lc "{}"
done

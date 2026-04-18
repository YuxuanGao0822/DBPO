#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MANIFEST_DIR="${REPO_ROOT}/testing/manifests"

export DBPO_DATA_DIR="${DBPO_DATA_DIR:-${REPO_ROOT}/data}"
export DBPO_LOG_DIR="${DBPO_LOG_DIR:-${REPO_ROOT}/outputs}"
export DBPO_PRETRAIN_DIR="${DBPO_PRETRAIN_DIR:-${DBPO_LOG_DIR}/server_full}"
export DBPO_TEST_REPORT_DIR="${DBPO_TEST_REPORT_DIR:-${REPO_ROOT}/reports}"
export DBPO_NUM_GPUS="${DBPO_NUM_GPUS:-1}"
export DBPO_NUM_CPUS="${DBPO_NUM_CPUS:-8}"

mkdir -p "${DBPO_TEST_REPORT_DIR}"

common_usage_note() {
    cat <<'EOF'
Common options:
  --run                  Execute commands.
  --print-only           Print planned commands without executing them.
  --family <family>      One of native, native_representative,
                         adroit_metaworld, adroit_metaworld_representative,
                         d4rl_gym, gym_representative, all.
  --task <task>          Single released task.
  --tasks-file <file>    File with one task per line.
EOF
}

parse_mode() {
    local default_mode="$1"
    MODE="${default_mode}"
    while [[ $# -gt 1 ]]; do
        shift
        case "$1" in
            --run)
                MODE="run"
                ;;
            --print-only)
                MODE="print-only"
                ;;
        esac
    done
}

gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
    else
        printf '%s\n' "${DBPO_NUM_GPUS}"
    fi
}

gpu_for_index() {
    local index="$1"
    local gpu_list
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
    else
        gpu_list=()
        local i
        for ((i=0; i<DBPO_NUM_GPUS; i++)); do
            gpu_list+=("${i}")
        done
    fi
    local n="${#gpu_list[@]}"
    if [[ "${n}" -eq 0 ]]; then
        printf '0\n'
        return
    fi
    printf '%s\n' "${gpu_list[$((index % n))]}"
}

task_family() {
    local task="$1"
    case "${task}" in
        can_*|lift_*|square_*|tool_hang_*|transport_*|pusht_*|blockpush_*|kitchen_*)
            printf 'native\n'
            ;;
        adroit_*|metaworld_*)
            printf 'adroit_metaworld\n'
            ;;
        hopper-medium-v2|ant-medium-expert-v2|walker2d-medium-v2)
            printf 'd4rl_gym\n'
            ;;
        *)
            return 1
            ;;
    esac
}

native_suite_for_task() {
    local task="$1"
    case "${task}" in
        can_*|lift_*|square_*|tool_hang_*|transport_*)
            printf 'robomimic\n'
            ;;
        pusht_*)
            printf 'pusht\n'
            ;;
        blockpush_*)
            printf 'blockpush\n'
            ;;
        kitchen_*)
            printf 'kitchen\n'
            ;;
        *)
            return 1
            ;;
    esac
}

manifest_for_family() {
    local family="$1"
    case "${family}" in
        native)
            printf '%s\n' "${MANIFEST_DIR}/native_all.txt"
            ;;
        native_representative)
            printf '%s\n' "${MANIFEST_DIR}/native_representative.txt"
            ;;
        adroit_metaworld)
            printf '%s\n' "${MANIFEST_DIR}/adroit_metaworld_all.txt"
            ;;
        adroit_metaworld_representative)
            printf '%s\n' "${MANIFEST_DIR}/adroit_metaworld_representative.txt"
            ;;
        d4rl_gym)
            printf '%s\n' "${MANIFEST_DIR}/gym_all.txt"
            ;;
        gym_representative)
            printf '%s\n' "${MANIFEST_DIR}/gym_representative.txt"
            ;;
        smoke)
            printf '%s\n' "${MANIFEST_DIR}/smoke_tasks.txt"
            ;;
        *)
            return 1
            ;;
    esac
}

read_tasks_from_manifest() {
    local manifest="$1"
    grep -v '^[[:space:]]*$' "${manifest}"
}

resolve_tasks() {
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
            "$(manifest_for_family native)" \
            "$(manifest_for_family adroit_metaworld)" \
            "$(manifest_for_family d4rl_gym)"
        return
    fi
    read_tasks_from_manifest "$(manifest_for_family "${family}")"
}

validate_task_exists() {
    local task="$1"
    local cfg="${REPO_ROOT}/configs/task/${task}.yaml"
    if [[ ! -f "${cfg}" ]]; then
        echo "Missing task config: ${cfg}" >&2
        return 1
    fi
}

validate_task_list() {
    local task
    while IFS= read -r task; do
        [[ -z "${task}" ]] && continue
        validate_task_exists "${task}"
    done
}

collect_family_tasks() {
    local family="$1"
    local task
    while IFS= read -r task; do
        [[ -z "${task}" ]] && continue
        if [[ "$(task_family "${task}")" == "${family}" ]]; then
            printf '%s\n' "${task}"
        fi
    done
}

quote_cmd() {
    printf '%q ' "$@"
}

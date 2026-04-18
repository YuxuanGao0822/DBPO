#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/testing/01_prepare_data.sh [--run|--print-only] [--dry-run] [--force-download]
      [--family native|adroit_metaworld|d4rl_gym|all]
      [--task <task> | --tasks-file <file>]

Prepare released stage-1 datasets and write data preparation reports.
EOF
    common_usage_note
}

SELECTED_FAMILY="all"
SELECTED_TASK=""
TASKS_FILE=""
MODE="run"
DRY_RUN=0
FORCE_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) MODE="run"; shift ;;
        --print-only) MODE="print-only"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --force-download) FORCE_DOWNLOAD=1; shift ;;
        --family) SELECTED_FAMILY="${2:-}"; shift 2 ;;
        --task) SELECTED_TASK="${2:-}"; shift 2 ;;
        --tasks-file) TASKS_FILE="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

tasks="$(resolve_tasks "${SELECTED_FAMILY}")"
validate_task_list <<< "${tasks}"

COMMAND_FILE="${DBPO_TEST_REPORT_DIR}/data_prepare_commands.sh"
mkdir -p "${DBPO_TEST_REPORT_DIR}"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
} > "${COMMAND_FILE}"

emit_family_command() {
    local family="$1"
    local command="$2"
    local report="${DBPO_TEST_REPORT_DIR}/data_prepare_${family}.txt"
    echo "${command}" >> "${COMMAND_FILE}"
    if [[ "${MODE}" == "run" ]]; then
        {
            echo "===== ${command} ====="
            bash -lc "${command}"
            echo
        } >> "${report}" 2>&1
        echo "Wrote ${report}"
    else
        printf '%s\n' "${command}"
    fi
}

prepare_native() {
    local suites_text="$1"
    local suite
    while IFS= read -r suite; do
        [[ -z "${suite}" ]] && continue
        local need_abs="$2"
        local command="bash ${REPO_ROOT}/scripts/prepare_datasets.sh native --suite ${suite} --data-dir ${DBPO_DATA_DIR}"
        if [[ "${need_abs}" == "1" ]]; then
            command="${command} --prepare-abs"
        fi
        if [[ "${FORCE_DOWNLOAD}" -eq 1 ]]; then
            command="${command} --force-download"
        fi
        emit_family_command "native" "${command}"
    done <<< "${suites_text}"
}

prepare_adroit_metaworld() {
    local suites_text="$1"
    local suite
    while IFS= read -r suite; do
        [[ -z "${suite}" ]] && continue
        local command="bash ${REPO_ROOT}/scripts/prepare_datasets.sh adroit_metaworld --suite ${suite} --all --data-dir ${DBPO_DATA_DIR}"
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            command="${command} --dry-run"
        fi
        emit_family_command "adroit_metaworld" "${command}"
    done <<< "${suites_text}"
}

prepare_gym() {
    local envs_text="$1"
    local env
    while IFS= read -r env; do
        [[ -z "${env}" ]] && continue
        local command="bash ${REPO_ROOT}/scripts/prepare_datasets.sh d4rl_gym --env ${env} --data-dir ${DBPO_DATA_DIR}"
        emit_family_command "d4rl_gym" "${command}"
    done <<< "${envs_text}"
}

if [[ -n "${SELECTED_TASK}" || -n "${TASKS_FILE}" ]]; then
    native_suites=""
    adroit_metaworld_suites=""
    gym_envs=""
    need_abs=0
    while IFS= read -r task; do
        [[ -z "${task}" ]] && continue
        family="$(task_family "${task}")"
        case "${family}" in
            native)
                native_suites="${native_suites}"$'\n'"$(native_suite_for_task "${task}")"
                [[ "${task}" == *_abs ]] && need_abs=1
                ;;
            adroit_metaworld)
                if [[ "${task}" == adroit_* ]]; then
                    adroit_metaworld_suites="${adroit_metaworld_suites}"$'\n'"adroit"
                else
                    adroit_metaworld_suites="${adroit_metaworld_suites}"$'\n'"metaworld"
                fi
                ;;
            d4rl_gym)
                gym_envs="${gym_envs}"$'\n'"${task}"
                ;;
        esac
    done <<< "${tasks}"
    [[ -n "${native_suites}" ]] && prepare_native "$(printf '%s\n' "${native_suites}" | sort -u)" "${need_abs}"
    [[ -n "${adroit_metaworld_suites}" ]] && prepare_adroit_metaworld "$(printf '%s\n' "${adroit_metaworld_suites}" | sort -u)"
    [[ -n "${gym_envs}" ]] && prepare_gym "$(printf '%s\n' "${gym_envs}" | sort -u)"
else
    case "${SELECTED_FAMILY}" in
        all)
            prepare_native "all" "1"
            prepare_adroit_metaworld $'adroit\nmetaworld'
            prepare_gym "$(cat "$(manifest_for_family d4rl_gym)")"
            ;;
        native)
            prepare_native "all" "1"
            ;;
        adroit_metaworld)
            prepare_adroit_metaworld $'adroit\nmetaworld'
            ;;
        d4rl_gym)
            prepare_gym "$(cat "$(manifest_for_family d4rl_gym)")"
            ;;
    esac
fi

chmod +x "${COMMAND_FILE}"
echo "Wrote ${COMMAND_FILE}"

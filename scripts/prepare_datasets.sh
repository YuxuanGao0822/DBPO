#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DBPO_DATA_DIR:-${REPO_ROOT}/data}"
NATIVE_TASK_DATA_URL="https://diffusion-policy.cs.columbia.edu/data/training"
D4RL_RELEASE_ENVS=("hopper-medium-v2" "ant-medium-expert-v2" "walker2d-medium-v2")
ROBOMIMIC_TASKS=("can" "lift" "square" "tool_hang" "transport")

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/prepare_datasets.sh <command> [options]

Commands:
  native   Prepare native simulated manipulation tasks used by DBP pretraining.
  adroit_metaworld   Prepare Adroit / MetaWorld point-cloud datasets.
  d4rl_gym           Prepare the released gym datasets.

native options:
  --suite all|robomimic|pusht|blockpush|kitchen
  --prepare-abs      Convert RoboMimic and BlockPush abs-action datasets after base download.
  --force-download   Re-download archives even if the repository-facing data contract is already satisfied.

adroit_metaworld options:
  --suite adroit|metaworld|all
  --task <task>      Single task name. Omit only with --all.
  --all              Expand to all supported tasks in the chosen suite.
  --num-episodes <n> Number of episodes to generate, useful for smoke tests.
  --dry-run          Print the upstream command without running point-cloud generation.

d4rl_gym options:
  --env <env>        Public gym set: hopper-medium-v2, ant-medium-expert-v2, walker2d-medium-v2
  --all              Prepare the full public gym set.

Common options:
  --data-dir <dir>   Override DBPO_DATA_DIR for this command.

Notes:
  - This script only standardizes the repository-facing data contract.
  - Point-cloud generation and any heavyweight simulator validation still require server verification.
  - Adroit expert checkpoints are searched under ${DBPO_ASSET_DIR:-<repo>/.assets}/adroit/vrl3_ckpts.
  - By default, missing Adroit checkpoints are fetched from the public VRL3 checkpoint archive.
  - Override Adroit checkpoint retrieval with DBPO_ADROIT_VRL3_CKPT_URL,
    DBPO_ADROIT_VRL3_CKPT_ONEDRIVE_URL, DBPO_ADROIT_VRL3_CKPT_ARCHIVE, or
    DBPO_ADROIT_VRL3_CKPT_DIR.
  - kitchen_demos_multitask is a required path for kitchen abs-action pretraining and is documented, not generated locally here.
EOF
}

require_value() {
    local name="$1"
    local value="${2:-}"
    if [[ -z "${value}" ]]; then
        echo "Missing value for ${name}" >&2
        exit 1
    fi
}

prepare_temp_dir() {
    mktemp -d "${TMPDIR:-/tmp}/dbpo-stage1.XXXXXX"
}

find_single_file() {
    local search_root="$1"
    local env_name="$2"
    local pattern="$3"
    local candidate=""
    candidate="$(find "${search_root}" -type f -path "*/${env_name}/*" -name "${pattern}" | head -n 1 || true)"
    if [[ -z "${candidate}" ]]; then
        candidate="$(find "${search_root}" -type f -iname "*${env_name}*" -name "${pattern}" | head -n 1 || true)"
    fi
    if [[ -z "${candidate}" ]]; then
        echo "Failed to locate ${pattern} for ${env_name} under ${search_root}" >&2
        exit 1
    fi
    printf '%s\n' "${candidate}"
}

download_and_extract_zip() {
    local url="$1"
    local extract_dir="$2"
    local archive_path="${extract_dir}/archive.zip"
    mkdir -p "${extract_dir}"
    wget -q --show-progress -c -O "${archive_path}" "${url}"
    unzip -q -o "${archive_path}" -d "${extract_dir}"
    rm -f "${archive_path}"
}

log_prepare_status() {
    local tag="$1"
    shift
    printf '%s: %s\n' "${tag}" "$*"
}

robomimic_base_ready() {
    local env
    for env in "${ROBOMIMIC_TASKS[@]}"; do
        [[ -f "${DATA_DIR}/robomimic/${env}/low_dim.hdf5" ]] || return 1
        [[ -f "${DATA_DIR}/robomimic/${env}/image.hdf5" ]] || return 1
    done
}

robomimic_abs_ready() {
    local env
    for env in "${ROBOMIMIC_TASKS[@]}"; do
        [[ -f "${DATA_DIR}/robomimic/${env}/low_dim_abs.hdf5" ]] || return 1
        [[ -f "${DATA_DIR}/robomimic/${env}/image_abs.hdf5" ]] || return 1
    done
}

pusht_ready() {
    [[ -d "${DATA_DIR}/pusht/pusht_cchi_v7_replay.zarr" ]]
}

canonicalize_blockpush_layout() {
    local legacy_dir="${DATA_DIR}/block_pushing"
    local canonical_dir="${DATA_DIR}/blockpush"

    if [[ ! -e "${canonical_dir}" && -d "${legacy_dir}" ]]; then
        ln -s "${legacy_dir}" "${canonical_dir}"
        log_prepare_status "NORMALIZED_BLOCKPUSH_LAYOUT" "${canonical_dir} -> ${legacy_dir}"
    fi

    if [[ -e "${canonical_dir}/multimodal_push_seed_abs.zarr" && ! -e "${canonical_dir}/blockpush_abs.zarr" ]]; then
        ln -s "${canonical_dir}/multimodal_push_seed_abs.zarr" "${canonical_dir}/blockpush_abs.zarr"
        log_prepare_status "NORMALIZED_BLOCKPUSH_LAYOUT" "${canonical_dir}/blockpush_abs.zarr -> ${canonical_dir}/multimodal_push_seed_abs.zarr"
    fi
}

blockpush_base_ready() {
    canonicalize_blockpush_layout
    [[ -d "${DATA_DIR}/blockpush/multimodal_push_seed.zarr" ]]
}

blockpush_abs_ready() {
    canonicalize_blockpush_layout
    [[ -d "${DATA_DIR}/blockpush/blockpush_abs.zarr" ]]
}

kitchen_ready() {
    [[ -f "${DATA_DIR}/kitchen/observations_seq.npy" ]] \
        && [[ -f "${DATA_DIR}/kitchen/actions_seq.npy" ]] \
        && [[ -f "${DATA_DIR}/kitchen/existence_mask.npy" ]]
}

prepare_robomimic_base() {
    if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && robomimic_base_ready; then
        log_prepare_status "SKIP_ALREADY_PREPARED" "robomimic base contract already satisfied under ${DATA_DIR}/robomimic"
        return
    fi
    local lowdim_tmp image_tmp
    lowdim_tmp="$(prepare_temp_dir)"
    image_tmp="$(prepare_temp_dir)"
    download_and_extract_zip "${NATIVE_TASK_DATA_URL}/robomimic_lowdim.zip" "${lowdim_tmp}"
    download_and_extract_zip "${NATIVE_TASK_DATA_URL}/robomimic_image.zip" "${image_tmp}"

    for env in "${ROBOMIMIC_TASKS[@]}"; do
        mkdir -p "${DATA_DIR}/robomimic/${env}"
        cp "$(find_single_file "${lowdim_tmp}" "${env}" 'low_dim*.hdf5')" \
            "${DATA_DIR}/robomimic/${env}/low_dim.hdf5"
        cp "$(find_single_file "${image_tmp}" "${env}" 'image*.hdf5')" \
            "${DATA_DIR}/robomimic/${env}/image.hdf5"
    done
    rm -rf "${lowdim_tmp}" "${image_tmp}"
    log_prepare_status "DOWNLOADED" "robomimic base datasets prepared under ${DATA_DIR}/robomimic"
}

prepare_robomimic_abs() {
    if ! robomimic_base_ready; then
        echo "robomimic base datasets are required before generating abs-action variants." >&2
        exit 1
    fi
    if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && robomimic_abs_ready; then
        log_prepare_status "SKIP_ALREADY_PREPARED" "robomimic abs contract already satisfied under ${DATA_DIR}/robomimic"
        return
    fi
    for env in "${ROBOMIMIC_TASKS[@]}"; do
        local env_dir="${DATA_DIR}/robomimic/${env}"
        mkdir -p "${env_dir}"
        if [[ -f "${env_dir}/low_dim.hdf5" ]]; then
            python "${SCRIPT_DIR}/convert_robomimic_abs.py" \
                --input "${env_dir}/low_dim.hdf5" \
                --output "${env_dir}/low_dim_abs.hdf5"
        fi
        if [[ -f "${env_dir}/image.hdf5" ]]; then
            python "${SCRIPT_DIR}/convert_robomimic_abs.py" \
                --input "${env_dir}/image.hdf5" \
                --output "${env_dir}/image_abs.hdf5"
        fi
    done
    log_prepare_status "GENERATED_ABS" "robomimic abs datasets prepared under ${DATA_DIR}/robomimic"
}

prepare_pusht() {
    if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && pusht_ready; then
        log_prepare_status "SKIP_ALREADY_PREPARED" "pusht contract already satisfied at ${DATA_DIR}/pusht/pusht_cchi_v7_replay.zarr"
        return
    fi
    mkdir -p "${DATA_DIR}/pusht"
    wget -q --show-progress -c -O "${DATA_DIR}/pusht/pusht.zip" "${NATIVE_TASK_DATA_URL}/pusht.zip"
    unzip -q -o "${DATA_DIR}/pusht/pusht.zip" -d "${DATA_DIR}/pusht/"
    rm -f "${DATA_DIR}/pusht/pusht.zip"
    log_prepare_status "DOWNLOADED" "pusht dataset prepared under ${DATA_DIR}/pusht"
}

prepare_blockpush_base() {
    canonicalize_blockpush_layout
    if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && blockpush_base_ready; then
        log_prepare_status "SKIP_ALREADY_PREPARED" "blockpush base contract already satisfied under ${DATA_DIR}/blockpush"
        return
    fi
    mkdir -p "${DATA_DIR}/blockpush"
    wget -q --show-progress -c -O "${DATA_DIR}/blockpush/blockpush.zip" "${NATIVE_TASK_DATA_URL}/block_pushing.zip"
    unzip -q -o "${DATA_DIR}/blockpush/blockpush.zip" -d "${DATA_DIR}/blockpush/"
    rm -f "${DATA_DIR}/blockpush/blockpush.zip"
    canonicalize_blockpush_layout
    log_prepare_status "DOWNLOADED" "blockpush base dataset prepared under ${DATA_DIR}/blockpush"
}

prepare_blockpush_abs() {
    canonicalize_blockpush_layout
    if ! blockpush_base_ready; then
        echo "blockpush base dataset is required before generating the abs-action variant." >&2
        exit 1
    fi
    if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && blockpush_abs_ready; then
        log_prepare_status "SKIP_ALREADY_PREPARED" "blockpush abs contract already satisfied under ${DATA_DIR}/blockpush"
        return
    fi
    if [[ -d "${DATA_DIR}/blockpush/multimodal_push_seed.zarr" && ! -d "${DATA_DIR}/blockpush/blockpush_abs.zarr" ]]; then
        python "${SCRIPT_DIR}/convert_blockpush_abs.py" \
            --input "${DATA_DIR}/blockpush/multimodal_push_seed.zarr" \
            --output "${DATA_DIR}/blockpush/blockpush_abs.zarr"
        log_prepare_status "GENERATED_ABS" "blockpush abs dataset prepared under ${DATA_DIR}/blockpush"
    fi
}

prepare_kitchen_base() {
    if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && kitchen_ready; then
        log_prepare_status "SKIP_ALREADY_PREPARED" "kitchen contract already satisfied under ${DATA_DIR}/kitchen"
        return
    fi
    local kitchen_tmp
    kitchen_tmp="$(prepare_temp_dir)"
    mkdir -p "${DATA_DIR}/kitchen"
    download_and_extract_zip "${NATIVE_TASK_DATA_URL}/kitchen.zip" "${kitchen_tmp}"
    for filename in observations_seq.npy actions_seq.npy existence_mask.npy; do
        local src_file
        src_file="$(find "${kitchen_tmp}" -type f -name "${filename}" | head -n 1 || true)"
        if [[ -z "${src_file}" ]]; then
            echo "Failed to locate ${filename} inside the native kitchen archive." >&2
            exit 1
        fi
        cp "${src_file}" "${DATA_DIR}/kitchen/${filename}"
    done
    local summary_file
    summary_file="$(find "${kitchen_tmp}" -type f -name 'dataset_summary.json' | head -n 1 || true)"
    if [[ -n "${summary_file}" ]]; then
        cp "${summary_file}" "${DATA_DIR}/kitchen/dataset_summary.json"
    fi
    rm -rf "${kitchen_tmp}"
    log_prepare_status "DOWNLOADED" "kitchen dataset prepared under ${DATA_DIR}/kitchen"
}

ensure_supported_d4rl_env() {
    local env_name="$1"
    for released_env in "${D4RL_RELEASE_ENVS[@]}"; do
        if [[ "${released_env}" == "${env_name}" ]]; then
            return 0
        fi
    done
    echo "Unsupported D4RL env '${env_name}'." >&2
    echo "Released tasks: ${D4RL_RELEASE_ENVS[*]}" >&2
    exit 1
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift

SUITE="all"
ENV_NAME=""
TASK_NAME=""
RUN_ALL=0
DRY_RUN=0
PREPARE_ABS=0
NUM_EPISODES=20
FORCE_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            require_value "$1" "${2:-}"
            DATA_DIR="$2"
            shift 2
            ;;
        --suite)
            require_value "$1" "${2:-}"
            SUITE="$2"
            shift 2
            ;;
        --env)
            require_value "$1" "${2:-}"
            ENV_NAME="$2"
            shift 2
            ;;
        --task)
            require_value "$1" "${2:-}"
            TASK_NAME="$2"
            shift 2
            ;;
        --num-episodes)
            require_value "$1" "${2:-}"
            NUM_EPISODES="$2"
            shift 2
            ;;
        --prepare-abs)
            PREPARE_ABS=1
            shift
            ;;
        --force-download)
            FORCE_DOWNLOAD=1
            shift
            ;;
        --all)
            RUN_ALL=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

case "${COMMAND}" in
    native)
        case "${SUITE}" in
            all|robomimic)
                echo "Preparing native RoboMimic tasks under ${DATA_DIR}/robomimic"
                prepare_robomimic_base
                if [[ "${PREPARE_ABS}" -eq 1 ]]; then
                    prepare_robomimic_abs
                fi
                ;;
        esac
        case "${SUITE}" in
            all|pusht)
                echo "Preparing native PushT tasks under ${DATA_DIR}/pusht"
                prepare_pusht
                ;;
        esac
        case "${SUITE}" in
            all|blockpush)
                echo "Preparing native BlockPush tasks under ${DATA_DIR}/blockpush"
                prepare_blockpush_base
                if [[ "${PREPARE_ABS}" -eq 1 ]]; then
                    prepare_blockpush_abs
                fi
                ;;
        esac
        case "${SUITE}" in
            all|kitchen)
                echo "Preparing native kitchen tasks under ${DATA_DIR}/kitchen"
                prepare_kitchen_base
                if [[ "${PREPARE_ABS}" -eq 1 ]]; then
                    echo "kitchen_demos_multitask must be prepared separately and is only documented here." >&2
                fi
                ;;
        esac
        ;;
    adroit_metaworld)
        if [[ "${RUN_ALL}" -eq 1 ]]; then
            if [[ "${SUITE}" == "adroit" || "${SUITE}" == "all" ]]; then
                for task in door hammer pen; do
                    python "${SCRIPT_DIR}/prepare_pointcloud.py" --suite adroit --task "${task}" --num-episodes "${NUM_EPISODES}" --data-dir "${DATA_DIR}" $([[ "${DRY_RUN}" -eq 1 ]] && echo "--dry-run")
                done
            fi
            if [[ "${SUITE}" == "metaworld" || "${SUITE}" == "all" ]]; then
                for task_cfg in "${REPO_ROOT}"/configs/task/metaworld_*.yaml; do
                    task_name="$(basename "${task_cfg}" .yaml | sed 's/^metaworld_//')"
                    python "${SCRIPT_DIR}/prepare_pointcloud.py" --suite metaworld --task "${task_name}" --num-episodes "${NUM_EPISODES}" --data-dir "${DATA_DIR}" $([[ "${DRY_RUN}" -eq 1 ]] && echo "--dry-run")
                done
            fi
        else
            require_value "--task" "${TASK_NAME}"
            if [[ "${SUITE}" != "adroit" && "${SUITE}" != "metaworld" ]]; then
                echo "adroit_metaworld requires --suite adroit or --suite metaworld when --all is not used." >&2
                exit 1
            fi
            python "${SCRIPT_DIR}/prepare_pointcloud.py" \
                --suite "${SUITE}" \
                --task "${TASK_NAME}" \
                --num-episodes "${NUM_EPISODES}" \
                --data-dir "${DATA_DIR}" \
                $([[ "${DRY_RUN}" -eq 1 ]] && echo "--dry-run")
        fi
        ;;
    d4rl_gym)
        if [[ "${RUN_ALL}" -eq 1 ]]; then
            for env_name in "${D4RL_RELEASE_ENVS[@]}"; do
                python "${SCRIPT_DIR}/prepare_d4rl.py" --env "${env_name}" --output_dir "${DATA_DIR}/gym/${env_name}"
            done
        else
            require_value "--env" "${ENV_NAME}"
            ensure_supported_d4rl_env "${ENV_NAME}"
            python "${SCRIPT_DIR}/prepare_d4rl.py" --env "${ENV_NAME}" --output_dir "${DATA_DIR}/gym/${ENV_NAME}"
        fi
        ;;
    *)
        echo "Unknown command: ${COMMAND}" >&2
        print_usage
        exit 1
        ;;
esac

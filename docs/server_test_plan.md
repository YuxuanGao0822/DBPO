# Server Test Pack

This document defines the server-side test order for:

- stage 1: `DBP pretrain`
- stage 2: `DBPO gym finetune`

Interpret server results in two layers:

- `TRAIN_SMOKE_OK`: dataset loading, one-batch loss, short pretrain, and
  checkpoint verification pass
- `RUNTIME_ENV_PENDING`: rollout backends such as `robosuite`, `Kitchen`, or
  MuJoCo EGL still need extra server-side rendering support

## Phase 0: Installation And Environment Snapshot

```bash
bash scripts/install_default.sh

export DBPO_DATA_DIR=/path/to/data
export DBPO_LOG_DIR=/path/to/outputs
export DBPO_TEST_REPORT_DIR=/path/to/reports

bash scripts/testing/00_check_env.sh
```

## Phase 1: Adroit / MetaWorld Family

Representative tasks:

- `adroit_hammer`
- `metaworld_assembly`

Commands:

```bash
bash scripts/testing/03_instantiate_smoke.sh --task adroit_hammer --device cuda:0
bash scripts/testing/03_instantiate_smoke.sh --task metaworld_assembly --device cuda:0
bash scripts/testing/04_short_pretrain.sh --task adroit_hammer
bash scripts/testing/04_short_pretrain.sh --task metaworld_assembly
```

## Phase 2: Native Simulated Task Family

Prepare data:

```bash
bash scripts/testing/01_prepare_data.sh --family native --dry-run
bash scripts/prepare_datasets.sh native --suite robomimic --prepare-abs
bash scripts/prepare_datasets.sh native --suite pusht
bash scripts/prepare_datasets.sh native --suite blockpush --prepare-abs
bash scripts/prepare_datasets.sh native --suite kitchen
```

Compose:

```bash
bash scripts/testing/02_compose_tasks.sh --family native
```

Representative tasks:

- `can_lowdim`
- `can_image`
- `pusht_image`
- `blockpush_lowdim_seed`
- `kitchen_lowdim`

## Phase 3: D4RL Gym Family

Compose:

```bash
bash scripts/testing/02_compose_tasks.sh --family d4rl_gym
```

Instantiate:

```bash
bash scripts/testing/03_instantiate_smoke.sh --family d4rl_gym --device cuda:0
```

Short pretrain:

```bash
bash scripts/testing/04_short_pretrain.sh --task hopper-medium-v2
bash scripts/testing/04_short_pretrain.sh --task ant-medium-expert-v2
bash scripts/testing/04_short_pretrain.sh --task walker2d-medium-v2
```

## Phase 4: Full Matrix Generation

```bash
bash scripts/testing/05_full_pretrain_matrix.sh --family all --print-only
```

## Phase 5: Stage-2 Gym Finetune

```bash
export DBPO_PRETRAIN_DIR=${DBPO_LOG_DIR}/server_full
bash scripts/testing/06_finetune_compose.sh
bash scripts/testing/07_finetune_instantiate_smoke.sh --family d4rl_gym --device cuda:0
bash scripts/testing/08_short_finetune.sh --task hopper-medium-v2
bash scripts/testing/08_short_finetune.sh --task ant-medium-expert-v2
bash scripts/testing/08_short_finetune.sh --task walker2d-medium-v2
bash scripts/testing/09_finetune_matrix.sh --print-only
```

The default public stage-2 smoke set is:

- `hopper-medium-v2`
- `ant-medium-expert-v2`
- `walker2d-medium-v2`

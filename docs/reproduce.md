# Reproduce

This document defines the released reproduction path for:

1. stage 1: `DBP pretrain`
2. stage 2: `DBPO gym finetune`

## 1. Install

```bash
conda env create -f environment.yml
conda activate dbpo
bash scripts/install_default.sh
```

Run static checks:

```bash
python scripts/check_environment.py --group native_suite
python scripts/check_environment.py --group adroit_metaworld_suite
python scripts/check_environment.py --group gym_suite
```

## 2. Prepare Data

```bash
export DBPO_DATA_DIR=/path/to/data
export DBPO_LOG_DIR=/path/to/outputs
```

### Native Simulated Manipulation Tasks

```bash
bash scripts/prepare_datasets.sh native --suite robomimic --prepare-abs
bash scripts/prepare_datasets.sh native --suite pusht
bash scripts/prepare_datasets.sh native --suite blockpush --prepare-abs
bash scripts/prepare_datasets.sh native --suite kitchen
```

### Adroit / MetaWorld Tasks

```bash
bash scripts/prepare_datasets.sh adroit_metaworld --suite adroit --task door --num-episodes 1
bash scripts/prepare_datasets.sh adroit_metaworld --suite metaworld --task assembly --num-episodes 1
```

### Gym Tasks

```bash
bash scripts/prepare_datasets.sh d4rl_gym --env hopper-medium-v2
bash scripts/prepare_datasets.sh d4rl_gym --env ant-medium-expert-v2
bash scripts/prepare_datasets.sh d4rl_gym --env walker2d-medium-v2
```

## 3. Run DBP Pretrain

Representative commands:

```bash
python pretrain.py task=can_lowdim
python pretrain.py task=can_image
python pretrain.py task=pusht_lowdim
python pretrain.py task=pusht_image
python pretrain.py task=blockpush_lowdim_seed
python pretrain.py task=kitchen_lowdim
python pretrain.py task=adroit_hammer
python pretrain.py task=metaworld_assembly
python pretrain.py task=hopper-medium-v2
python pretrain.py task=ant-medium-expert-v2
python pretrain.py task=walker2d-medium-v2
```

## 4. Run DBPO Gym Finetune

Stage-2 is publicly released for the three D4RL locomotion tasks:

```bash
python finetune.py task=hopper-medium-v2 actor_policy_path=/path/to/server_full/hopper-medium-v2/checkpoint/last.pt
python finetune.py task=ant-medium-expert-v2 actor_policy_path=/path/to/server_full/ant-medium-expert-v2/checkpoint/last.pt
python finetune.py task=walker2d-medium-v2 actor_policy_path=/path/to/server_full/walker2d-medium-v2/checkpoint/last.pt
```

## 5. Local Static Preparation vs Server Validation

Local preparation covers:

- public repository surface
- task/config organization
- dataset directory contracts
- public data commands
- static dependency checks
- unified checkpoint metadata
- public stage-2 gym finetune entrypoint and testing scaffold

Server validation is still required for:

- all real data downloads
- point-cloud data generation
- any training run
- any benchmark or rollout
- simulator runtime behavior for `PushT`, `BlockPush`, and `Kitchen`
- rollout-backed runtime behavior for stage-2 `DBPO gym finetune`

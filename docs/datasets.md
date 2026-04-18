# Datasets

This document defines the dataset contract for `DBP pretrain`.

All paths are rooted at `DBPO_DATA_DIR`. If `DBPO_DATA_DIR` is unset, the
repository defaults to `<repo>/data`.

## Native Simulated Manipulation Tasks

Public entrypoint:

```bash
bash scripts/prepare_datasets.sh native --suite <suite> [--prepare-abs]
```

Supported suites:

- `robomimic`
- `pusht`
- `blockpush`
- `kitchen`
- `all`

Output contract:

- `robomimic`
  - `${DBPO_DATA_DIR}/robomimic/<task>/low_dim.hdf5`
  - `${DBPO_DATA_DIR}/robomimic/<task>/low_dim_abs.hdf5`
  - `${DBPO_DATA_DIR}/robomimic/<task>/image.hdf5`
  - `${DBPO_DATA_DIR}/robomimic/<task>/image_abs.hdf5`
- `pusht`
  - `${DBPO_DATA_DIR}/pusht/pusht_cchi_v7_replay.zarr`
- `blockpush`
  - `${DBPO_DATA_DIR}/blockpush/multimodal_push_seed.zarr`
  - `${DBPO_DATA_DIR}/blockpush/blockpush_abs.zarr`
- `kitchen`
  - `${DBPO_DATA_DIR}/kitchen/observations_seq.npy`
  - `${DBPO_DATA_DIR}/kitchen/actions_seq.npy`
  - `${DBPO_DATA_DIR}/kitchen/existence_mask.npy`

Representative commands:

```bash
bash scripts/prepare_datasets.sh native --suite robomimic --prepare-abs
bash scripts/prepare_datasets.sh native --suite pusht
bash scripts/prepare_datasets.sh native --suite blockpush --prepare-abs
bash scripts/prepare_datasets.sh native --suite kitchen
```

## Adroit / MetaWorld Tasks

Public entrypoint:

```bash
bash scripts/prepare_datasets.sh adroit_metaworld --suite <suite> --task <task> [--dry-run]
```

Supported suites:

- `adroit`
- `metaworld`
- `all`

Output contract:

- `${DBPO_DATA_DIR}/adroit/<task>/train.zarr`
- `${DBPO_DATA_DIR}/metaworld/<task>/train.zarr`

Required zarr keys:

- `data/state`
- `data/action`
- `data/point_cloud`
- `meta/episode_ends`

Representative commands:

```bash
bash scripts/prepare_datasets.sh adroit_metaworld --suite adroit --task door --num-episodes 1
bash scripts/prepare_datasets.sh adroit_metaworld --suite metaworld --task assembly --num-episodes 1
```

## Gym Tasks

Public entrypoint:

```bash
bash scripts/prepare_datasets.sh d4rl_gym --env <env>
```

Public environments:

- `hopper-medium-v2`
- `ant-medium-expert-v2`
- `walker2d-medium-v2`

Output contract:

- `${DBPO_DATA_DIR}/gym/<env>/train.npz`
- `${DBPO_DATA_DIR}/gym/<env>/normalization.npz`

`train.npz` contains:

- `states`
- `actions`
- `rewards`
- `terminals`
- `traj_lengths`

`normalization.npz` contains:

- `obs_min`
- `obs_max`
- `action_min`
- `action_max`

Representative commands:

```bash
bash scripts/prepare_datasets.sh d4rl_gym --env hopper-medium-v2
bash scripts/prepare_datasets.sh d4rl_gym --env ant-medium-expert-v2
bash scripts/prepare_datasets.sh d4rl_gym --env walker2d-medium-v2
bash scripts/prepare_datasets.sh d4rl_gym --all
```

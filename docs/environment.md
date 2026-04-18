# Environment

The repository uses one canonical environment definition:
[environment.yml](/Users/ethangao/Code/opensource/DBPO/environment.yml).

## Canonical Baseline

- `python=3.10`
- `pytorch=2.4.1`
- `torchvision=0.19.1`
- `hydra-core=1.3.2`
- `omegaconf=2.3.0`
- `gym==0.23.1`
- `mujoco==3.1.6`
- `mujoco-py==2.1.2.14`
- `dm_control==1.0.16`
- `robomimic==0.3.0`
- `robosuite==1.4.1`
- `open3d==0.18.0`
- vendored `metaworld` from `third_party/metaworld`

The public repository is organized around two workflows:

- stage 1: `DBP pretrain`
- stage 2: `DBPO gym finetune`

## Installation

```bash
conda env create -f environment.yml
conda activate dbpo
bash scripts/install_default.sh
```

`scripts/install_default.sh` performs:

- `pip install -e .`
- install validated `mjrl` and `d4rl` zipball sources
- install vendored local packages:
  - `third_party/metaworld`
  - `third_party/mj_envs`
  - `third_party/adroit_metaworld_support`
- install runtime Python dependencies commonly missing in reused environments:
  - `scikit-image`
  - `pybullet`
- run a minimal Python import smoke check

Run static checks with:

```bash
python scripts/check_environment.py --group native_suite
python scripts/check_environment.py --group adroit_metaworld_suite
python scripts/check_environment.py --group gym_suite
```

## Scope Notes

- `PushT`, `BlockPush`, and `Kitchen` environment code is included for stage-1
  pretraining.
- Point-cloud tasks depend on vendored `metaworld`, `mj_envs`, and
  `adroit_metaworld_support`.
- Gym tasks require the D4RL and MuJoCo stack for both stage-1 pretrain and
  stage-2 finetune.
- Full runtime behavior still requires server-side validation.

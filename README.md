# Drift-Based Policy Optimization (DBPO)

**Official implementation of [Drift-Based Policy Optimization: Native One-Step Policy Learning for Online Robot Control](https://arxiv.org/abs/2604.03540)**

*Yuxuan Gao, Yedong Shen, Shiqi Zhang, Wenhao Yu, Yifan Duan, Jia Pan, Jiajia Wu, Jiajun Deng, Yanyong Zhang*  
*University of Science and Technology of China, iFLYTEK*

---

<p align="center">
  <img src="docs/figures/intro.png" alt="Method Comparison" width="100%">
</p>

## Overview

Robotic manipulation requires policies that can execute complex visuomotor tasks while maintaining high control frequencies. Multi-step generative policies such as Diffusion Policy achieve strong performance by modeling multimodal action distributions, but their iterative denoising process requires tens to hundreds of network function evaluations (NFEs) per action. This computational cost creates a fundamental bottleneck for high-frequency closed-loop control and online reinforcement learning.

This repository introduces a training-centric solution that shifts iterative refinement from inference to training. The **Drift-Based Policy (DBP)** internalizes corrective dynamics through fixed-point drifting objectives during training, yielding a generator that produces high-quality multimodal actions in exactly one forward pass. The **Drift-Based Policy Optimization (DBPO)** framework extends this backbone to online reinforcement learning through a minimal stochastic interface that enables exact-likelihood PPO updates while preserving the deterministic one-step deployment path.

<p align="center">
  <img src="docs/figures/framework.png" alt="Method Framework" width="90%">
</p>

The approach addresses three core requirements: (1) native one-step generation without distillation or post-hoc acceleration, (2) multimodal action modeling capacity, and (3) stable online policy improvement. Empirical validation demonstrates that DBP matches or exceeds multi-step diffusion baselines at 100× lower inference cost, achieves state-of-the-art performance among one-step methods on point-cloud manipulation benchmarks, and enables effective online fine-tuning. Real-world deployment on a dual-arm UR5 robot confirms practical feasibility at 105.2 Hz control frequency.

## Key Contributions

- **Native One-Step Generation**: Achieves 1-NFE inference through training-time drift-field internalization rather than distillation or auxiliary corrections
- **Multimodal Action Modeling**: Preserves expressive action distributions through attraction-repulsion dynamics learned during training
- **Online RL Extension**: Enables exact-likelihood on-policy updates via minimal stochastic adapter while maintaining deterministic one-step deployment
- **Comprehensive Validation**: Evaluated across offline imitation, online fine-tuning, simulation benchmarks, and real-world hardware

## Highlights

**Efficiency vs. Multi-Step Baselines**  
On the 12-task Diffusion Policy simulation suite, DBP achieves 0.83 average success rate compared to 0.79 for 100-step Diffusion Policy, while reducing inference cost from 100 NFE to 1 NFE—a 100× speedup.

**State-of-the-Art One-Step Performance**  
On 37 point-cloud manipulation tasks (Adroit + Meta-World), DBP achieves 88.4% average success rate, outperforming prior one-step methods OMP (82.3%) and MP1 (78.9%) under identical evaluation protocols.

**Stable Online Fine-Tuning**  
DBPO successfully applies PPO to the one-step backbone on RoboMimic and D4RL benchmarks, improving task rewards and state-space coverage without sacrificing deployment efficiency.

**Real-World High-Frequency Control**  
Physical dual-arm UR5 deployment achieves 75% success rate across Lift, Can, and bimanual Transport tasks at 105.2 Hz control frequency with 9.5 ms end-to-end latency.

## Release Scope

This public repository releases validated workflows for offline training and online fine-tuning in simulation. The following components are included:

**Stage 1: DBP Pretrain (Offline Learning)**
- Native simulated manipulation: Push-T, BlockPush, Kitchen, RoboMimic tasks (low-dim and image observations)
- Point-cloud manipulation: Adroit (Door, Hammer, Pen) and Meta-World task families
- D4RL Gym locomotion: Hopper, Walker2d, Ant (medium/medium-expert datasets)

**Stage 2: DBPO Finetune (Online Learning)**
- D4RL Gym continuous control: `hopper-medium-v2`, `walker2d-medium-v2`, `ant-medium-expert-v2`

**Not Included in Public Release**
- Real-robot hardware drivers and teleoperation interfaces
- Vision-based online RL components for manipulation tasks
- Point-cloud preprocessing pipelines for custom datasets

## Quick Start

Minimal path to environment setup, dataset preparation, and training:

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate dbpo
bash scripts/install_default.sh

# 2. Prepare a representative dataset (Push-T)
bash scripts/prepare_datasets.sh native --suite pusht

# 3. Train DBP offline backbone (Stage 1)
python pretrain.py task=pusht_lowdim

# 4. (Optional) Prepare D4RL dataset and run online fine-tuning (Stage 2)
bash scripts/prepare_datasets.sh d4rl_gym --env hopper-medium-v2
python finetune.py task=hopper-medium-v2 actor_policy_path=<path/to/stage1/checkpoint.pt>
```

## Installation

### Environment Setup

The codebase requires a CUDA-capable machine. We recommend Anaconda or Miniconda for environment management.

```bash
conda env create -f environment.yml
conda activate dbpo
bash scripts/install_default.sh
```

### Verification

Verify that simulators and dependencies are correctly configured:

```bash
python scripts/check_environment.py --group native_suite
python scripts/check_environment.py --group adroit_metaworld_suite
python scripts/check_environment.py --group gym_suite
```

Detailed dependency notes and troubleshooting guidance are available in [docs/environment.md](docs/environment.md).

## Data Preparation

### Directory Configuration

Optionally define environment variables to specify dataset and logging directories:

```bash
export DBPO_DATA_DIR=/path/to/data
export DBPO_LOG_DIR=/path/to/outputs
```

If `DBPO_DATA_DIR` is unset, datasets default to `<repo>/data`.

### Dataset Download and Preprocessing

Use `scripts/prepare_datasets.sh` to download and preprocess public datasets. Select a task family (`native`, `adroit_metaworld`, or `d4rl_gym`):

**Native Manipulation Tasks**
```bash
# RoboMimic (Can, Lift, Square, Tool Hang, Transport)
bash scripts/prepare_datasets.sh native --suite robomimic --prepare-abs

# Push-T
bash scripts/prepare_datasets.sh native --suite pusht

# BlockPush
bash scripts/prepare_datasets.sh native --suite blockpush --prepare-abs

# Kitchen
bash scripts/prepare_datasets.sh native --suite kitchen
```

**Point-Cloud Manipulation**
```bash
# Adroit (Door, Hammer, Pen)
bash scripts/prepare_datasets.sh adroit_metaworld --suite adroit --task door --num-episodes 1

# Meta-World (Assembly, Button Press, etc.)
bash scripts/prepare_datasets.sh adroit_metaworld --suite metaworld --task assembly --num-episodes 1
```

**D4RL Gym Locomotion**
```bash
# Single environment
bash scripts/prepare_datasets.sh d4rl_gym --env hopper-medium-v2

# All supported environments
bash scripts/prepare_datasets.sh d4rl_gym --all
```

Detailed data contracts and task specifications are documented in [docs/datasets.md](docs/datasets.md) and [docs/task_matrix.md](docs/task_matrix.md).

## Training

### Stage 1: DBP Pretrain

Train the native one-step backbone from offline demonstrations using `pretrain.py`. The stage-1 checkpoint format is compatible with stage-2 fine-tuning.

**Basic Usage**
```bash
python pretrain.py task=<task_name>
```

**Representative Commands**
```bash
# Image-based manipulation
python pretrain.py task=pusht_image
python pretrain.py task=can_image

# Low-dimensional state
python pretrain.py task=blockpush_lowdim_seed
python pretrain.py task=kitchen_lowdim

# Point-cloud manipulation
python pretrain.py task=adroit_hammer
python pretrain.py task=metaworld_assembly

# Locomotion
python pretrain.py task=walker2d-medium-v2
python pretrain.py task=hopper-medium-v2
```

**Supported Task Families**
- Native manipulation: `pusht_lowdim`, `pusht_image`, `blockpush_lowdim_seed`, `blockpush_lowdim_seed_abs`, `kitchen_lowdim`, `kitchen_lowdim_abs`
- RoboMimic: `can`, `lift`, `square`, `tool_hang`, `transport` (with `_lowdim`, `_lowdim_abs`, `_image`, `_image_abs` suffixes)
- Adroit: `adroit_door`, `adroit_hammer`, `adroit_pen`
- Meta-World: Released `metaworld_*` configs (see [docs/task_matrix.md](docs/task_matrix.md))
- D4RL Gym: `hopper-medium-v2`, `walker2d-medium-v2`, `ant-medium-expert-v2`

### Stage 2: DBPO Finetune

Extend a pretrained stage-1 checkpoint with online PPO fine-tuning using `finetune.py`. Currently supported for D4RL Gym locomotion tasks.

**Basic Usage**
```bash
python finetune.py task=<gym_task> actor_policy_path=<path/to/stage1/checkpoint.pt>
```

**Representative Commands**
```bash
python finetune.py task=hopper-medium-v2 \
    actor_policy_path=/path/to/checkpoints/hopper-medium-v2/last.pt

python finetune.py task=walker2d-medium-v2 \
    actor_policy_path=/path/to/checkpoints/walker2d-medium-v2/last.pt

python finetune.py task=ant-medium-expert-v2 \
    actor_policy_path=/path/to/checkpoints/ant-medium-expert-v2/last.pt
```

**Note**: Online fine-tuning for vision-based manipulation tasks is not included in this release.

## Main Results

### DBP vs. Multi-Step Diffusion Policy

Comparison on the 12-task Diffusion Policy simulation suite. Results averaged over last 10 checkpoints and 3 training seeds.

| Task | Diffusion Policy (100 NFE) | DBP (1 NFE) |
|------|----------------------------|-------------|
| Push-T (Image) | **0.91** | 0.89 |
| Push-T (Low-Dim) | 0.85 | **0.87** |
| BlockPush (P1/P2) | 0.24 | **0.43** |
| RoboMimic (Low-Dim) | 0.80 | **0.92** |
| RoboMimic (Image) | **0.91** | 0.87 |
| Kitchen (P1/P2/P3/P4) | **1.00** | **1.00** |
| **Average** | **0.79** | **0.83** |

DBP achieves higher average performance while reducing inference cost from 100 NFE to 1 NFE.

### One-Step Baseline Comparison

Point-cloud manipulation on 37 Adroit and Meta-World tasks under the MP1/OMP protocol. Results report success rate (%, mean ± std over 3 seeds).

| Method | NFE | Adroit (3 tasks) | Meta-World Easy (21) | Meta-World Medium (4) | Meta-World Hard (4) | Meta-World Very Hard (5) | **Average (37 tasks)** |
|--------|-----|------------------|----------------------|-----------------------|---------------------|--------------------------|------------------------|
| DP3 | 10 | 67.3 ± 5.0 | 87.3 ± 2.2 | 44.5 ± 8.7 | 32.7 ± 7.7 | 39.4 ± 9.0 | 68.7 ± 4.7 |
| MP1 | 1 | 75.7 ± 2.3 | 88.2 ± 1.1 | 68.0 ± 3.1 | 58.1 ± 5.0 | 67.2 ± 2.7 | 78.9 ± 2.1 |
| OMP | 1 | 76.0 ± 2.3 | 89.7 ± 0.7 | 77.4 ± 2.2 | 62.5 ± 3.1 | 77.8 ± 3.0 | 82.3 ± 1.6 |
| **DBP (Ours)** | **1** | **83.3 ± 2.7** | **91.7 ± 1.7** | **90.3 ± 3.6** | **75.2 ± 6.1** | **86.7 ± 5.8** | **88.4 ± 3.1** |

DBP establishes state-of-the-art performance among one-step methods, with particularly strong gains on Medium, Hard, and Very Hard task categories.

### Real-World Deployment

Physical dual-arm UR5 robot with tri-camera setup (dual wrist-mounted RealSense L515 + Orbbec Gemini head camera). Results from 20 trials per task.

| Task | Success / Total | Success Rate |
|------|----------------|--------------|
| Lift (Single-arm) | 18 / 20 | 90.0% |
| Can (Single-arm) | 16 / 20 | 80.0% |
| Transport (Bimanual) | 11 / 20 | 55.0% |
| **Overall** | **45 / 60** | **75.0%** |

Average end-to-end latency: 9.5 ms (105.2 Hz control frequency)

## Repository Structure

```
DBPO/
├── configs/              # Hydra configuration files
│   ├── task/            # Task-specific configs
│   ├── model/           # Model architecture configs
│   ├── pretrain.yaml    # Stage 1 training config
│   └── finetune.yaml    # Stage 2 training config
├── dbpo/                # Core implementation
│   ├── algorithm/       # DBP and DBPO algorithms
│   ├── model/           # Network architectures
│   ├── dataset/         # Data loading and preprocessing
│   ├── env/             # Environment wrappers
│   └── policy/          # Policy interfaces
├── scripts/             # Dataset preparation and utilities
│   ├── prepare_datasets.sh
│   ├── check_environment.py
│   └── testing/         # Validation test suite
├── docs/                # Documentation
│   ├── datasets.md      # Data preparation guide
│   ├── environment.md   # Installation details
│   ├── task_matrix.md   # Task specifications
│   └── reproduce.md     # Reproduction instructions
├── pretrain.py          # Stage 1 training entrypoint
├── finetune.py          # Stage 2 training entrypoint
└── environment.yml      # Conda environment specification
```

## Developer Validation

Comprehensive testing suite for repository development and continuous integration:

```bash
# Environment verification
bash scripts/testing/00_check_env.sh

# Dataset preparation validation
bash scripts/testing/01_prepare_data.sh --family all --dry-run

# Task configuration composition
bash scripts/testing/02_compose_tasks.sh --family all

# Smoke tests for instantiation
bash scripts/testing/03_instantiate_smoke.sh --family all --device cuda:0

# Short training runs
bash scripts/testing/04_short_pretrain.sh --family all --print-only

# Full training matrix (print commands)
bash scripts/testing/05_full_pretrain_matrix.sh --family all --print-only

# Fine-tuning validation
bash scripts/testing/06_finetune_compose.sh
bash scripts/testing/07_finetune_instantiate_smoke.sh --family d4rl_gym --device cuda:0
bash scripts/testing/08_short_finetune.sh --family d4rl_gym
bash scripts/testing/09_finetune_matrix.sh --print-only
```

Detailed testing protocols and reproduction instructions are documented in [docs/server_test_plan.md](docs/server_test_plan.md) and [docs/reproduce.md](docs/reproduce.md).

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{gao2026drift,
  title={Drift-Based Policy Optimization: Native One-Step Policy Learning for Online Robot Control},
  author={Gao, Yuxuan and Shen, Yedong and Zhang, Shiqi and Yu, Wenhao and Duan, Yifan and Pan, Jia and Wu, Jiajia and Deng, Jiajun and Zhang, Yanyong},
  journal={arXiv preprint arXiv:2604.03540},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work builds upon the Diffusion Policy codebase and evaluation protocols from the robotics community. We thank the authors of Diffusion Policy, DP3 and Reinflow for their open-source implementations and benchmark contributions.

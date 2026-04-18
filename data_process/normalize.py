"""
DBPO D4RL locomotion dataset preparation.

Outputs:
  - train.npz
  - normalization.npz
"""
from __future__ import annotations

import os

import numpy as np

try:
    import d4rl  # noqa: F401
    import gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install gym and d4rl to process D4RL datasets.") from exc


def _episode_lengths(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    dones = terminals | timeouts
    traj_lengths = []
    current_len = 0
    for done in dones:
        current_len += 1
        if done:
            traj_lengths.append(current_len)
            current_len = 0
    if current_len > 0:
        traj_lengths.append(current_len)
    return np.asarray(traj_lengths, dtype=np.int32)


def prepare_d4rl_dataset(env_name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading '{env_name}' from D4RL...")
    env = gym.make(env_name)
    dataset = env.get_dataset()

    obs = dataset["observations"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    rewards = dataset["rewards"].astype(np.float32)
    terminals = dataset["terminals"].astype(bool)
    timeouts = dataset.get("timeouts", np.zeros_like(terminals, dtype=bool)).astype(bool)

    traj_lengths = _episode_lengths(terminals, timeouts)
    total_steps = int(traj_lengths.sum())
    assert total_steps == len(obs), f"Mismatch in stitched lengths: {total_steps} vs {len(obs)}"

    obs_min = np.min(obs, axis=0)
    obs_max = np.max(obs, axis=0)
    action_min = np.min(actions, axis=0)
    action_max = np.max(actions, axis=0)

    obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
    action_norm = 2 * (actions - action_min) / (action_max - action_min + 1e-6) - 1

    train_path = os.path.join(output_dir, "train.npz")
    np.savez(
        train_path,
        states=obs_norm.astype(np.float32),
        actions=action_norm.astype(np.float32),
        rewards=rewards,
        terminals=terminals,
        traj_lengths=traj_lengths,
    )

    normalization_path = os.path.join(output_dir, "normalization.npz")
    np.savez(
        normalization_path,
        obs_min=obs_min.astype(np.float32),
        obs_max=obs_max.astype(np.float32),
        action_min=action_min.astype(np.float32),
        action_max=action_max.astype(np.float32),
    )

    print(f"[{env_name}] Saved {len(traj_lengths)} episodes ({len(obs)} total steps)")
    print(f"  - train.npz -> {train_path}")
    print(f"  - normalization.npz -> {normalization_path}")


def process_d4rl_gym(env_name: str, output_dir: str) -> None:
    prepare_d4rl_dataset(env_name=env_name, output_dir=output_dir)

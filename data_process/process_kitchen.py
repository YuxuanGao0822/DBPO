"""
D4RL FrankaKitchen dataset conversion for stage-1 DBPO.
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import d4rl  # noqa: F401
    import gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install gym and d4rl to process D4RL Kitchen datasets.") from exc


def prepare_kitchen_dataset(env_name: str, output_dir: str, write_summary: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading '{env_name}' from D4RL...")

    env = gym.make(env_name)
    dataset = env.get_dataset()

    obs = dataset["observations"]
    actions = dataset["actions"]
    terminals = dataset["terminals"]
    timeouts = dataset["timeouts"]

    dones = terminals | timeouts
    episodes_obs = []
    episodes_act = []
    current_obs = []
    current_act = []

    for i in range(len(dones)):
        current_obs.append(obs[i])
        current_act.append(actions[i])
        if dones[i]:
            episodes_obs.append(np.array(current_obs))
            episodes_act.append(np.array(current_act))
            current_obs = []
            current_act = []

    if current_obs:
        episodes_obs.append(np.array(current_obs))
        episodes_act.append(np.array(current_act))

    max_len = max(len(ep) for ep in episodes_obs)
    n_eps = len(episodes_obs)
    obs_dim = obs.shape[-1]
    act_dim = actions.shape[-1]

    obs_seq = np.zeros((n_eps, max_len, obs_dim), dtype=np.float32)
    act_seq = np.zeros((n_eps, max_len, act_dim), dtype=np.float32)
    existence_mask = np.zeros((n_eps, max_len), dtype=np.float32)

    for i in range(n_eps):
        ep_len = len(episodes_obs[i])
        obs_seq[i, :ep_len] = episodes_obs[i]
        act_seq[i, :ep_len] = episodes_act[i]
        existence_mask[i, :ep_len] = 1.0

    np.save(os.path.join(output_dir, "observations_seq.npy"), obs_seq)
    np.save(os.path.join(output_dir, "actions_seq.npy"), act_seq)
    np.save(os.path.join(output_dir, "existence_mask.npy"), existence_mask)

    if write_summary:
        summary_path = os.path.join(output_dir, "dataset_summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "env": env_name,
                    "num_episodes": int(n_eps),
                    "max_episode_length": int(max_len),
                    "obs_shape": list(obs_seq.shape),
                    "action_shape": list(act_seq.shape),
                },
                file,
                indent=2,
                sort_keys=True,
            )

    print(f"[{env_name}] Processed {n_eps} padded episodes into {output_dir}/")
    print(f"  - observations_seq.npy: {obs_seq.shape}")
    print(f"  - actions_seq.npy: {act_seq.shape}")
    print(f"  - existence_mask.npy: {existence_mask.shape}")
    if write_summary:
        print("  - dataset_summary.json")


def process_kitchen(env_name: str, output_dir: str) -> None:
    prepare_kitchen_dataset(env_name=env_name, output_dir=output_dir, write_summary=False)

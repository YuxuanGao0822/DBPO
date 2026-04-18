"""Minimal vectorized gym env factory for released DBPO stage-2 finetune."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os

import numpy as np

from dbpo.envs.gym_utils.wrapper import wrapper_dict


def _stack_obs(observations):
    first = observations[0]
    if isinstance(first, Mapping):
        return {key: _stack_obs([obs[key] for obs in observations]) for key in first.keys()}
    return np.stack(observations, axis=0)


def _normalize_step_result(step_result):
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, reward, terminated, truncated, info
    observation, reward, done, info = step_result
    info = dict(info)
    terminated = bool(info.pop("terminated", done))
    truncated = bool(info.pop("truncated", False))
    if done and not terminated and not truncated:
        terminated = True
    return observation, reward, terminated, truncated, info


class SyncVectorEnvCompat:
    """Small vector-env adapter sufficient for released D4RL finetune tasks."""

    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        if not self.envs:
            raise ValueError("env_fns must not be empty")
        self.num_envs = len(self.envs)
        self.metadata = getattr(self.envs[0], "metadata", {})
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = getattr(self.envs[0], "observation_space", None)
        self.action_space = getattr(self.envs[0], "action_space", None)

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None] * self.num_envs
        elif isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        if len(seeds) != self.num_envs:
            raise ValueError("seed list length must equal num_envs")
        for env, seed in zip(self.envs, seeds):
            if hasattr(env, "seed"):
                env.seed(seed)

    def reset_arg(self, options_list):
        if len(options_list) != self.num_envs:
            raise ValueError("options_list length must equal num_envs")
        observations = []
        for env, options in zip(self.envs, options_list):
            observation = env.reset(options=options)
            if isinstance(observation, tuple):
                observation = observation[0]
            observations.append(observation)
        return _stack_obs(observations)

    def step(self, actions):
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        for env, action in zip(self.envs, actions):
            observation, reward, terminated, truncated, info = _normalize_step_result(env.step(action))
            observations.append(observation)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return (
            _stack_obs(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminateds, dtype=np.bool_),
            np.asarray(truncateds, dtype=np.bool_),
            infos,
        )

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def render(self, *args, **kwargs):
        return tuple(env.render(*args, **kwargs) for env in self.envs)


def make_async(
    env_name: str,
    num_envs: int = 1,
    asynchronous: bool = True,
    wrappers=None,
    render: bool = False,
    obs_dim: int = 23,
    action_dim: int = 7,
    env_type: str = "gym",
    max_episode_steps=None,
    robomimic_env_cfg_path=None,
    dataset_path=None,
    use_image_obs: bool = False,
    render_offscreen: bool = False,
    reward_shaping: bool = False,
    shape_meta=None,
    **kwargs,
):
    """Create the public stage-2 vectorized env path.

    The current public stage-2 workflow only supports the public D4RL low-dimensional
    locomotion tasks. The factory keeps the historical ``make_async`` name for
    compatibility with the existing workspace.
    """
    del asynchronous, render, obs_dim, action_dim, max_episode_steps, robomimic_env_cfg_path
    del dataset_path, use_image_obs, render_offscreen, reward_shaping, shape_meta

    if env_type not in {"d4rl", "gym"}:
        raise ValueError(
            f"Public stage-2 finetune only supports env_type=d4rl/gym, got {env_type!r}."
        )

    def _make_env():
        if env_type == "d4rl":
            import d4rl.gym_mujoco  # noqa: F401
        from gym.envs import make as make_

        env_kwargs = dict(kwargs)
        env = make_(env_name, **env_kwargs)

        if wrappers is not None:
            for wrapper_name, wrapper_args in wrappers.items():
                env = wrapper_dict[wrapper_name](env, **wrapper_args)
        return env

    return SyncVectorEnvCompat([_make_env for _ in range(num_envs)])

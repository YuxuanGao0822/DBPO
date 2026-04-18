"""Low-dimensional D4RL MuJoCo wrapper for stage-2 gym finetune."""

from __future__ import annotations

import gym
from gym import spaces
import numpy as np


class MujocoLocomotionLowdimWrapper(gym.Env):
    """Normalize low-dimensional observations and unnormalize actions."""

    def __init__(self, env, normalization_path: str):
        self.env = env
        self.action_space = env.action_space

        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

        obs_example = self.env.reset()
        if isinstance(obs_example, tuple):
            obs_example = obs_example[0]
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = spaces.Dict()
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def seed(self, seed=None):
        np.random.seed(seed=seed if seed is not None else None)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        action = (action + 1) / 2
        return action * (self.action_max - self.action_min) + self.action_min

    def reset(self, **kwargs):
        options = kwargs.get("options", {})
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        raw_obs = self.env.reset()
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        return {"state": self.normalize_obs(raw_obs)}

    def step(self, action):
        raw_action = self.unnormalize_action(action)
        step_result = self.env.step(raw_action)
        if len(step_result) == 5:
            raw_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            info = dict(info)
            info.setdefault("terminated", terminated)
            info.setdefault("truncated", truncated)
        else:
            raw_obs, reward, done, info = step_result
        return {"state": self.normalize_obs(raw_obs)}, reward, done, info

    def render(self, **kwargs):
        return self.env.render()

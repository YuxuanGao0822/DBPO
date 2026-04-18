"""Multi-step action wrapper with stacked observations."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import gym
from gym import spaces
import numpy as np


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    if isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    return {key: take_last_n(value, n) for key, value in x.items()}


def aggregate(data, method="max"):
    if method == "max":
        return np.max(data)
    if method == "min":
        return np.min(data)
    if method == "mean":
        return np.mean(data)
    if method == "sum":
        return np.sum(data)
    raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        result[:start_idx] = result[start_idx]
    return result


class MultiStep(gym.Wrapper):
    """Execute multi-step actions and expose stacked observations."""

    def __init__(
        self,
        env,
        n_obs_steps: int = 1,
        n_action_steps: int = 1,
        max_episode_steps=None,
        reward_agg_method: str = "sum",
        prev_action: bool = True,
        reset_within_step: bool = False,
        pass_full_observations: bool = False,
        **kwargs,
    ):
        super().__init__(env)
        del kwargs
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations

    def reset(self, seed=None, return_info=False, options={}):
        obs = self.env.reset(seed=seed, options=options, return_info=return_info)
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque([self._single_action_space.sample()], maxlen=self.n_obs_steps)
        self.reward = []
        self.done = []
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        self.cnt = 0
        return self._get_obs(self.n_obs_steps)

    def step(self, action):
        if action.ndim == 1:
            action = action[None]
        truncated = False
        terminated = False
        act_step = 0
        for act_step, act in enumerate(action):
            self.cnt += 1
            if terminated or truncated:
                break
            observation, reward, done, info = self.env.step(act)
            self.obs.append(observation)
            if self.prev_action:
                self.action.append(act)
            self.reward.append(reward)
            if "TimeLimit.truncated" not in info:
                if done:
                    terminated = True
                elif self.max_episode_steps is not None and self.cnt >= self.max_episode_steps:
                    truncated = True
            else:
                truncated = info["TimeLimit.truncated"]
                terminated = done
            done = truncated or terminated
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        info = dict_take_last_n(self.info, self.n_obs_steps)
        if self.pass_full_observations:
            info["full_obs"] = self._get_obs(act_step + 1)

        if self.reset_within_step and self.done[-1]:
            if truncated:
                info["final_obs"] = observation
            observation = self.reset()

        self.reward = []
        self.done = []
        return observation, reward, terminated, truncated, info

    def _get_obs(self, n_steps=1):
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        if isinstance(self.observation_space, spaces.Dict):
            return {
                key: stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
                for key in self.observation_space.keys()
            }
        raise RuntimeError("Unsupported space type")

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

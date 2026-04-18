"""PPO rollout buffer for single-step DBPO fine-tuning."""

from __future__ import annotations

import logging

import numpy as np
import torch


log = logging.getLogger(__name__)


class RolloutBuffer:
    """CPU-side rollout buffer for single-step DBPO PPO fine-tuning."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        horizon_steps: int,
        act_steps: int,
        action_dim: int,
        n_cond_step: int,
        obs_dim: int,
        furniture_sparse_reward: bool,
        best_reward_threshold_for_success: float,
        reward_scale_running: bool,
        gamma: float,
        gae_lambda: float,
        reward_scale_const: float,
        device: torch.device,
        save_full_observation: bool = False,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.n_cond_step = n_cond_step
        self.obs_dim = obs_dim
        self.furniture_sparse_reward = furniture_sparse_reward
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.reward_scale_running = reward_scale_running
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale_const = reward_scale_const
        self.device = device
        self.save_full_observation = save_full_observation
        self.ft_generation_steps = 1

    def reset(self):
        self.obs_trajs = {
            "state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))
        }
        self.chains_trajs = np.zeros(
            (self.n_steps, self.n_envs, 2, self.horizon_steps, self.action_dim)
        )
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        self.value_trajs = np.zeros((self.n_steps, self.n_envs))
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs))

    def add(
        self,
        step,
        state_venv,
        chains_venv,
        reward_venv,
        terminated_venv,
        truncated_venv,
        value_venv,
        logprob_venv,
    ):
        done_venv = terminated_venv | truncated_venv
        self.obs_trajs["state"][step] = state_venv
        self.chains_trajs[step] = chains_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv
        self.value_trajs[step] = value_venv
        self.logprobs_trajs[step] = logprob_venv

    @torch.no_grad()
    def update(self, obs_venv: dict, critic: torch.nn.Module):
        self._normalize_reward()
        self._compute_gae(obs_venv, critic)

    @torch.no_grad()
    def _compute_gae(self, obs_venv: dict, critic: torch.nn.Module):
        obs_ts = {"state": torch.from_numpy(obs_venv["state"]).float().to(self.device)}
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs))
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextvalues = critic(obs_ts).reshape(1, -1).cpu().numpy()
            else:
                nextvalues = self.value_trajs[t + 1]
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            self.advantages_trajs[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        self.returns_trajs = self.advantages_trajs + self.value_trajs

    def _normalize_reward(self):
        if self.reward_scale_running:
            std = self.reward_trajs.std() + 1e-8
            self.reward_trajs = self.reward_trajs / std

    def make_dataset(self):
        obs = torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0, 1)
        chains = torch.tensor(self.chains_trajs, device=self.device).float().flatten(0, 1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0, 1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0, 1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0, 1)
        logprobs = torch.tensor(self.logprobs_trajs, device=self.device).float().flatten(0, 1)
        return obs, chains, returns, values, advantages, logprobs

    @torch.no_grad()
    def get_explained_var(self, values, returns):
        y_pred = values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        return float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    @torch.no_grad()
    def summarize_episode_reward(self):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(self.firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start, end = env_steps[i], env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                self.reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array([np.sum(r) for r in reward_trajs_split])
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [np.max(r) / self.act_steps for r in reward_trajs_split]
                )
            self.avg_episode_reward = float(np.mean(episode_reward))
            self.avg_best_reward = float(np.mean(episode_best_reward))
            self.success_rate = float(
                np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
            )
            self.std_episode_reward = float(np.std(episode_reward))
            self.std_best_reward = float(np.std(episode_best_reward))
            self.std_success_rate = float(
                np.std(episode_best_reward >= self.best_reward_threshold_for_success)
            )
            episode_lengths = np.array(
                [end - start + 1 for _, start, end in episodes_start_end]
            ) * self.act_steps
            self.avg_episode_length = float(np.mean(episode_lengths))
            self.std_episode_length = float(np.std(episode_lengths))
        else:
            self.num_episode_finished = 0
            self.avg_episode_reward = 0.0
            self.avg_best_reward = 0.0
            self.success_rate = 0.0
            self.avg_episode_length = 0.0
            self.std_episode_reward = 0.0
            self.std_best_reward = 0.0
            self.std_success_rate = 0.0
            self.std_episode_length = 0.0
            log.warning("No episode completed within the iteration.")

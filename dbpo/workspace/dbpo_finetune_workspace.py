"""
DBPO online fine-tuning workspace (PPO).

PPO is implemented here as a proof-of-concept feasibility validation of the
DBPO paradigm. It is not presented as the definitive fine-tuning algorithm
within the DBPO framework.

Logging namespace:
  train/pg_loss         — PPO policy gradient loss
  train/value_loss      — critic value loss
  train/entropy_loss    — entropy regularisation loss
  train/anchor_loss     — anchor regularisation to the pretrained DBP manifold
  train/approx_kl       — approximate KL divergence
  train/clipfrac        — fraction of clipped ratios
  train/noise_std       — mean action noise std
  train/actor_lr        — actor learning rate
  train/critic_lr       — critic learning rate
  eval/success_rate     — task success rate
  eval/avg_episode_reward
  eval/avg_episode_length
"""
from __future__ import annotations

import logging
import os
import pickle
import random
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from dbpo.utils.logging_util import init_wandb, log_metrics, finish_wandb, create_bordered_text
from dbpo.utils.scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)


class DBPOFinetuneWorkspace:
    """
    DBPO online fine-tuning workspace.

    Config contract
    ---------------
    cfg.model           : DBPOPolicyOptimizer (actor + critic)
    cfg.env             : vectorised env config (name, n_envs, wrappers, …)
    cfg.train           : PPO hyperparameters (see finetune_ppo.yaml)
    cfg.wandb           : WandB config (optional)
    cfg.logdir          : output directory
    cfg.seed            : random seed
    cfg.device          : torch device string
    cfg.actor_policy_path : path to pre-trained DBP checkpoint (required)
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = OmegaConf.create(kwargs)
        elif kwargs:
            merged = OmegaConf.create(kwargs)
            cfg = OmegaConf.merge(cfg, merged)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.render_dir, exist_ok=True)
        init_wandb(cfg.get("wandb", None))

        # Build DBPOPPOWrapper model
        import hydra
        self.model = hydra.utils.instantiate(cfg.model)
        log.info("DBPOPPOWrapper model built. params=%.2fM",
                 sum(p.numel() for p in self.model.parameters()) / 1e6)

        # Vectorised environment
        from dbpo.envs.gym_utils import make_async
        env_cfg = cfg.env
        self.venv = make_async(
            env_name=env_cfg.name,
            num_envs=env_cfg.n_envs,
            asynchronous=True,
            max_episode_steps=env_cfg.max_episode_steps,
            wrappers=env_cfg.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            dataset_path=cfg.task.get("env_dataset_path", cfg.task.get("dataset_path", None)),
            env_type=env_cfg.get("env_type", "gym"),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=env_cfg.get("use_image_obs", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
        )
        self.venv.seed([seed + i for i in range(env_cfg.n_envs)])

        # Training params
        train = cfg.train
        self.n_train_itr = train.n_train_itr
        self.n_steps = train.n_steps
        self.n_envs = env_cfg.n_envs
        self.batch_size = train.batch_size
        self.update_epochs = train.update_epochs
        self.gamma = train.gamma
        self.gae_lambda = train.get("gae_lambda", 0.95)
        self.ent_coef = train.get("ent_coef", 0.01)
        self.vf_coef = train.get("vf_coef", 0.5)
        self.max_grad_norm = train.get("max_grad_norm", None)
        self.target_kl = train.get("target_kl", None)
        self.n_critic_warmup_itr = train.get("n_critic_warmup_itr", 0)
        self.val_freq = train.val_freq
        self.save_model_freq = train.save_model_freq
        self.log_freq = train.get("log_freq", 1)
        self.normalize_act_space_dim = train.get("normalize_act_space_dimension", True)
        self.clip_intermediate_actions = train.get("clip_intermediate_actions", True)
        self.reward_scale_const = train.get("reward_scale_const", 1.0)
        self.best_reward_threshold = env_cfg.get("best_reward_threshold_for_success", 1.0)
        self.furniture_sparse_reward = False

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=train.actor_lr,
            weight_decay=train.get("actor_weight_decay", 1e-6),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=train.critic_lr,
            weight_decay=train.get("critic_weight_decay", 1e-6),
        )
        alr_cfg = train.actor_lr_scheduler
        clr_cfg = train.critic_lr_scheduler
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=alr_cfg.first_cycle_steps,
            max_lr=train.actor_lr,
            min_lr=alr_cfg.min_lr,
            warmup_steps=alr_cfg.warmup_steps,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=clr_cfg.first_cycle_steps,
            max_lr=train.critic_lr,
            min_lr=clr_cfg.min_lr,
            warmup_steps=clr_cfg.warmup_steps,
        )

        # State
        self.itr = 0
        self.cnt_train_step = 0
        self.current_best_reward = float("-inf")
        self.run_results = []

        # Resume
        resume_path = cfg.get("resume_path", None)
        if resume_path:
            self._resume(resume_path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        from dbpo.methods.dbpo.rollout_buffer import RolloutBuffer

        buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            horizon_steps=self.cfg.horizon_steps,
            act_steps=self.cfg.act_steps,
            action_dim=self.cfg.action_dim,
            n_cond_step=self.cfg.cond_steps,
            obs_dim=self.cfg.obs_dim,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold,
            reward_scale_running=self.cfg.train.get("reward_scale_running", False),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
        )

        prev_obs = self.venv.reset_arg([{} for _ in range(self.n_envs)])
        done_venv = np.zeros(self.n_envs)
        last_itr_eval = False

        try:
            while self.itr < self.n_train_itr:
                eval_mode = (self.itr % self.val_freq == 0)
                self.model.eval() if eval_mode else self.model.train()
                buffer.reset()

                if eval_mode or last_itr_eval:
                    prev_obs = self.venv.reset_arg([{} for _ in range(self.n_envs)])
                    buffer.firsts_trajs[0] = 1
                else:
                    buffer.firsts_trajs[0] = done_venv

                # --- Rollout ---
                for step in range(self.n_steps):
                    with torch.no_grad():
                        cond = {"state": torch.tensor(
                            prev_obs["state"], device=self.device, dtype=torch.float32
                        )}
                        value_venv = self.model.critic(cond).view(-1).cpu().numpy()
                        actions, chains, logprobs = self.model.get_actions(
                            cond=cond,
                            eval_mode=eval_mode,
                            save_chains=True,
                            normalize_act_space_dimension=self.normalize_act_space_dim,
                            clip_intermediate_actions=self.clip_intermediate_actions,
                        )
                        action_np = actions.cpu().numpy()
                        chains_np = chains.cpu().numpy()
                        logprobs_np = logprobs.cpu().numpy()

                    action_venv = action_np[:, : self.cfg.act_steps]
                    obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = \
                        self.venv.step(action_venv)
                    done_venv = terminated_venv | truncated_venv

                    buffer.add(
                        step,
                        prev_obs["state"],
                        chains_np,
                        reward_venv,
                        terminated_venv,
                        truncated_venv,
                        value_venv,
                        logprobs_np,
                    )
                    prev_obs = obs_venv
                    if not eval_mode:
                        self.cnt_train_step += self.n_envs * self.cfg.act_steps

                buffer.summarize_episode_reward()

                # --- Update ---
                if not eval_mode:
                    buffer.update(obs_venv, self.model.critic)
                    train_metrics = self._ppo_update(buffer)
                    train_metrics.update({
                        "train/actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                        "train/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                        "train/total_env_steps": self.cnt_train_step,
                        "train/success_rate": buffer.success_rate,
                        "train/avg_episode_reward": buffer.avg_episode_reward,
                    })
                    if self.itr % self.log_freq == 0:
                        log_metrics(train_metrics, step=self.itr)

                # --- Eval logging ---
                if eval_mode and self.itr % self.log_freq == 0:
                    eval_metrics = {
                        "eval/success_rate": buffer.success_rate,
                        "eval/avg_episode_reward": buffer.avg_episode_reward,
                        "eval/avg_best_reward": buffer.avg_best_reward,
                        "eval/avg_episode_length": buffer.avg_episode_length,
                        "eval/num_episodes": buffer.num_episode_finished,
                    }
                    log_metrics(eval_metrics, step=self.itr)
                    log.info(create_bordered_text(
                        f"Eval itr {self.itr}\n" +
                        "\n".join(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}"
                                  for k, v in eval_metrics.items())
                    ))
                    if buffer.avg_episode_reward > self.current_best_reward:
                        self.current_best_reward = buffer.avg_episode_reward
                        self._save_best()

                self._update_lr()
                self._save_model()
                self.run_results.append({"itr": self.itr, "step": self.cnt_train_step})
                with open(self.result_path, "wb") as f:
                    pickle.dump(self.run_results, f)

                last_itr_eval = eval_mode
                self.itr += 1

            log.info("Fine-tuning complete.")
        finally:
            if hasattr(self.venv, "close"):
                self.venv.close()
            finish_wandb()

    # ------------------------------------------------------------------
    # PPO gradient update
    # ------------------------------------------------------------------
    def _ppo_update(self, buffer) -> dict:
        obs, chains, returns, oldvalues, advantages, oldlogprobs = buffer.make_dataset()
        total_steps = self.n_steps * self.n_envs
        clipfracs, noise_stds = [], []

        for update_epoch in range(self.update_epochs):
            indices = torch.randperm(total_steps, device=self.device)
            kl_too_large = False
            for start in range(0, total_steps, self.batch_size):
                inds = indices[start : start + self.batch_size]
                minibatch = (
                    {"state": obs[inds]},
                    chains[inds],
                    returns[inds],
                    oldvalues[inds],
                    advantages[inds],
                    oldlogprobs[inds],
                )
                (pg_loss, entropy_loss, v_loss, anchor_loss, clipfrac,
                 approx_kl, ratio, noise_std) = self.model.loss(
                    *minibatch,
                    normalize_act_space_dimension=self.normalize_act_space_dim,
                    verbose=False,
                    clip_intermediate_actions=self.clip_intermediate_actions,
                )
                loss = (pg_loss + entropy_loss * self.ent_coef
                        + v_loss * self.vf_coef
                        + anchor_loss * self.model.anchor_loss_coeff)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.actor_ft.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                if self.itr >= self.n_critic_warmup_itr:
                    self.actor_optimizer.step()

                clipfracs.append(clipfrac)
                noise_stds.append(float(noise_std))
                if self.target_kl and approx_kl > self.target_kl:
                    kl_too_large = True
                    break
            if kl_too_large:
                break

        return {
            "train/pg_loss": pg_loss.item(),
            "train/value_loss": v_loss.item(),
            "train/entropy_loss": entropy_loss.item(),
            "train/anchor_loss": anchor_loss.item(),
            "train/approx_kl": approx_kl,
            "train/clipfrac": float(np.mean(clipfracs)),
            "train/noise_std": float(np.mean(noise_stds)),
            "train/ratio": ratio,
        }

    def _update_lr(self):
        self.critic_lr_scheduler.step()
        if self.itr >= self.n_critic_warmup_itr:
            self.actor_lr_scheduler.step()

    def _save_model(self):
        data = {
            "itr": self.itr,
            "cnt_train_steps": self.cnt_train_step,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
        }
        torch.save(data, os.path.join(self.checkpoint_dir, "last.pt"))
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, path)
            log.info("Saved checkpoint to %s", path)

    def _save_best(self):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
        }
        path = os.path.join(self.checkpoint_dir, "best.pt")
        torch.save(data, path)
        log.info("Saved best model (reward=%.3f) to %s", self.current_best_reward, path)

    def _resume(self, path: str):
        log.info("Resuming fine-tuning from %s", path)
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.itr = data.get("itr", 0)
        self.cnt_train_step = data.get("cnt_train_steps", 0)
        self.model.load_state_dict(data["model"])
        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        if "actor_lr_scheduler" in data:
            self.actor_lr_scheduler.load_state_dict(data["actor_lr_scheduler"])
            self.critic_lr_scheduler.load_state_dict(data["critic_lr_scheduler"])
        log.info("Resumed from itr=%d, steps=%d", self.itr, self.cnt_train_step)

"""DBPO stochastic adapter with exact-likelihood PPO updates."""

from __future__ import annotations

import copy
import logging
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class StateConditionedLogStdHead(nn.Module):
    """State-conditioned diagonal log-std head for the DBPO actor."""

    def __init__(self, cond_dim: int, output_dim: int, init_log_std: float):
        super().__init__()
        self.proj = nn.Linear(cond_dim, output_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_log_std)

    def forward(self, cond_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(cond_emb)


class DBPOActor(nn.Module):
    """Wrap a pretrained DBP policy with a state-conditioned Gaussian head."""

    def __init__(self, policy: nn.Module, init_log_std: float):
        super().__init__()
        self.policy = policy
        cond_dim = getattr(policy.model, "cond_enc_dim")
        output_dim = policy.horizon * policy.action_dim
        self.logstd_head = StateConditionedLogStdHead(
            cond_dim=cond_dim,
            output_dim=output_dim,
            init_log_std=init_log_std,
        )

    @staticmethod
    def _policy_obs_dict(cond: dict) -> dict:
        if "obs" in cond:
            return cond
        if "state" in cond:
            return {"obs": cond["state"]}
        return cond

    def forward(
        self,
        cond: dict,
        z: torch.Tensor,
        clip_intermediate_actions: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result, cond_emb = self.policy.predict_action(
            cond=self._policy_obs_dict(cond),
            noise=z,
            output_embedding=True,
        )
        mean = result["action_pred"]
        log_std = self.logstd_head(cond_emb).view_as(mean)
        return mean, log_std


class DBPOPolicyOptimizer(nn.Module):
    """DBPO online fine-tuning wrapper using PPO + anchor regularization."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_policy_path: str | None = None,
        device: torch.device | str = "cuda:0",
        horizon_steps: int = 4,
        action_dim: int = 3,
        act_min: float = -1.0,
        act_max: float = 1.0,
        executed_act_steps: int | None = None,
        anchor_loss_coeff: float = 1.0,
        min_sampling_std: float = 0.03,
        min_logprob_std: float = 0.03,
        max_logprob_std: float = 0.10,
        init_logprob_std: float | None = None,
        logprob_min: float = -2.0,
        logprob_max: float | None = None,
        clip_ploss_coef: float = 0.02,
        clip_vloss_coef: float | None = None,
        randn_clip_value: float = 3.0,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            log.info("Ignoring unused DBPOPolicyOptimizer config keys: %s", sorted(kwargs.keys()))
        self.device = torch.device(device)
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.act_dim_total = horizon_steps * action_dim
        self.anchor_loss_coeff = float(anchor_loss_coeff)

        if executed_act_steps is None:
            executed_act_steps = horizon_steps
        self.executed_act_steps = int(executed_act_steps)
        if not 1 <= self.executed_act_steps <= self.horizon_steps:
            raise ValueError(
                "executed_act_steps must satisfy 1 <= executed_act_steps <= horizon_steps"
            )
        self.executed_act_dim_total = self.executed_act_steps * action_dim

        self.act_min = act_min
        self.act_max = act_max
        self.min_sampling_std = float(min_sampling_std)
        self.min_logprob_std = float(min_logprob_std)
        self.max_logprob_std = float(max_logprob_std)
        if init_logprob_std is None:
            init_logprob_std = self.min_logprob_std
        self.init_logprob_std = float(
            min(max(init_logprob_std, self.min_logprob_std), self.max_logprob_std)
        )
        self.logprob_min = float(logprob_min)
        self.logprob_max = None if logprob_max is None else float(logprob_max)
        self.clip_ploss_coef = float(clip_ploss_coef)
        self.clip_vloss_coef = clip_vloss_coef
        self.randn_clip_value = float(randn_clip_value)
        self.norm_adv = norm_adv

        self.actor_old = actor.to(self.device)
        if actor_policy_path:
            self.load_policy(actor_policy_path, use_ema=True)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.eval()

        actor_copy = copy.deepcopy(self.actor_old)
        for param in actor_copy.parameters():
            param.requires_grad = True
        self.actor_ft = DBPOActor(
            policy=actor_copy,
            init_log_std=float(torch.log(torch.tensor(self.init_logprob_std)).item()),
        ).to(self.device)
        self.critic = critic.to(self.device)
        self._report_network_params()

    def _report_network_params(self) -> None:
        log.info(
            "DBPO parameters — total=%.3fM actor_old=%.3fM actor_ft=%.3fM critic=%.3fM",
            sum(p.numel() for p in self.parameters()) / 1e6,
            sum(p.numel() for p in self.actor_old.parameters()) / 1e6,
            sum(p.numel() for p in self.actor_ft.parameters()) / 1e6,
            sum(p.numel() for p in self.critic.parameters()) / 1e6,
        )

    def _extract_actor_state_dict(self, checkpoint: dict, use_ema: bool) -> dict:
        if use_ema and checkpoint.get("ema") is not None:
            return checkpoint["ema"]
        if "policy" in checkpoint:
            return checkpoint["policy"]
        if "model" in checkpoint:
            actor_policy_state_dict = {}
            for key, value in checkpoint["model"].items():
                if key.startswith("actor_old."):
                    actor_policy_state_dict[key.replace("actor_old.", "")] = value
                elif key.startswith("actor_ft.policy."):
                    actor_policy_state_dict[key.replace("actor_ft.policy.", "")] = value
            if actor_policy_state_dict:
                return actor_policy_state_dict
            return checkpoint["model"]
        raise KeyError(
            "Unsupported actor checkpoint schema. Expected one of: ema, policy, model."
        )

    def load_policy(self, network_path: str, use_ema: bool = True) -> None:
        log.info("Loading stage-1 DBP policy from %s", network_path)
        checkpoint = torch.load(network_path, map_location=self.device, weights_only=True)
        state_dict = self._extract_actor_state_dict(checkpoint, use_ema=use_ema)
        missing, unexpected = self.actor_old.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("Missing keys when loading actor: %s", missing)
        if unexpected:
            log.info("Unexpected keys when loading actor: %s", unexpected)

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.horizon_steps, self.action_dim, device=self.device)

    def _distribution(
        self,
        cond: dict,
        z: torch.Tensor,
        clip_intermediate_actions: bool = True,
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.actor_ft(
            cond=cond,
            z=z,
            clip_intermediate_actions=clip_intermediate_actions,
        )
        log_std = raw_log_std.clamp(
            min=torch.log(torch.tensor(self.min_logprob_std, device=self.device)),
            max=torch.log(torch.tensor(self.max_logprob_std, device=self.device)),
        )
        std = log_std.exp()
        return Normal(mean, std), mean, std

    def _executed_prefix(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., : self.executed_act_steps, :]

    def _reduce_executed_prefix(
        self,
        tensor: torch.Tensor,
        normalize_act_space_dimension: bool = False,
    ) -> torch.Tensor:
        reduced = self._executed_prefix(tensor).sum(dim=(-2, -1))
        if normalize_act_space_dimension:
            reduced = reduced / self.executed_act_dim_total
        return reduced

    def _mean_executed_prefix(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._executed_prefix(tensor).mean()

    def get_logprobs(
        self,
        cond: dict,
        chains: torch.Tensor,
        get_entropy: bool = False,
        normalize_act_space_dimension: bool = False,
        clip_intermediate_actions: bool = True,
        verbose_entropy_stats: bool = True,
        get_chains_stds: bool = True,
    ):
        z = chains[:, 0]
        actions = chains[:, 1]
        dist, _, std = self._distribution(
            cond=cond,
            z=z,
            clip_intermediate_actions=clip_intermediate_actions,
        )
        logprob = self._reduce_executed_prefix(
            dist.log_prob(actions),
            normalize_act_space_dimension=normalize_act_space_dimension,
        )
        entropy = None
        if get_entropy:
            entropy = self._reduce_executed_prefix(
                dist.entropy(),
                normalize_act_space_dimension=normalize_act_space_dimension,
            )
            if verbose_entropy_stats:
                log.info(
                    "entropy=%s 10%%=%.2f 50%%=%.2f 90%%=%.2f",
                    tuple(entropy.shape),
                    entropy.quantile(0.1).item(),
                    entropy.median().item(),
                    entropy.quantile(0.9).item(),
                )

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy, self._mean_executed_prefix(std)
            return logprob, entropy
        if get_chains_stds:
            return logprob, self._mean_executed_prefix(std)
        return logprob

    @torch.no_grad()
    def get_actions(
        self,
        cond: dict,
        eval_mode: bool,
        save_chains: bool = False,
        normalize_act_space_dimension: bool = False,
        clip_intermediate_actions: bool = True,
        ret_logprob: bool = True,
    ):
        batch_size = next(iter(cond.values())).shape[0]
        z = self._sample_noise(batch_size)
        dist, mean, std = self._distribution(
            cond=cond,
            z=z,
            clip_intermediate_actions=clip_intermediate_actions,
        )
        sampling_std = std.clamp(min=self.min_sampling_std)
        sampling_dist = Normal(mean, sampling_std)

        if eval_mode:
            actions = mean
        else:
            sample = sampling_dist.sample()
            sample = sample.clamp_(
                sampling_dist.loc - self.randn_clip_value * sampling_std,
                sampling_dist.loc + self.randn_clip_value * sampling_std,
            )
            actions = sample
        actions = actions.clamp_(self.act_min, self.act_max)

        log_prob = None
        if ret_logprob:
            log_prob = self._reduce_executed_prefix(
                sampling_dist.log_prob(actions),
                normalize_act_space_dimension=normalize_act_space_dimension,
            )

        chains = None
        if save_chains:
            chains = torch.stack([z, actions], dim=1)

        if ret_logprob and save_chains:
            return actions, chains, log_prob
        if ret_logprob:
            return actions, log_prob
        if save_chains:
            return actions, chains
        return actions

    def loss(
        self,
        obs: dict,
        chains: torch.Tensor,
        returns: torch.Tensor,
        oldvalues: torch.Tensor,
        advantages: torch.Tensor,
        oldlogprobs: torch.Tensor,
        normalize_act_space_dimension: bool = False,
        verbose: bool = True,
        clip_intermediate_actions: bool = True,
    ):
        newlogprobs, entropy, noise_std = self.get_logprobs(
            obs,
            chains,
            get_entropy=True,
            normalize_act_space_dimension=normalize_act_space_dimension,
            clip_intermediate_actions=clip_intermediate_actions,
            verbose_entropy_stats=verbose,
        )

        if self.logprob_max is None:
            newlogprobs = newlogprobs.clamp(min=self.logprob_min)
            oldlogprobs = oldlogprobs.clamp(min=self.logprob_min)
        else:
            newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
            oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1 - self.clip_ploss_coef,
            1 + self.clip_ploss_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef is not None:
            v_clipped = torch.clamp(
                newvalues,
                oldvalues - self.clip_vloss_coef,
                oldvalues + self.clip_vloss_coef,
            )
            v_loss = 0.5 * torch.max(
                (newvalues - returns) ** 2,
                (v_clipped - returns) ** 2,
            ).mean()

        entropy_loss = -entropy.mean()

        z = chains[:, 0]
        policy_obs = DBPOActor._policy_obs_dict(obs)
        with torch.no_grad():
            old_actions = self.actor_old.predict_action(cond=policy_obs, noise=z)["action_pred"]
        new_actions = self.actor_ft.policy.predict_action(cond=policy_obs, noise=z)["action_pred"]
        anchor_loss = F.mse_loss(new_actions, old_actions)

        return (
            pg_loss,
            entropy_loss,
            v_loss,
            anchor_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            noise_std.detach(),
        )


DBPOPPOWrapper = DBPOPolicyOptimizer

"""DBP policies organized by modality."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from dbpo.algorithm.dbp_loss import compute_dbp_loss
from dbpo.model.common.normalizer import LinearNormalizer
from dbpo.model.unet.dbp_unet1d import DBPUNet1D
from dbpo.utils.pytorch_util import dict_apply


def _resolve_obs_dict(
    obs_dict: Optional[Dict[str, torch.Tensor]] = None,
    cond: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    if obs_dict is not None:
        return obs_dict
    if cond is not None:
        return cond
    raise ValueError("Either obs_dict or cond must be provided.")


class _BaseDBPPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        temperatures: Optional[list] = None,
        per_timestep_loss: bool = False,
        gen_per_label: int = 4,
    ):
        super().__init__()
        if temperatures is None:
            temperatures = [0.02, 0.05, 0.2]
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.temperatures = temperatures
        self.per_timestep_loss = per_timestep_loss
        self.gen_per_label = gen_per_label

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _sample_noise(self, batch_size: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is not None:
            return noise.to(device=self.device, dtype=self.dtype)
        return torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

    def _slice_action(self, naction_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def _compute_dbp_loss(self, pred_actions: torch.Tensor, nactions: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size = nactions.shape[0]
        R_list = tuple(self.temperatures)
        if self.per_timestep_loss:
            horizon = nactions.shape[1]
            total_loss = 0.0
            metrics = {}
            for t in range(horizon):
                gen_t = pred_actions[:, :, t, :]
                pos_t = nactions[:, t, :].unsqueeze(1)
                loss_t, info_t = compute_dbp_loss(gen_t, pos_t, temp_schedule=R_list)
                total_loss = total_loss + loss_t.mean()
                for key, value in info_t.items():
                    metrics[key] = metrics.get(key, 0.0) + value.item() / horizon
            return total_loss / horizon, metrics

        gen = pred_actions.reshape(batch_size, self.gen_per_label, -1)
        pos = nactions.reshape(batch_size, 1, -1)
        loss, info = compute_dbp_loss(gen, pos, temp_schedule=R_list)
        return loss.mean(), {k: v.item() for k, v in info.items()}


class LowdimDBPPolicy(_BaseDBPPolicy):
    """DBP policy for low-dimensional observations."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        backbone_kwargs: Optional[dict] = None,
        temperatures: Optional[list] = None,
        per_timestep_loss: bool = False,
        gen_per_label: int = 4,
    ):
        super().__init__(
            action_dim=action_dim,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            temperatures=temperatures,
            per_timestep_loss=per_timestep_loss,
            gen_per_label=gen_per_label,
        )
        if backbone_kwargs is None:
            backbone_kwargs = {}
        else:
            backbone_kwargs = dict(backbone_kwargs)
        self.obs_dim = obs_dim
        self.model = DBPUNet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim * n_obs_steps,
            **backbone_kwargs,
        )

    def predict_action(
        self,
        obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        output_embedding: bool = False,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_dict = _resolve_obs_dict(obs_dict=obs_dict, cond=cond)
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        batch_size = nobs.shape[0]
        global_cond = nobs[:, : self.n_obs_steps].reshape(batch_size, -1)
        naction_pred = self.model(self._sample_noise(batch_size, noise), global_cond=global_cond)
        result = self._slice_action(naction_pred)
        if output_embedding:
            return result, global_cond
        return result

    def compute_loss(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch["obs"]
        nactions = nbatch["action"]
        batch_size = nactions.shape[0]

        global_cond = nobs[:, : self.n_obs_steps].reshape(batch_size, -1)
        global_cond_rep = global_cond.repeat_interleave(self.gen_per_label, dim=0)
        noise = torch.randn(
            batch_size * self.gen_per_label,
            nactions.shape[1],
            nactions.shape[2],
            device=nactions.device,
        )
        pred_all = self.model(noise, global_cond=global_cond_rep)
        pred_actions = pred_all.reshape(batch_size, self.gen_per_label, nactions.shape[1], nactions.shape[2])
        return self._compute_dbp_loss(pred_actions, nactions)


class _EncodedObsDBPPolicy(_BaseDBPPolicy):
    def __init__(
        self,
        obs_encoder: nn.Module,
        action_dim: int,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        backbone_kwargs: Optional[dict] = None,
        temperatures: Optional[list] = None,
        per_timestep_loss: bool = False,
        gen_per_label: int = 4,
    ):
        super().__init__(
            action_dim=action_dim,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            temperatures=temperatures,
            per_timestep_loss=per_timestep_loss,
            gen_per_label=gen_per_label,
        )
        if backbone_kwargs is None:
            backbone_kwargs = {}
        else:
            backbone_kwargs = dict(backbone_kwargs)
        self.obs_encoder = obs_encoder
        obs_feature_dim = int(getattr(obs_encoder, "output_dim"))
        self.obs_feature_dim = obs_feature_dim
        self.model = DBPUNet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            **backbone_kwargs,
        )

    def _encode_obs(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def predict_action(
        self,
        obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        output_embedding: bool = False,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_dict = _resolve_obs_dict(obs_dict=obs_dict, cond=cond)
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        global_cond = self._encode_obs(nobs)
        naction_pred = self.model(self._sample_noise(batch_size, noise), global_cond=global_cond)
        result = self._slice_action(naction_pred)
        if output_embedding:
            return result, global_cond
        return result

    def compute_loss(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        global_cond = self._encode_obs(nobs)
        global_cond_rep = global_cond.repeat_interleave(self.gen_per_label, dim=0)
        noise = torch.randn(
            batch_size * self.gen_per_label,
            nactions.shape[1],
            nactions.shape[2],
            device=nactions.device,
        )
        pred_all = self.model(noise, global_cond=global_cond_rep)
        pred_actions = pred_all.reshape(batch_size, self.gen_per_label, nactions.shape[1], nactions.shape[2])
        return self._compute_dbp_loss(pred_actions, nactions)


class ImageDBPPolicy(_EncodedObsDBPPolicy):
    """DBP policy for image + low-dim observations."""


class PointCloudDBPPolicy(_EncodedObsDBPPolicy):
    """DBP policy for point-cloud + state observations."""


DBPLowdimPolicy = LowdimDBPPolicy
DBPHybridImagePolicy = ImageDBPPolicy

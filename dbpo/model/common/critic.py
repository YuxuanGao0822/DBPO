"""Value critic used by the released DBPO stage-2 PPO adapter."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueCritic(nn.Module):
    """Simple MLP value function on flattened low-dimensional state history."""

    def __init__(
        self,
        obs_dim: int,
        n_obs_steps: int = 1,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        input_dim = obs_dim * n_obs_steps
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Mish()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"]
        if state.ndim == 3:
            state = state.flatten(1)
        return self.net(state)

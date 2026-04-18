"""
Mask generators for sequence sampling compatibility.
"""
from typing import Sequence, Optional
import torch
from dbpo.model.common.module_attr_mixin import ModuleAttrMixin


class DummyMaskGenerator(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, shape):
        return torch.ones(size=shape, dtype=torch.bool, device=self.device)


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim,
        obs_dim,
        max_n_obs_steps=2,
        fix_obs_steps=True,
        action_visible=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps + 1, size=(B,), generator=rng, device=device
            )

        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)
        obs_mask = obs_mask & is_obs_dim

        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        return mask

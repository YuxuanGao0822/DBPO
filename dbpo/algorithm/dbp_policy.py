"""Compatibility exports for the reorganized DBP policy module."""

from dbpo.methods.dbp.policies import (
    DBPHybridImagePolicy,
    DBPLowdimPolicy,
    ImageDBPPolicy,
    LowdimDBPPolicy,
    PointCloudDBPPolicy,
)


def build_dbp_lowdim_policy(
    obs_dim: int,
    action_dim: int,
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
    down_dims: list | None = None,
    kernel_size: int = 5,
    n_groups: int = 8,
    cond_predict_scale: bool = True,
    temperatures: list | None = None,
    gen_per_label: int = 4,
    per_timestep_loss: bool = False,
) -> LowdimDBPPolicy:
    backbone_kwargs = {
        "down_dims": down_dims or [256, 512, 1024],
        "kernel_size": kernel_size,
        "n_groups": n_groups,
        "cond_predict_scale": cond_predict_scale,
    }
    return LowdimDBPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        temperatures=temperatures,
        gen_per_label=gen_per_label,
        per_timestep_loss=per_timestep_loss,
        backbone_kwargs=backbone_kwargs,
    )


__all__ = [
    "LowdimDBPPolicy",
    "ImageDBPPolicy",
    "PointCloudDBPPolicy",
    "DBPLowdimPolicy",
    "DBPHybridImagePolicy",
    "build_dbp_lowdim_policy",
]

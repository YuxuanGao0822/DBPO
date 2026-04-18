"""Point-cloud observation encoders for DBP policies."""

from __future__ import annotations

import torch.nn as nn

from dbpo.model.vision.pointnet_extractor import PointCloudEncoder


class PointCloudObsEncoder(nn.Module):
    """Thin wrapper over the DBPO point-cloud encoder."""

    def __init__(
        self,
        state_dim: int,
        point_channels: int = 3,
        point_feature_dim: int = 256,
        state_feature_dim: int = 64,
        point_hidden_dims: tuple[int, ...] = (64, 128, 256),
        state_hidden_dims: tuple[int, ...] = (64,),
    ):
        super().__init__()
        observation_space = {
            "agent_pos": (state_dim,),
            "point_cloud": (1024, point_channels),
        }
        self.encoder = PointCloudEncoder(
            observation_space=observation_space,
            out_channel=point_feature_dim,
            state_mlp_size=tuple(state_hidden_dims) + (state_feature_dim,),
            pointcloud_encoder_cfg={
                "out_channels": point_feature_dim,
                "use_layernorm": False,
                "final_norm": "none",
                "use_projection": True,
            },
            use_pc_color=(point_channels == 6),
            pointnet_type="mlp",
        )
        self.output_dim = int(self.encoder.output_shape())

    def forward(self, obs_dict: dict):
        return self.encoder(obs_dict)

import torch
import torch.nn as nn

from typing import Dict, List, Type


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        **kwargs,
    ):
        super().__init__()
        block_channel = [64, 128, 256, 512]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        return self.final_projection(x)


class PointNetEncoderXYZ(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"PointNetEncoderXYZ only supports 3 channels, got {in_channels}")
        block_channel = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        if not use_projection:
            self.final_projection = nn.Identity()

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        return self.final_projection(x)


class PointCloudEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="mlp",
    ):
        super().__init__()
        del img_crop_shape
        self.imagination_key = "imagin_robot"
        self.state_key = "agent_pos"
        self.point_cloud_key = "point_cloud"
        self.n_output_channels = out_channel

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.state_shape = observation_space[self.state_key]
        if pointcloud_encoder_cfg is None:
            pointcloud_encoder_cfg = {}
        else:
            pointcloud_encoder_cfg = dict(pointcloud_encoder_cfg)

        self.use_pc_color = use_pc_color
        self.point_in_channels = 6 if use_pc_color else 3
        if pointnet_type != "mlp":
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        if use_pc_color:
            pointcloud_encoder_cfg["in_channels"] = 6
            self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
        else:
            pointcloud_encoder_cfg["in_channels"] = 3
            self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)

        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        output_dim = state_mlp_size[-1]
        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
        )

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        # Keep stage-1 point-cloud pretraining tolerant to dataset/config drift.
        # Adroit/MetaWorld-generated zarr currently stores XYZRGB point clouds. Some task
        # configs or stale checkpoints may still instantiate a 3-channel
        # PointNet. In that case, use XYZ only. Conversely, if a 6-channel
        # extractor sees XYZ-only input, pad zeros for RGB.
        if points.shape[-1] != self.point_in_channels:
            if self.point_in_channels == 3 and points.shape[-1] >= 3:
                points = points[..., :3]
            elif self.point_in_channels == 6 and points.shape[-1] == 3:
                pad = torch.zeros(
                    *points.shape[:-1],
                    3,
                    device=points.device,
                    dtype=points.dtype,
                )
                points = torch.cat([points, pad], dim=-1)
            else:
                raise ValueError(
                    f"Unsupported point-cloud channel mismatch: "
                    f"expected {self.point_in_channels}, got {points.shape[-1]}"
                )
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., : points.shape[-1]]
            points = torch.concat([points, img_points], dim=1)
        pn_feat = self.extractor(points)
        state_feat = self.state_mlp(observations[self.state_key])
        return torch.cat([pn_feat, state_feat], dim=-1)

    def output_shape(self):
        return self.n_output_channels

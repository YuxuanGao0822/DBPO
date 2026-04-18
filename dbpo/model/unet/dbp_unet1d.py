"""
DBPUNet1D — the native one-step generative backbone network for DBPO.

This is the central backbone implementation.
Consistent with the Drift-Based Policy (DBP) methodology, this network maps
(noise, global_cond) directly to an action sequence in a single forward pass.
No timestep or diffusion schedule parameters are used within the network.
"""
from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from dbpo.model.unet.unet_components import Downsample1d, Upsample1d, Conv1dBlock

logger = logging.getLogger(__name__)


class ResidualBlock1D(nn.Module):
    """
    Conditional residual block with FiLM modulation.

    Applies two Conv1d layers with GroupNorm and Mish activation, conditioned
    on a global feature vector via FiLM (scale+bias or additive bias).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        
        # In pure DBP, if cond_dim == 0, we can't project. We assume cond_dim > 0.
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x    : [B, in_channels, horizon]
        cond : [B, cond_dim]
        out  : [B, out_channels, horizon]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class DBPUNet1D(nn.Module):
    """
    Native pure one-step backbone for Drift-Based Policies.

    Architecture
    ------------
    Encoder-decoder UNet with FiLM-conditioned residual blocks. The global
    conditioning vector (flattened observation history) is injected at every
    residual block via FiLM modulation. No timestep parameters exist.

    Parameters
    ----------
    input_dim : int
        Action dimensionality (D_a).
    local_cond_dim : int or None
        Per-timestep local conditioning dimension (set None for global-only).
    global_cond_dim : int or None
        Global conditioning vector dimension (obs_dim * n_obs_steps).
    down_dims : list[int]
        Channel widths for each encoder stage.
    kernel_size : int
        Convolution kernel size.
    n_groups : int
        GroupNorm group count.
    cond_predict_scale : bool
        If True, use FiLM scale+bias; otherwise additive bias only.

    Attributes
    ----------
    cond_enc_dim : int
        Total conditioning dimension (global_cond_dim).
        Exposed for the DBPO PPO actor log-std head sizing.
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim=None,
        global_cond_dim=None,
        down_dims=None,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        assert global_cond_dim is not None and global_cond_dim > 0, "DBPUNet1D assumes global condition vectors exist."
        
        if down_dims is None:
            down_dims = [256, 512, 1024]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        cond_dim = global_cond_dim
        self.cond_enc_dim = cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            local_cond_encoder = nn.ModuleList([
                ResidualBlock1D(local_cond_dim, dim_out, cond_dim=cond_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale),
                ResidualBlock1D(local_cond_dim, dim_out, cond_dim=cond_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale),
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale),
            ResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale),
                ResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale),
                ResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "DBPUNet1D — parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        sample: torch.Tensor,
        global_cond: torch.Tensor,
        local_cond: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Pure one-step generation forward pass.

        Parameters
        ----------
        sample : Tensor [B, T, input_dim]
            Input action sequence (Gaussian noise z).
        global_cond : Tensor [B, global_cond_dim]
            Flattened observation conditioning vector.
        local_cond : Tensor [B, T, local_cond_dim] or None

        Returns
        -------
        Tensor [B, T, input_dim]
            Predicted action sequence.
        """
        x = einops.rearrange(sample, "b h t -> b t h")
        
        # In DBP, there is no step injection; global_cond directly acts as the conditioning vector
        global_feature = global_cond

        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        last_up_idx = len(self.up_modules) - 1
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == last_up_idx and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x

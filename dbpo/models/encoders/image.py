"""Image observation encoders for DBP policies."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from dbpo.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class ResNetImageBackbone(nn.Module):
    """ResNet backbone that returns a flat feature vector."""

    def __init__(self, output_dim: int = 128, pretrained: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        backbone = torchvision.models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_features, output_dim)
        self.output_dim = output_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)
        return self.proj(feats)


class MultiImageResNetEncoder(nn.Module):
    """RGB + low-dim encoder backed by ResNet18 and `MultiImageObsEncoder`."""

    def __init__(
        self,
        shape_meta: dict,
        rgb_feature_dim: int = 128,
        crop_shape: tuple[int, int] | None = (84, 84),
        use_group_norm: bool = True,
        eval_fixed_crop: bool = True,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        rgb_model = ResNetImageBackbone(
            output_dim=rgb_feature_dim,
            pretrained=pretrained_backbone,
        )
        self.encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=crop_shape,
            random_crop=not eval_fixed_crop,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
        )
        self.output_dim = int(self.encoder.output_shape()[0])

    def forward(self, obs_dict: dict) -> torch.Tensor:
        return self.encoder(obs_dict)

    def output_shape(self):
        return self.encoder.output_shape()

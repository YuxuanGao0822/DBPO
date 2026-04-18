"""
MetaWorld manipulation dataset.

I/O: returns {"obs": {"point_cloud": [T,N,3], "agent_pos": [T,D]}, "action": [T,Da]}
Compatible with DBP point-cloud policies, or low-dimensional DBP policies when using agent_pos only.
"""
from typing import Dict
import torch
import numpy as np
import copy
from dbpo.utils.pytorch_util import dict_apply
from dbpo.dataset.replay_buffer import ReplayBuffer
from dbpo.dataset.sampler import SequenceSampler, get_val_mask, downsample_mask
from dbpo.model.common.normalizer import LinearNormalizer
from dbpo.dataset.base_dataset import BaseLowdimDataset


class MetaworldDataset(BaseLowdimDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["state", "action", "point_cloud"]
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:, ].astype(np.float32)
        point_cloud = sample["point_cloud"][:, ].astype(np.float32)
        return {
            "obs": {"point_cloud": point_cloud, "agent_pos": agent_pos},
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

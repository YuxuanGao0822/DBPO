"""
BlockPush low-dimensional dataset.

I/O: returns {"obs": [T, Do], "action": [T, Da]}
Compatible with low-dimensional DBP policies.
"""
from typing import Dict
import logging
import os
import torch
import numpy as np
import copy
import zarr
from dbpo.utils.pytorch_util import dict_apply
from dbpo.dataset.replay_buffer import ReplayBuffer
from dbpo.dataset.sampler import SequenceSampler, get_val_mask, downsample_mask
from dbpo.model.common.normalizer import LinearNormalizer
from dbpo.dataset.base_dataset import BaseLowdimDataset

log = logging.getLogger(__name__)


class BlockPushLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key="obs",
        action_key="action",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        abs_action=True,
    ):
        super().__init__()
        self.replay_buffer = self._load_replay_buffer(
            zarr_path=zarr_path, obs_key=obs_key, action_key=action_key
        )
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.abs_action = abs_action

    @staticmethod
    def _resolve_zarr_path(zarr_path: str) -> str:
        expanded = os.path.expanduser(zarr_path)
        if os.path.exists(expanded):
            return expanded
        if "/blockpush/" in expanded:
            legacy_path = expanded.replace("/blockpush/", "/block_pushing/")
            if os.path.exists(legacy_path):
                log.info("Using legacy block_pushing dataset layout: %s", legacy_path)
                return legacy_path
        return expanded

    @classmethod
    def _load_replay_buffer(cls, zarr_path: str, obs_key: str, action_key: str) -> ReplayBuffer:
        resolved_zarr_path = cls._resolve_zarr_path(zarr_path)
        try:
            return ReplayBuffer.copy_from_path(
                resolved_zarr_path, keys=[obs_key, action_key]
            )
        except zarr.errors.PathNotFoundError:
            if "/blockpush/" not in resolved_zarr_path:
                raise
            legacy_path = resolved_zarr_path.replace("/blockpush/", "/block_pushing/")
            if legacy_path == resolved_zarr_path:
                raise
            log.info("Using legacy block_pushing dataset layout: %s", legacy_path)
            return ReplayBuffer.copy_from_path(
                legacy_path, keys=[obs_key, action_key]
            )

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
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key]
        return {"obs": obs, "action": sample[self.action_key]}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

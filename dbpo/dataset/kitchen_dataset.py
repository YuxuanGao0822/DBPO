"""
FrankaKitchen (D4RL) low-dimensional dataset.

I/O: returns {"obs": [T, 60], "action": [T, 9]}
Compatible with low-dimensional DBP policies (obs_dim=60, action_dim=9).
"""
from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from dbpo.utils.pytorch_util import dict_apply
from dbpo.dataset.replay_buffer import ReplayBuffer
from dbpo.dataset.sampler import SequenceSampler, get_val_mask
from dbpo.model.common.normalizer import LinearNormalizer
from dbpo.dataset.base_dataset import BaseLowdimDataset


class KitchenLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
    ):
        super().__init__()
        data_directory = pathlib.Path(dataset_dir)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in range(len(masks)):
            eps_len = int(masks[i].sum())
            self.replay_buffer.add_episode({
                "obs": observations[i, :eps_len].astype(np.float32),
                "action": actions[i, :eps_len].astype(np.float32),
            })

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
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
        data = {"obs": self.replay_buffer["obs"], "action": self.replay_buffer["action"]}
        if "range_eps" not in kwargs:
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(sample, torch.from_numpy)

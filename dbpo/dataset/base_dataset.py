"""
Abstract base dataset classes.

I/O contract:
  - Low-dim datasets return: {"obs": Tensor[T, Do], "action": Tensor[T, Da]}
  - Image datasets return:   {"obs": {"key": Tensor[T, ...]}, "action": Tensor[T, Da]}
"""
from typing import Dict
import torch
import torch.nn
from dbpo.model.common.normalizer import LinearNormalizer


class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseLowdimDataset":
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns: {"obs": [T, Do], "action": [T, Da]}"""
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseImageDataset":
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns: {"obs": {key: [T, ...]}, "action": [T, Da]}"""
        raise NotImplementedError()

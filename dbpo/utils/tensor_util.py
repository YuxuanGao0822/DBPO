"""
Tensor utility functions used by the image observation encoder and dataset helpers.
"""
import collections
import numpy as np
import torch


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    assert begin_axis <= end_axis
    assert begin_axis >= 0
    assert end_axis < len(x.shape)
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return reshape_dimensions_single(x, begin_axis, end_axis, target_dims)
    raise TypeError(f"Unsupported type: {type(x)}")


def join_dimensions(x, begin_axis, end_axis):
    if isinstance(x, torch.Tensor):
        return reshape_dimensions_single(x, begin_axis, end_axis, [-1])
    raise TypeError(f"Unsupported type: {type(x)}")


def flatten(x, begin_axis=1):
    if isinstance(x, torch.Tensor):
        fixed_size = x.size()[:begin_axis]
        return x.reshape(*list(fixed_size) + [-1])
    raise TypeError(f"Unsupported type: {type(x)}")


def unsqueeze_expand_at(x, size, dim):
    x = x.unsqueeze(dim)
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)

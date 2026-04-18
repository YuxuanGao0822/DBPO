"""
Normalizer factory helpers for datasets.
"""
import numpy as np
from dbpo.model.common.normalizer import SingleFieldLinearNormalizer
from dbpo.utils.pytorch_util import dict_apply


def _dict_apply_split(x, split_func):
    """Apply split_func to each value in dict, then re-group by split key."""
    import collections
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def _dict_apply_reduce(x_list, reduce_func):
    result = {}
    for key in x_list[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x_list])
    return result


def array_to_stats(arr: np.ndarray) -> dict:
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)


def get_image_range_normalizer():
    """Normalizer that maps [0, 1] images to [-1, 1]."""
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)


def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = _dict_apply_split(stat, lambda x: {"pos": x[..., :3], "other": x[..., 3:]})

    def _pos_param(s, output_max=1, output_min=-1, range_eps=1e-7):
        input_range = s["max"] - s["min"]
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * s["min"]
        offset[ignore_dim] = (output_max + output_min) / 2 - s["min"][ignore_dim]
        return {"scale": scale, "offset": offset}, s

    def _other_param(s):
        example = s["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {"max": np.ones_like(example), "min": np.full_like(example, -1),
                "mean": np.zeros_like(example), "std": np.ones_like(example)}
        return {"scale": scale, "offset": offset}, info

    pos_param, pos_info = _pos_param(result["pos"])
    other_param, other_info = _other_param(result["other"])
    param = _dict_apply_reduce([pos_param, other_param], lambda x: np.concatenate(x, axis=-1))
    info = _dict_apply_reduce([pos_info, other_info], lambda x: np.concatenate(x, axis=-1))
    return SingleFieldLinearNormalizer.create_manual(scale=param["scale"], offset=param["offset"], input_stats_dict=info)


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    Da = stat["max"].shape[-1]
    Dah = Da // 2
    result = _dict_apply_split(stat, lambda x: {
        "pos0": x[..., :3], "other0": x[..., 3:Dah],
        "pos1": x[..., Dah:Dah + 3], "other1": x[..., Dah + 3:],
    })

    def _pos_param(s, output_max=1, output_min=-1, range_eps=1e-7):
        input_range = s["max"] - s["min"]
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * s["min"]
        offset[ignore_dim] = (output_max + output_min) / 2 - s["min"][ignore_dim]
        return {"scale": scale, "offset": offset}, s

    def _other_param(s):
        example = s["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {"max": np.ones_like(example), "min": np.full_like(example, -1),
                "mean": np.zeros_like(example), "std": np.ones_like(example)}
        return {"scale": scale, "offset": offset}, info

    p0, i0 = _pos_param(result["pos0"])
    p1, i1 = _pos_param(result["pos1"])
    o0, oi0 = _other_param(result["other0"])
    o1, oi1 = _other_param(result["other1"])
    param = _dict_apply_reduce([p0, o0, p1, o1], lambda x: np.concatenate(x, axis=-1))
    info = _dict_apply_reduce([i0, oi0, i1, oi1], lambda x: np.concatenate(x, axis=-1))
    return SingleFieldLinearNormalizer.create_manual(scale=param["scale"], offset=param["offset"], input_stats_dict=info)

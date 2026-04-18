"""
Robomimic image replay dataset (lift/can/square/transport with RGB observations).

I/O: returns {"obs": {key: [T, ...]}, "action": [T, Da]}
Compatible with image-conditioned DBP policies.
"""
from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import concurrent.futures
import multiprocessing
from threadpoolctl import threadpool_limits
from filelock import FileLock
from dbpo.utils.pytorch_util import dict_apply
from dbpo.dataset.base_dataset import BaseImageDataset
from dbpo.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from dbpo.dataset.replay_buffer import ReplayBuffer
from dbpo.dataset.sampler import SequenceSampler, get_val_mask
from dbpo.dataset.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

try:
    from imagecodecs.numcodecs import register_codecs, Jpeg2k
    register_codecs()
    _HAS_JPEG2K = True
except ImportError:
    _HAS_JPEG2K = False


class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        abs_action=False,
        rotation_rep="rotation_6d",
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
    ):
        from dbpo.utils.rotation_transformer import RotationTransformer
        rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                        )
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

        rgb_keys, lowdim_keys = [], []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

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

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            if self.use_legacy_normalizer:
                this_normalizer = _normalizer_from_stat(stat)
        else:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith("pos"):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat") or key.endswith("qpos"):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported lowdim key: {key}")
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)
        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]
        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }


def _normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer,
                                  n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys, lowdim_keys = [], []
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        demos = file["data"]
        episode_ends = []
        prev_end = 0
        for i in range(len(demos)):
            episode_length = demos[f"demo_{i}"]["actions"].shape[0]
            prev_end += episode_length
            episode_ends.append(prev_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)

        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
            data_key = "obs/" + key if key != "action" else "actions"
            this_data = np.concatenate(
                [demos[f"demo_{i}"][data_key][:].astype(np.float32) for i in range(len(demos))], axis=0
            )
            if key == "action" and abs_action:
                is_dual_arm = this_data.shape[-1] == 14
                if is_dual_arm:
                    this_data = this_data.reshape(-1, 2, 7)
                pos = this_data[..., :3]
                rot = rotation_transformer.forward(this_data[..., 3:6])
                gripper = this_data[..., 6:]
                this_data = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
                if is_dual_arm:
                    this_data = this_data.reshape(-1, 20)
            data_group.array(name=key, data=this_data, shape=this_data.shape,
                             chunks=this_data.shape, compressor=None, dtype=this_data.dtype)

        if _HAS_JPEG2K:
            def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
                try:
                    zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                    _ = zarr_arr[zarr_idx]
                    return True
                except Exception:
                    return False

            with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = set()
                    for key in rgb_keys:
                        shape = tuple(shape_meta["obs"][key]["shape"])
                        c, h, w = shape
                        img_arr = data_group.require_dataset(
                            name=key, shape=(n_steps, h, w, c), chunks=(1, h, w, c),
                            compressor=Jpeg2k(level=50), dtype=np.uint8,
                        )
                        for ep_idx in range(len(demos)):
                            hdf5_arr = demos[f"demo_{ep_idx}"]["obs"][key]
                            for hdf5_idx in range(hdf5_arr.shape[0]):
                                if len(futures) >= max_inflight_tasks:
                                    completed, futures = concurrent.futures.wait(
                                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                                    )
                                    for f in completed:
                                        if not f.result():
                                            raise RuntimeError("Failed to encode image!")
                                    pbar.update(len(completed))
                                zarr_idx = episode_starts[ep_idx] + hdf5_idx
                                futures.add(executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                    completed, futures = concurrent.futures.wait(futures)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                    pbar.update(len(completed))

    return ReplayBuffer(root)

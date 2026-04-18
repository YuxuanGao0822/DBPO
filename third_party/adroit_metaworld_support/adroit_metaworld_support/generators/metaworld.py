"""Generate MetaWorld point-cloud demonstrations using DBPO policies."""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import zarr
from metaworld.policies import *  # noqa: F403
from adroit_metaworld_runtime.env import MetaWorldEnv
from termcolor import cprint


def load_mw_policy(task_name: str):
    if task_name == "peg-insert-side":
        return SawyerPegInsertionSideV2Policy()  # noqa: F405
    task_tokens = [token.capitalize() for token in task_name.split("-")]
    policy_name = "Sawyer" + "".join(task_tokens) + "V2Policy"
    return eval(policy_name)()  # noqa: S307


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get("DBPO_DATA_DIR", "data"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = os.path.join(args.root_dir, f"metaworld_{args.env_name}_expert.zarr")
    if os.path.exists(save_dir):
        cprint(f"Overwriting existing dataset at {save_dir}", "red")
        os.system(f'rm -rf "{save_dir}"')
    os.makedirs(save_dir, exist_ok=True)

    env = MetaWorldEnv(args.env_name, device="cuda:0", use_point_crop=True)
    policy = load_mw_policy(args.env_name)

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    full_state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    episode_idx = 0
    while episode_idx < args.num_episodes:
        raw_state = env.reset()["full_state"]
        obs_dict = env.get_visual_obs()
        done = False
        ep_reward = 0.0
        ep_success = False
        ep_success_times = 0

        img_arrays_sub = []
        point_cloud_arrays_sub = []
        depth_arrays_sub = []
        state_arrays_sub = []
        full_state_arrays_sub = []
        action_arrays_sub = []
        total_count_sub = 0

        while not done:
            total_count_sub += 1
            img_arrays_sub.append(obs_dict["image"])
            point_cloud_arrays_sub.append(obs_dict["point_cloud"])
            depth_arrays_sub.append(obs_dict["depth"])
            state_arrays_sub.append(obs_dict["agent_pos"])
            full_state_arrays_sub.append(raw_state)

            action = policy.get_action(raw_state)
            action_arrays_sub.append(action)
            obs_dict, reward, done, info = env.step(action)
            raw_state = obs_dict["full_state"]
            ep_reward += reward
            ep_success = ep_success or info["success"]
            ep_success_times += info["success"]

        if not ep_success or ep_success_times < 5:
            cprint(
                f"Episode {episode_idx} failed with reward {ep_reward} and success times {ep_success_times}",
                "red",
            )
            continue

        total_count += total_count_sub
        episode_ends_arrays.append(copy.deepcopy(total_count))
        img_arrays.extend(copy.deepcopy(img_arrays_sub))
        point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
        depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
        state_arrays.extend(copy.deepcopy(state_arrays_sub))
        action_arrays.extend(copy.deepcopy(action_arrays_sub))
        full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
        cprint(
            f"Episode: {episode_idx}, Reward: {ep_reward}, Success Times: {ep_success_times}",
            "green",
        )
        episode_idx += 1

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3:
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    state_arrays = np.stack(state_arrays, axis=0)
    full_state_arrays = np.stack(full_state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "img",
        data=img_arrays,
        chunks=(100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3]),
        dtype="uint8",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "full_state",
        data=full_state_arrays,
        chunks=(100, full_state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud_arrays,
        chunks=(100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "depth",
        data=depth_arrays,
        chunks=(100, depth_arrays.shape[1], depth_arrays.shape[2]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=(100, action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    cprint(f"Saved zarr file to {save_dir}", "green")


if __name__ == "__main__":
    main()

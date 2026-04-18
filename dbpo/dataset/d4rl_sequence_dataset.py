"""
D4RL / Gym stitched sequence dataset for DBP pre-training and DBPO fine-tuning.

I/O: returns Batch(actions=[T, Da], conditions={"state": [cond_steps, Do]})
Compatible with low-dimensional DBP policies via the training agent's collation logic.

Supports both .npz and .pkl dataset files.
"""
from collections import namedtuple
import numpy as np
import torch
import logging
import pickle
import random
import copy
from tqdm import tqdm
from dbpo.model.common.normalizer import LinearNormalizer

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple("Transition", "actions conditions rewards dones reward_to_gos")


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions from .npz or .pkl files.

    Each sample is a Batch(actions, conditions) where:
      - actions: [horizon_steps, action_dim]
      - conditions["state"]: [cond_steps, obs_dim]  (most recent at the end)

    This is the primary dataset for D4RL gym tasks (hopper, walker2d, ant, kitchen-*)
    and Robomimic lowdim tasks.

    Dataset storage stays on CPU so DataLoader workers can slice safely.
    Minibatches are moved to GPU later by the workspace.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon_steps: int = 64,
        cond_steps: int = 1,
        img_cond_steps: int = 1,
        max_n_episodes: int = -1,
        use_img: bool = False,
        device: str = "cuda:0",
    ):
        assert img_cond_steps <= cond_steps, "img_cond_steps must be <= cond_steps"

        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        if max_n_episodes == -1:
            max_n_episodes = len(dataset["traj_lengths"])
            log.info(f"max_n_episodes=-1, using all {max_n_episodes} episodes")

        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = int(np.sum(traj_lengths))
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        self.states = torch.from_numpy(dataset["states"][:total_num_steps]).float()
        self.actions = torch.from_numpy(dataset["actions"][:total_num_steps]).float()

        log.info(f"Loaded {dataset_path}: {len(traj_lengths)} episodes, {total_num_steps} steps")
        log.info(f"states: {self.states.shape}, actions: {self.actions.shape}")

        if use_img:
            self.images = torch.from_numpy(dataset["images"][:total_num_steps])
            log.info(f"images: {self.images.shape}")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "obs": self.states,
            "action": self.actions,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return self.actions

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps

        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]

        # Stack obs history; most recent is at the end
        states = torch.stack(
            [states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))]
        )
        conditions = {"state": states}

        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [images[max(num_before_start - t, 0)] for t in reversed(range(self.img_cond_steps))]
            )
            conditions["rgb"] = images

        return {"action": actions, "obs": conditions["state"]}

    def make_indices(self, traj_lengths, horizon_steps):
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [(i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset with rewards and dones for Q-learning.
    Returns Transition(actions, conditions, rewards, dones).
    """

    def __init__(
        self,
        dataset_path: str,
        max_n_episodes: int = 10000,
        discount_factor: float = 1.0,
        device: str = "cuda:0",
        get_mc_return: bool = False,
        **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = int(np.sum(traj_lengths))
        self.discount_factor = discount_factor

        self.rewards = torch.from_numpy(dataset["rewards"][:total_num_steps]).float()
        self.dones = torch.from_numpy(dataset["terminals"][:total_num_steps]).float()

        super().__init__(dataset_path=dataset_path, max_n_episodes=max_n_episodes, device=device, **kwargs)

        self.get_mc_return = get_mc_return
        if get_mc_return:
            self.reward_to_go = torch.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            for i, traj_length in tqdm(enumerate(cumulative_traj_length), desc="Computing reward-to-go"):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = torch.zeros_like(traj_rewards)
                prev_return = 0
                for t in range(len(traj_rewards)):
                    returns[-t - 1] = traj_rewards[-t - 1] + self.discount_factor * prev_return
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length

    def make_indices(self, traj_lengths, horizon_steps):
        """Skip last step of truncated episodes."""
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:
                max_start -= 1
                num_skip += 1
            indices += [(i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)]
            cur_traj_index += traj_length
        log.info(f"Skipped {num_skip} truncated episode endings")
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps

        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start : (start + 1)]
        dones = self.dones[start : (start + 1)]

        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps) : start + 1 + self.horizon_steps
            ]
        else:
            next_states = torch.zeros_like(states)

        states = torch.stack(
            [states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))]
        )
        next_states = torch.stack(
            [next_states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))]
        )
        conditions = {"state": states, "next_state": next_states}

        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [images[max(num_before_start - t, 0)] for t in reversed(range(self.img_cond_steps))]
            )
            conditions["rgb"] = images

        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start : (start + 1)]
            return TransitionWithReturn(actions, conditions, rewards, dones, reward_to_gos)
        return Transition(actions, conditions, rewards, dones)

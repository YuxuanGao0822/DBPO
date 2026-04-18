from __future__ import annotations

import gym
import numpy as np
import metaworld
import random

from gym import spaces
from termcolor import cprint

from adroit_metaworld_runtime.gym_util.mujoco_point_cloud import PointCloudGenerator
from adroit_metaworld_runtime.gym_util.mjpc_wrapper import point_cloud_sampling


TASK_BOUNDS = {
    "default": [-0.5, -1.5, -0.795, 1, -0.4, 100],
}


class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        task_name: str,
        device: str = "cuda",
        use_point_crop: bool = True,
        num_points: int = 1024,
    ):
        super().__init__()
        del device

        self.task_name = task_name
        self.env, self._api_mode = self._make_env(task_name)
        self._base_env = self._resolve_base_env(self.env)
        self._sim = self._resolve_sim(self._base_env)
        if self._sim is None:
            raise AttributeError(
                f"Failed to resolve a MetaWorld base env with MuJoCo sim/model for task {task_name}."
            )

        if hasattr(self._base_env, "_freeze_rand_vec"):
            self._base_env._freeze_rand_vec = False
        self._sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        self._sim.model.vis.map.znear = 0.1
        self._sim.model.vis.map.zfar = 1.5

        self.device_id = 0
        self.image_size = 128
        self.pc_generator = PointCloudGenerator(
            sim=self._sim, cam_names=["corner2"], img_size=self.image_size
        )
        self.use_point_crop = use_point_crop
        cprint(f"[MetaWorldEnv] use_point_crop: {self.use_point_crop}", "cyan")
        self.num_points = num_points

        x_angle = 61.4
        y_angle = -7
        self.pc_transform = np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
                [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))],
            ]
        ) @ np.array(
            [
                [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
                [0, 1, 0],
                [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))],
            ]
        )

        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        bounds_key = self._normalize_task_key(task_name)
        x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUNDS.get(
            bounds_key, TASK_BOUNDS["default"]
        )
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        self.episode_length = self._max_episode_steps = 200
        self.action_space = self._base_env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.image_size, self.image_size),
                    dtype=np.float32,
                ),
                "depth": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_size, self.image_size),
                    dtype=np.float32,
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_sensor_dim,),
                    dtype=np.float32,
                ),
                "point_cloud": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_points, 3),
                    dtype=np.float32,
                ),
                "full_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(39,),
                    dtype=np.float32,
                ),
            }
        )

    def _normalize_task_key(self, task_name: str) -> str:
        normalized = task_name
        for suffix in ("-v2-goal-observable", "-v2", "-v3", "-goal-observable"):
            normalized = normalized.removesuffix(suffix)
        return normalized

    def _normalize_task_name(self, task_name: str, version: str) -> str:
        task_name = self._normalize_task_key(task_name)
        if version == "v2":
            return f"{task_name}-v2-goal-observable"
        if version == "v3":
            return f"{task_name}-v3"
        raise ValueError(f"Unsupported MetaWorld version: {version}")

    def _resolve_base_env(self, env):
        visited = set()
        queue = [env]
        while queue:
            current = queue.pop(0)
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)
            if hasattr(current, "sim") or (hasattr(current, "model") and hasattr(current, "render")):
                return current
            for attr in ("unwrapped", "env", "_env"):
                if hasattr(current, attr):
                    child = getattr(current, attr)
                    if child is not None and id(child) not in visited:
                        queue.append(child)
            if hasattr(current, "__dict__"):
                for child in current.__dict__.values():
                    if child is None:
                        continue
                    if isinstance(child, (str, bytes, int, float, bool, tuple, list, dict, set)):
                        continue
                    if id(child) not in visited:
                        queue.append(child)
        return env

    def _resolve_sim(self, base_env):
        if hasattr(base_env, "sim"):
            return base_env.sim
        if hasattr(base_env, "model") and hasattr(base_env, "render"):
            class _SimAdapter:
                def __init__(self, env):
                    self._env = env
                    self.model = env.model

                def render(self, width, height, camera_name=None, depth=False, device_id=0):
                    try:
                        return self._env.render(
                            width=width,
                            height=height,
                            camera_name=camera_name,
                            depth=depth,
                            device_id=device_id,
                        )
                    except TypeError:
                        try:
                            return self._env.render()
                        except TypeError as exc:
                            raise RuntimeError(
                                "Failed to call MetaWorld render through the compatibility adapter."
                            ) from exc

            return _SimAdapter(base_env)
        return None

    def _make_mt1_env(self, task_name: str):
        if not hasattr(metaworld, "MT1"):
            return None
        env_name = self._normalize_task_name(task_name, "v3")
        try:
            benchmark = metaworld.MT1(env_name)
            env_cls = benchmark.train_classes[env_name]
            env = env_cls()
            matching_tasks = [task for task in benchmark.train_tasks if task.env_name == env_name]
            if matching_tasks and hasattr(env, "set_task"):
                env.set_task(random.choice(matching_tasks))
            return env
        except Exception:  # noqa: BLE001
            return None

    def _make_env(self, task_name: str):
        v2_name = self._normalize_task_name(task_name, "v2")
        env_dict = getattr(metaworld.envs, "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE", None)
        if env_dict is not None and v2_name in env_dict:
            return env_dict[v2_name](), "legacy_v2"

        mt1_env = self._make_mt1_env(task_name)
        if mt1_env is not None:
            return mt1_env, "mt1_v3"

        try:
            import gymnasium
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "MetaWorld v3 requires gymnasium. Install metaworld with its gymnasium dependency."
            ) from exc

        v3_name = self._normalize_task_name(task_name, "v3")
        env = gymnasium.make(
            "Meta-World/MT1",
            env_name=v3_name,
            render_mode=None,
        )
        return env, "gymnasium_v3"

    def get_robot_state(self):
        eef_pos = self._base_env.get_endeff_pos()
        finger_right = self._base_env._get_site_pos("rightEndEffector")
        finger_left = self._base_env._get_site_pos("leftEndEffector")
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        return self._sim.render(
            width=512,
            height=512,
            camera_name="corner2",
            device_id=self.device_id,
        )

    def render_high_res(self, resolution=1024):
        return self._sim.render(
            width=resolution,
            height=resolution,
            camera_name="corner2",
            device_id=self.device_id,
        )

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(
            device_id=self.device_id
        )

        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, "fps")
        depth = depth[::-1]
        return point_cloud, depth

    def _build_obs(self, raw_state):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        return {
            "image": obs_pixels,
            "depth": depth,
            "agent_pos": robot_state,
            "point_cloud": point_cloud,
            "full_state": raw_state,
        }

    def get_visual_obs(self):
        obs = self._build_obs(np.zeros((39,), dtype=np.float32))
        obs.pop("full_state")
        return obs

    def step(self, action: np.ndarray):
        step_result = self.env.step(action)
        self.cur_step += 1

        if len(step_result) == 5:
            raw_state, reward, terminated, truncated, env_info = step_result
            done = terminated or truncated
        else:
            raw_state, reward, done, env_info = step_result

        obs_dict = self._build_obs(raw_state)
        done = done or self.cur_step >= self.episode_length
        return obs_dict, reward, done, env_info

    def reset(self):
        self.cur_step = 0
        if self._api_mode == "legacy_v2":
            self._base_env.reset()
            self._base_env.reset_model()
            raw_obs = self._base_env.reset()
        elif self._api_mode == "mt1_v3":
            raw_obs = self.env.reset()
        else:
            raw_obs, _ = self.env.reset()
        return self._build_obs(raw_obs)

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def set_seed(self, seed=None):
        self.seed(seed)

    def render(self, mode="rgb_array"):
        del mode
        return self.get_rgb()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

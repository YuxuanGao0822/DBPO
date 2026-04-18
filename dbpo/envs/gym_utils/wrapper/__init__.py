"""Wrapper registry for the public stage-2 gym finetune path."""

from dbpo.envs.gym_utils.wrapper.multi_step import MultiStep
from dbpo.envs.gym_utils.wrapper.mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper

wrapper_dict = {
    "multi_step": MultiStep,
    "mujoco_locomotion_lowdim": MujocoLocomotionLowdimWrapper,
}

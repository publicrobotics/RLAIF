# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# this is just for the min height
def object_is_lifted_push_task(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    height = (object.data.root_pos_w[:, 2] - env.event_manager.get_term_cfg("reset_objects").func.init_object_state[:, 2])
    # print("HEIGHT", height)
    return torch.where(height > minimal_height, 1.0, 0.0)


def object_is_lifted_lift_task(
        env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    height = (object.data.root_pos_w[:, 2] - env.event_manager.get_term_cfg("reset_objects").func.init_object_state[:, 2])
    # print("HIEGHT", height)
    return torch.where(height > minimal_height, 1.0, 0.0)


# This method will use height in intervals
def object_height_shaped(
    env: ManagerBasedRLEnv,
    goal_height: float,
    min_height: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    kernel: str = "tanh",
    std: float = 0.05
) -> torch.Tensor:
    """
    Returns a value in [0, 1] that increases as the object’s Z approaches
    `goal_height`.  At/above the goal, the reward saturates at 1.

    • "linear":   0     …→ 1  between min_height and goal_height
    • "tanh"  :   smooth sigmoid‑like curve controlled by `std`
    """
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]

    if kernel == "linear":
        # clamp to [0, 1]
        return torch.clamp((z - min_height) / (goal_height - min_height), 0.0, 1.0)

    elif kernel == "tanh":
        # positive distance still to go (≤0 means we’re at/above the goal)
        d = torch.clamp(goal_height - z, min=0.0)
        return 1.0 - torch.tanh(d / std)

    else:
        raise ValueError(f"Unknown kernel '{kernel}'")


# use this for push

# no need to have the minimal distance be a thing
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


# # do one where you then go from object to object_pose (goal position)
# # lift by some minimal height
# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# torch.where((object.data.root_pos_w[:, 2] - env.event_manager.get_term_cfg("reset_object_state").func.init_object_state[:, 2]) > minimal_height, 1.0, 0.0)
# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     height = (object.data.root_pos_w[:, 2] - env.event_manager.get_term_cfg("reset_object_state").func.init_object_state[:, 2])
#     return torch.where(height > minimal_height, 1.0, 0.0) * (1 - torch.tanh(distance / std))


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_object_cfg: SceneEntityCfg = SceneEntityCfg("goal_object"),
) -> torch.Tensor:
    """Reward the agent for bringing `object` to `goal_object` using a tanh-kernel.
    
    Reward = 1{object lifted above threshold} * (1 - tanh(||p_obj - p_goal|| / std))
    """
    # scene objects
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_object_cfg.name]

    # print("OBJ POS: ", obj.data.root_pos_w[:, :3])
    # print("GOAL POS: ", goal.data.root_pos_w[:, :3])

    # Euclidean distance between object and goal_object (world frame)
    distance = torch.norm(goal.data.root_pos_w[:, :3] - obj.data.root_pos_w[:, :3], dim=1)

    # Lift gate: object must be above its initial height by minimal_height
    init_z = env.event_manager.get_term_cfg("reset_objects").func.init_object_state[:, 2]
    height = obj.data.root_pos_w[:, 2] - init_z

    # Tanh kernel (ensure std > 0)
    pos_reward = 1.0 - torch.tanh(distance / std)
    gate = (height > minimal_height).to(pos_reward.dtype)   # or .float()
    return gate * pos_reward


def object_goal_distance_xy_no_lift(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_object_cfg: SceneEntityCfg = SceneEntityCfg("goal_object"),
) -> torch.Tensor:
    """Planar XY tracking reward between two scene objects.
    No commands, no lift gate — just minimize XY distance between `object` and `goal_object`."""
    # typed handles
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_object_cfg.name]

    # positions in world
    obj_xy = obj.data.root_pos_w[:, :2]   # (N, 2)
    goal_xy = goal.data.root_pos_w[:, :2]   # (N, 2)

    # print("obj_xy: ", obj_xy)
    # print("goal_xy: ", goal_xy)

    object_ee_distance = torch.norm(goal_xy - obj_xy, dim=1)

    # print("object_ee_distance: ", object_ee_distance)
    # print("std: ", std)
    # print("Reward: ", torch.tanh(object_ee_distance / std))

    # print("Rewards: ", (1 - torch.tanh(object_ee_distance / std)))

    return 1 - torch.tanh(object_ee_distance / std)


# just to get the right z distance
def object_goal_distance_z(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_object_cfg: SceneEntityCfg = SceneEntityCfg("goal_object"),
) -> torch.Tensor:
    """Z‑axis tracking reward between two scene objects (no XY term).

    Returns 1 − tanh(|z_obj − z_goal| / std), which is:
        • ≈1 when the two Z positions are equal,
        • smoothly decreases as the vertical distance grows.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_object_cfg.name]

    # extract Z coordinates (N, )
    obj_z = obj.data.root_pos_w[:, 2]
    goal_z = goal.data.root_pos_w[:, 2]

    z_dist = torch.abs(goal_z - obj_z)

    return 1.0 - torch.tanh(z_dist / std)

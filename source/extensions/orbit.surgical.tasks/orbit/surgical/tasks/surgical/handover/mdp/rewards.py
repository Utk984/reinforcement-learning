# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the handover environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def finger_toggle(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    finger_idx = -1   # assumes last joint is gripper
    if not hasattr(robot, "_prev_finger"):
        robot._prev_finger = torch.zeros_like(robot.data.joint_pos[:, finger_idx])
    toggled = (robot.data.joint_pos[:, finger_idx] - robot._prev_finger).abs() > 0.01
    robot._prev_finger.copy_(robot.data.joint_pos[:, finger_idx])
    return toggled.float()    # 1 if toggled, else 0

def needle_ang_vel(env, object_cfg):
    obj = env.scene[object_cfg.name]
    return obj.data.root_ang_vel_w.square().sum(1)   # (N,)

# ---------------------------------------------------------------------
#  rewards.py   (add below your other functions)
# ---------------------------------------------------------------------

import torch
from omni.isaac.lab.managers import SceneEntityCfg

def _update_hold_flags(env, robot_cfg, object_cfg, close_thresh=0.02, lift_thresh=0.06):
    """
    Keep an internal boolean tensor `env._is_holding` of shape (num_envs,)
    that is True while the needle is within `close_thresh` metres of the
    gripper AND the finger is closed AND the needle Z is above table by
    `lift_thresh`.
    """
    # lazy init
    if not hasattr(env, "_is_holding"):
        env._is_holding = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    obj  = env.scene[object_cfg.name]
    ee   = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)

    finger_closed = env.scene[robot_cfg.name].data.joint_pos[:, -1] > 0.5
    lifted        = obj.data.root_pos_w[:, 2] > lift_thresh

    currently_holding = (dist < close_thresh) & finger_closed & lifted

    was_holding = env._is_holding.clone()
    env._is_holding.copy_(currently_holding)

    return was_holding, currently_holding


def hold_bonus(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    reward_per_step: float = 0.02,
):
    """
    +0.02 every sim-step while the needle remains grasped and lifted.
    """
    _, holding = _update_hold_flags(env, robot_cfg, object_cfg)
    return holding.float() * reward_per_step


def drop_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    penalty: float = -1.0,
):
    """
    −1 the first step the robot *was* holding the needle last frame
    but is *not* holding it now.
    """
    was, holding = _update_hold_flags(env, robot_cfg, object_cfg)
    dropped = was & (~holding)
    return dropped.float() * penalty


def needle_height(env, object_cfg):
    return env.scene[object_cfg.name].data.root_pos_w[:, 2:3]

def finger_state(env, asset_cfg):
    return env.scene[asset_cfg.name].data.joint_pos[:, -1:]  # assumes last joint = gripper

def smart_grasp_bonus(env, proximity_threshold, robot_cfg, object_cfg):
    # distance check
    d = torch.norm(
        env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :] -
        env.scene[object_cfg.name].data.root_pos_w, dim=1
    )
    # finger closed?
    closed = finger_state(env, robot_cfg).squeeze(1) > 0.5
    return ((d < proximity_threshold) & closed).float()

def lifted_success(env, min_height, hold_steps):
    obj = env.scene["object"]
    # a small circular buffer in obj.data can track “how long”
    if not hasattr(obj, "_lift_counter"):
        obj._lift_counter = torch.zeros_like(obj.data.root_pos_w[:, 0], dtype=torch.int32)
    lifted = obj.data.root_pos_w[:, 2] > min_height
    obj._lift_counter = torch.where(lifted, obj._lift_counter + 1, 0)
    return obj._lift_counter >= hold_steps

def arm_to_needle_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_name: str = "end_effector"
) -> torch.Tensor:
    """Reward the agent for bringing the arm closer to the needle using tanh-kernel."""
    # Get the object (needle)
    object = env.scene[object_cfg.name]
    # Get the robot
    robot_name = robot_cfg.name
    # Get the ee_frame from the scene
    ee_frame = env.scene[f"ee_{robot_name[-1]}_frame"]
    
    # Needle position: (num_envs, 3)
    needle_pos = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    # Distance between end-effector and needle: (num_envs,)
    distance = torch.norm(needle_pos - ee_pos, dim=1)
    
    # Convert to reward (closer = higher reward)
    return 1 - torch.tanh(distance / std)


def needle_height_reward(
    env: "ManagerBasedRLEnv", 
    min_height: float = 0.08,  # Target height above table
    max_reward: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward for lifting the needle above a certain height."""
    object = env.scene[object_cfg.name]
    needle_height = object.data.root_pos_w[:, 2]  # z-coordinate
    return torch.clamp(max_reward * (needle_height - min_height) / 0.05, min=0.0, max=max_reward)


def needle_proximity_bonus(
    env: "ManagerBasedRLEnv",
    proximity_threshold: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Extra reward when needle is close to the end-effector."""
    object = env.scene[object_cfg.name]
    robot_name = robot_cfg.name
    ee_frame = env.scene[f"ee_{robot_name[-1]}_frame"]
    
    needle_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    distance = torch.norm(needle_pos - ee_pos, dim=1)
    return torch.where(distance < proximity_threshold, 1.0, 0.0)


def grasp_success(
    env: "ManagerBasedRLEnv",
    min_height: float = 0.08,
    grasp_duration: float = 1.0,  # Seconds needle must be held
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward for successfully grasping and holding the needle."""
    object = env.scene[object_cfg.name]
    robot_name = robot_cfg.name
    ee_frame = env.scene[f"ee_{robot_name[-1]}_frame"]
    
    needle_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    # Check if needle is close to gripper and above minimum height
    distance = torch.norm(needle_pos - ee_pos, dim=1)
    is_close = distance < 0.02
    is_lifted = needle_pos[:, 2] > min_height
    
    # Track how long the needle has been held
    if not hasattr(env, "_grasp_timer"):
        env._grasp_timer = torch.zeros(env.num_envs, device=env.device)
    
    dt = env.step_dt
    # Increment timer where conditions are met, reset otherwise
    env._grasp_timer = torch.where(
        is_close & is_lifted,
        env._grasp_timer + dt,
        torch.zeros_like(env._grasp_timer)
    )
    
    # Reward increases with duration held
    reward = torch.clamp(env._grasp_timer / grasp_duration, 0.0, 1.0)
    return reward


def needle_stability(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward stable needle orientation (less rotation)."""
    object = env.scene[object_cfg.name]
    angular_vel = object.data.root_ang_vel_w
    
    # Penalize high angular velocity
    return torch.exp(-torch.norm(angular_vel, dim=1)) 

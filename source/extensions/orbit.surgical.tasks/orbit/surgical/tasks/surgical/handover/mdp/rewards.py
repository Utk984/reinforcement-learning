# rewards.py
# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# ------------------------------
# Holding State Tracker
# ------------------------------
def _update_hold_flags(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, close_thresh=0.02, lift_thresh=0.06):
    if not hasattr(env, "_is_holding"):
        env._is_holding = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    obj = env.scene[object_cfg.name]
    ee = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    finger_closed = env.scene[robot_cfg.name].data.joint_pos[:, -1] > 0.5
    lifted = obj.data.root_pos_w[:, 2] > lift_thresh
    currently_holding = (dist < close_thresh) & finger_closed & lifted
    was_holding = env._is_holding.clone()
    env._is_holding.copy_(currently_holding)
    return was_holding, currently_holding

# ------------------------------
# Rewards (Normalized 0-1)
# ------------------------------
def hold_bonus(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    _, holding = _update_hold_flags(env, robot_cfg, object_cfg)
    return holding.float()

def drop_penalty(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    was, holding = _update_hold_flags(env, robot_cfg, object_cfg)
    dropped = was & (~holding)
    return dropped.float()

def smart_grasp_bonus(env: ManagerBasedRLEnv, proximity_threshold: float, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    d = torch.norm(
        env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :] -
        env.scene[object_cfg.name].data.root_pos_w, dim=1
    )
    closed = env.scene[robot_cfg.name].data.joint_pos[:, -1] > 0.5
    return ((d < proximity_threshold) & closed).float()

def lifted_success(env: ManagerBasedRLEnv, min_height: float, hold_steps: int) -> torch.Tensor:
    obj = env.scene["object"]
    if not hasattr(obj, "_lift_counter"):
        obj._lift_counter = torch.zeros_like(obj.data.root_pos_w[:, 0], dtype=torch.int32)
    lifted = obj.data.root_pos_w[:, 2] > min_height
    obj._lift_counter = torch.where(lifted, obj._lift_counter + 1, 0)
    return (obj._lift_counter >= hold_steps)

def arm_to_needle_distance(env: ManagerBasedRLEnv, std: float, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    needle_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(needle_pos - ee_pos, dim=1)
    return 1 - torch.tanh(distance / std)

def needle_height_reward(env: ManagerBasedRLEnv, min_height: float, max_reward: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    needle_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    return torch.clamp((needle_z - min_height) / 0.05, 0.0, 1.0)

def needle_proximity_bonus(env: ManagerBasedRLEnv, proximity_threshold: float, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    needle_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(needle_pos - ee_pos, dim=1)
    return (distance < proximity_threshold).float()

def grasp_success(env: ManagerBasedRLEnv, min_height: float, grasp_duration: float, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    needle_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(needle_pos - ee_pos, dim=1)
    is_close = distance < 0.02
    is_lifted = needle_pos[:, 2] > min_height
    if not hasattr(env, "_grasp_timer"):
        env._grasp_timer = torch.zeros(env.num_envs, device=env.device)
    dt = env.step_dt
    env._grasp_timer = torch.where(
        is_close & is_lifted,
        env._grasp_timer + dt,
        torch.zeros_like(env._grasp_timer)
    )
    return torch.clamp(env._grasp_timer / grasp_duration, 0.0, 1.0)

def needle_stability(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    angular_vel = env.scene[object_cfg.name].data.root_ang_vel_w
    return torch.exp(-torch.norm(angular_vel, dim=1))

def finger_toggle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    finger_idx = -1
    if not hasattr(robot, "_prev_finger"):
        robot._prev_finger = torch.zeros_like(robot.data.joint_pos[:, finger_idx])
    toggled = (robot.data.joint_pos[:, finger_idx] - robot._prev_finger).abs() > 0.01
    robot._prev_finger.copy_(robot.data.joint_pos[:, finger_idx])
    return toggled.float()


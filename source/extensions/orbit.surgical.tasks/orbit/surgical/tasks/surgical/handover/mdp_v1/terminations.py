# terminations_enhanced.py
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
# Utility Functions
# ------------------------------
def modify_termination_param(env, term_name: str, param_name: str, value):
    """Modify a parameter of a termination term."""
    term_cfg = env.termination_cfg[term_name]
    term_cfg.params[param_name] = value

# ------------------------------
# Time-Based Terminations
# ------------------------------
def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate episode after maximum time steps."""
    return env.episode_length_buf >= env.max_episode_length

# ------------------------------
# Success Terminations
# ------------------------------
def handover_complete(env: ManagerBasedRLEnv, min_duration: float = 0.5) -> torch.Tensor:
    """Terminate when handover is successfully completed and stable."""
    if not hasattr(env, "_handover_complete_time"):
        env._handover_complete_time = torch.zeros(env.num_envs, device=env.device)
    
    # Check if robot 2 is holding and robot 1 is not
    r1_holding = env._is_holding_r1 if hasattr(env, "_is_holding_r1") else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    r2_holding = env._is_holding_r2 if hasattr(env, "_is_holding_r2") else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    successful_handover = r2_holding & (~r1_holding)
    
    # Track duration of successful handover
    env._handover_complete_time = torch.where(
        successful_handover,
        env._handover_complete_time + env.step_dt,
        torch.zeros_like(env._handover_complete_time)
    )
    
    # Terminate when handover has been stable for the minimum duration
    return env._handover_complete_time >= min_duration

def object_reached_goal_height(env: ManagerBasedRLEnv, min_height: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when object is lifted to a certain height by robot 2."""
    r2_holding = env._is_holding_r2 if hasattr(env, "_is_holding_r2") else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    object_height = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    
    return (object_height > min_height) & r2_holding

# ------------------------------
# Failure Terminations
# ------------------------------
def root_height_below_minimum(env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if object falls below a minimum height (e.g., drops to floor)."""
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2] < minimum_height

def object_velocity_exceeded(env: ManagerBasedRLEnv, max_linear_vel: float, max_angular_vel: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if object is moving too fast (indicating loss of control or throwing)."""
    obj = env.scene[object_cfg.name]
    linear_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    angular_speed = torch.norm(obj.data.root_ang_vel_w, dim=1)
    
    return (linear_speed > max_linear_vel) | (angular_speed > max_angular_vel)

def robots_collision(env: ManagerBasedRLEnv, min_distance: float) -> torch.Tensor:
    """Terminate if robots collide (excluding intentional close proximity for handover)."""
    # Get positions of all non-tip links
    robot1_states = env.scene["robot_1"].data.body_state_w
    robot2_states = env.scene["robot_2"].data.body_state_w
    
    # Exclude the end-effector links during handover
    # This is a simplification - ideally you'd check all pairs of bodies
    r1_base_pos = robot1_states[:, 0, :3]  # Use base link as example
    r2_base_pos = robot2_states[:, 0, :3]
    
    distance = torch.norm(r1_base_pos - r2_base_pos, dim=1)
    return distance < min_distance

def object_unstable(env: ManagerBasedRLEnv, max_angular_vel: float, duration: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if object is unstable for too long (spinning rapidly)."""
    obj = env.scene[object_cfg.name]
    
    if not hasattr(obj, "_unstable_duration"):
        obj._unstable_duration = torch.zeros(env.num_envs, device=env.device)
    
    angular_speed = torch.norm(obj.data.root_ang_vel_w, dim=1)
    is_unstable = angular_speed > max_angular_vel
    
    obj._unstable_duration = torch.where(
        is_unstable,
        obj._unstable_duration + env.step_dt,
        torch.zeros_like(obj._unstable_duration)
    )
    
    return obj._unstable_duration > duration

def both_robots_not_holding(env: ManagerBasedRLEnv, timeout: float) -> torch.Tensor:
    """Terminate if neither robot is holding the object for too long during handover."""
    if not hasattr(env, "_no_hold_duration"):
        env._no_hold_duration = torch.zeros(env.num_envs, device=env.device)
    
    r1_holding = env._is_holding_r1 if hasattr(env, "_is_holding_r1") else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    r2_holding = env._is_holding_r2 if hasattr(env, "_is_holding_r2") else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # After handover has started (robot 1 has held the object), if neither robot is holding, track duration
    handover_started = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)  # Simplified condition
    neither_holding = (~r1_holding) & (~r2_holding)
    
    env._no_hold_duration = torch.where(
        handover_started & neither_holding,
        env._no_hold_duration + env.step_dt,
        torch.zeros_like(env._no_hold_duration)
    )
    
    return env._no_hold_duration > timeout
# observations_enhanced.py
# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_mul, quat_conjugate

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# ------------------------------
# Quaternion to Rotation Matrix
# ------------------------------
def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    xx, yy, zz, ww = x*x, y*y, z*z, w*w
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return torch.stack((
        ww + xx - yy - zz, 2*(xy - wz), 2*(xz + wy),
        2*(xy + wz), ww - xx + yy - zz, 2*(yz - wx),
        2*(xz - wy), 2*(yz + wx), ww - xx - yy + zz
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))

# ------------------------------
# Basic Observations
# ------------------------------
# def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Return the last action executed by the policy."""
#     if not hasattr(env, "_last_action") or env._last_action is None:
#         env._last_action = torch.zeros((env.num_envs, 14), device=env.device)
#     return env._last_action

def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the last action executed by the policy."""
    if not hasattr(env, "_last_action") or env._last_action is None:
        # At initialization time, action_space may not be available yet
        # Try to get action dimension from the action manager if available
        if hasattr(env, "action_manager") and env.action_manager is not None:
            action_dim = env.action_manager.action_dim
        else:
            # Fallback to a default value (14 based on your action shape log)
            action_dim = 14
        
        env._last_action = torch.zeros((env.num_envs, action_dim), device=env.device)
    return env._last_action

def joint_positions(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the joint positions for the specified asset."""
    return env.scene[asset_cfg.name].data.joint_pos

def joint_velocities(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the joint velocities for the specified asset."""
    return env.scene[asset_cfg.name].data.joint_vel

def ee_pose(env: ManagerBasedRLEnv, robot_name: str) -> torch.Tensor:
    """Return the end-effector pose (position and orientation)."""
    ee_frame = env.scene[f"ee_{robot_name[-1]}_frame"]
    pos = ee_frame.data.target_pos_w[..., 0, :]
    quat = ee_frame.data.target_rot_w[..., 0, :]
    return torch.cat([pos, quat], dim=-1)

# ------------------------------
# Object-Related Observations
# ------------------------------
def object_pose(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the object pose (position and orientation)."""
    obj = env.scene[object_cfg.name]
    return torch.cat([obj.data.root_pos_w, obj.data.root_quat_w], dim=-1)

def needle_vel(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the object velocity (linear and angular)."""
    obj = env.scene[object_cfg.name]
    return torch.cat([obj.data.root_lin_vel_w, obj.data.root_ang_vel_w], dim=1)

def needle_height(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the object height above the ground."""
    return env.scene[object_cfg.name].data.root_pos_w[:, 2:3]

def needle_ends_positions(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the positions of the needle tip and tail."""
    obj = env.scene[object_cfg.name]
    tip, tail = obj.data.body_state_w[:, 0, :3], obj.data.body_state_w[:, -1, :3]
    return torch.cat([tip, tail], dim=1)

# ------------------------------
# Relational Observations
# ------------------------------
def ee_to_needle_relative_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the relative position from end-effector to object."""
    needle_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    return needle_pos - ee_pos

def ee_to_needle_distance(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the distance from end-effector to object."""
    rel_pos = ee_to_needle_relative_pos(env, robot_cfg, object_cfg)
    return torch.norm(rel_pos, dim=1, keepdim=True)

def ee_to_needle_ori_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the orientation error between end-effector and object."""
    ee_R = quat_to_matrix(env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_rot_w)
    obj_R = quat_to_matrix(env.scene[object_cfg.name].data.root_quat_w)
    ee_x = ee_R[..., 0, :]
    needle_z = obj_R[..., 2, :]
    return (ee_x * needle_z).sum(-1, keepdim=True)

def ee_to_needle_ends(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the distances from end-effector to the needle ends."""
    obj = env.scene[object_cfg.name]
    tip, tail = obj.data.body_state_w[:, 0, :3], obj.data.body_state_w[:, -1, :3]
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    d_tip = torch.norm(ee_pos - tip, dim=1, keepdim=True)
    d_tail = torch.norm(ee_pos - tail, dim=1, keepdim=True)
    return torch.cat([d_tip, d_tail], dim=1)

def ee_to_ee_relative_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the relative position from end-effector 1 to end-effector 2."""
    ee1_pos = env.scene["ee_1_frame"].data.target_pos_w[..., 0, :]
    ee2_pos = env.scene["ee_2_frame"].data.target_pos_w[..., 0, :]
    return ee2_pos - ee1_pos

def ee_to_ee_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the distance between the end-effectors."""
    rel_pos = ee_to_ee_relative_pos(env)
    return torch.norm(rel_pos, dim=1, keepdim=True)

# ------------------------------
# State Tracking Observations
# ------------------------------
def finger_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the finger state (gripper aperture)."""
    return env.scene[asset_cfg.name].data.joint_pos[:, -1:]

def is_object_grasped(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, dist_threshold: float = 0.02) -> torch.Tensor:
    """Return whether the object is grasped by the specified robot."""
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    obj_pos = env.scene[object_cfg.name].data.root_pos_w
    distance = torch.norm(ee_pos - obj_pos, dim=1)
    fingers_closed = env.scene[robot_cfg.name].data.joint_pos[:, -1] > 0.1
    return (distance < dist_threshold).float() * fingers_closed.float()

def handover_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current phase of the handover task.
    
    Phases:
    0: initial phase (robot 1 needs to grasp)
    1: robot 1 has grasped, moving to handover position
    2: robot 2 approaching for handover
    3: robot 2 has grasped, robot 1 can release
    4: handover complete, robot 2 holding object
    
    Returns:
        One-hot encoded phase vector
    """
    if not hasattr(env, "_handover_phase"):
        env._handover_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Create aliases for grasp state
    if not hasattr(env, "_is_holding_r1") or not hasattr(env, "_is_holding_r2"):
        # If these don't exist yet, initialize them
        env._is_holding_r1 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._is_holding_r2 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    r1_holding = env._is_holding_r1
    r2_holding = env._is_holding_r2
    
    # Get EE positions for distance calculation
    ee1_pos = env.scene["ee_1_frame"].data.target_pos_w[..., 0, :]
    ee2_pos = env.scene["ee_2_frame"].data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(ee1_pos - ee2_pos, dim=1)
    
    # Define handover proximity threshold
    handover_proximity = 0.05  # 5cm
    
    # Get object position for checking if it's lifted
    obj_height = env.scene["object"].data.root_pos_w[:, 2]
    min_lift_height = 0.03  # 3cm above table
    
    # Track how long each robot has been holding
    if not hasattr(env, "_r1_hold_duration"):
        env._r1_hold_duration = torch.zeros(env.num_envs, device=env.device)
        env._r2_hold_duration = torch.zeros(env.num_envs, device=env.device)
    
    # Update hold durations
    env._r1_hold_duration = torch.where(
        r1_holding,
        env._r1_hold_duration + env.step_dt,
        torch.zeros_like(env._r1_hold_duration)
    )
    
    env._r2_hold_duration = torch.where(
        r2_holding,
        env._r2_hold_duration + env.step_dt,
        torch.zeros_like(env._r2_hold_duration)
    )
    
    # Stable hold thresholds (require holding for at least this time to confirm phase)
    stable_hold_time = 0.2  # 200ms
    r1_stable_hold = env._r1_hold_duration > stable_hold_time
    r2_stable_hold = env._r2_hold_duration > stable_hold_time
    
    # Phase transition logic
    current_phase = env._handover_phase.clone()
    
    # Phase 0 -> 1: Robot 1 grasps object and lifts it
    phase0_to_1 = (current_phase == 0) & r1_stable_hold & (obj_height > min_lift_height)
    
    # Phase 1 -> 2: Robot 1 brings object close to Robot 2
    phase1_to_2 = (current_phase == 1) & r1_stable_hold & (ee_distance < handover_proximity)
    
    # Phase 2 -> 3: Robot 2 grasps the object while Robot 1 still holds it
    phase2_to_3 = (current_phase == 2) & r1_holding & r2_holding
    
    # Phase 3 -> 4: Robot 1 releases, Robot 2 is stably holding
    phase3_to_4 = (current_phase == 3) & (~r1_holding) & r2_stable_hold & (obj_height > min_lift_height)
    
    # Apply phase transitions
    new_phase = current_phase.clone()
    new_phase = torch.where(phase0_to_1, torch.ones_like(current_phase), new_phase)
    new_phase = torch.where(phase1_to_2, 2 * torch.ones_like(current_phase), new_phase)
    new_phase = torch.where(phase2_to_3, 3 * torch.ones_like(current_phase), new_phase)
    new_phase = torch.where(phase3_to_4, 4 * torch.ones_like(current_phase), new_phase)
    
    # Handle failure cases - go back to appropriate phase
    
    # If robot 1 drops the object in phases 1 or 2, go back to phase 0
    drop_to_phase0 = ((current_phase == 1) | (current_phase == 2)) & (~r1_holding) & (obj_height < min_lift_height)
    new_phase = torch.where(drop_to_phase0, torch.zeros_like(current_phase), new_phase)
    
    # If robot 2 drops after grasping (phase 3), go back to phase 2
    drop_to_phase2 = (current_phase == 3) & (~r2_holding) & r1_holding
    new_phase = torch.where(drop_to_phase2, 2 * torch.ones_like(current_phase), new_phase)
    
    # If both robots drop the object after phase 3 started, go back to phase 0
    both_drop = (current_phase >= 3) & (~r1_holding) & (~r2_holding) & (obj_height < min_lift_height)
    new_phase = torch.where(both_drop, torch.zeros_like(current_phase), new_phase)
    
    # Update the stored phase
    env._handover_phase.copy_(new_phase)
    
    # Return one-hot encoded phase
    return torch.nn.functional.one_hot(env._handover_phase, num_classes=5).float()

# def time_remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Return normalized time remaining in episode."""
#     progress = env.episode_length_buf.float() / env.max_episode_length
#     return 1.0 - progress.unsqueeze(1)

def time_remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized time remaining in episode."""
    # Check if we have access to these attributes
    if hasattr(env, "episode_length_buf"):
        progress = env.episode_length_buf.float() / env.max_episode_length
    else:
        # Alternative implementation using available attributes
        # Most ORBIT environments have step_count or episode_length
        if hasattr(env, "step_count"):
            progress = env.step_count.float() / env.max_episode_length
        elif hasattr(env, "episode_length"):
            progress = env.episode_length.float() / env.max_episode_length
        else:
            # Fallback if no progress tracking is available
            # Create a dummy progress variable if it doesn't exist
            if not hasattr(env, "_episode_progress"):
                env._episode_progress = torch.zeros(env.num_envs, device=env.device)
                env._max_steps = int(env.episode_length_s / env.step_dt)
            
            # Increment progress counter
            env._episode_progress += 1.0 / env._max_steps
            # Ensure it's clamped between 0 and 1
            env._episode_progress = torch.clamp(env._episode_progress, 0.0, 1.0)
            progress = env._episode_progress
    
    return 1.0 - progress.unsqueeze(1)
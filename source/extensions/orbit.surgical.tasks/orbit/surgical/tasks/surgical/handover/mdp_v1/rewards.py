from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# ------------------------------
# Utility Functions
# ------------------------------
def _update_hold_flags(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, close_thresh=0.02, lift_thresh=0.06):
    """Track whether the object is being held by the specified robot."""
    if not hasattr(env, "_is_holding_r1") or not hasattr(env, "_is_holding_r2"):
        env._is_holding_r1 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._is_holding_r2 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
    obj = env.scene[object_cfg.name]
    ee = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    
    # Consider both gripper joints (assuming they're the last two joints)
    finger_closed = env.scene[robot_cfg.name].data.joint_pos[:, -1] > 0.05
    lifted = obj.data.root_pos_w[:, 2] > lift_thresh
    
    currently_holding = (dist < close_thresh) & finger_closed & lifted
    
    # Store holding state for the appropriate robot
    is_robot1 = "1" in robot_cfg.name
    was_holding = env._is_holding_r1.clone() if is_robot1 else env._is_holding_r2.clone()
    
    if is_robot1:
        env._is_holding_r1.copy_(currently_holding)
    else:
        env._is_holding_r2.copy_(currently_holding)
        
    return was_holding, currently_holding

def _is_handover_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if handover is complete (robot 1 was holding, now robot 2 is holding)."""
    if not hasattr(env, "_handover_complete"):
        env._handover_complete = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Update handover completion status if robot 2 is holding and robot 1 is not
    handover_happened = env._is_holding_r2 & (~env._is_holding_r1)
    env._handover_complete = env._handover_complete | handover_happened
    
    return env._handover_complete

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute the L2 norm of the action rate."""
    if not hasattr(env, "_prev_actions") or env._prev_actions is None:
        env._prev_actions = torch.zeros_like(env._last_action)
        return torch.zeros(env.num_envs, device=env.device)
    
    # Compute action rate and update previous actions
    action_rate = torch.norm(env._last_action - env._prev_actions, dim=1)
    env._prev_actions.copy_(env._last_action)
    
    return action_rate

# ------------------------------
# Basic Rewards
# ------------------------------
def hold_bonus(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for holding the object."""
    _, holding = _update_hold_flags(env, robot_cfg, object_cfg)
    return holding.float()

def drop_penalty(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for dropping the object after holding it."""
    was, holding = _update_hold_flags(env, robot_cfg, object_cfg)
    dropped = was & (~holding)
    return -dropped.float()  # Negative reward as it's a penalty

def arm_to_needle_distance(env: ManagerBasedRLEnv, std: float, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward based on proximity of end-effector to object."""
    needle_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(needle_pos - ee_pos, dim=1)
    return 1 - torch.tanh(distance / std)

def needle_height_reward(env: ManagerBasedRLEnv, min_height: float, max_height: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward based on object height above the ground."""
    needle_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    norm_height = (needle_z - min_height) / (max_height - min_height)
    return torch.clamp(norm_height, 0.0, 1.0)

def needle_stability(env: ManagerBasedRLEnv, scale: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the object stable (minimal angular velocity)."""
    angular_vel = env.scene[object_cfg.name].data.root_ang_vel_w
    angular_speed = torch.norm(angular_vel, dim=1)
    return torch.exp(-scale * angular_speed)

# ------------------------------
# Task-Specific Rewards
# ------------------------------
def smart_grasp_bonus(env: ManagerBasedRLEnv, proximity_threshold: float, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for attempting to grasp when close to the object."""
    d = torch.norm(
        env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :] -
        env.scene[object_cfg.name].data.root_pos_w, dim=1
    )
    closed = env.scene[robot_cfg.name].data.joint_pos[:, -1] > 0.05
    return ((d < proximity_threshold) & closed).float()

def lifted_success(env: ManagerBasedRLEnv, min_height: float, hold_steps: int, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for successfully lifting the object for a sustained period."""
    obj = env.scene[object_cfg.name]
    if not hasattr(obj, "_lift_counter"):
        obj._lift_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    
    lifted = obj.data.root_pos_w[:, 2] > min_height
    obj._lift_counter = torch.where(lifted, obj._lift_counter + 1, torch.zeros_like(obj._lift_counter))
    
    return (obj._lift_counter >= hold_steps).float()

def handover_approach(env: ManagerBasedRLEnv, optimal_distance: float, std: float) -> torch.Tensor:
    """Reward for bringing end-effectors close enough for handover."""
    ee1_pos = env.scene["ee_1_frame"].data.target_pos_w[..., 0, :]
    ee2_pos = env.scene["ee_2_frame"].data.target_pos_w[..., 0, :]
    
    distance = torch.norm(ee1_pos - ee2_pos, dim=1)
    proximity = torch.exp(-((distance - optimal_distance) / std)**2)
    
    # Only reward approach when robot 1 is holding the object
    return proximity * env._is_holding_r1.float()

def handover_completion(env: ManagerBasedRLEnv, bonus_scale: float = 10.0) -> torch.Tensor:
    """Large reward for completing the handover (robot 2 holding what robot 1 was holding)."""
    # Check if handover was just completed this timestep
    if not hasattr(env, "_prev_handover_complete"):
        env._prev_handover_complete = torch.zeros_like(env._handover_complete)
    
    just_completed = _is_handover_complete(env) & (~env._prev_handover_complete)
    env._prev_handover_complete.copy_(env._handover_complete)
    
    return just_completed.float() * bonus_scale

def coordinated_grasp_timing(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for coordinated grasping during handover (robot 2 grasps before robot 1 releases)."""
    robot1 = env.scene["robot_1"]
    robot2 = env.scene["robot_2"]
    
    # Check finger states (assuming gripper control is the last action dimension)
    r1_closing = robot1.data.joint_pos[:, -1] > 0.05
    r2_closing = robot2.data.joint_pos[:, -1] > 0.05
    
    # Reward when both robots are simultaneously grasping during handover phase
    both_grasping = r1_closing & r2_closing
    near_handover = torch.norm(
        env.scene["ee_1_frame"].data.target_pos_w[..., 0, :] - 
        env.scene["ee_2_frame"].data.target_pos_w[..., 0, :], 
        dim=1
    ) < 0.05
    
    return (both_grasping & near_handover & env._is_holding_r1).float()

def movement_efficiency(env: ManagerBasedRLEnv, scale: float = 0.1) -> torch.Tensor:
    """Reward for efficient movement (minimizing joint velocities)."""
    robot1_vel = torch.norm(env.scene["robot_1"].data.joint_vel, dim=1)
    robot2_vel = torch.norm(env.scene["robot_2"].data.joint_vel, dim=1)
    
    # Penalize high velocities
    return torch.exp(-scale * (robot1_vel + robot2_vel))

def finger_toggle_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for excessive gripper toggling (to encourage deliberate grasping)."""
    robot = env.scene[asset_cfg.name]
    finger_idx = -1
    
    if not hasattr(robot, "_prev_finger"):
        robot._prev_finger = torch.zeros_like(robot.data.joint_pos[:, finger_idx])
    
    toggled = (robot.data.joint_pos[:, finger_idx] - robot._prev_finger).abs() > 0.01
    robot._prev_finger.copy_(robot.data.joint_pos[:, finger_idx])
    
    return -toggled.float() * 0.1  # Small penalty for toggling

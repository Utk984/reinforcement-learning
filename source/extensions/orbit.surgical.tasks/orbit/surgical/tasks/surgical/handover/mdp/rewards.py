from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg

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
# Holding State Tracker
# ------------------------------
def _update_hold_flags(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    close_thresh=0.02,
    lift_thresh=0.01):
    # make a unique attribute name per robot
    attr = f"_is_holding_{robot_cfg.name}"
    if not hasattr(env, attr):
        setattr(env, attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    buf = getattr(env, attr)
    obj = env.scene[object_cfg.name]
    ee = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    finger_closed = env.scene[robot_cfg.name].data.joint_pos[:, -1] < 0.3
    lifted = obj.data.root_pos_w[:, 2] > lift_thresh
    currently_holding = finger_closed & lifted
    was_holding = buf.clone()
    buf.copy_(currently_holding)
    return was_holding, currently_holding

# ------------------------------
# Rewards (Normalized 0-1)
# ------------------------------
def needle_lift_with_grip(env, robot_cfg, object_cfg, min_height=0.02, max_reward=1.0):
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    needle_height = obj.data.root_pos_w[:, 2]  # (N,)
    jp = robot.data.joint_pos[:, -1]
    finger_closed = torch.abs(jp) < 0.3
    lifted = needle_height > min_height

    reward = (lifted & finger_closed).float() * max_reward
    return reward

def grasp_success(
    env: ManagerBasedRLEnv,
    min_height: float,
    grasp_duration: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    close_thresh: float = 0.02,
) -> torch.Tensor:

    # unique timer per robot
    timer_attr = f"_grasp_timer_{robot_cfg.name}"
    if (not hasattr(env, timer_attr)) or (getattr(env, timer_attr).shape[0] != env.num_envs):
        setattr(env, timer_attr, torch.zeros(env.num_envs, device=env.device))
    timer = getattr(env, timer_attr)

    obj  = env.scene[object_cfg.name]
    ee   = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    robot = env.scene[robot_cfg.name]

    jp = robot.data.joint_pos[:, -1]
    finger_closed = torch.abs(jp) < 0.3
    is_lifted     = obj.data.root_pos_w[:, 2] > min_height

    timer.copy_(torch.where(
        is_lifted & finger_closed, 
        timer + env.step_dt,
        torch.zeros_like(timer)
    ))
    return torch.clamp(timer / grasp_duration, 0.0, 1.0)

def needle_stability(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    angular_vel = env.scene[object_cfg.name].data.root_ang_vel_w
    return torch.exp(-torch.norm(angular_vel, dim=1))





def ee_to_cube_distance_reward(
    env,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
    cube_cfg:  SceneEntityCfg = SceneEntityCfg("pass_target"),
    object_cfg:  SceneEntityCfg = SceneEntityCfg("object"),
):
    """
    Reward = holding_flag * (1 - tanh(dist(EE_tip, cube)/std)).
    This drives the end-effector tip (ee_2_frame) straight onto the cube.
    """

    obj  = env.scene[object_cfg.name]
    is_lifted     = obj.data.root_pos_w[:, 2] > 0.01

    # 2) Get EE frame (true tool tip) in world
    ee_sensor = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    ee_pos    = ee_sensor.data.target_pos_w[..., 0, :]   # (N,3)

    # 3) Get cube root pos in world
    cube      = env.scene[cube_cfg.name]
    cube_pos  = cube.data.root_pos_w[:, :3]             # (N,3)

    # 4) Distance & tanh shaping
    dist   = torch.norm(ee_pos - cube_pos, dim=1)       # (N,)
    shaped = 1.0 - torch.tanh(dist / std)               # in [0,1]

    # 5) Only reward once we’re actually holding
    return is_lifted.float() * shaped



from omni.isaac.lab.managers import SceneEntityCfg


def tip_to_pass_target_distance(
    env, 
    std: float, 
    needle_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("pass_target"),
):
    # 1) Needle tip world position
    needle = env.scene[needle_cfg.name]
    tip_pos = needle.data.body_state_w[:, -1, :3]         # (N,3)

    # 2) Pass target world position
    target = env.scene[target_cfg.name]
    tgt_pos = target.data.root_pos_w[:, :3]              # (N,3)

    # 3) Tanh‐shaped distance reward
    dist = torch.norm(tip_pos - tgt_pos, dim=1)          # (N,)
    return 1.0 - torch.tanh(dist / std)


from omni.isaac.lab.managers import SceneEntityCfg


def needle_to_cube_distance_reward(
    env,
    std: float,
    needle_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cube_cfg:   SceneEntityCfg = SceneEntityCfg("pass_target"),
) -> torch.Tensor:
    # 1) needle root position in world
    needle = env.scene[needle_cfg.name]
    needle_pos = needle.data.root_pos_w[:, :3]          # (N,3)

    # 2) cube root position in world
    cube = env.scene[cube_cfg.name]
    cube_pos = cube.data.root_pos_w[:, :3]              # (N,3)

    # 3) distance & tanh shaping
    dist = torch.norm(needle_pos - cube_pos, dim=1)     # (N,)
    return 1.0 - torch.tanh(dist / std)

def lifted_success(env: ManagerBasedRLEnv, min_height: float, hold_steps: int) -> torch.Tensor:
    obj = env.scene["object"]
    if not hasattr(obj, "_lift_counter"):
        obj._lift_counter = torch.zeros_like(obj.data.root_pos_w[:, 0], dtype=torch.int32)
    lifted = obj.data.root_pos_w[:, 2] > min_height
    obj._lift_counter = torch.where(lifted, obj._lift_counter + 1, 0)
    return (obj._lift_counter >= hold_steps)

def needle_height_reward(env: ManagerBasedRLEnv, min_height: float, max_reward: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    needle_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    return torch.clamp((needle_z - min_height) / 0.05, 0.0, 1.0)

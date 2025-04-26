# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the handover environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


# ------------------------------------------------------------------
# Local fallback: quaternion (x,y,z,w)  ->  3×3 rotation matrix
# ------------------------------------------------------------------
def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    xx = x * x; yy = y * y; zz = z * z; ww = w * w
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz

    R = torch.stack(
        (m00, m01, m02,
         m10, m11, m12,
         m20, m21, m22),
        dim=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))



def ee_to_needle_relative_pos(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Return the relative position vector from end-effector to needle."""
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
    
    # Relative position: (num_envs, 3)
    return needle_pos - ee_pos 

def needle_vel(env, object_cfg):
    obj = env.scene[object_cfg.name]
    lin = obj.data.root_lin_vel_w                 # (N, 3)
    ang = obj.data.root_ang_vel_w                 # (N, 3)
    return torch.cat([lin, ang], dim=1)           # (N, 6)

def ee_to_needle_ori_error(env, robot_cfg, object_cfg):
    ee  = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    obj = env.scene[object_cfg.name]
    ee_R  = quat_to_matrix(ee.data.target_rot_w)
    obj_R = quat_to_matrix(obj.data.root_quat_w)
    ee_x  = ee_R[..., 0, :]       # EE forward
    needle_z = obj_R[..., 2, :]   # needle axis
    cos = (ee_x * needle_z).sum(-1, keepdim=True)   # (N,1)
    return cos                    # +1 = aligned, –1 = opposite

def ee_to_needle_ends(env, robot_cfg, object_cfg):
    ee   = env.scene[f"ee_{robot_cfg.name[-1]}_frame"]
    obj  = env.scene[object_cfg.name]
    tip  = obj.data.body_state_w[:, 0, :3]   # first body = tip
    tail = obj.data.body_state_w[:, -1, :3]  # last  body = tail
    ee_p = ee.data.target_pos_w[..., 0, :]
    d_tip  = torch.norm(ee_p - tip,  dim=1, keepdim=True)
    d_tail = torch.norm(ee_p - tail, dim=1, keepdim=True)
    return torch.cat([d_tip, d_tail], dim=1)


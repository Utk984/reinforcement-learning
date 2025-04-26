# observations.py
# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
# Observations
# ------------------------------
def ee_to_needle_relative_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    needle_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    return needle_pos - ee_pos

def needle_vel(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    return torch.cat([obj.data.root_lin_vel_w, obj.data.root_ang_vel_w], dim=1)

def ee_to_needle_ori_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_R = quat_to_matrix(env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_rot_w)
    obj_R = quat_to_matrix(env.scene[object_cfg.name].data.root_quat_w)
    ee_x = ee_R[..., 0, :]
    needle_z = obj_R[..., 2, :]
    return (ee_x * needle_z).sum(-1, keepdim=True)

def ee_to_needle_ends(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    tip, tail = obj.data.body_state_w[:, 0, :3], obj.data.body_state_w[:, -1, :3]
    ee_pos = env.scene[f"ee_{robot_cfg.name[-1]}_frame"].data.target_pos_w[..., 0, :]
    d_tip = torch.norm(ee_pos - tip, dim=1, keepdim=True)
    d_tail = torch.norm(ee_pos - tail, dim=1, keepdim=True)
    return torch.cat([d_tip, d_tail], dim=1)

def needle_height(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_pos_w[:, 2:3]

def finger_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos[:, -1:]


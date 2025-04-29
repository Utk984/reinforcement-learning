# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import torch
from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
import inspect
import time
from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the handover scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_1_frame: FrameTransformerCfg = MISSING
    ee_2_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_1_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_1",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.12, -0.08),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    ee_2_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_2",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.12, -0.08),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_1_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_1_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING
    body_2_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_2_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # TODO
        end_dist = ObsTerm(
            func=mdp.ee_to_needle_ends,
            params={"robot_cfg": SceneEntityCfg("robot_2"), "object_cfg": SceneEntityCfg("object")}
        )

        needle_vel = ObsTerm(
            func=mdp.needle_vel,
            params={"object_cfg": SceneEntityCfg("object")}
        )

        # Height of needle (1 value)
        needle_height = ObsTerm(
            func=mdp.needle_height,
            params={"object_cfg": SceneEntityCfg("object")}
        )   

        # Finger (gripper) joint position: 0=open  1=closed
        finger_1_state = ObsTerm(
            func=mdp.finger_state,
            params={"asset_cfg": SceneEntityCfg("robot_2")}
        )

        # Relative position observations
        ee_1_to_needle = ObsTerm(
            func=mdp.ee_to_needle_relative_pos,
            params={"robot_cfg": SceneEntityCfg("robot_1"), "object_cfg": SceneEntityCfg("object")}
        )
        ee_2_to_needle = ObsTerm(
            func=mdp.ee_to_needle_relative_pos,
            params={"robot_cfg": SceneEntityCfg("robot_2"), "object_cfg": SceneEntityCfg("object")}
        )

        # Last actions
        joint_positions_1 = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot_1")}
        )

        joint_positions_2 = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot_2")}
        )

        joint_velocities_1 = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot_1")}
        )

        joint_velocities_2 = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot_2")}
        )

        last_action = ObsTerm(func=mdp.last_action)
        needle_position = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")}
        )
        needle_linear_velocity = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("object")}
        )

        #contact_forces = ObsTerm(func=mdp.contact_forces)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.1, -0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Needle approach rewards
    arm_1_to_needle = RewTerm(
        func=mdp.arm_to_needle_distance,
        weight=0.1,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot_1"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    arm_2_to_needle = RewTerm(
        func=mdp.arm_to_needle_distance,
        weight=0.1,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot_2"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    # Needle lifting reward
    needle_lifted = RewTerm(
        func=mdp.needle_height_reward,
        weight=3.0,
        params={
            "min_height": 0.02,      # Needle must be 2cm above table
            "max_reward": 1.0,        # Already normalized inside function
            "object_cfg": SceneEntityCfg("object")
        }
    )

    # Holding bonus
    hold_bonus = RewTerm(
        func=mdp.hold_bonus,
        weight=5.0,                      # 5× reward per step
        params={
            "robot_cfg": SceneEntityCfg("robot_2"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    hold_bonus2 = RewTerm(
        func=mdp.hold_bonus,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot_1"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    # Drop penalty
    drop_penalty = RewTerm(
        func=mdp.drop_penalty,
        weight=-1.0,                     # NEGATIVE weight
        params={
            "robot_cfg": SceneEntityCfg("robot_2"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    drop_penalty2 = RewTerm(
        func=mdp.drop_penalty,
        weight=-1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot_1"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    #alignment_bonus = RewTerm(
    #    func=mdp.ee_to_needle_orientation_alignment,  # Again, you need to add this helper if missing
    #    weight=1.0,
    #    params={
    #        "robot_cfg": SceneEntityCfg("robot_2"),
    #        "object_cfg": SceneEntityCfg("object")
    #    }
    #)

    # Needle stability (minimize spin)
    needle_stability = RewTerm(
        func=mdp.needle_stability,
        weight=0.05,
        params={
            "object_cfg": SceneEntityCfg("object")
        }
    )

    joint_velocity_smoothness = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00005,
        params={
            "asset_cfg": SceneEntityCfg("robot_2")
        }
    )

    # Grasp success reward
    grasp_success = RewTerm(
        func=mdp.grasp_success,
        weight=10.0,
        params={
            "min_height": 0.06,            # 6cm lifted
            "grasp_duration": 1.0,          # 1 second holding
            "robot_cfg": SceneEntityCfg("robot_1"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    grasp_success2 = RewTerm(
        func=mdp.grasp_success,
        weight=10.0,
        params={
            "min_height": 0.06,
            "grasp_duration": 1.0,
            "robot_cfg": SceneEntityCfg("robot_2"),
            "object_cfg": SceneEntityCfg("object")
        }
    )

    # Stability reward (keep needle stable after grasp)
    needle_stability = RewTerm(
        func=mdp.needle_stability,
        weight=0.05,
        params={
            "object_cfg": SceneEntityCfg("object")
        }
    )

    correct_grip = RewTerm(
    func=mdp.correct_grip_bonus,
    weight=0.5,  # Moderate reward
    params={
        "robot_cfg": SceneEntityCfg("robot_2"),
        "object_cfg": SceneEntityCfg("object"),
        "proximity_threshold": 0.01
        }
    )

    smooth_action = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    #small_action = RewTerm(func=mdp.action_l2, weight=-1e-3)
    #safe_joint_positions = RewTerm(func=mdp.joint_pos_limit_normalized, weight=-0.5)
    #undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0)


    # Finger toggle penalty (optional - can uncomment later if needed)
    finger_spam = RewTerm(
         func=mdp.finger_toggle,
         weight=-0.0002,
         params={"asset_cfg": SceneEntityCfg("robot_2")}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    lifted_success = DoneTerm(
        func=mdp.lifted_success,
        params={
            "min_height": 0.08,
            "hold_steps": 500 
        }
    )

def modify_termination_params(env, term_name: str, num_steps: int, **new_params):
    """
    Modify multiple parameters of a termination term at a specific training step.

    Arguments:
    """
    if env.total_steps >= num_steps:
        term_cfg = env.termination_cfg[term_name]
        for key, value in new_params.items():
            term_cfg.params[key] = value

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Smoothness increase: strengthen action rate penalty
    smoother_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "smooth_action",
            "weight": -0.0015,
            "num_steps": 10_000
        }
    )
    
    smoother_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "smooth_action",
            "weight": -0.01,
            "num_steps": 12_000
        }
    )


    # Introduce correct grip bonus reward later (optional, advanced)
    # enable_correct_grip = CurrTerm(
    #    func=mdp.modify_reward_weight,
    #    params={
    #        "term_name": "correct_grip_bonus",
    #        "weight": 2.0,
    #        "num_steps": 20_000             # Only after robot starts lifting
    #    }
    #)



@configclass
class HandoverEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the handover environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        self.sim.dt = 0.01  # 100Hz

        # simulation settings
        self.viewer.eye = (1.25, 0.5, 0.3)
        self.viewer.lookat = (1.25, 0.0, 0.05)

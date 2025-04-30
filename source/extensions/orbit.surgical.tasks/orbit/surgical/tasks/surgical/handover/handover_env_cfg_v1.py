from dataclasses import MISSING

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
        
        # Robot state observations
        robot_1_joints = ObsTerm(func=mdp.joint_positions, params={"asset_cfg": SceneEntityCfg("robot_1")})
        robot_2_joints = ObsTerm(func=mdp.joint_positions, params={"asset_cfg": SceneEntityCfg("robot_2")})
        
        ee_1_to_object = ObsTerm(
            func=mdp.ee_to_needle_relative_pos, 
            params={"robot_cfg": SceneEntityCfg("robot_1"), "object_cfg": SceneEntityCfg("object")}
        )
        ee_2_to_object = ObsTerm(
            func=mdp.ee_to_needle_relative_pos, 
            params={"robot_cfg": SceneEntityCfg("robot_2"), "object_cfg": SceneEntityCfg("object")}
        )
        
        # Object state observations
        object_pose = ObsTerm(func=mdp.object_pose, params={"object_cfg": SceneEntityCfg("object")})
        object_velocity = ObsTerm(func=mdp.needle_vel, params={"object_cfg": SceneEntityCfg("object")})
        object_height = ObsTerm(func=mdp.needle_height, params={"object_cfg": SceneEntityCfg("object")})
        
        # Relational observations
        ee_to_ee = ObsTerm(func=mdp.ee_to_ee_relative_pos)
        ee_1_object_ends = ObsTerm(
            func=mdp.ee_to_needle_ends, 
            params={"robot_cfg": SceneEntityCfg("robot_1"), "object_cfg": SceneEntityCfg("object")}
        )
        ee_2_object_ends = ObsTerm(
            func=mdp.ee_to_needle_ends, 
            params={"robot_cfg": SceneEntityCfg("robot_2"), "object_cfg": SceneEntityCfg("object")}
        )
        
        # Task state observations
        finger_state_1 = ObsTerm(func=mdp.finger_state, params={"asset_cfg": SceneEntityCfg("robot_1")})
        finger_state_2 = ObsTerm(func=mdp.finger_state, params={"asset_cfg": SceneEntityCfg("robot_2")})
        is_grasped_r1 = ObsTerm(
            func=mdp.is_object_grasped, 
            params={
                "robot_cfg": SceneEntityCfg("robot_1"), 
                "object_cfg": SceneEntityCfg("object"), 
                "dist_threshold": 0.02
            }
        )
        is_grasped_r2 = ObsTerm(
            func=mdp.is_object_grasped, 
            params={
                "robot_cfg": SceneEntityCfg("robot_2"), 
                "object_cfg": SceneEntityCfg("object"), 
                "dist_threshold": 0.02
            }
        )
        
        # Timing information
        # time_info = ObsTerm(func=mdp.time_remaining)
        
        # Previous actions
        actions = ObsTerm(func=mdp.last_action)

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

    # Reaching rewards
    robot_1_reach = RewTerm(
        func=mdp.arm_to_needle_distance, 
        params={
            "std": 0.05, 
            "robot_cfg": SceneEntityCfg("robot_1"), 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=0.3
    )
    
    robot_2_reach = RewTerm(
        func=mdp.arm_to_needle_distance, 
        params={
            "std": 0.05, 
            "robot_cfg": SceneEntityCfg("robot_2"), 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=0.3
    )
    
    # Grasping rewards
    robot_1_hold = RewTerm(
        func=mdp.hold_bonus, 
        params={
            "robot_cfg": SceneEntityCfg("robot_1"), 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=0.5
    )
    
    robot_2_hold = RewTerm(
        func=mdp.hold_bonus, 
        params={
            "robot_cfg": SceneEntityCfg("robot_2"), 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=0.5
    )
    
    drop_penalty_r1 = RewTerm(
        func=mdp.drop_penalty, 
        params={
            "robot_cfg": SceneEntityCfg("robot_1"), 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=1.0
    )
    
    drop_penalty_r2 = RewTerm(
        func=mdp.drop_penalty, 
        params={
            "robot_cfg": SceneEntityCfg("robot_2"), 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=1.0
    )
    
    # Height rewards
    height_reward = RewTerm(
        func=mdp.needle_height_reward, 
        params={
            "min_height": 0.03, 
            "max_height": 0.15, 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=0.2
    )
    
    # Stability rewards
    object_stability = RewTerm(
        func=mdp.needle_stability, 
        params={
            "scale": 0.2, 
            "object_cfg": SceneEntityCfg("object")
        },
        weight=0.3
    )
    
    # Handover coordination rewards
    handover_approach = RewTerm(
        func=mdp.handover_approach, 
        params={
            "optimal_distance": 0.04, 
            "std": 0.02
        },
        weight=0.4
    )
    
    coordination = RewTerm(
        func=mdp.coordinated_grasp_timing,
        weight=0.5
    )
    
    handover_success = RewTerm(
        func=mdp.handover_completion, 
        params={"bonus_scale": 5.0},
        weight=1.0
    )
    
    # Efficiency rewards
    movement_efficiency = RewTerm(
        func=mdp.movement_efficiency, 
        params={"scale": 0.1},
        weight=0.2
    )
    
    # Action penalties
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.05
    )
    
    finger_toggle_r1 = RewTerm(
        func=mdp.finger_toggle_penalty, 
        params={"asset_cfg": SceneEntityCfg("robot_1")},
        weight=1.0
    )
    
    finger_toggle_r2 = RewTerm(
        func=mdp.finger_toggle_penalty, 
        params={"asset_cfg": SceneEntityCfg("robot_2")},
        weight=1.0
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Success terminations
    handover_complete = DoneTerm(
        func=mdp.handover_complete, 
        params={"min_duration": 0.5}# ,
        # success=True
    )
    
    object_goal_reached = DoneTerm(
        func=mdp.object_reached_goal_height, 
        params={
            "min_height": 0.08, 
            "object_cfg": SceneEntityCfg("object")
        }#,
        # success=True
    )
    
    # Failure terminations
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )
    
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={
            "minimum_height": -0.05, 
            "asset_cfg": SceneEntityCfg("object")
        }
    )
    
    object_too_fast = DoneTerm(
        func=mdp.object_velocity_exceeded, 
        params={
            "max_linear_vel": 1.0, 
            "max_angular_vel": 10.0, 
            "object_cfg": SceneEntityCfg("object")
        }
    )
    
    robots_collision = DoneTerm(
        func=mdp.robots_collision, 
        params={"min_distance": 0.05}
    )
    
    unstable_object = DoneTerm(
        func=mdp.object_unstable, 
        params={
            "max_angular_vel": 5.0, 
            "duration": 1.0, 
            "object_cfg": SceneEntityCfg("object")
        }
    )
    
    dropped_during_handover = DoneTerm(
        func=mdp.both_robots_not_holding, 
        params={"timeout": 1.0}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.01, "num_steps": 10000}
    )

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


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_tasks.manager_based.manipulation.bernie_proj.ycb import mdp
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
import os
import pathlib
workspace = pathlib.Path(os.getenv("WORKSPACE_FOLDER", pathlib.Path.cwd()))

##
# Scene definition
##


@configclass
class YCBLiftSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    ee_frame: FrameTransformerCfg = MISSING

    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    goal_object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # adjust the table placement

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.665, 0.5, 0.475], rot=[-0.7071, 0, 0, 0.7071]),
        spawn=UsdFileCfg(usd_path=str(workspace) + "/assets/rlaif_table.usd", scale=(1.5, 1.5, 0.765),),
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

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_left_arm_6",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        # objects pose
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        # goal objects pose
        goal_object_position = ObsTerm(func=mdp.goal_object_position_in_robot_root_frame)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset scene assets
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    """
    reset robot state
    """
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    # reset start and goal state positions
    reset_objects = EventTerm(
        func=mdp.reset_object_state_goal_state_uniform,
        mode="reset",
        params={
            "pose_range":
            {
                "x": (0.000, 0.010),
                "y": (0.35, 0.45),
                "z": (0.546, 0.546),
            },
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted_lift_task, params={"minimal_height": 0.04}, weight=100.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance_z,
    #     params={"std": 0.3},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance_z,
    #     params={"std": 0.05},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##


@configclass
class YCBLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: YCBLiftSceneCfg = YCBLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 100000

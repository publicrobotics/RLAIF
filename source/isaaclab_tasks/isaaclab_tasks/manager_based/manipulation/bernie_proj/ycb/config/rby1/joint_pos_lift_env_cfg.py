# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.bernie_proj.ycb.ycb_lift_env_cfg import YCBLiftEnvCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_tasks.manager_based.manipulation.bernie_proj.ycb import mdp
import os
import pathlib
workspace = pathlib.Path(os.getenv("WORKSPACE_FOLDER", pathlib.Path.cwd()))

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, DifferentialIKControllerCfg


@configclass
class JointLiftEnvCfg(YCBLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set RBY1 as robot
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(workspace) + "/assets/RBY1.usd",
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
                copy_from_source=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "left_arm_0": 0.0,
                    "left_arm_1": 0.0,
                    "left_arm_2": 0.0,
                    "left_arm_3": 0.0,
                    "left_arm_4": 0.0,
                    "left_arm_5": 0.0,
                    "left_arm_6": 0.0,
                    "right_arm_0": 0.0,
                    "right_arm_1": 0.0,
                    "right_arm_2": 0.0,
                    "right_arm_3": 0.0,
                    "right_arm_4": 0.0,
                    "right_arm_5": 0.0,
                    "right_arm_6": 0.0,
                    "gripper_finger_l1": -0.05,
                    "gripper_finger_l2": 0.05,
                    "gripper_finger_r1": -0.05,
                    "gripper_finger_r2": 0.05,
                    "torso_0": 0.0,
                    "torso_1": 0.0,
                    "torso_2": 0.0,
                    "torso_3": 0.0,
                    "torso_4": 0.0,
                    "torso_5": 0.0,
                    "head_0": 0.0,
                    "head_1": 0.0,
                },
            ),
            actuators={
                "rby1_leftarm": ImplicitActuatorCfg(
                    joint_names_expr=["left_arm_[0-6]"],
                    effort_limit_sim=87.0,
                    velocity_limit_sim=2.175,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "rby1_rightarm": ImplicitActuatorCfg(
                    joint_names_expr=["right_arm_[0-6]"],
                    effort_limit_sim=87.0,
                    velocity_limit_sim=2.175,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "rby1_leftgripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper_finger_l.*"],
                    effort_limit_sim=200.0,
                    velocity_limit_sim=0.2,
                    stiffness=2e3,
                    damping=1e2,
                ),
                "rby1_rightgripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper_finger_r.*"],
                    effort_limit_sim=200.0,
                    velocity_limit_sim=0.2,
                    stiffness=2e3,
                    damping=1e2,
                ),
                "rby1_torso": ImplicitActuatorCfg(
                    joint_names_expr=["torso_[0-5]"],
                    effort_limit=0.0,
                    velocity_limit=0.0,
                    effort_limit_sim=0.0,
                    velocity_limit_sim=0.0,
                    stiffness=1e7,
                    damping=1e7,
                ),
                "rby1_head": ImplicitActuatorCfg(
                    joint_names_expr=["head_[0-1]"],
                    effort_limit=0.0,
                    velocity_limit=0.0,
                    effort_limit_sim=0.0,
                    velocity_limit_sim=0.0,
                    stiffness=1e7,
                    damping=1e7,
                ),
                "rby1_wheel": ImplicitActuatorCfg(
                    joint_names_expr=["left_wheel", "right_wheel"],
                    effort_limit=0.0,
                    velocity_limit=0.0,
                    effort_limit_sim=0.0,
                    velocity_limit_sim=0.0,
                    stiffness=1e7,
                    damping=1e7,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        # RBY1 Arm Actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_arm_[0-6]"],
            body_name="link_left_arm_6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_finger_l1", "gripper_finger_l2"],
            open_command_expr={
                "gripper_finger_l1": 0.0,
                "gripper_finger_l2": 0.0,
            },
            close_command_expr={
                "gripper_finger_l1": -0.05,
                "gripper_finger_l2": 0.05,
            },
        )

        # here I will be specifying the ee_frame position
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link_left_arm_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link_left_arm_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.275],
                    ),
                ),
            ],
        )


@configclass
class JointLiftEnvCfg_PLAY(JointLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

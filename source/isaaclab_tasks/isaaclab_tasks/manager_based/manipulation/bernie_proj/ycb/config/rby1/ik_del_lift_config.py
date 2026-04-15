# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.bernie_proj.ycb import mdp
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.manager_based.manipulation.bernie_proj.ycb.ycb_lift_env_cfg import YCBLiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

##
# Pre-defined configs
##
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, DifferentialIKControllerCfg
from isaaclab.assets import (
    RigidObjectCfg,
)

# For path names
import os
import pathlib
workspace = pathlib.Path(os.getenv("WORKSPACE_FOLDER", pathlib.Path.cwd()))
from pathlib import Path
from typing import Literal
ASSET_ROOT = Path("/home/davin123/PoliGen/assets/ycb")  # folder where *.usd files live

@configclass
class RBY1TeleopLift(YCBLiftEnvCfg):
    # --------------- NEW FIELDS ---------------
    object_name: Literal[
        "banana", "cup", "block", "bottle", "dice", "pitcher", "rubrik"
    ] = "banana"

    # optional per‑object scale overrides
    object_scale: dict[str, tuple[float, float, float]] = {
        "banana": (0.8, 0.8, 0.8),
        "cup": (1.0, 1.0, 1.0),
        "block": (0.6, 0.6, 0.6),
        "bottle": (0.9, 0.9, 0.9),
        "dice": (0.5, 0.5, 0.5),
        "pitcher": (0.7, 0.7, 0.7),
        "rubrik": (0.8, 0.8, 0.8),
    }
    # ------------------------------------------

    def __post_init__(self):
        super().__post_init__()

        usd_path = ASSET_ROOT / f"{self.object_name}_goal.usd"
        self.scene.goal_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/goal_object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.10, 0.00, 0.30], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=str(usd_path),
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=False),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.45, 0.88, 0.67))
            ),
        )

        # build the USD path for whichever object is requested
        usd_path = ASSET_ROOT / f"{self.object_name}.usd"
        if not usd_path.exists():
            raise FileNotFoundError(f"Could not find asset: {usd_path}")

        # ---------- object definition ----------
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.10, 0.0, 0.055], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=str(usd_path),
                scale=self.object_scale.get(self.object_name, (1.0, 1.0, 1.0)),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

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

                    # this will be for the 1st main joint
                    "left_arm_0": -0.40,
                    "left_arm_1": 0.0,
                    "left_arm_2": 0.0,

                    # this will be for the 2nd main joint
                    "left_arm_3": -0.5,

                    # this will be for the 3rd main joint
                    "left_arm_4": 0.0,
                    "left_arm_5": 0.90,
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
                    stiffness=1000.0,
                    damping=250.0,
                ),
                "rby1_rightarm": ImplicitActuatorCfg(
                    joint_names_expr=["right_arm_[0-6]"],
                    effort_limit_sim=87.0,
                    velocity_limit_sim=2.175,
                    stiffness=400.0,
                    damping=100.0,
                ),
                "rby1_leftgripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper_finger_l.*"],
                    effort_limit_sim=200.0,
                    velocity_limit_sim=0.2,
                    stiffness=4e3,
                    damping=2e2,
                ),
                "rby1_rightgripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper_finger_r.*"],
                    effort_limit_sim=200.0,
                    velocity_limit_sim=0.2,
                    stiffness=4e3,
                    damping=2e2,
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
            scale=0.25,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.275]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_finger_l1", "gripper_finger_l2"],
            open_command_expr={
                "gripper_finger_l1": 0.0,
                "gripper_finger_l2": 0.0,
            },
            close_command_expr={
                "gripper_finger_l1": -0.025,
                "gripper_finger_l2": 0.025,
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
class RBY1TeleopLiftBanana(RBY1TeleopLift):
    object_name: str = "banana"


@configclass
class RBY1TeleopLiftCup(RBY1TeleopLift):
    object_name: str = "cup"


@configclass
class RBY1TeleopLiftBlock(RBY1TeleopLift):
    object_name: str = "block"


@configclass
class RBY1TeleopLiftBottle(RBY1TeleopLift):
    object_name: str = "bottle"


@configclass
class RBY1TeleopLiftDice(RBY1TeleopLift):
    object_name: str = "dice"


@configclass
class RBY1TeleopLiftPitcher(RBY1TeleopLift):
    object_name: str = "pitcher"


@configclass
class RBY1TeleopLiftRubrik(RBY1TeleopLift):
    object_name: str = "rubrik"

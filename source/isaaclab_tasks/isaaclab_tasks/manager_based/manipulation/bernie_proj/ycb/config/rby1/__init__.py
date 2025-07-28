# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##


"""
Teleoperation
""" 
gym.register(
    id="YCBS-Teleop-PickAndPlace",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_del_pick_and_place_config:RBY1TeleopPickAndPlace",
    },
    disable_env_checker=True,
)

gym.register(
    id="YCBS-Teleop-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_del_lift_config:RBY1TeleopLift",
    },
    disable_env_checker=True,
)

gym.register(
    id="YCBS-Teleop-Push",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_del_push_config:RBY1TeleopPush",
    },
    disable_env_checker=True,
)


"""
Task Environemnts
"""


"""
YCB RBY1 Lift 
"""
YCB_ENVS = [
    ("banana",  "RBY1YCBLiftBananaEnvCfg"),
    ("block",   "RBY1YCBLiftBlockEnvCfg"),
    ("bottle",  "RBY1YCBLiftBottleEnvCfg"),
    ("cup",     "RBY1YCBLiftCupEnvCfg"),
    ("dice",    "RBY1YCBLiftDiceEnvCfg"),
    ("pitcher", "RBY1YCBLiftPitcherEnvCfg"),
    ("rubriks", "RBY1YCBLiftRubriksEnvCfg"),
]

for env_name, env_class in YCB_ENVS:
    gym.register(
        id=f"ycb-lift-{env_name}",             
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            # Points Gym to the corresponding EnvCfg class
            "env_cfg_entry_point": (
                f"{__name__}.ycb_rby1_lift_env_cfg:{env_class}"
            ),
            # PPO runner shared by all single‑object tasks
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:YCBPPORunnerCfg"
            ),
        },
    )


"""
YCB RBY1 Push
"""
YCB_ENVS = [
    ("banana",  "RBY1YCBPushBananaEnvCfg"),
    ("block",   "RBY1YCBPushBlockEnvCfg"),
    ("bottle",  "RBY1YCBPushBottleEnvCfg"),
    ("cup",     "RBY1YCBPushCupEnvCfg"),
    ("dice",    "RBY1YCBPushDiceEnvCfg"),
    ("pitcher", "RBY1YCBPushPitcherEnvCfg"),
    ("rubriks", "RBY1YCBPushRubriksEnvCfg"),
]

for env_name, env_class in YCB_ENVS:
    gym.register(
        id=f"ycb-push-{env_name}",             
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            # Points Gym to the corresponding EnvCfg class
            "env_cfg_entry_point": (
                f"{__name__}.ycb_rby1_push_env_cfg:{env_class}"
            ),
            # PPO runner shared by all single‑object tasks
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:YCBPPORunnerCfg"
            ),
        },
    )



"""
YCB RBY1 Pick And Place
"""
YCB_ENVS = [
    ("banana",  "RBY1YCBPickAndPlaceBananaEnvCfg"),
    ("block",   "RBY1YCBPickAndPlaceBlockEnvCfg"),
    ("bottle",  "RBY1YCBPickAndPlaceBottleEnvCfg"),
    ("cup",     "RBY1YCBPickAndPlaceCupEnvCfg"),
    ("dice",    "RBY1YCBPickAndPlaceDiceEnvCfg"),
    ("pitcher", "RBY1YCBPickAndPlacePitcherEnvCfg"),
    ("rubriks", "RBY1YCBPickAndPlaceRubriksEnvCfg"),
]

for env_name, env_class in YCB_ENVS:
    gym.register(
        id=f"ycb-pick-and-place-{env_name}",             
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            # Points Gym to the corresponding EnvCfg class
            "env_cfg_entry_point": (
                f"{__name__}.ycb_rby1_pick_and_place_env_cfg:{env_class}"
            ),
            # PPO runner shared by all single‑object tasks
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:YCBPPORunnerCfg"
            ),
        },
    )

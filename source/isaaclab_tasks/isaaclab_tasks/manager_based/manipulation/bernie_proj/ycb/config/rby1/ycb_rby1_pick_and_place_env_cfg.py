from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab_tasks.manager_based.manipulation.bernie_proj.ycb.config.rby1.joint_pos_pick_and_place_env_cfg import JointPickAndPlaceEnvCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
import os
import pathlib
workspace = pathlib.Path(os.getenv("WORKSPACE_FOLDER", pathlib.Path.cwd()))


@configclass
class YCBRBY1PickAndPlaceEnvCfg(JointPickAndPlaceEnvCfg):
    """
    Class Description: This is class that all of our enviornments will extend
                         - it enables a user to specify the start and goal object
                           that a user wants to use for a task
                       This class itself extends FrankaYCBEnvCfg
                         - this class sets up the enviornment
    """
    def __init__(self, object_name: str, scale=(1.0, 1.0, 1.0)):
        """
        parameters:
        - object_name: string for the YCB object name
        - scale: scaling of the object
        """
        self.object_name = object_name
        self.scale = scale
        super().__init__()

    def __post_init__(self):
        super().__post_init__()

        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0, 0.75], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="/home/davin123/PoliGen/assets/box.usd",
                scale=(0.8, 0.8, 0.8),
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

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=str(workspace) + str("/assets/ycb/") + str(self.object_name) + ".usd",
                scale=(0.8, 0.8, 0.8),
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


"""
Description: Below are all of the classes
that represent all of the enviornments that
we will be working with.

Enviornment structure:
- Robot: Panda Franka
- Object
- Goal State
- Table: where actions
         will take place
"""

# banana, block, bottle, cup, dice, pitcher, rubrik

@configclass
class RBY1YCBPickAndPlaceBananaEnvCfg(YCBRBY1PickAndPlaceEnvCfg):
    def __init__(self):
        super().__init__("banana")


@configclass
class RBY1YCBPickAndPlaceBlockEnvCfg(YCBRBY1PickAndPlaceEnvCfg):
    def __init__(self):
        super().__init__("block")


@configclass
class RBY1YCBPickAndPlaceBottleEnvCfg(YCBRBY1PickAndPlaceEnvCfg):
    def __init__(self):
        super().__init__("bottle")


@configclass
class RBY1YCBPickAndPlaceCupEnvCfg(YCBRBY1PickAndPlaceEnvCfg):
    def __init__(self):
        super().__init__("cup")


@configclass
class RBY1YCBPickAndPlaceDiceEnvCfg(YCBRBY1PickAndPlaceEnvCfg):
    def __init__(self):
        super().__init__("dice")


@configclass
class RBY1YCBPickAndPlacePitcherEnvCfg(YCBRBY1PickAndPlaceEnvCfg):
    def __init__(self):
        super().__init__("pitcher")


@configclass
class RBY1YCBPickAndPlaceRubriksEnvCfg(YCBRBY1PickAndPlaceEnvCfg):
    def __init__(self):
        super().__init__("rubrik")

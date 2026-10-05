from pathlib import Path
import threading
import time

import numpy as np
import pybullet
import pybullet_data

from .config import (
    ASSETS_DIR,
    BOUNDS,
    PICK_TARGETS,
    PIXEL_SIZE,
    PLACE_TARGETS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = ASSETS_DIR


# ============================================================================
# LOCAL ENVIRONMENT CONSTANTS
# ============================================================================

# Object colors are specific to the PyBullet rendering layer and therefore
# remain local to the environment. Shared task/workspace configuration lives
# in src/config.py and is imported above.
COLORS = {
    "blue": (78 / 255, 121 / 255, 167 / 255, 255 / 255),
    "red": (255 / 255, 87 / 255, 89 / 255, 255 / 255),
    "green": (89 / 255, 169 / 255, 79 / 255, 255 / 255),
    "yellow": (237 / 255, 201 / 255, 72 / 255, 255 / 255),
    "orange": (242 / 255, 142 / 255, 43 / 255, 255 / 255),
    "default": (255 / 255, 255 / 255, 255 / 255),
}


# Object dimensions.
BLOCK_SIZES = {
    "small": [0.017, 0.017, 0.02],
    "medium": [0.02, 0.02, 0.02],
    "big": [0.035, 0.035, 0.02],
}


# ============================================================================
# TASK-SUCCESS EVALUATION PARAMETERS
# ============================================================================

# These parameters are deliberately separated from robot motion.
# The robot can complete a motion without actually completing the requested
# manipulation, so task success is evaluated from the final simulated state.

PLACEMENT_XY_TOLERANCE = 0.035
BOWL_XY_TOLERANCE = 0.050
SPATIAL_XY_TOLERANCE = 0.040

MIN_PICK_DISPLACEMENT = 0.040
MIN_LIFT_HEIGHT = 0.040

VERTICAL_TOLERANCE = 0.015


# ============================================================================
# ROBOTIQ 2F85 GRIPPER
# ============================================================================

class Robotiq2F85:
    """Gripper handling for the Robotiq 2F85."""

    def __init__(self, robot, tool):
        self.robot = robot
        self.tool = tool

        pos = [
            0.1339999999999999,
            -0.49199999999872496,
            0.5,
        ]

        rot = pybullet.getQuaternionFromEuler(
            [np.pi, 0, np.pi]
        )

        urdf = str(
            ASSETS_ROOT
            / "robotiq_2f_85"
            / "robotiq_2f_85.urdf"
        )

        self.body = pybullet.loadURDF(
            urdf,
            pos,
            rot,
        )

        self.n_joints = pybullet.getNumJoints(
            self.body
        )

        self.activated = False

        # Connect gripper base to robot tool.
        pybullet.createConstraint(
            self.robot,
            tool,
            self.body,
            0,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, -0.07],
            childFrameOrientation=pybullet.getQuaternionFromEuler(
                [0, 0, np.pi / 2]
            ),
        )

        # Set friction coefficients for gripper fingers.
        for i in range(pybullet.getNumJoints(self.body)):
            pybullet.changeDynamics(
                self.body,
                i,
                lateralFriction=10.0,
                spinningFriction=1.0,
                rollingFriction=1.0,
                frictionAnchor=True,
            )

        # Background thread for gripper constraints.
        self.motor_joint = 1
        self.running = True

        self.constraints_thread = threading.Thread(
            target=self.step,
            daemon=True,
        )

        self.constraints_thread.start()

    def step(self):
        """Continuously enforce gripper finger constraints."""

        while self.running:
            try:
                currj = [
                    pybullet.getJointState(
                        self.body,
                        i,
                    )[0]
                    for i in range(self.n_joints)
                ]

                indj = [6, 3, 8, 5, 10]

                targj = [
                    currj[1],
                    -currj[1],
                    -currj[1],
                    currj[1],
                    currj[1],
                ]

                pybullet.setJointMotorControlArray(
                    self.body,
                    indj,
                    pybullet.POSITION_CONTROL,
                    targj,
                    positionGains=np.ones(5),
                )

            except Exception:
                return

            time.sleep(0.001)

    def activate(self):
        """Close gripper fingers."""

        pybullet.setJointMotorControl2(
            self.body,
            self.motor_joint,
            pybullet.VELOCITY_CONTROL,
            targetVelocity=1,
            force=10,
        )

        self.activated = True

    def release(self):
        """Open gripper fingers."""

        pybullet.setJointMotorControl2(
            self.body,
            self.motor_joint,
            pybullet.VELOCITY_CONTROL,
            targetVelocity=-1,
            force=10,
        )

        self.activated = False

    def detect_contact(self):
        """Check whether the gripper has external contact."""

        obj, _, ray_frac = self.check_proximity()

        if self.activated:
            empty = self.grasp_width() < 0.01

            cbody = self.body if empty else obj

            if obj == self.body or obj == 0:
                return False

            return self.external_contact(cbody)

    def external_contact(self, body=None):
        """Return whether the body contacts something other than gripper."""

        if body is None:
            body = self.body

        pts = pybullet.getContactPoints(
            bodyA=body
        )

        pts = [
            pt
            for pt in pts
            if pt[2] != self.body
        ]

        return len(pts) > 0

    def check_grasp(self):
        """Check whether the gripper opening indicates a grasp."""

        # The previous implementation called an undefined ``moving()``
        # helper. Grasp success is already evaluated from the gripper width
        # elsewhere in the environment, so no undefined motion wait is
        # necessary here.
        return self.grasp_width() > 0.01

    def grasp_width(self):
        """Return current gripper opening width."""

        lpad = np.array(
            pybullet.getLinkState(
                self.body,
                4,
            )[0]
        )

        rpad = np.array(
            pybullet.getLinkState(
                self.body,
                9,
            )[0]
        )

        dist = (
            np.linalg.norm(lpad - rpad)
            - 0.047813
        )

        return dist

    def check_proximity(self):
        """Check proximity between end effector and gripper."""

        ee_pos = np.array(
            pybullet.getLinkState(
                self.robot,
                self.tool,
            )[0]
        )

        tool_pos = np.array(
            pybullet.getLinkState(
                self.body,
                0,
            )[0]
        )

        vec = (
            tool_pos - ee_pos
        ) / np.linalg.norm(
            tool_pos - ee_pos
        )

        ee_targ = ee_pos + vec

        ray_data = pybullet.rayTest(
            ee_pos,
            ee_targ,
        )[0]

        obj = ray_data[0]
        link = ray_data[1]
        ray_frac = ray_data[2]

        return obj, link, ray_frac

    def stop(self):
        """Stop the background gripper constraint thread safely."""

        self.running = False

        if (
            hasattr(self, "constraints_thread")
            and self.constraints_thread.is_alive()
            and threading.current_thread()
            is not self.constraints_thread
        ):
            self.constraints_thread.join(
                timeout=2.0
            )


# ============================================================================
# PICK-AND-PLACE ENVIRONMENT
# ============================================================================

class PickPlaceEnv:
    """PyBullet environment for robotic pick-and-place manipulation."""

    def __init__(self, gui=False):
        self.dt = 1 / 480
        self.gui = gui
        self.sim_step = 0

        # Start PyBullet.
        if pybullet.isConnected():
            pybullet.disconnect()

        pybullet.connect(
            pybullet.GUI
            if self.gui
            else pybullet.DIRECT
        )

        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_GUI,
            0,
        )

        pybullet.setPhysicsEngineParameter(
            enableFileCaching=0
        )

        pybullet.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        pybullet.setTimeStep(
            self.dt
        )

        self.home_joints = (
            np.pi / 2,
            -np.pi / 2,
            np.pi / 2,
            -np.pi / 2,
            3 * np.pi / 2,
            0,
        )

        self.home_ee_euler = (
            np.pi,
            0,
            np.pi,
        )

        self.ee_link_id = 9
        self.tip_link_id = 10

        self.gripper = None

        # State used by task-success evaluation.
        self.initial_obj_positions = {}
        self.last_task_result = None
        self.last_action = None
        self.pick_lift_position = None

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self, config):
        """Reset the simulation and build the scene described by config."""

        if not isinstance(config, dict):
            raise TypeError(
                "config must be a dictionary with "
                "'pick' and 'place' lists."
            )

        if "pick" not in config or "place" not in config:
            raise ValueError(
                "config must contain both 'pick' and 'place' keys."
            )

        if (
            not isinstance(config["pick"], (list, tuple))
            or not isinstance(config["place"], (list, tuple))
        ):
            raise TypeError(
                "'pick' and 'place' must be lists or tuples."
            )

        if len(config["pick"]) == 0:
            raise ValueError(
                "'pick' must contain at least one target."
            )

        # ``place`` may be empty.
        if (
            len(config["pick"]) > 1
            and len(config["place"]) > 1
        ):
            raise ValueError(
                "Use at most one explicit place target "
                "per pick-and-place episode; multiple "
                "pick entries are allowed as scene candidates."
            )

        required_assets = {
            "UR5e": ASSETS_ROOT / "ur5e" / "ur5e.urdf",
            "Robotiq 2F85": (
                ASSETS_ROOT
                / "robotiq_2f_85"
                / "robotiq_2f_85.urdf"
            ),
            "Bowl": ASSETS_ROOT / "bowl" / "bowl.urdf",
        }

        missing_assets = [
            name
            for name, path in required_assets.items()
            if not path.exists()
        ]

        if missing_assets:
            raise FileNotFoundError(
                "Missing simulation asset(s): "
                + ", ".join(missing_assets)
            )

        pybullet.resetSimulation(
            pybullet.RESET_USE_DEFORMABLE_WORLD
        )

        pybullet.setGravity(
            0,
            0,
            -9.8,
        )

        self.cache_video = []
        self.initial_obj_positions = {}
        self.last_task_result = None
        self.last_action = None
        self.pick_lift_position = None

        # Temporarily disable rendering.
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RENDERING,
            0,
        )

        # Add robot.
        pybullet.loadURDF(
            str(
                Path(
                    pybullet_data.getDataPath()
                )
                / "plane.urdf"
            ),
            [0, 0, -0.001],
        )

        self.robot_id = pybullet.loadURDF(
            str(
                ASSETS_ROOT
                / "ur5e"
                / "ur5e.urdf"
            ),
            [0, 0, 0],
            flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )

        self.ghost_id = pybullet.loadURDF(
            str(
                ASSETS_ROOT
                / "ur5e"
                / "ur5e.urdf"
            ),
            [0, 0, -10],
        )

        self.joint_ids = [
            pybullet.getJointInfo(
                self.robot_id,
                i,
            )
            for i in range(
                pybullet.getNumJoints(
                    self.robot_id
                )
            )
        ]

        self.joint_ids = [
            j[0]
            for j in self.joint_ids
            if j[2] == pybullet.JOINT_REVOLUTE
        ]

        # Move robot to home configuration.
        for i in range(
            len(self.joint_ids)
        ):
            pybullet.resetJointState(
                self.robot_id,
                self.joint_ids[i],
                self.home_joints[i],
            )

        # Add gripper.
        if self.gripper is not None:
            while (
                self.gripper.constraints_thread.is_alive()
            ):
                self.gripper.stop()

        self.gripper = Robotiq2F85(
            self.robot_id,
            self.ee_link_id,
        )

        self.gripper.release()

        # Add workspace.
        plane_shape = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[
                0.3,
                0.3,
                0.001,
            ],
        )

        plane_visual = pybullet.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=[
                0.3,
                0.3,
                0.001,
            ],
        )

        plane_id = pybullet.createMultiBody(
            0,
            plane_shape,
            plane_visual,
            basePosition=[
                0,
                -0.5,
                0,
            ],
        )

        pybullet.changeVisualShape(
            plane_id,
            -1,
            rgbaColor=[
                0.2,
                0.2,
                0.2,
                1.0,
            ],
        )

        # Load objects.
        self.config = config
        self.obj_name_to_id = {}

        obj_names = (
            list(self.config["pick"])
            + list(self.config["place"])
        )

        obj_xyz = np.zeros(
            (0, 3)
        )

        for obj_name in obj_names:

            if (
                "block" in obj_name
                or "bowl" in obj_name
                or "sphere" in obj_name
            ):

                # Random position at least 15 cm from
                # previously placed objects.
                while True:

                    rand_x = np.random.uniform(
                        BOUNDS[0, 0] + 0.1,
                        BOUNDS[0, 1] - 0.1,
                    )

                    rand_y = np.random.uniform(
                        BOUNDS[1, 0] + 0.1,
                        BOUNDS[1, 1] - 0.1,
                    )

                    rand_xyz = np.float32(
                        [
                            rand_x,
                            rand_y,
                            0.03,
                        ]
                    ).reshape(
                        1,
                        3,
                    )

                    if len(obj_xyz) == 0:
                        obj_xyz = np.concatenate(
                            (
                                obj_xyz,
                                rand_xyz,
                            ),
                            axis=0,
                        )
                        break

                    nn_dist = np.min(
                        np.linalg.norm(
                            obj_xyz - rand_xyz,
                            axis=1,
                        )
                    ).squeeze()

                    if nn_dist > 0.15:
                        obj_xyz = np.concatenate(
                            (
                                obj_xyz,
                                rand_xyz,
                            ),
                            axis=0,
                        )
                        break

                object_color = COLORS[
                    obj_name.split(" ")[0]
                ]

                object_type = (
                    obj_name.split(" ")[1]
                )

                object_position = (
                    rand_xyz.squeeze()
                )

                if object_type == "block":

                    object_shape = (
                        pybullet.createCollisionShape(
                            pybullet.GEOM_BOX,
                            halfExtents=[
                                0.02,
                                0.02,
                                0.02,
                            ],
                        )
                    )

                    object_visual = (
                        pybullet.createVisualShape(
                            pybullet.GEOM_BOX,
                            halfExtents=[
                                0.02,
                                0.02,
                                0.02,
                            ],
                        )
                    )

                    object_id = (
                        pybullet.createMultiBody(
                            0.01,
                            object_shape,
                            object_visual,
                            basePosition=object_position,
                        )
                    )

                elif object_type == "bowl":

                    object_position[2] = 0

                    object_id = (
                        pybullet.loadURDF(
                            str(
                                ASSETS_ROOT
                                / "bowl"
                                / "bowl.urdf"
                            ),
                            object_position,
                            useFixedBase=1,
                        )
                    )

                elif object_type == "sphere":

                    radius = 0.02

                    object_shape = (
                        pybullet.createCollisionShape(
                            pybullet.GEOM_SPHERE,
                            radius=radius,
                        )
                    )

                    object_visual = (
                        pybullet.createVisualShape(
                            pybullet.GEOM_SPHERE,
                            radius=radius,
                        )
                    )

                    object_id = (
                        pybullet.createMultiBody(
                            0.01,
                            object_shape,
                            object_visual,
                            basePosition=object_position,
                        )
                    )

                pybullet.changeVisualShape(
                    object_id,
                    -1,
                    rgbaColor=object_color,
                )

                self.obj_name_to_id[
                    obj_name
                ] = object_id

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RENDERING,
            1,
        )

        for _ in range(200):
            pybullet.stepSimulation()

        # Record settled initial state.
        self.initial_obj_positions = {
            name: np.float32(
                pybullet.getBasePositionAndOrientation(
                    object_id
                )[0]
            )
            for name, object_id
            in self.obj_name_to_id.items()
        }

        return self.get_observation()

    # ========================================================================
    # ROBOT CONTROL
    # ========================================================================

    def servoj(self, joints):
        """Move to target joint positions with position control."""

        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[0.01] * 6,
        )

    def movep(self, position):
        """Move end effector to target XYZ position."""

        joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tip_link_id,
            targetPosition=position,
            targetOrientation=pybullet.getQuaternionFromEuler(
                self.home_ee_euler
            ),
            maxNumIterations=100,
        )

        self.servoj(joints)

    # ========================================================================
    # EXECUTION
    # ========================================================================

    def step(self, action=None):
        """Execute one pick-and-place action and evaluate task success.

        ``action`` must contain:

            pick  -> XYZ coordinates
            place -> XYZ coordinates

        It may additionally contain:

            pick_name  -> semantic object name
            place_name -> semantic target name

        When semantic names are supplied, they are used directly for task
        evaluation. This is important for multi-step tasks where two objects
        can occupy nearly identical coordinates after stacking.

        If semantic names are not supplied, the environment falls back to
        coordinate-based target inference for compatibility with existing
        callers.
        """

        if action is None or not isinstance(action, dict):
            raise TypeError(
                "action must be a dictionary containing "
                "'pick' and 'place'."
            )

        if (
            "pick" not in action
            or "place" not in action
        ):
            raise ValueError(
                "action must contain both "
                "'pick' and 'place'."
            )

        pick_xyz = np.asarray(
            action["pick"],
            dtype=np.float32,
        ).copy()

        place_xyz = np.asarray(
            action["place"],
            dtype=np.float32,
        ).copy()

        if (
            pick_xyz.shape != (3,)
            or place_xyz.shape != (3,)
        ):
            raise ValueError(
                "'pick' and 'place' must each contain "
                "exactly three XYZ values."
            )

        if (
            not np.all(np.isfinite(pick_xyz))
            or not np.all(np.isfinite(place_xyz))
        ):
            raise ValueError(
                "'pick' and 'place' must contain "
                "finite XYZ values."
            )

        # --------------------------------------------------------------------
        # NEW: Optional semantic action names
        # --------------------------------------------------------------------

        semantic_pick_name = action.get(
            "pick_name"
        )

        semantic_place_name = action.get(
            "place_name"
        )

        if (
            semantic_pick_name is not None
            and not isinstance(
                semantic_pick_name,
                str,
            )
        ):
            raise TypeError(
                "'pick_name' must be a string "
                "when provided."
            )

        if (
            semantic_place_name is not None
            and not isinstance(
                semantic_place_name,
                str,
            )
        ):
            raise TypeError(
                "'place_name' must be a string "
                "when provided."
            )

        if semantic_pick_name is not None:

            semantic_pick_name = (
                semantic_pick_name
                .strip()
                .lower()
            )

            if (
                semantic_pick_name
                not in self.obj_name_to_id
            ):
                raise ValueError(
                    f"Unknown pick_name: "
                    f"{semantic_pick_name}"
                )

        if semantic_place_name is not None:

            semantic_place_name = (
                semantic_place_name
                .strip()
                .lower()
            )

            if (
                semantic_place_name
                not in self.obj_name_to_id
                and semantic_place_name
                not in PLACE_TARGETS
            ):
                raise ValueError(
                    f"Unknown place_name: "
                    f"{semantic_place_name}"
                )

        # --------------------------------------------------------------------
        # Determine semantic targets.
        #
        # If the caller supplied semantic names, those names take priority.
        # Otherwise retain the original coordinate-based inference.
        # --------------------------------------------------------------------

        inferred_pick_name, inferred_place_name = (
            self._infer_action_targets(
                pick_xyz,
                place_xyz,
            )
        )

        pick_name = (
            semantic_pick_name
            if semantic_pick_name is not None
            else inferred_pick_name
        )

        place_name = (
            semantic_place_name
            if semantic_place_name is not None
            else inferred_place_name
        )

        # Record the complete action for later evaluation.
        self.last_action = {
            "pick": pick_xyz.copy(),
            "place": place_xyz.copy(),
            "pick_name": pick_name,
            "place_name": place_name,
        }

        self.pick_lift_position = None
        self.last_task_result = None

        # --------------------------------------------------------------------
        # Low-level motion primitive
        # --------------------------------------------------------------------

        hover_xyz = (
            pick_xyz.copy()
            + np.float32([0, 0, 0.2])
        )

        pick_xyz[2] = 0.03
        place_xyz[2] = 0.15

        # Move above object.
        ee_xyz = np.float32(
            pybullet.getLinkState(
                self.robot_id,
                self.tip_link_id,
            )[0]
        )

        while (
            np.linalg.norm(
                hover_xyz - ee_xyz
            )
            > 0.01
        ):
            self.movep(hover_xyz)
            self.step_sim_and_render()

            ee_xyz = np.float32(
                pybullet.getLinkState(
                    self.robot_id,
                    self.tip_link_id,
                )[0]
            )

        # Move to object.
        while (
            np.linalg.norm(
                pick_xyz - ee_xyz
            )
            > 0.01
        ):
            self.movep(pick_xyz)
            self.step_sim_and_render()

            ee_xyz = np.float32(
                pybullet.getLinkState(
                    self.robot_id,
                    self.tip_link_id,
                )[0]
            )

        # Pick up object.
        self.gripper.activate()

        for _ in range(240):
            self.step_sim_and_render()

        # Lift object.
        while (
            np.linalg.norm(
                hover_xyz - ee_xyz
            )
            > 0.01
        ):
            self.movep(hover_xyz)
            self.step_sim_and_render()

            ee_xyz = np.float32(
                pybullet.getLinkState(
                    self.robot_id,
                    self.tip_link_id,
                )[0]
            )

        if pick_name in self.obj_name_to_id:

            self.pick_lift_position = np.float32(
                pybullet.getBasePositionAndOrientation(
                    self.obj_name_to_id[
                        pick_name
                    ]
                )[0]
            )

        # Move to placement location.
        while (
            np.linalg.norm(
                place_xyz - ee_xyz
            )
            > 0.01
        ):
            self.movep(place_xyz)
            self.step_sim_and_render()

            ee_xyz = np.float32(
                pybullet.getLinkState(
                    self.robot_id,
                    self.tip_link_id,
                )[0]
            )

        # Place object.
        while (
            not self.gripper.detect_contact()
            and place_xyz[2] > 0.03
        ):
            place_xyz[2] -= 0.001

            self.movep(place_xyz)

            for _ in range(3):
                self.step_sim_and_render()

        self.gripper.release()

        for _ in range(240):
            self.step_sim_and_render()

        # Retreat vertically.
        place_xyz[2] = 0.2

        ee_xyz = np.float32(
            pybullet.getLinkState(
                self.robot_id,
                self.tip_link_id,
            )[0]
        )

        while (
            np.linalg.norm(
                place_xyz - ee_xyz
            )
            > 0.01
        ):
            self.movep(place_xyz)
            self.step_sim_and_render()

            ee_xyz = np.float32(
                pybullet.getLinkState(
                    self.robot_id,
                    self.tip_link_id,
                )[0]
            )

        # Return home.
        home_xyz = np.float32(
            [0, -0.5, 0.2]
        )

        while (
            np.linalg.norm(
                home_xyz - ee_xyz
            )
            > 0.01
        ):
            self.movep(home_xyz)
            self.step_sim_and_render()

            ee_xyz = np.float32(
                pybullet.getLinkState(
                    self.robot_id,
                    self.tip_link_id,
                )[0]
            )

        # --------------------------------------------------------------------
        # TASK EVALUATION
        # --------------------------------------------------------------------

        self.last_task_result = (
            self.check_task_success(
                pick_name=pick_name,
                place_name=place_name,
            )
        )

        observation = self.get_observation()

        reward = self.get_reward()

        done = bool(
            self.last_task_result["success"]
        )

        info = {
            "task_success": bool(
                self.last_task_result["success"]
            ),
            "task_result": self.last_task_result,
        }

        return (
            observation,
            reward,
            done,
            info,
        )

    # ========================================================================
    # SEMANTIC TARGET INFERENCE
    # ========================================================================

    def _infer_action_targets(
        self,
        pick_xyz,
        place_xyz,
    ):
        """Infer semantic pick/place objects from action coordinates.

        This remains as a compatibility fallback.

        Pick identification uses initial scene positions.

        Place identification uses current object positions so that sequential
        multi-step tasks remain valid after earlier manipulations.
        """

        if not self.obj_name_to_id:
            raise RuntimeError(
                "No scene objects are available "
                "for action evaluation."
            )

        # --------------------------------------------------------------------
        # Pick target
        # --------------------------------------------------------------------

        pick_candidates = [
            name
            for name in self.config.get(
                "pick",
                [],
            )
            if name in self.obj_name_to_id
        ]

        if not pick_candidates:
            pick_candidates = list(
                self.obj_name_to_id.keys()
            )

        def initial_xy(name):
            return self.initial_obj_positions[
                name
            ][:2]

        pick_name = min(
            pick_candidates,
            key=lambda name: float(
                np.linalg.norm(
                    initial_xy(name)
                    - pick_xyz[:2]
                )
            ),
        )

        # --------------------------------------------------------------------
        # Place target
        # --------------------------------------------------------------------

        explicit_places = [
            name
            for name in self.config.get(
                "place",
                [],
            )
            if (
                name in self.obj_name_to_id
                or name in PLACE_TARGETS
            )
        ]

        if len(explicit_places) == 1:

            place_name = explicit_places[0]

        elif len(explicit_places) > 1:

            place_name = min(
                explicit_places,
                key=lambda name:
                self._distance_to_target_xy(
                    name,
                    place_xyz[:2],
                ),
            )

        else:

            # No explicit target was supplied.
            #
            # Infer destination from current object positions.

            place_candidates = [
                name
                for name in self.obj_name_to_id.keys()
                if name != pick_name
            ]

            if not place_candidates:
                place_candidates = list(
                    self.obj_name_to_id.keys()
                )

            def current_xy(name):

                position = (
                    self._get_object_position(
                        name
                    )
                )

                if position is None:
                    return np.array(
                        [np.inf, np.inf],
                        dtype=np.float32,
                    )

                return position[:2]

            place_name = min(
                place_candidates,
                key=lambda name: float(
                    np.linalg.norm(
                        current_xy(name)
                        - place_xyz[:2]
                    )
                ),
            )

        return pick_name, place_name

    # ========================================================================
    # TARGET POSITION HELPERS
    # ========================================================================

    def _distance_to_target_xy(
        self,
        target_name,
        xy,
    ):
        """Return XY distance to a named object/workspace target."""

        if target_name in self.obj_name_to_id:

            target_xy = (
                self.initial_obj_positions[
                    target_name
                ][:2]
            )

        elif (
            target_name in PLACE_TARGETS
            and PLACE_TARGETS[target_name]
            is not None
        ):

            target_xy = np.asarray(
                PLACE_TARGETS[target_name],
                dtype=np.float32,
            )[:2]

        else:
            return float("inf")

        return float(
            np.linalg.norm(
                target_xy
                - np.asarray(
                    xy,
                    dtype=np.float32,
                )
            )
        )

    def _get_object_position(
        self,
        object_name,
    ):
        """Return current XYZ position of a simulated object."""

        if object_name not in self.obj_name_to_id:
            return None

        position = (
            pybullet.getBasePositionAndOrientation(
                self.obj_name_to_id[
                    object_name
                ]
            )[0]
        )

        return np.float32(position)

    def _get_place_target_position(
        self,
        place_name,
    ):
        """Return XYZ location used to evaluate placement."""

        if place_name in self.obj_name_to_id:

            return self._get_object_position(
                place_name
            )

        target = PLACE_TARGETS.get(
            place_name
        )

        if target is None:
            return None

        return np.float32(target)

    def _get_placement_tolerance(
        self,
        place_name,
    ):
        """Return XY placement tolerance for target type."""

        if "bowl" in place_name:
            return BOWL_XY_TOLERANCE

        if place_name in self.obj_name_to_id:
            return PLACEMENT_XY_TOLERANCE

        return SPATIAL_XY_TOLERANCE

    # ========================================================================
    # TASK SUCCESS EVALUATION
    # ========================================================================

    def check_task_success(
        self,
        pick_name=None,
        place_name=None,
    ):
        """Evaluate whether the requested manipulation succeeded.

        Success requires:

        1. The picked object was lifted meaningfully.
        2. The picked object moved from its initial position.
        3. The final XY position is within target tolerance.
        4. The final Z position is physically compatible with the target.
        """

        if (
            pick_name is None
            or place_name is None
        ):

            inferred_pick, inferred_place = (
                self._infer_action_targets(
                    self.last_action["pick"],
                    self.last_action["place"],
                )
            )

            if pick_name is None:
                pick_name = inferred_pick

            if place_name is None:
                place_name = inferred_place

        initial_position = (
            self.initial_obj_positions.get(
                pick_name
            )
        )

        final_position = (
            self._get_object_position(
                pick_name
            )
        )

        target_position = (
            self._get_place_target_position(
                place_name
            )
        )

        result = {
            "success": False,
            "pick_name": pick_name,
            "place_name": place_name,
            "grasp_success": False,
            "movement_success": False,
            "placement_success": False,
            "initial_position": None,
            "lift_position": None,
            "final_position": None,
            "target_position": None,
            "lift_height": None,
            "pick_displacement": None,
            "placement_xy_error": None,
            "vertical_error": None,
            "placement_tolerance": float(
                self._get_placement_tolerance(
                    place_name
                )
            ),
            "reason": "",
            "action_pick_xyz": (
                [
                    float(v)
                    for v in self.last_action["pick"]
                ]
                if self.last_action is not None
                else None
            ),
            "action_place_xyz": (
                [
                    float(v)
                    for v in self.last_action["place"]
                ]
                if self.last_action is not None
                else None
            ),
        }

        if initial_position is None:

            result["reason"] = (
                f"No initial position recorded "
                f"for '{pick_name}'."
            )

            return result

        if final_position is None:

            result["reason"] = (
                f"Pick target '{pick_name}' "
                f"is not present in the scene."
            )

            return result

        if target_position is None:

            result["reason"] = (
                f"Place target '{place_name}' "
                f"is not present or has no position."
            )

            return result

        result["initial_position"] = [
            float(v)
            for v in initial_position
        ]

        result["final_position"] = [
            float(v)
            for v in final_position
        ]

        result["target_position"] = [
            float(v)
            for v in target_position
        ]

        if self.pick_lift_position is not None:

            result["lift_position"] = [
                float(v)
                for v in self.pick_lift_position
            ]

            lift_height = float(
                self.pick_lift_position[2]
                - initial_position[2]
            )

        else:

            lift_height = 0.0

        pick_displacement = float(
            np.linalg.norm(
                final_position
                - initial_position
            )
        )

        placement_xy_error = float(
            np.linalg.norm(
                final_position[:2]
                - target_position[:2]
            )
        )

        vertical_error = float(
            final_position[2]
            - target_position[2]
        )

        result["lift_height"] = lift_height
        result["pick_displacement"] = (
            pick_displacement
        )
        result["placement_xy_error"] = (
            placement_xy_error
        )
        result["vertical_error"] = (
            vertical_error
        )

        grasp_success = (
            lift_height
            >= MIN_LIFT_HEIGHT
        )

        movement_success = (
            pick_displacement
            >= MIN_PICK_DISPLACEMENT
        )

        tolerance = (
            self._get_placement_tolerance(
                place_name
            )
        )

        placement_success = (
            placement_xy_error <= tolerance
            and vertical_error
            >= -VERTICAL_TOLERANCE
        )

        result["grasp_success"] = bool(
            grasp_success
        )

        result["movement_success"] = bool(
            movement_success
        )

        result["placement_success"] = bool(
            placement_success
        )

        result["success"] = bool(
            grasp_success
            and movement_success
            and placement_success
        )

        if result["success"]:

            result["reason"] = (
                "Object was lifted, moved, "
                "and placed within the target tolerance."
            )

        elif not grasp_success:

            result["reason"] = (
                "The object was not lifted enough "
                "to confirm a successful grasp."
            )

        elif not movement_success:

            result["reason"] = (
                "The object did not move far enough "
                "from its initial position."
            )

        elif not placement_success:

            result["reason"] = (
                "The object reached the placement area "
                "but is outside the allowed "
                "XY/vertical target tolerance."
            )

        return result

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def set_alpha_transparency(
        self,
        alpha: float,
    ):

        for id in range(20):

            visual_shape_data = (
                pybullet.getVisualShapeData(id)
            )

            for i in range(
                len(visual_shape_data)
            ):

                (
                    object_id,
                    link_index,
                    _,
                    _,
                    _,
                    _,
                    _,
                    rgba_color,
                ) = visual_shape_data[i]

                rgba_color = (
                    list(rgba_color[0:3])
                    + [alpha]
                )

                pybullet.changeVisualShape(
                    self.robot_id,
                    linkIndex=i,
                    rgbaColor=rgba_color,
                )

                pybullet.changeVisualShape(
                    self.gripper.body,
                    linkIndex=i,
                    rgbaColor=rgba_color,
                )

    def step_sim_and_render(self):

        pybullet.stepSimulation()

        self.sim_step += 1

        if self.sim_step % 60 == 0:

            self.cache_video.append(
                self.get_camera_image()
            )

    def get_camera_image(self):

        image_size = (
            240,
            240,
        )

        intrinsics = (
            120.0,
            0,
            120.0,
            0,
            120.0,
            120.0,
            0,
            0,
            1,
        )

        color, _, _, _, _ = (
            self.render_image(
                image_size,
                intrinsics,
            )
        )

        return color

    def get_camera_image_top(
        self,
        image_size=(240, 240),
        intrinsics=(
            2000.0,
            0,
            2000.0,
            0,
            2000.0,
            2000.0,
            0,
            0,
            1,
        ),
        position=(0, -0.5, 5),
        orientation=(
            0,
            np.pi,
            -np.pi / 2,
        ),
        zrange=(0.01, 1.0),
        set_alpha=True,
    ):

        set_alpha and self.set_alpha_transparency(
            0
        )

        color, _, _, _, _ = (
            self.render_image_top(
                image_size,
                intrinsics,
                position,
                orientation,
                zrange,
            )
        )

        set_alpha and self.set_alpha_transparency(
            1
        )

        return color

    # ========================================================================
    # CAMERA
    # ========================================================================

    def render_image(
        self,
        image_size=(720, 720),
        intrinsics=(
            360.0,
            0,
            360.0,
            0,
            360.0,
            360.0,
            0,
            0,
            1,
        ),
    ):

        position = (
            0,
            -0.85,
            0.4,
        )

        orientation = (
            np.pi / 4 + np.pi / 48,
            np.pi,
            np.pi,
        )

        orientation = (
            pybullet.getQuaternionFromEuler(
                orientation
            )
        )

        zrange = (
            0.01,
            10.0,
        )

        noise = True

        lookdir = np.float32(
            [0, 0, 1]
        ).reshape(
            3,
            1,
        )

        updir = np.float32(
            [0, -1, 0]
        ).reshape(
            3,
            1,
        )

        rotation = (
            pybullet.getMatrixFromQuaternion(
                orientation
            )
        )

        rotm = np.float32(
            rotation
        ).reshape(
            3,
            3,
        )

        lookdir = (
            rotm @ lookdir
        ).reshape(-1)

        updir = (
            rotm @ updir
        ).reshape(-1)

        lookat = position + lookdir

        focal_len = intrinsics[0]

        fovh = (
            image_size[0] / 2
        ) / focal_len

        fovh = (
            180
            * np.arctan(fovh)
            * 2
            / np.pi
        )

        aspect_ratio = (
            image_size[1]
            / image_size[0]
        )

        viewm = pybullet.computeViewMatrix(
            position,
            lookat,
            updir,
        )

        projm = (
            pybullet.computeProjectionMatrixFOV(
                fovh,
                aspect_ratio,
                0.01,
                10.0,
            )
        )

        _, _, color, depth, segm = (
            pybullet.getCameraImage(
                width=image_size[1],
                height=image_size[0],
                viewMatrix=viewm,
                projectionMatrix=projm,
                shadow=1,
                flags=(
                    pybullet
                    .ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                ),
                renderer=(
                    pybullet
                    .ER_BULLET_HARDWARE_OPENGL
                ),
            )
        )

        color_image_size = (
            image_size[0],
            image_size[1],
            4,
        )

        color = np.array(
            color,
            dtype=np.uint8,
        ).reshape(
            color_image_size
        )

        color = color[:, :, :3]

        if noise:

            color = np.int32(
                color
            )

            color += np.int32(
                np.random.normal(
                    0,
                    3,
                    color.shape,
                )
            )

            color = np.uint8(
                np.clip(
                    color,
                    0,
                    255,
                )
            )

        depth_image_size = (
            image_size[0],
            image_size[1],
        )

        zbuffer = np.float32(
            depth
        ).reshape(
            depth_image_size
        )

        znear, zfar = (
            0.01,
            10.0,
        )

        depth = (
            zfar
            + znear
            - (
                2 * zbuffer - 1
            )
            * (
                zfar - znear
            )
        )

        depth = (
            2
            * znear
            * zfar
        ) / depth

        if noise:

            depth += np.random.normal(
                0,
                0.003,
                depth.shape,
            )

        intrinsics = np.float32(
            intrinsics
        ).reshape(
            3,
            3,
        )

        return (
            color,
            depth,
            position,
            orientation,
            intrinsics,
        )

    def render_image_top(
        self,
        image_size=(240, 240),
        intrinsics=(
            2000.0,
            0,
            2000.0,
            0,
            2000.0,
            2000.0,
            0,
            0,
            1,
        ),
        position=(0, -0.5, 5),
        orientation=(
            0,
            np.pi,
            -np.pi / 2,
        ),
        zrange=(0.01, 1.0),
    ):

        orientation = (
            pybullet.getQuaternionFromEuler(
                orientation
            )
        )

        noise = True

        lookdir = np.float32(
            [0, 0, 1]
        ).reshape(
            3,
            1,
        )

        updir = np.float32(
            [0, -1, 0]
        ).reshape(
            3,
            1,
        )

        rotation = (
            pybullet.getMatrixFromQuaternion(
                orientation
            )
        )

        rotm = np.float32(
            rotation
        ).reshape(
            3,
            3,
        )

        lookdir = (
            rotm @ lookdir
        ).reshape(-1)

        updir = (
            rotm @ updir
        ).reshape(-1)

        lookat = (
            position + lookdir
        )

        focal_len = intrinsics[0]

        fovh = (
            image_size[0] / 2
        ) / focal_len

        fovh = (
            180
            * np.arctan(fovh)
            * 2
            / np.pi
        )

        aspect_ratio = (
            image_size[1]
            / image_size[0]
        )

        viewm = pybullet.computeViewMatrix(
            position,
            lookat,
            updir,
        )

        projm = (
            pybullet.computeProjectionMatrixFOV(
                fovh,
                aspect_ratio,
                0.01,
                10.0,
            )
        )

        _, _, color, depth, segm = (
            pybullet.getCameraImage(
                width=image_size[1],
                height=image_size[0],
                viewMatrix=viewm,
                projectionMatrix=projm,
                shadow=1,
                flags=(
                    pybullet
                    .ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                ),
                renderer=(
                    pybullet
                    .ER_BULLET_HARDWARE_OPENGL
                ),
            )
        )

        color_image_size = (
            image_size[0],
            image_size[1],
            4,
        )

        color = np.array(
            color,
            dtype=np.uint8,
        ).reshape(
            color_image_size
        )

        color = color[:, :, :3]

        if noise:

            color = np.int32(
                color
            )

            color += np.int32(
                np.random.normal(
                    0,
                    3,
                    color.shape,
                )
            )

            color = np.uint8(
                np.clip(
                    color,
                    0,
                    255,
                )
            )

        depth_image_size = (
            image_size[0],
            image_size[1],
        )

        zbuffer = np.float32(
            depth
        ).reshape(
            depth_image_size
        )

        znear, zfar = (
            0.01,
            10.0,
        )

        depth = (
            zfar
            + znear
            - (
                2 * zbuffer - 1
            )
            * (
                zfar - znear
            )
        )

        depth = (
            2
            * znear
            * zfar
        ) / depth

        if noise:

            depth += np.random.normal(
                0,
                0.003,
                depth.shape,
            )

        intrinsics = np.float32(
            intrinsics
        ).reshape(
            3,
            3,
        )

        return (
            color,
            depth,
            position,
            orientation,
            intrinsics,
        )

    # ========================================================================
    # POINT CLOUD / HEIGHTMAP
    # ========================================================================

    def get_pointcloud(
        self,
        depth,
        intrinsics,
    ):
        """Convert perspective depth image to 3D point cloud."""

        height, width = depth.shape

        xlin = np.linspace(
            0,
            width - 1,
            width,
        )

        ylin = np.linspace(
            0,
            height - 1,
            height,
        )

        px, py = np.meshgrid(
            xlin,
            ylin,
        )

        px = (
            px
            - intrinsics[0, 2]
        ) * (
            depth
            / intrinsics[0, 0]
        )

        py = (
            py
            - intrinsics[1, 2]
        ) * (
            depth
            / intrinsics[1, 1]
        )

        points = np.float32(
            [
                px,
                py,
                depth,
            ]
        ).transpose(
            1,
            2,
            0,
        )

        return points

    def transform_pointcloud(
        self,
        points,
        transform,
    ):
        """Apply rigid transformation to point cloud."""

        padding = (
            (0, 0),
            (0, 0),
            (0, 1),
        )

        homogen_points = np.pad(
            points.copy(),
            padding,
            "constant",
            constant_values=1,
        )

        for i in range(3):

            points[..., i] = np.sum(
                transform[i, :]
                * homogen_points,
                axis=-1,
            )

        return points

    def get_heightmap(
        self,
        points,
        colors,
        bounds,
        pixel_size,
    ):
        """Generate top-down orthographic heightmap."""

        width = int(
            np.round(
                (
                    bounds[0, 1]
                    - bounds[0, 0]
                )
                / pixel_size
            )
        )

        height = int(
            np.round(
                (
                    bounds[1, 1]
                    - bounds[1, 0]
                )
                / pixel_size
            )
        )

        heightmap = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.float32,
        )

        colormap = np.zeros(
            (
                height,
                width,
                colors.shape[-1],
            ),
            dtype=np.uint8,
        )

        xyzmap = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.float32,
        )

        ix = (
            points[..., 0]
            >= bounds[0, 0]
        ) & (
            points[..., 0]
            < bounds[0, 1]
        )

        iy = (
            points[..., 1]
            >= bounds[1, 0]
        ) & (
            points[..., 1]
            < bounds[1, 1]
        )

        iz = (
            points[..., 2]
            >= bounds[2, 0]
        ) & (
            points[..., 2]
            < bounds[2, 1]
        )

        valid = ix & iy & iz

        points = points[valid]
        colors = colors[valid]

        # Sort by z to simulate z-buffering.
        iz = np.argsort(
            points[:, -1]
        )

        points = points[iz]
        colors = colors[iz]

        px = np.int32(
            np.floor(
                (
                    points[:, 0]
                    - bounds[0, 0]
                )
                / pixel_size
            )
        )

        py = np.int32(
            np.floor(
                (
                    points[:, 1]
                    - bounds[1, 0]
                )
                / pixel_size
            )
        )

        px = np.clip(
            px,
            0,
            width - 1,
        )

        py = np.clip(
            py,
            0,
            height - 1,
        )

        heightmap[
            py,
            px
        ] = (
            points[:, 2]
            - bounds[2, 0]
        )

        for c in range(
            colors.shape[-1]
        ):

            colormap[
                py,
                px,
                c
            ] = colors[:, c]

            xyzmap[
                py,
                px,
                c
            ] = points[:, c]

        colormap = colormap[
            ::-1,
            :,
            :
        ]

        xv, yv = np.meshgrid(
            np.linspace(
                BOUNDS[0, 0],
                BOUNDS[0, 1],
                height,
            ),
            np.linspace(
                BOUNDS[1, 0],
                BOUNDS[1, 1],
                width,
            ),
        )

        xyzmap[:, :, 0] = xv
        xyzmap[:, :, 1] = yv

        xyzmap = xyzmap[
            ::-1,
            :,
            :
        ]

        heightmap = heightmap[
            ::-1,
            :,
        ]

        return (
            heightmap,
            colormap,
            xyzmap,
        )

    # ========================================================================
    # OBSERVATION / REWARD
    # ========================================================================

    def get_reward(self):
        """Return binary task-completion reward."""

        if self.last_task_result is None:
            return 0.0

        return (
            1.0
            if self.last_task_result["success"]
            else 0.0
        )

    def get_observation(self):

        observation = {}

        color, depth, position, orientation, intrinsics = (
            self.render_image()
        )

        points = self.get_pointcloud(
            depth,
            intrinsics,
        )

        position = np.float32(
            position
        ).reshape(
            3,
            1,
        )

        rotation = (
            pybullet.getMatrixFromQuaternion(
                orientation
            )
        )

        rotation = np.float32(
            rotation
        ).reshape(
            3,
            3,
        )

        transform = np.eye(4)

        transform[:3, :] = np.hstack(
            (
                rotation,
                position,
            )
        )

        points = self.transform_pointcloud(
            points,
            transform,
        )

        heightmap, colormap, xyzmap = (
            self.get_heightmap(
                points,
                color,
                BOUNDS,
                PIXEL_SIZE,
            )
        )

        observation["image"] = colormap
        observation["xyzmap"] = xyzmap
        observation["pick"] = list(
            self.config["pick"]
        )
        observation["place"] = list(
            self.config["place"]
        )

        return observation

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def close(self):
        """Stop background resources and disconnect PyBullet safely."""

        if pybullet.isConnected():

            try:

                if self.gripper is not None:
                    self.gripper.stop()

            except Exception:
                pass

            pybullet.disconnect()

        self.gripper = None
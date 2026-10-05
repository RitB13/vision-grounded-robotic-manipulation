"""
Controlled failure injection for robotic manipulation experiments.

This module modifies an otherwise valid physical action in order
to create reproducible execution failures for evaluation.

Supported failure modes:

    none
        Execute the original action unchanged.

    movement_failure
        Replace the place position with the pick position.
        The robot can grasp the object, but it cannot move it
        toward the intended target.

    placement_failure
        Move the object to a location near, but outside, the
        target placement tolerance.
        The object should move successfully but fail the final
        placement check.

    grasp_failure
        Move the pick position to a dynamically selected empty
        workspace location so that the robot should fail to
        grasp the intended object without intentionally
        interacting with another object.

Important:

    This module does not modify the PyBullet environment or the
    execution checker. It only modifies the physical action
    immediately before env.step().
"""


# ============================================================
# SUPPORTED FAILURE MODES
# ============================================================

FAILURE_MODES = {
    "none",
    "movement_failure",
    "placement_failure",
    "grasp_failure",
}


# ============================================================
# FAILURE INJECTOR
# ============================================================

class FailureInjector:
    """
    Inject controlled physical execution failures.

    Parameters
    ----------
    mode : str
        Failure mode.

    placement_offset : float
        XY displacement used for placement_failure.

    grasp_offset : float
        Minimum XY distance between the injected grasp point
        and the intended object.

    safe_margin : float
        Minimum XY distance that the injected grasp point should
        maintain from every known object.

    workspace_bounds : tuple
        Workspace bounds:

            (
                (xmin, xmax),
                (ymin, ymax)
            )

        The injected grasp point is selected from this region.
    """

    def __init__(
        self,
        mode="none",
        placement_offset=0.08,
        grasp_offset=0.12,
        safe_margin=0.08,
        workspace_bounds=(
            (-0.30, 0.30),
            (-0.80, -0.20),
        ),
    ):
        mode = str(mode).strip().lower()

        if mode not in FAILURE_MODES:
            raise ValueError(
                f"Unsupported failure mode: {mode}. "
                f"Supported modes: {sorted(FAILURE_MODES)}"
            )

        self.mode = mode

        self.placement_offset = float(
            placement_offset
        )

        self.grasp_offset = float(
            grasp_offset
        )

        self.safe_margin = float(
            safe_margin
        )

        self.workspace_bounds = (
            (
                float(workspace_bounds[0][0]),
                float(workspace_bounds[0][1]),
            ),
            (
                float(workspace_bounds[1][0]),
                float(workspace_bounds[1][1]),
            ),
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def apply(
        self,
        action,
        enabled=False,
        object_positions=None,
    ):
        """
        Apply the configured failure to an action.

        Parameters
        ----------
        action : dict
            Physical robot action containing at least:

                {
                    "pick": [x, y, z],
                    "place": [x, y, z]
                }

        enabled : bool
            Whether the failure should actually be injected
            for this particular execution step.

        object_positions : dict or None
            Optional mapping from object names to XYZ positions.

            Example:

                {
                    "blue block": [0.01, -0.40, 0.06],
                    "green block": [-0.02, -0.36, 0.02],
                    "red block": [-0.20, -0.65, 0.02]
                }

            This is used only for grasp_failure so that the
            injected pick point can be selected away from
            all known objects.

        Returns
        -------
        tuple
            (
                modified_action,
                metadata
            )

        The original action dictionary is not modified in place.
        """

        # ----------------------------------------------------
        # Make a shallow copy so that the original prediction
        # remains available for experiment logging.
        # ----------------------------------------------------

        modified_action = dict(
            action
        )

        metadata = {
            "enabled": bool(enabled),
            "mode": (
                self.mode
                if enabled
                else "none"
            ),
            "injected": False,
            "description": None,
        }

        # ----------------------------------------------------
        # No failure requested.
        # ----------------------------------------------------

        if not enabled:
            return (
                modified_action,
                metadata,
            )

        if self.mode == "none":
            return (
                modified_action,
                metadata,
            )

        # ----------------------------------------------------
        # Validate required coordinates.
        # ----------------------------------------------------

        if "pick" not in action:
            raise KeyError(
                "FailureInjector requires "
                "action['pick']."
            )

        if "place" not in action:
            raise KeyError(
                "FailureInjector requires "
                "action['place']."
            )

        pick = self._to_xyz(
            action["pick"],
            "pick",
        )

        place = self._to_xyz(
            action["place"],
            "place",
        )

        # ====================================================
        # MOVEMENT FAILURE
        # ====================================================

        if self.mode == "movement_failure":

            modified_action["place"] = pick.copy()

            metadata.update(
                {
                    "injected": True,
                    "description": (
                        "Place position was replaced "
                        "with the pick position so the "
                        "object should be grasped but "
                        "not moved toward the target."
                    ),
                    "original_place": place,
                    "modified_place": pick.copy(),
                }
            )

            return (
                modified_action,
                metadata,
            )

        # ====================================================
        # PLACEMENT FAILURE
        # ====================================================

        if self.mode == "placement_failure":

            modified_place = place.copy()

            # Shift only the XY coordinates.
            #
            # The default 0.08 m offset is deliberately
            # larger than the environment's 0.035 m
            # placement tolerance.

            modified_place[0] += (
                self.placement_offset
            )

            modified_action["place"] = (
                modified_place
            )

            metadata.update(
                {
                    "injected": True,
                    "description": (
                        "Place position was shifted "
                        "outside the target placement "
                        "tolerance while preserving "
                        "a meaningful object movement."
                    ),
                    "original_place": place,
                    "modified_place": (
                        modified_place.copy()
                    ),
                    "offset_xy": [
                        self.placement_offset,
                        0.0,
                    ],
                }
            )

            return (
                modified_action,
                metadata,
            )

        # ====================================================
        # GRASP FAILURE
        # ====================================================

        if self.mode == "grasp_failure":

            safe_pick = self._find_safe_pick_position(
                original_pick=pick,
                object_positions=object_positions,
            )

            modified_action["pick"] = (
                safe_pick.copy()
            )

            metadata.update(
                {
                    "injected": True,
                    "description": (
                        "Pick position was moved to a "
                        "dynamically selected empty "
                        "workspace location that is "
                        "separated from the known objects. "
                        "This is intended to produce a "
                        "clean grasp failure without "
                        "intentionally contacting another "
                        "object."
                    ),
                    "original_pick": pick,
                    "modified_pick": (
                        safe_pick.copy()
                    ),
                    "safe_margin": (
                        self.safe_margin
                    ),
                    "object_positions_used": (
                        object_positions
                        if object_positions is not None
                        else {}
                    ),
                }
            )

            return (
                modified_action,
                metadata,
            )

        # This should never be reached because the mode was
        # validated during initialization.
        raise RuntimeError(
            f"Unhandled failure mode: {self.mode}"
        )

    # ========================================================
    # SAFE GRASP-FAILURE POSITION
    # ========================================================

    def _find_safe_pick_position(
        self,
        original_pick,
        object_positions=None,
    ):
        """
        Select an empty workspace point for grasp_failure.

        The point must satisfy:

            1. It is inside the configured workspace.
            2. It is at least grasp_offset away from the
               original intended pick point.
            3. It is at least safe_margin away from every
               known object.

        Candidate points are deterministic, so experiments
        remain reproducible for the same scene.
        """

        object_xy = []

        if object_positions:
            for position in object_positions.values():

                try:
                    xyz = self._to_xyz(
                        position,
                        "object_position",
                    )
                except ValueError:
                    continue

                object_xy.append(
                    (
                        float(xyz[0]),
                        float(xyz[1]),
                    )
                )

        xmin, xmax = self.workspace_bounds[0]
        ymin, ymax = self.workspace_bounds[1]

        # Keep a small margin from the workspace boundaries.
        boundary_margin = 0.025

        xmin += boundary_margin
        xmax -= boundary_margin
        ymin += boundary_margin
        ymax -= boundary_margin

        # Deterministic candidate grid.
        #
        # These points cover the workspace corners and
        # interior regions. The first valid candidate is used.

        x_values = [
            xmin,
            xmin + 0.05,
            0.0,
            xmax - 0.05,
            xmax,
        ]

        y_values = [
            ymin,
            ymin + 0.05,
            -0.50,
            ymax - 0.05,
            ymax,
        ]

        candidates = []

        for y in y_values:
            for x in x_values:
                candidates.append(
                    [x, y, float(original_pick[2])]
                )

        # Also include the four workspace corners first
        # because they are normally the furthest locations
        # from objects in the manipulation workspace.

        candidates = [
            [xmin, ymin, float(original_pick[2])],
            [xmin, ymax, float(original_pick[2])],
            [xmax, ymin, float(original_pick[2])],
            [xmax, ymax, float(original_pick[2])],
        ] + candidates

        # Remove duplicates while preserving order.
        unique_candidates = []

        for candidate in candidates:
            if not any(
                self._xy_distance(
                    candidate,
                    existing,
                ) < 1e-9
                for existing in unique_candidates
            ):
                unique_candidates.append(
                    candidate
                )

        # ----------------------------------------------------
        # First pass: strict safety constraints.
        # ----------------------------------------------------

        for candidate in unique_candidates:

            if self._xy_distance(
                candidate,
                original_pick,
            ) < self.grasp_offset:

                continue

            if any(
                self._xy_distance(
                    candidate,
                    object_xy,
                ) < self.safe_margin
                for object_xy in object_xy
            ):
                continue

            return candidate

        # ----------------------------------------------------
        # Second pass: maximize distance from objects.
        #
        # This is a deterministic fallback for unusually
        # crowded scenes.
        # ----------------------------------------------------

        best_candidate = None
        best_score = -float("inf")

        for candidate in unique_candidates:

            distance_from_pick = (
                self._xy_distance(
                    candidate,
                    original_pick,
                )
            )

            if distance_from_pick < self.grasp_offset:
                continue

            if object_xy:

                nearest_object_distance = min(
                    self._xy_distance(
                        candidate,
                        position,
                    )
                    for position in object_xy
                )

            else:
                nearest_object_distance = float(
                    "inf"
                )

            score = min(
                distance_from_pick,
                nearest_object_distance,
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate

        # ----------------------------------------------------
        # Final deterministic fallback.
        #
        # This should only occur if the configured workspace
        # is invalid or completely occupied.
        # ----------------------------------------------------

        return [
            xmin,
            ymin,
            float(original_pick[2]),
        ]

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _xy_distance(
        first,
        second,
    ):
        """
        Euclidean XY distance between two XYZ-like values.
        """

        dx = (
            float(first[0])
            - float(second[0])
        )

        dy = (
            float(first[1])
            - float(second[1])
        )

        return (
            dx * dx
            + dy * dy
        ) ** 0.5

    @staticmethod
    def _to_xyz(
        value,
        name,
    ):
        """
        Convert a coordinate-like value into a mutable list.

        Returning a list keeps the injector independent from
        NumPy and PyTorch types.
        """

        if value is None:
            raise ValueError(
                f"Action '{name}' coordinate "
                f"cannot be None."
            )

        try:
            coordinates = [
                float(value[0]),
                float(value[1]),
                float(value[2]),
            ]
        except (
            TypeError,
            IndexError,
            KeyError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"Action '{name}' must contain "
                f"three numeric coordinates."
            ) from exc

        return coordinates


# ============================================================
# STANDALONE TEST
# ============================================================

def main():

    print("=" * 70)
    print(
        "FAILURE INJECTOR STANDALONE TEST"
    )
    print("=" * 70)

    base_action = {
        "pick": [
            0.10,
            -0.30,
            0.04,
        ],
        "place": [
            0.02,
            -0.56,
            0.08,
        ],
        "pick_name": "red block",
        "place_name": "blue block",
    }

    object_positions = {
        "blue block": [
            -0.03,
            -0.36,
            0.06,
        ],
        "green block": [
            -0.03,
            -0.36,
            0.02,
        ],
        "red block": [
            -0.20,
            -0.65,
            0.02,
        ],
    }

    for mode in sorted(
        FAILURE_MODES
    ):

        print(
            f"\nTesting mode: {mode}"
        )

        injector = FailureInjector(
            mode=mode
        )

        modified_action, metadata = (
            injector.apply(
                base_action,
                enabled=(
                    mode != "none"
                ),
                object_positions=(
                    object_positions
                ),
            )
        )

        print(
            "  Original pick :",
            base_action["pick"],
        )

        print(
            "  Modified pick :",
            modified_action["pick"],
        )

        print(
            "  Original place:",
            base_action["place"],
        )

        print(
            "  Modified place:",
            modified_action["place"],
        )

        print(
            "  Injected      :",
            metadata["injected"],
        )

        print(
            "  Description   :",
            metadata["description"],
        )

    print(
        "\n✅ FailureInjector standalone test passed."
    )


if __name__ == "__main__":
    main()
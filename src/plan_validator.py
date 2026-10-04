from typing import Any, Dict, List, Optional, Set


class PlanValidator:
    """
    Deterministic validator for high-level robotic manipulation plans.

    The LLM generates the plan.
    This class decides whether that plan is logically and
    structurally valid for the current world state.

    It does NOT:
        - call the LLM
        - control the robot
        - generate coordinates
        - perform inverse kinematics
        - execute PyBullet actions
    """

    VALID_ACTION_TYPES = {"pick", "place"}

    def validate(
        self,
        plan: Dict[str, Any],
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate a complete symbolic task plan.

        Returns
        -------
        dict
            {
                "valid": bool,
                "errors": [...],
                "warnings": [...],
                "validated_actions": [...]
            }
        """

        errors: List[str] = []
        warnings: List[str] = []

        # ---------------------------------------------------------
        # 1. Validate the world state
        # ---------------------------------------------------------

        available_objects = self._extract_objects(
            world_state,
            errors,
        )

        # ---------------------------------------------------------
        # 2. Validate the plan structure
        # ---------------------------------------------------------

        if not isinstance(plan, dict):
            errors.append("Plan must be a dictionary.")
            return self._result(errors, warnings, [])

        if not isinstance(plan.get("goal"), str):
            errors.append("'goal' must be a string.")

        actions = plan.get("actions")

        if not isinstance(actions, list):
            errors.append("'actions' must be a list.")
            return self._result(errors, warnings, [])

        if len(actions) == 0:
            errors.append("Plan contains no actions.")
            return self._result(errors, warnings, [])

        # ---------------------------------------------------------
        # 3. Validate individual actions
        # ---------------------------------------------------------

        validated_actions = []

        for index, action in enumerate(actions):
            action_errors = self._validate_action(
                action=action,
                index=index,
                available_objects=available_objects,
            )

            errors.extend(action_errors)

            if not action_errors:
                validated_actions.append(action)

        # ---------------------------------------------------------
        # 4. Validate action sequence
        # ---------------------------------------------------------

        errors.extend(
            self._validate_sequence(actions)
        )

        # ---------------------------------------------------------
        # 5. Detect contradictory object usage
        # ---------------------------------------------------------

        errors.extend(
            self._validate_object_flow(actions)
        )

        # ---------------------------------------------------------
        # 6. Warnings
        # ---------------------------------------------------------

        warnings.extend(
            self._generate_warnings(actions)
        )

        return self._result(
            errors,
            warnings,
            validated_actions,
        )

    # =============================================================
    # World-state handling
    # =============================================================

    @staticmethod
    def _extract_objects(
        world_state: Dict[str, Any],
        errors: List[str],
    ) -> Set[str]:
        """
        Extract available object names from the world state.

        Currently supports:

            {
                "objects": [
                    "blue block",
                    "green block"
                ]
            }

        Later this can be extended to richer object dictionaries.
        """

        if not isinstance(world_state, dict):
            errors.append("World state must be a dictionary.")
            return set()

        objects = world_state.get("objects")

        if not isinstance(objects, list):
            errors.append(
                "World state must contain an 'objects' list."
            )
            return set()

        normalized_objects: Set[str] = set()

        for obj in objects:
            if isinstance(obj, str):
                name = obj.strip().lower()

                if name:
                    normalized_objects.add(name)

            elif isinstance(obj, dict):
                name = obj.get("name")

                if isinstance(name, str) and name.strip():
                    normalized_objects.add(
                        name.strip().lower()
                    )

        if not normalized_objects:
            errors.append(
                "World state contains no valid objects."
            )

        return normalized_objects

    # =============================================================
    # Individual action validation
    # =============================================================

    def _validate_action(
        self,
        action: Any,
        index: int,
        available_objects: Set[str],
    ) -> List[str]:

        errors: List[str] = []

        if not isinstance(action, dict):
            return [
                f"Action {index} must be a dictionary."
            ]

        action_type = action.get("type")
        object_name = action.get("object")
        target = action.get("target")

        # ---------------------------------------------------------
        # Action type
        # ---------------------------------------------------------

        if action_type not in self.VALID_ACTION_TYPES:
            errors.append(
                f"Action {index} has unsupported type "
                f"'{action_type}'."
            )

        # ---------------------------------------------------------
        # Object
        # ---------------------------------------------------------

        if not isinstance(object_name, str):
            errors.append(
                f"Action {index} must contain a string 'object'."
            )
        else:
            normalized_object = object_name.strip().lower()

            if not normalized_object:
                errors.append(
                    f"Action {index} has an empty object name."
                )

            elif normalized_object not in available_objects:
                errors.append(
                    f"Action {index} references unavailable "
                    f"object '{object_name}'."
                )

        # ---------------------------------------------------------
        # Pick-specific rules
        # ---------------------------------------------------------

        if action_type == "pick":

            if target is not None:
                errors.append(
                    f"Pick action {index} must have target=null."
                )

        # ---------------------------------------------------------
        # Place-specific rules
        # ---------------------------------------------------------

        if action_type == "place":

            if not isinstance(target, str):
                errors.append(
                    f"Place action {index} must contain "
                    f"a string target."
                )

            else:
                normalized_target = target.strip().lower()

                if not normalized_target:
                    errors.append(
                        f"Place action {index} has "
                        f"an empty target."
                    )

                elif normalized_target not in available_objects:
                    errors.append(
                        f"Action {index} references unavailable "
                        f"target '{target}'."
                    )

            # Prevent placing an object onto itself.
            if (
                isinstance(object_name, str)
                and isinstance(target, str)
                and object_name.strip().lower()
                == target.strip().lower()
            ):
                errors.append(
                    f"Action {index} attempts to place "
                    f"'{object_name}' onto itself."
                )

        return errors

    # =============================================================
    # Sequence validation
    # =============================================================

    @staticmethod
    def _validate_sequence(
        actions: List[Dict[str, Any]]
    ) -> List[str]:

        errors: List[str] = []

        if not actions:
            return errors

        # ---------------------------------------------------------
        # A place cannot happen before the first pick.
        # ---------------------------------------------------------

        picked_object: Optional[str] = None

        for index, action in enumerate(actions):

            if not isinstance(action, dict):
                continue

            action_type = action.get("type")
            object_name = action.get("object")

            if action_type == "pick":

                if picked_object is not None:
                    errors.append(
                        f"Action {index} attempts to pick "
                        f"'{object_name}' while "
                        f"'{picked_object}' is already held."
                    )

                else:
                    picked_object = (
                        object_name.strip().lower()
                        if isinstance(object_name, str)
                        else None
                    )

            elif action_type == "place":

                if picked_object is None:
                    errors.append(
                        f"Action {index} is a place action "
                        f"but no object has been picked."
                    )

                else:
                    place_object = (
                        object_name.strip().lower()
                        if isinstance(object_name, str)
                        else None
                    )

                    if place_object != picked_object:
                        errors.append(
                            f"Action {index} attempts to place "
                            f"'{place_object}' while holding "
                            f"'{picked_object}'."
                        )

                    picked_object = None

        # ---------------------------------------------------------
        # Something was picked but never placed.
        # ---------------------------------------------------------

        if picked_object is not None:
            errors.append(
                f"Object '{picked_object}' was picked "
                f"but never placed."
            )

        return errors

    # =============================================================
    # Object-flow validation
    # =============================================================

    @staticmethod
    def _validate_object_flow(
        actions: List[Dict[str, Any]]
    ) -> List[str]:

        errors: List[str] = []

        placed_objects = set()

        for index, action in enumerate(actions):

            if not isinstance(action, dict):
                continue

            if action.get("type") != "place":
                continue

            object_name = action.get("object")
            target = action.get("target")

            if not isinstance(object_name, str):
                continue

            object_name = object_name.strip().lower()

            if object_name in placed_objects:
                errors.append(
                    f"Object '{object_name}' is placed "
                    f"multiple times without being "
                    f"picked again."
                )

            placed_objects.add(object_name)

            # A target can itself have been placed earlier.
            # That is not automatically invalid because stacking
            # objects is a valid manipulation operation.
            #
            # Therefore we intentionally do NOT reject it here.

        return errors

    # =============================================================
    # Warnings
    # =============================================================

    @staticmethod
    def _generate_warnings(
        actions: List[Dict[str, Any]]
    ) -> List[str]:

        warnings: List[str] = []

        if len(actions) > 10:
            warnings.append(
                "Plan contains more than 10 actions; "
                "consider checking whether the task can "
                "be simplified."
            )

        return warnings

    # =============================================================
    # Result formatting
    # =============================================================

    @staticmethod
    def _result(
        errors: List[str],
        warnings: List[str],
        validated_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_actions": validated_actions,
        }


# =================================================================
# Standalone test
# =================================================================

if __name__ == "__main__":

    validator = PlanValidator()

    world_state = {
        "objects": [
            "blue block",
            "green block",
            "red block",
        ]
    }

    # -------------------------------------------------------------
    # Test 1: Valid plan
    # -------------------------------------------------------------

    valid_plan = {
        "goal": "blue block placed on green block",
        "actions": [
            {
                "type": "pick",
                "object": "blue block",
                "target": None,
            },
            {
                "type": "place",
                "object": "blue block",
                "target": "green block",
            },
        ],
    }

    result = validator.validate(
        valid_plan,
        world_state,
    )

    print("\n" + "=" * 60)
    print("TEST 1 — VALID PLAN")
    print("=" * 60)
    print(result)

    # -------------------------------------------------------------
    # Test 2: Non-existent object
    # -------------------------------------------------------------

    invalid_object_plan = {
        "goal": "purple block placed on green block",
        "actions": [
            {
                "type": "pick",
                "object": "purple block",
                "target": None,
            },
            {
                "type": "place",
                "object": "purple block",
                "target": "green block",
            },
        ],
    }

    result = validator.validate(
        invalid_object_plan,
        world_state,
    )

    print("\n" + "=" * 60)
    print("TEST 2 — NON-EXISTENT OBJECT")
    print("=" * 60)
    print(result)

    # -------------------------------------------------------------
    # Test 3: Place before pick
    # -------------------------------------------------------------

    invalid_sequence_plan = {
        "goal": "blue block placed on green block",
        "actions": [
            {
                "type": "place",
                "object": "blue block",
                "target": "green block",
            },
            {
                "type": "pick",
                "object": "blue block",
                "target": None,
            },
        ],
    }

    result = validator.validate(
        invalid_sequence_plan,
        world_state,
    )

    print("\n" + "=" * 60)
    print("TEST 3 — INVALID SEQUENCE")
    print("=" * 60)
    print(result)

    # -------------------------------------------------------------
    # Test 4: Place object onto itself
    # -------------------------------------------------------------

    self_place_plan = {
        "goal": "blue block placed on itself",
        "actions": [
            {
                "type": "pick",
                "object": "blue block",
                "target": None,
            },
            {
                "type": "place",
                "object": "blue block",
                "target": "blue block",
            },
        ],
    }

    result = validator.validate(
        self_place_plan,
        world_state,
    )

    print("\n" + "=" * 60)
    print("TEST 4 — SELF PLACEMENT")
    print("=" * 60)
    print(result)
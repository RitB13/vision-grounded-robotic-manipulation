from typing import Any, Dict, List, Optional, Set, Tuple


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

    In addition to structural validation, the validator can optionally
    enforce REQUIRED manipulation pairs.

    Example:

        required_pairs = [
            ("red block", "blue block")
        ]

    Then a candidate plan containing:

        green block -> blue block
        red block -> blue block

    will be rejected because the green -> blue manipulation is not
    required by the remaining task.
    """

    VALID_ACTION_TYPES = {"pick", "place"}

    # Current research/evaluation scope: place targets must be named
    # objects present in the symbolic world state. Workspace-location
    # targets (for example, "middle" or "top left corner") remain
    # supported by the simulation environment but are intentionally
    # outside the current LLM planning/validation domain.
    TARGET_DOMAIN = "named_objects"

    def validate(
        self,
        plan: Dict[str, Any],
        world_state: Dict[str, Any],
        required_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a complete symbolic task plan.

        Parameters
        ----------
        plan:
            LLM-generated symbolic plan.

        world_state:
            Current symbolic world state.

        required_pairs:
            Optional list of required pick->place pairs.

            Example:
                [
                    ("red block", "blue block")
                ]

            When supplied, every manipulation pair in the candidate
            plan must correspond to one of these required pairs.

            This prevents the LLM from introducing unnecessary
            manipulations during replanning.

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

        validated_actions: List[Dict[str, Any]] = []

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
        # 6. Validate manipulation relevance
        #
        # Only enabled when required_pairs is supplied.
        # ---------------------------------------------------------

        if required_pairs is not None:
            errors.extend(
                self._validate_required_pairs(
                    actions=actions,
                    required_pairs=required_pairs,
                )
            )

        # ---------------------------------------------------------
        # 7. Warnings
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

        Supports:

            {
                "objects": [
                    "blue block",
                    "green block"
                ]
            }

        and:

            {
                "objects": [
                    {"name": "blue block"},
                    {"name": "green block"}
                ]
            }
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
                        f"target '{target}'. The current planning domain "
                        f"supports named objects as placement targets only."
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
        # A place cannot happen before a pick.
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

        return errors

    # =============================================================
    # Required manipulation validation
    # =============================================================

    @staticmethod
    def _normalize_pair(
        object_name: Any,
        target: Any,
    ) -> Optional[Tuple[str, str]]:
        """
        Normalize a manipulation pair.

        Returns:
            ("blue block", "green block")

        or None when the pair is malformed.
        """

        if not isinstance(object_name, str):
            return None

        if not isinstance(target, str):
            return None

        object_name = object_name.strip().lower()
        target = target.strip().lower()

        if not object_name or not target:
            return None

        return object_name, target

    @classmethod
    def _extract_manipulation_pairs(
        cls,
        actions: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:

        pairs: List[Tuple[str, str]] = []

        if not actions:
            return pairs

        for index in range(len(actions) - 1):

            pick_action = actions[index]
            place_action = actions[index + 1]

            if not isinstance(pick_action, dict):
                continue

            if not isinstance(place_action, dict):
                continue

            if pick_action.get("type") != "pick":
                continue

            if place_action.get("type") != "place":
                continue

            pick_object = pick_action.get("object")
            place_object = place_action.get("object")
            target = place_action.get("target")

            if not isinstance(pick_object, str):
                continue

            if not isinstance(place_object, str):
                continue

            if not isinstance(target, str):
                continue

            normalized_pick = pick_object.strip().lower()
            normalized_place = place_object.strip().lower()
            normalized_target = target.strip().lower()

            # Only consider a true pick -> place pair where the
            # same object was picked and subsequently placed.

            if normalized_pick != normalized_place:
                continue

            pairs.append(
                (
                    normalized_pick,
                    normalized_target,
                )
            )

        return pairs

    @classmethod
    def _normalize_required_pairs(
        cls,
        required_pairs: List[Tuple[str, str]],
    ) -> Set[Tuple[str, str]]:

        normalized_pairs: Set[Tuple[str, str]] = set()

        for pair in required_pairs:

            if not isinstance(pair, (tuple, list)):
                continue

            if len(pair) != 2:
                continue

            normalized_pair = cls._normalize_pair(
                pair[0],
                pair[1],
            )

            if normalized_pair is not None:
                normalized_pairs.add(normalized_pair)

        return normalized_pairs

    @classmethod
    def _validate_required_pairs(
        cls,
        actions: List[Dict[str, Any]],
        required_pairs: List[Tuple[str, str]],
    ) -> List[str]:
        """
        Ensure that a candidate plan contains only manipulation
        pairs that are required by the remaining task.

        This is intentionally deterministic.

        Example:

            required_pairs = [
                ("red block", "blue block")
            ]

        Candidate:

            green -> blue
            red   -> blue

        Result:

            INVALID

        because green -> blue is not part of the required task.

        We do NOT require the candidate to contain every required pair
        here. That is intentional: a separate recovery/replanning
        attempt may contain only a subset while the caller is deciding
        what remains. The important guarantee is that the LLM cannot
        introduce unrelated manipulation pairs.
        """

        errors: List[str] = []

        normalized_required = cls._normalize_required_pairs(
            required_pairs
        )

        if not normalized_required:

            errors.append(
                "Required manipulation pairs were supplied "
                "but none were valid."
            )

            return errors

        candidate_pairs = cls._extract_manipulation_pairs(
            actions
        )

        if not candidate_pairs:

            errors.append(
                "Plan does not contain a valid pick->place "
                "manipulation pair."
            )

            return errors

        for pair in candidate_pairs:

            if pair not in normalized_required:

                object_name, target = pair

                required_text = ", ".join(
                    f"{obj} -> {tgt}"
                    for obj, tgt in sorted(normalized_required)
                )

                errors.append(
                    f"Manipulation '{object_name} -> {target}' "
                    f"is not required by the remaining task. "
                    f"Allowed manipulation pair(s): "
                    f"{required_text}."
                )

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

    assert result["valid"] is True

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

    assert result["valid"] is False

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

    assert result["valid"] is False

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

    assert result["valid"] is False

    # -------------------------------------------------------------
    # Test 5: Required-pair validation — valid
    # -------------------------------------------------------------

    required_pair_plan = {
        "goal": "red block placed on blue block",
        "actions": [
            {
                "type": "pick",
                "object": "red block",
                "target": None,
            },
            {
                "type": "place",
                "object": "red block",
                "target": "blue block",
            },
        ],
    }

    result = validator.validate(
        required_pair_plan,
        world_state,
        required_pairs=[
            ("red block", "blue block"),
        ],
    )

    print("\n" + "=" * 60)
    print("TEST 5 — REQUIRED PAIR VALID")
    print("=" * 60)
    print(result)

    assert result["valid"] is True

    # -------------------------------------------------------------
    # Test 6: Required-pair validation — unnecessary action
    # -------------------------------------------------------------

    unnecessary_action_plan = {
        "goal": "red block placed on blue block",
        "actions": [
            {
                "type": "pick",
                "object": "green block",
                "target": None,
            },
            {
                "type": "place",
                "object": "green block",
                "target": "blue block",
            },
            {
                "type": "pick",
                "object": "red block",
                "target": None,
            },
            {
                "type": "place",
                "object": "red block",
                "target": "blue block",
            },
        ],
    }

    result = validator.validate(
        unnecessary_action_plan,
        world_state,
        required_pairs=[
            ("red block", "blue block"),
        ],
    )

    print("\n" + "=" * 60)
    print("TEST 6 — UNNECESSARY ACTION")
    print("=" * 60)
    print(result)

    assert result["valid"] is False

    # Make sure the error specifically identifies the unwanted pair.

    assert any(
        "green block -> blue block" in error
        for error in result["errors"]
    )

    # -------------------------------------------------------------
    # Test 7: Multiple required pairs
    # -------------------------------------------------------------

    multi_step_plan = {
        "goal": (
            "blue block placed on green block, "
            "then red block placed on blue block"
        ),
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
            {
                "type": "pick",
                "object": "red block",
                "target": None,
            },
            {
                "type": "place",
                "object": "red block",
                "target": "blue block",
            },
        ],
    }

    result = validator.validate(
        multi_step_plan,
        world_state,
        required_pairs=[
            ("blue block", "green block"),
            ("red block", "blue block"),
        ],
    )

    print("\n" + "=" * 60)
    print("TEST 7 — MULTI-STEP REQUIRED PAIRS")
    print("=" * 60)
    print(result)

    assert result["valid"] is True

    print("\n" + "=" * 60)
    print("ALL PLAN VALIDATOR TESTS PASSED")
    print("=" * 60)
from typing import Any, Dict, List


def extract_pick_place_pairs(
    plan: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Extract sequential pick -> place pairs from a validated symbolic plan.

    Example:

        pick blue
        place blue -> green
        pick red
        place red -> blue

    becomes:

        [
            {
                "pick": "blue block",
                "place": "green block",
            },
            {
                "pick": "red block",
                "place": "blue block",
            },
        ]

    This function only handles symbolic action structure.
    It does not perform visual grounding or robot execution.
    """

    actions = plan.get("actions", [])

    if not actions:
        raise ValueError("Plan contains no actions.")

    pairs: List[Dict[str, str]] = []

    index = 0

    while index < len(actions):

        action = actions[index]

        if not isinstance(action, dict):
            raise ValueError(
                f"Action {index} must be a dictionary."
            )

        action_type = action.get("type")
        obj = action.get("object")
        target = action.get("target")

        # -----------------------------------------------------
        # Pick must be followed by its corresponding place.
        # -----------------------------------------------------

        if action_type == "pick":

            if target is not None:
                raise ValueError(
                    f"Pick action {index} must have target=None."
                )

            if index + 1 >= len(actions):
                raise ValueError(
                    f"Pick action {index} has no following place action."
                )

            next_action = actions[index + 1]

            if not isinstance(next_action, dict):
                raise ValueError(
                    f"Action {index + 1} must be a dictionary."
                )

            if next_action.get("type") != "place":
                raise ValueError(
                    f"Pick action {index} must be immediately "
                    f"followed by a place action."
                )

            if next_action.get("object") != obj:
                raise ValueError(
                    f"Pick action {index} picks '{obj}', but "
                    f"the following place action operates on "
                    f"'{next_action.get('object')}'."
                )

            next_target = next_action.get("target")

            if not isinstance(next_target, str) or not next_target.strip():
                raise ValueError(
                    f"Place action {index + 1} requires a target."
                )

            pairs.append(
                {
                    "pick": obj,
                    "place": next_target,
                }
            )

            index += 2
            continue

        # -----------------------------------------------------
        # A standalone place is not supported by the current
        # robot primitive because env.step() performs a complete
        # pick-and-place operation.
        # -----------------------------------------------------

        if action_type == "place":
            raise ValueError(
                f"Place action {index} is not preceded by its "
                f"corresponding pick action."
            )

        raise ValueError(
            f"Unsupported action type at index {index}: "
            f"{action_type}"
        )

    return pairs


def plan_to_instruction(plan: Dict[str, Any]) -> str:
    """
    Convert a validated symbolic manipulation plan into the
    natural-language instruction expected by CLIPort.

    This function is retained for the existing single-step
    pipeline and for compatibility with the current system.
    """

    pairs = extract_pick_place_pairs(plan)

    instructions = []

    for pair in pairs:

        instructions.append(
            f"Pick the {pair['pick']} and place it on the {pair['place']}"
        )

    return " and ".join(instructions) + "."


def pair_to_instruction(pair: Dict[str, str]) -> str:
    """
    Convert one pick/place pair into the CLIPort-compatible
    instruction used for one physical execution step.
    """

    if not isinstance(pair, dict):
        raise ValueError("pair must be a dictionary.")

    pick = pair.get("pick")
    place = pair.get("place")

    if not isinstance(pick, str) or not pick.strip():
        raise ValueError("pair must contain a valid 'pick' object.")

    if not isinstance(place, str) or not place.strip():
        raise ValueError("pair must contain a valid 'place' target.")

    return (
        f"Pick the {pick} and place it on the {place}."
    )


if __name__ == "__main__":

    example_plan = {
        "goal": (
            "Place the blue block on the green block, "
            "then place the red block on the blue block."
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

    pairs = extract_pick_place_pairs(example_plan)

    print("\nExtracted pick/place pairs:")
    for index, pair in enumerate(pairs, start=1):
        print(f"  Pair {index}: {pair}")

    print("\nIndividual CLIPort instructions:")

    for index, pair in enumerate(pairs, start=1):
        print(
            f"  Action {index}: "
            f"{pair_to_instruction(pair)}"
        )

    print("\nFull-plan CLIPort instruction:")
    print(plan_to_instruction(example_plan))
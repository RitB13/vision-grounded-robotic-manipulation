from typing import Any, Dict, List


def plan_to_instruction(plan: Dict[str, Any]) -> str:
    """
    Convert a validated symbolic manipulation plan into the natural-language
    instruction expected by CLIPort.

    This adapter does not perform robotics or coordinate reasoning.
    It only converts symbolic actions into a CLIPort-compatible instruction.
    """

    actions: List[Dict[str, Any]] = plan.get("actions", [])

    if not actions:
        raise ValueError("Plan contains no actions.")

    instructions = []

    index = 0

    while index < len(actions):
        action = actions[index]

        action_type = action.get("type")
        obj = action.get("object")
        target = action.get("target")

        # ---------------------------------------------------------
        # Pick followed immediately by place
        # ---------------------------------------------------------
        if action_type == "pick":

            if target is not None:
                raise ValueError(
                    f"Pick action {index} must have target=None."
                )

            # Check whether the next action places the same object.
            if index + 1 < len(actions):
                next_action = actions[index + 1]

                if (
                    next_action.get("type") == "place"
                    and next_action.get("object") == obj
                ):
                    next_target = next_action.get("target")

                    if not next_target:
                        raise ValueError(
                            f"Place action {index + 1} requires a target."
                        )

                    # Preserve the phrasing that worked in the original
                    # CLIPort baseline.
                    instructions.append(
                        f"Pick the {obj} and place it on the {next_target}"
                    )

                    index += 2
                    continue

            # Standalone pick.
            instructions.append(
                f"Pick the {obj}"
            )

        # ---------------------------------------------------------
        # Standalone place
        # ---------------------------------------------------------
        elif action_type == "place":

            if not target:
                raise ValueError(
                    f"Place action {index} requires a target."
                )

            instructions.append(
                f"Place the {obj} on the {target}"
            )

        else:
            raise ValueError(
                f"Unsupported action type: {action_type}"
            )

        index += 1

    return " and ".join(instructions) + "."


if __name__ == "__main__":

    example_plan = {
        "goal": "Place the blue block on the green block.",
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

    instruction = plan_to_instruction(example_plan)

    print("\nGenerated CLIPort instruction:")
    print(instruction)
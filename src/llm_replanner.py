import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-120b"


class LLMTaskReplanner:
    """
    High-level task replanner using GPT-OSS-120B through Groq.

    The replanner receives:
        - the original user instruction
        - the current world state
        - the previous symbolic plan
        - execution feedback
        - optional validation errors
        - the exact manipulation pairs that remain required

    It produces a revised symbolic task plan.

    The LLM does NOT:
        - generate coordinates
        - generate joint angles
        - control the robot
        - perform inverse kinematics
        - execute PyBullet actions
    """

    def __init__(self, model: str = MODEL_NAME):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY was not found.\n"
                "Create a .env file in the project root containing:\n\n"
                "GROQ_API_KEY=your_api_key_here"
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    @staticmethod
    def _build_system_prompt() -> str:
        return """
You are the high-level task replanner for a simulated robotic
manipulation system.

Your job is to revise a previously generated symbolic manipulation
plan after either:

1. the robot reports an execution failure, or
2. a deterministic validator rejects a previously generated plan.

You receive:

- the original user instruction
- the current world state
- the previous symbolic plan
- structured execution feedback
- optional validation errors
- REQUIRED MANIPULATION PAIRS describing exactly which
  manipulations remain necessary

Your task is to produce ONE valid revised symbolic plan that still
satisfies the user's original goal.

The robot operates in a tabletop environment containing named objects.

The available primitive actions are:

1. pick
   - Select one object to grasp.
   - object must contain the name of the object being picked.
   - target must be null.

2. place
   - Place the previously picked object onto or at a target.
   - object must contain the object being placed.
   - target must contain the destination object or location.

IMPORTANT RULES:

HIGH-LEVEL SYMBOLIC REASONING ONLY:

- Produce only a high-level symbolic plan.
- Never output robot coordinates.
- Never output joint angles.
- Never output motor commands.
- Never perform inverse kinematics.
- Never claim that an action has already succeeded.
- Do not execute the task.
- Do not use tools.

WORLD-STATE RULES:

- Use only objects present in the supplied world state.
- Do not invent objects.
- Preserve the user's original goal.

PICK-AND-PLACE RULES:

- Every place action must be preceded by a pick of the same
  object that is currently being held.
- A pick action must have target=null.
- A place action must have a string target.
- Never place an object twice without picking it again.
- Never append duplicate place actions.
- Never repeat the same manipulation pair unless that pair is
  explicitly listed as a required manipulation pair.
- Keep the plan as short as possible.

CRITICAL REQUIRED-PAIR RULE:

The field "required_pairs" is a HARD CONSTRAINT.

If required_pairs is provided and is not empty:

- You MUST output ONLY the manipulation pair(s) listed in
  required_pairs.
- Do NOT invent additional manipulation pairs.
- Do NOT add exploratory actions.
- Do NOT add recovery actions involving unrelated objects.
- Do NOT change the target of a required pair.
- Do NOT repeat a required pair.
- For each required pair, output exactly:
      pick <object>
      place <object> on <target>
- The object and target names must exactly match the supplied
  required_pairs.
- If there is exactly one required pair, the plan must contain
  exactly two actions: one pick followed immediately by one place.
- If there are multiple required pairs, output exactly one
  pick/place pair for each required pair, in the supplied order.

Example:

required_pairs:
[
    ["red block", "blue block"]
]

Correct:
    pick red block
    place red block on blue block

Incorrect:
    pick green block
    place green block on blue block
    pick red block
    place red block on blue block

Incorrect:
    pick red block
    place red block on blue block
    place red block on blue block

Incorrect:
    pick red block
    place red block on green block

The deterministic validator will reject any manipulation pair that
is not required.

VALIDATION ERRORS:

- If validation_errors are supplied, fix EVERY validation error.
- Do not repeat the exact invalid action sequence.
- Use the required_pairs field as the authoritative description
  of what manipulation work remains.
- Validation errors explain why the previous candidate failed;
  they do NOT authorize adding unrelated manipulation pairs.

EXECUTION FEEDBACK:

- Use execution feedback to understand why the previous physical
  attempt failed.
- Replan the remaining symbolic task.
- Do not claim that the failed action succeeded.
- Do not invent a different manipulation objective merely because
  execution failed.

The plan will be checked by a separate deterministic validator.

Return only the structured symbolic plan.
""".strip()

    @staticmethod
    def _build_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string"
                },
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "pick",
                                    "place"
                                ]
                            },
                            "object": {
                                "type": "string"
                            },
                            "target": {
                                "type": [
                                    "string",
                                    "null"
                                ]
                            }
                        },
                        "required": [
                            "type",
                            "object",
                            "target"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": [
                "goal",
                "actions"
            ],
            "additionalProperties": False
        }

    @staticmethod
    def _normalize_required_pairs(
        required_pairs: Optional[
            List[Tuple[str, str]]
        ],
    ) -> List[List[str]]:
        """
        Convert required manipulation pairs into a JSON-friendly
        representation.

        Example:
            [
                ("red block", "blue block")
            ]

        becomes:
            [
                ["red block", "blue block"]
            ]
        """

        if required_pairs is None:
            return []

        normalized = []

        for index, pair in enumerate(required_pairs):

            if not isinstance(pair, (list, tuple)):
                raise TypeError(
                    f"required_pairs[{index}] must be "
                    "a list or tuple containing "
                    "(object, target)."
                )

            if len(pair) != 2:
                raise ValueError(
                    f"required_pairs[{index}] must contain "
                    "exactly two values: (object, target)."
                )

            object_name, target_name = pair

            if not isinstance(object_name, str):
                raise TypeError(
                    f"required_pairs[{index}][0] must be a string."
                )

            if not isinstance(target_name, str):
                raise TypeError(
                    f"required_pairs[{index}][1] must be a string."
                )

            if not object_name.strip():
                raise ValueError(
                    f"required_pairs[{index}][0] cannot be empty."
                )

            if not target_name.strip():
                raise ValueError(
                    f"required_pairs[{index}][1] cannot be empty."
                )

            normalized.append(
                [
                    object_name,
                    target_name,
                ]
            )

        return normalized

    def replan(
        self,
        instruction: str,
        world_state: Dict[str, Any],
        previous_plan: Dict[str, Any],
        execution_feedback: Dict[str, Any],
        validation_errors: Optional[List[str]] = None,
        required_pairs: Optional[
            List[Tuple[str, str]]
        ] = None,
    ) -> Dict[str, Any]:

        if not instruction or not instruction.strip():
            raise ValueError(
                "instruction must be a non-empty string."
            )

        if not isinstance(world_state, dict):
            raise TypeError(
                "world_state must be a dictionary."
            )

        if not isinstance(previous_plan, dict):
            raise TypeError(
                "previous_plan must be a dictionary."
            )

        if not isinstance(execution_feedback, dict):
            raise TypeError(
                "execution_feedback must be a dictionary."
            )

        if validation_errors is None:
            validation_errors = []

        if not isinstance(validation_errors, list):
            raise TypeError(
                "validation_errors must be a list."
            )

        normalized_required_pairs = (
            self._normalize_required_pairs(
                required_pairs
            )
        )

        user_prompt = {
            "original_instruction": instruction,
            "world_state": world_state,
            "previous_plan": previous_plan,
            "execution_feedback": execution_feedback,
            "validation_errors": validation_errors,

            # This is the important addition.
            # It explicitly tells the LLM which manipulation
            # pairs remain and prevents unrelated recovery actions.
            "required_pairs": normalized_required_pairs,
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        user_prompt,
                        indent=2,
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_replanned_task",
                    "strict": True,
                    "schema": self._build_schema(),
                },
            },
            reasoning_effort="medium",
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty replanner response."
            )

        try:
            plan = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Groq returned content that could not be parsed as JSON.\n"
                f"Raw response:\n{content}"
            ) from exc

        self._basic_validate_plan(plan)

        return plan

    @staticmethod
    def _basic_validate_plan(
        plan: Dict[str, Any]
    ) -> None:

        if not isinstance(plan, dict):
            raise ValueError(
                "Replanner output must be a dictionary."
            )

        if "goal" not in plan:
            raise ValueError(
                "Replanner output is missing 'goal'."
            )

        if "actions" not in plan:
            raise ValueError(
                "Replanner output is missing 'actions'."
            )

        if not isinstance(plan["goal"], str):
            raise ValueError(
                "'goal' must be a string."
            )

        if not isinstance(plan["actions"], list):
            raise ValueError(
                "'actions' must be a list."
            )

        for index, action in enumerate(
            plan["actions"]
        ):

            if not isinstance(action, dict):
                raise ValueError(
                    f"Action {index} must be an object."
                )

            if action.get("type") not in {
                "pick",
                "place",
            }:
                raise ValueError(
                    f"Action {index} has invalid type: "
                    f"{action.get('type')}"
                )

            if not isinstance(
                action.get("object"),
                str,
            ):
                raise ValueError(
                    f"Action {index} has an invalid object."
                )

            if action["type"] == "pick":

                if action.get("target") is not None:
                    raise ValueError(
                        f"Pick action {index} "
                        "must have target=null."
                    )

            if action["type"] == "place":

                if not isinstance(
                    action.get("target"),
                    str,
                ):
                    raise ValueError(
                        f"Place action {index} "
                        "must have a string target."
                    )


if __name__ == "__main__":

    replanner = LLMTaskReplanner()

    instruction = (
        "Pick the blue block and place it on the green block."
    )

    world_state = {
        "objects": [
            {
                "name": "blue block",
                "type": "block",
                "color": "blue",
                "position": [
                    0.09,
                    -0.62,
                    0.021,
                ],
            },
            {
                "name": "green block",
                "type": "block",
                "color": "green",
                "position": [
                    -0.15,
                    -0.62,
                    0.021,
                ],
            },
            {
                "name": "red block",
                "type": "block",
                "color": "red",
                "position": [
                    0.03,
                    -0.33,
                    0.021,
                ],
            },
        ]
    }

    previous_plan = {
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

    execution_feedback = {
        "execution_success": False,
        "failure_type": "movement_failure",
        "failed_action": {
            "type": "place",
            "object": "blue block",
            "target": "green block",
        },
        "feedback": (
            "The robot successfully grasped and lifted "
            "the blue block, but failed to move it "
            "to the intended target, green block."
        ),
        "reason": (
            "The object did not move far enough "
            "from its initial position."
        ),
        "metrics": {
            "lift_height": 0.203,
            "pick_displacement": 0.001,
            "placement_xy_error": 0.239,
            "placement_tolerance": 0.035,
        },
    }

    validation_errors = [
        "Object 'blue block' is placed multiple times "
        "without being picked again.",
    ]

    required_pairs = [
        ("blue block", "green block")
    ]

    plan = replanner.replan(
        instruction=instruction,
        world_state=world_state,
        previous_plan=previous_plan,
        execution_feedback=execution_feedback,
        validation_errors=validation_errors,
        required_pairs=required_pairs,
    )

    print("\nReplanned task plan:")
    print(json.dumps(plan, indent=2))
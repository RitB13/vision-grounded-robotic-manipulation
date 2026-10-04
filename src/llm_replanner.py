import json
import os
from typing import Any, Dict, List, Optional

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
        - optional validation errors from a rejected replan

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

- Produce only a high-level symbolic plan.
- Never output robot coordinates.
- Never output joint angles.
- Never output motor commands.
- Never perform inverse kinematics.
- Never claim that an action has already succeeded.
- Do not execute the task.
- Do not use tools.
- Use only objects present in the supplied world state.
- Preserve the user's original goal.
- Do not invent objects.
- Every place action must be preceded by a pick of the same object
  that is currently being held.
- Never place an object twice without picking it again.
- Do not append duplicate place actions.
- A normal pick-and-place task should contain exactly:
    pick object
    place that same object on the intended target
- If validation errors are supplied, fix EVERY validation error.
- Do not repeat the exact invalid action sequence.
- Keep the plan as short as possible while satisfying the goal.
- The plan will be checked by a separate deterministic validator.

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

    def replan(
        self,
        instruction: str,
        world_state: Dict[str, Any],
        previous_plan: Dict[str, Any],
        execution_feedback: Dict[str, Any],
        validation_errors: Optional[List[str]] = None,
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

        user_prompt = {
            "original_instruction": instruction,
            "world_state": world_state,
            "previous_plan": previous_plan,
            "execution_feedback": execution_feedback,
            "validation_errors": validation_errors,
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
    def _basic_validate_plan(plan: Dict[str, Any]) -> None:

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

        for index, action in enumerate(plan["actions"]):

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
        "Action 2 is a place action but no object has been picked.",
        "Object 'blue block' is placed multiple times without being picked again.",
    ]

    plan = replanner.replan(
        instruction=instruction,
        world_state=world_state,
        previous_plan=previous_plan,
        execution_feedback=execution_feedback,
        validation_errors=validation_errors,
    )

    print("\nReplanned task plan:")
    print(json.dumps(plan, indent=2))
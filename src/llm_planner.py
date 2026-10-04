import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from the project's .env file.
load_dotenv()


MODEL_NAME = "openai/gpt-oss-120b"


class LLMTaskPlanner:
    """
    High-level task planner using GPT-OSS-120B through the Groq API.

    The LLM is responsible only for symbolic task planning.

    It does NOT:
        - generate robot joint angles
        - generate Cartesian coordinates
        - control the robot
        - perform inverse kinematics
        - execute PyBullet actions

    Those responsibilities remain in the robotics/vision layer.
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
You are the high-level task planner for a simulated robotic manipulation system.

Your job is to convert a user's natural-language manipulation instruction
into a structured symbolic task plan.

The robot operates in a tabletop environment containing named objects such as:

- blue block
- red block
- green block
- yellow block
- orange block
- blue sphere
- red sphere
- blue bowl
- red bowl
- green bowl
- yellow bowl
- orange bowl

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
- Never invent physical coordinates.
- Do not claim that an action has already succeeded.
- Do not execute the task.
- Do not use tools.
- Use only the available object names supplied by the user/system.
- Preserve the user's intended object identities.
- A normal pick-and-place task should contain a pick action followed by a place action.
- For multi-step tasks, generate the actions in the order they should be executed.
- If the instruction asks for an unsupported operation, represent the closest valid
  high-level interpretation only when it is unambiguous.

The plan will be validated by a separate deterministic validator before execution.
Therefore, focus on producing a clear, minimal, logically ordered symbolic plan.
""".strip()

    @staticmethod
    def _build_schema() -> Dict[str, Any]:
        """
        JSON Schema for the structured planner output.

        GPT-OSS-120B on Groq supports strict JSON Schema structured outputs.
        All fields are therefore required and objects disallow extra fields.
        """

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

    def plan(
        self,
        instruction: str,
        world_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured symbolic task plan.

        Parameters
        ----------
        instruction:
            Natural-language task given by the user.

        world_state:
            Optional description of the currently available scene.

        Returns
        -------
        dict
            Structured task plan generated by the LLM.
        """

        if not instruction or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")

        if world_state is None:
            world_state = {}

        user_prompt = {
            "instruction": instruction,
            "world_state": world_state,
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
                        indent=2
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_task_plan",
                    "strict": True,
                    "schema": self._build_schema(),
                },
            },
            reasoning_effort="medium",
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty planner response."
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
        """
        Lightweight structural validation.

        This is intentionally NOT the full plan validator.

        A separate plan_validator.py module will later perform:
            - object existence checks
            - action ordering checks
            - target validity checks
            - scene consistency checks
            - execution feasibility checks
        """

        if not isinstance(plan, dict):
            raise ValueError("Planner output must be a dictionary.")

        if "goal" not in plan:
            raise ValueError("Planner output is missing 'goal'.")

        if "actions" not in plan:
            raise ValueError("Planner output is missing 'actions'.")

        if not isinstance(plan["goal"], str):
            raise ValueError("'goal' must be a string.")

        if not isinstance(plan["actions"], list):
            raise ValueError("'actions' must be a list.")

        for index, action in enumerate(plan["actions"]):
            if not isinstance(action, dict):
                raise ValueError(
                    f"Action {index} must be an object."
                )

            if action.get("type") not in {"pick", "place"}:
                raise ValueError(
                    f"Action {index} has invalid type: "
                    f"{action.get('type')}"
                )

            if not isinstance(action.get("object"), str):
                raise ValueError(
                    f"Action {index} has an invalid object."
                )

            if action["type"] == "pick":
                if action.get("target") is not None:
                    raise ValueError(
                        f"Pick action {index} must have target=null."
                    )

            if action["type"] == "place":
                if not isinstance(action.get("target"), str):
                    raise ValueError(
                        f"Place action {index} must have a string target."
                    )


if __name__ == "__main__":
    planner = LLMTaskPlanner()

    instruction = (
        "Pick the blue block and place it on the green block."
    )

    world_state = {
        "objects": [
            "blue block",
            "green block",
            "red block",
        ]
    }

    plan = planner.plan(
        instruction=instruction,
        world_state=world_state,
    )

    print("\nGenerated task plan:")
    print(json.dumps(plan, indent=2))
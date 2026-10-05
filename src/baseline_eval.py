"""
Baseline evaluation for the vision-grounded robotic manipulation system.

Purpose
-------
Evaluate task execution when a controlled execution failure occurs,
WITHOUT invoking LLM-based replanning.

This provides the baseline against which the feedback-based
replanning system can later be compared.

Experiment flow
---------------
1. Initialize PyBullet environment.
2. Load CLIPort.
3. Reset environment with the experiment configuration.
4. Build initial world state.
5. Generate an LLM task plan.
6. Validate the plan deterministically.
7. Execute the first manipulation normally.
8. Inject a controlled failure into the second manipulation.
9. Record execution feedback.
10. STOP -- no replanning or recovery is attempted.
11. Save the complete experiment log as JSON.

Supported failure modes
-----------------------
- none
- movement_failure
- placement_failure
- grasp_failure
"""


from __future__ import annotations
import pybullet
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .environment import PickPlaceEnv
from .world_state import WorldStateBuilder
from .llm_planner import LLMTaskPlanner
from .plan_validator import PlanValidator
from .plan_to_cliport import extract_pick_place_pairs, pair_to_instruction
from .execution_feedback import ExecutionFeedbackBuilder
from .cliport_policy import CLIPortPolicy
from .config import CHECKPOINT_PATH
from .failure_injector import FailureInjector


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

FAILURE_MODE = "grasp_failure"
EXPERIMENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "100"))

# Physical manipulation step receiving the controlled failure.
FAILURE_STEP = 2

TASK_INSTRUCTION = (
    "Pick the blue block and place it on the green block, "
    "then pick the red block and place it on the blue block."
)

CONFIG = {
    "pick": [
        "blue block",
        "green block",
        "red block",
    ],
    "place": [],
}

RESULT_PATH = Path(
    "baseline_experiment_result.json"
)


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

def to_json_safe(value: Any) -> Any:
    """
    Convert NumPy and other non-JSON-native values into
    JSON-serializable Python values.
    """

    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if value is None:
        return None

    return str(value)


def save_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    """
    Save experiment data as formatted JSON.
    """

    safe_data = to_json_safe(data)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            safe_data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# PRINTING HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_world_state(
    world_state: Dict[str, Any],
) -> None:

    print()
    print("Current world state:")

    for obj in world_state.get(
        "objects",
        [],
    ):

        print(
            f"  {obj.get('name')}: "
            f"type={obj.get('type')}, "
            f"color={obj.get('color')}, "
            f"position={obj.get('position')}"
        )


def print_plan(
    plan: Dict[str, Any],
) -> None:

    print(
        json.dumps(
            plan,
            indent=2,
        )
    )


def print_validation(
    validation: Dict[str, Any],
) -> None:

    print(
        json.dumps(
            validation,
            indent=2,
        )
    )


def print_execution_result(
    task_result: Dict[str, Any],
) -> None:

    print()
    print("Task result:")

    fields = [
        "success",
        "pick_name",
        "place_name",
        "grasp_success",
        "movement_success",
        "placement_success",
        "initial_position",
        "lift_position",
        "final_position",
        "target_position",
        "lift_height",
        "pick_displacement",
        "placement_xy_error",
        "vertical_error",
        "placement_tolerance",
        "reason",
        "action_pick_xyz",
        "action_place_xyz",
    ]

    for field in fields:

        if field in task_result:
            print(
                f"  {field}: "
                f"{task_result[field]}"
            )


def print_feedback(
    feedback: Dict[str, Any],
) -> None:

    print()
    print("Structured execution feedback:")

    print(
        json.dumps(
            feedback,
            indent=2,
        )
    )


# ============================================================
# PHYSICAL EXECUTION
# ============================================================

def execute_pair(
    env: PickPlaceEnv,
    policy: CLIPortPolicy,
    failure_injector: FailureInjector,
    pair: Dict[str, str],
    step_number: int,
    inject_failure: bool,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    float,
    bool,
    Dict[str, Any],
]:
    """
    Execute one pick/place pair.

    No replanning is performed here.

    For grasp_failure, the failure injector receives the
    current world-state object positions so it can choose
    a safe empty workspace location rather than blindly
    shifting the pick coordinate toward another object.
    """

    instruction = pair_to_instruction(
        pair
    )

    print_header(
        f"PHYSICAL STEP {step_number}"
    )

    print(
        f"Semantic instruction: "
        f"{instruction}"
    )

    # --------------------------------------------------------
    # Obtain current observation
    # --------------------------------------------------------

    obs = env.get_observation()

    obs["_instruction"] = instruction

    # --------------------------------------------------------
    # Run CLIPort
    # --------------------------------------------------------

    print()
    print("Running CLIPort...")

    prediction = policy.predict(
        obs
    )

    print()
    print("CLIPort prediction:")

    print(
        f"  pick pixel : "
        f"{prediction['pick_yx']}"
    )

    print(
        f"  pick XYZ   : "
        f"{prediction['pick_xyz']}"
    )

    print(
        f"  place pixel: "
        f"{prediction['place_yx']}"
    )

    print(
        f"  place XYZ  : "
        f"{prediction['place_xyz']}"
    )

    # --------------------------------------------------------
    # Construct original CLIPort action
    # --------------------------------------------------------

    original_action = {
        "pick": prediction["pick_xyz"],
        "place": prediction["place_xyz"],
        "pick_name": pair["pick"],
        "place_name": pair["place"],
    }

    # --------------------------------------------------------
    # Build current object-position map for safe failure
    # injection.
    # --------------------------------------------------------

    object_positions = {}

    for name, body_id in env.obj_name_to_id.items():
        position, _ = (
            pybullet.getBasePositionAndOrientation(
                body_id
            )
        )

        object_positions[name] = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ]

    # --------------------------------------------------------
    # Apply controlled failure
    # --------------------------------------------------------

    failure_enabled = (
        inject_failure
        and FAILURE_MODE != "none"
    )

    executed_action, failure_metadata = (
        failure_injector.apply(
            original_action,
            enabled=failure_enabled,
            object_positions=object_positions,
        )
    )

    if failure_metadata.get(
        "injected",
        False,
    ):

        print()
        print(
            "⚠️ CONTROLLED FAILURE INJECTION"
        )

        print(
            f"Failure mode: "
            f"{failure_metadata['mode']}"
        )

        print(
            f"Description: "
            f"{failure_metadata['description']}"
        )

        if "original_pick" in failure_metadata:

            print(
                "Original pick XYZ: "
                f"{failure_metadata['original_pick']}"
            )

        if "modified_pick" in failure_metadata:

            print(
                "Modified pick XYZ: "
                f"{failure_metadata['modified_pick']}"
            )

        if "original_place" in failure_metadata:

            print(
                "Original place XYZ: "
                f"{failure_metadata['original_place']}"
            )

        if "modified_place" in failure_metadata:

            print(
                "Modified place XYZ: "
                f"{failure_metadata['modified_place']}"
            )

    else:

        print()
        print(
            "No failure injected. "
            "Using normal CLIPort action."
        )

    # --------------------------------------------------------
    # Execute in PyBullet
    # --------------------------------------------------------

    print()
    print(
        "Executing robot action..."
    )

    (
        new_obs,
        reward,
        done,
        info,
    ) = env.step(
        executed_action
    )

    task_result = info.get(
        "task_result"
    )

    if task_result is None:

        raise RuntimeError(
            "Environment did not return "
            "task_result."
        )

    # --------------------------------------------------------
    # Execution result
    # --------------------------------------------------------

    print()
    print(
        "Execution result:"
    )

    print(
        f"  Reward: {reward}"
    )

    print(
        f"  Task success: "
        f"{task_result.get('success')}"
    )

    print_execution_result(
        task_result
    )

    return (
        new_obs,
        prediction,
        original_action,
        executed_action,
        failure_metadata,
        reward,
        done,
        task_result,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "BASELINE FAILURE EVALUATION"
    )

    print()
    print("Task:")
    print(
        f"  {TASK_INSTRUCTION}"
    )

    print()
    print("Baseline configuration:")

    print(
        f"  Failure mode: "
        f"{FAILURE_MODE}"
    )

    print(
        f"  Failure step: "
        f"{FAILURE_STEP}"
    )

    print(
        "  Replanning: DISABLED"
    )

    # --------------------------------------------------------
    # Experiment log
    # --------------------------------------------------------

    experiment_log: Dict[str, Any] = {

        "experiment":
            "baseline_failure_evaluation",

        "timestamp":
            datetime.now().isoformat(),

        "task_instruction":
            TASK_INSTRUCTION,

        "failure_configuration": {
            "mode": FAILURE_MODE,
            "step": FAILURE_STEP,
        },

        "replanning_used":
            False,

        "initial_world_state":
            None,

        "experiment_seed": EXPERIMENT_SEED,

        "initial_plan":
            None,

        "initial_validation":
            None,

        "initial_pairs":
            [],

        "steps":
            [],

        "final_summary":
            {},
    }

    # --------------------------------------------------------
    # Initialize environment
    # --------------------------------------------------------

    print_header(
        "INITIALIZING ENVIRONMENT"
    )

    env = PickPlaceEnv(
        gui=True
    )

    try:

        print(
            "Environment object created."
        )

        # ----------------------------------------------------
        # Initialize components
        # ----------------------------------------------------

        world_state_builder = (
            WorldStateBuilder(env)
        )

        planner = LLMTaskPlanner()

        validator = PlanValidator()

        failure_injector = (
            FailureInjector(
                mode=FAILURE_MODE
            )
        )

        # ----------------------------------------------------
        # Load CLIPort
        # ----------------------------------------------------

        print_header(
            "LOADING CLIPORT"
        )

        if not CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                "Missing CLIPort checkpoint: "
                f"{CHECKPOINT_PATH}"
            )

        policy = CLIPortPolicy(
            CHECKPOINT_PATH
        )

        print(
            "CLIPort loaded."
        )

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        print_header(
            "RESETTING ENVIRONMENT"
        )

        np.random.seed(EXPERIMENT_SEED)
        print(f"Experiment seed: {EXPERIMENT_SEED}")
        obs = env.reset(
            CONFIG
        )

        print(
            "Environment initialized."
        )

        # ----------------------------------------------------
        # Initial world state
        # ----------------------------------------------------

        print_header(
            "BUILDING INITIAL WORLD STATE"
        )

        world_state = (
            world_state_builder.build()
        )

        experiment_log[
            "initial_world_state"
        ] = world_state

        print_world_state(
            world_state
        )

        # ----------------------------------------------------
        # Initial LLM plan
        # ----------------------------------------------------

        print_header(
            "GENERATING INITIAL PLAN"
        )

        initial_plan = planner.plan(
            instruction=TASK_INSTRUCTION,
            world_state=world_state,
        )

        experiment_log[
            "initial_plan"
        ] = initial_plan

        print()
        print(
            "Initial LLM plan:"
        )

        print_plan(
            initial_plan
        )

        # ----------------------------------------------------
        # Validate initial plan
        # ----------------------------------------------------

        print_header(
            "VALIDATING INITIAL PLAN"
        )

        initial_validation = (
            validator.validate(
                plan=initial_plan,
                world_state=world_state,
            )
        )

        experiment_log[
            "initial_validation"
        ] = initial_validation

        print_validation(
            initial_validation
        )

        if not initial_validation[
            "valid"
        ]:

            print()
            print(
                "❌ INITIAL PLAN REJECTED"
            )

            experiment_log[
                "final_summary"
            ] = {

                "failure_mode":
                    FAILURE_MODE,

                "failure_step":
                    FAILURE_STEP,

                "initial_plan_valid":
                    False,

                "initial_steps":
                    0,

                "completed_steps":
                    0,

                "failure_occurred":
                    False,

                "replanning_used":
                    False,

                "recovery_attempted":
                    False,

                "recovery_success":
                    False,

                "task_success":
                    False,

                "experiment_completed":
                    False,

                "reason":
                    "Initial plan failed deterministic validation.",
            }

            save_json(
                RESULT_PATH,
                experiment_log
            )

            print()
            print(
                "Experiment result saved to:"
            )

            print(
                f"  {RESULT_PATH}"
            )

            return

        print()
        print(
            "✅ Initial plan accepted."
        )

        # ----------------------------------------------------
        # Extract physical manipulation pairs
        # ----------------------------------------------------

        pairs = extract_pick_place_pairs(
            initial_plan
        )

        experiment_log[
            "initial_pairs"
        ] = pairs

        print()
        print(
            "Initial manipulation pairs:"
        )

        for index, pair in enumerate(
            pairs,
            start=1,
        ):

            print(
                f"  Step {index}: "
                f"{pair['pick']} -> "
                f"{pair['place']}"
            )

        if not pairs:

            raise RuntimeError(
                "No executable "
                "pick/place pairs found."
            )

        # ----------------------------------------------------
        # Execute baseline
        # ----------------------------------------------------

        completed_steps = 0
        failure_occurred = False

        for (
            step_number,
            pair,
        ) in enumerate(
            pairs,
            start=1,
        ):

            # Only the configured step gets
            # the controlled failure.

            inject_failure = (
                step_number == FAILURE_STEP
                and FAILURE_MODE != "none"
            )

            (
                new_obs,
                prediction,
                original_action,
                executed_action,
                failure_metadata,
                reward,
                done,
                task_result,
            ) = execute_pair(
                env=env,
                policy=policy,
                failure_injector=failure_injector,
                pair=pair,
                step_number=step_number,
                inject_failure=inject_failure,
            )

            feedback_builder = (
                ExecutionFeedbackBuilder()
            )

            feedback = (
                feedback_builder.build(
                    task_result
                )
            )

            step_log = {

                "step_number":
                    step_number,

                "pair":
                    pair,

                "instruction":
                    pair_to_instruction(
                        pair
                    ),

                "prediction":
                    prediction,

                "original_action":
                    original_action,

                "executed_action":
                    executed_action,

                "failure_injected":
                    failure_metadata.get(
                        "injected",
                        False,
                    ),

                "failure_mode":
                    failure_metadata.get(
                        "mode"
                    ),

                "failure_metadata":
                    failure_metadata,

                "reward":
                    reward,

                "done":
                    done,

                "task_result":
                    task_result,

                "execution_feedback":
                    feedback,
            }

            experiment_log[
                "steps"
            ].append(
                step_log
            )

            # ------------------------------------------------
            # Successful step
            # ------------------------------------------------

            if task_result.get(
                "success"
            ):

                completed_steps += 1

                print()
                print(
                    f"✅ STEP "
                    f"{step_number} "
                    f"SUCCEEDED"
                )

                if step_number < len(
                    pairs
                ):

                    world_state = (
                        world_state_builder.build()
                    )

                    print()
                    print(
                        "World state after "
                        "successful step:"
                    )

                    print_world_state(
                        world_state
                    )

                continue

            # ------------------------------------------------
            # Failure
            # ------------------------------------------------

            failure_occurred = True

            print()
            print(
                f"❌ STEP "
                f"{step_number} "
                f"FAILED"
            )

            print()
            print("=" * 70)
            print(
                "BASELINE STOPS AFTER FAILURE"
            )
            print("=" * 70)

            print()
            print(
                "Replanning is DISABLED "
                "for this experiment."
            )

            print(
                "No recovery attempt "
                "will be made."
            )

            break

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        task_success = (
            not failure_occurred
            and completed_steps == len(pairs)
        )

        experiment_log[
            "final_summary"
        ] = {

            "failure_mode":
                FAILURE_MODE,

            "failure_step":
                FAILURE_STEP,

            "initial_plan_valid":
                True,

            "initial_steps":
                len(pairs),

            "completed_steps":
                completed_steps,

            "failure_occurred":
                failure_occurred,

            "replanning_used":
                False,

            "recovery_attempted":
                False,

            "recovery_success":
                False,

            "task_success":
                task_success,

            "experiment_completed":
                True,
        }

        # ----------------------------------------------------
        # Print final summary
        # ----------------------------------------------------

        print_header(
            "BASELINE EXPERIMENT SUMMARY"
        )

        print(
            f"Failure mode           : "
            f"{FAILURE_MODE}"
        )

        print(
            f"Failure step           : "
            f"{FAILURE_STEP}"
        )

        print(
            "Initial plan valid     : "
            "True"
        )

        print(
            f"Initial steps          : "
            f"{len(pairs)}"
        )

        print(
            f"Completed steps        : "
            f"{completed_steps} / "
            f"{len(pairs)}"
        )

        print(
            f"Failure occurred       : "
            f"{failure_occurred}"
        )

        print(
            "Replanning used        : "
            "False"
        )

        print(
            "Recovery attempted     : "
            "False"
        )

        print(
            f"Task success           : "
            f"{task_success}"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        print()
        print(
            "Converting experiment log "
            "to JSON-safe format..."
        )

        save_json(
            RESULT_PATH,
            experiment_log
        )

        print()
        print(
            "Experiment result saved to:"
        )

        print(
            f"  {RESULT_PATH}"
        )

    finally:

        # Always close PyBullet, even if an
        # exception occurs during the experiment.

        env.close()


if __name__ == "__main__":
    main()
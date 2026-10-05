"""
Multi-step LLM replanning experiment.

Task:
    1. Pick the blue block and place it on the green block.
    2. Pick the red block and place it on the blue block.

Experiment:
    - Step 1 executes normally and must succeed.
    - Step 2 is deliberately forced to fail.
    - Structured execution feedback is generated.
    - The world state is rebuilt after the failure.
    - The LLM replans the remaining task.
    - The replanned symbolic plan is deterministically validated.
    - Invalid/unnecessary manipulation pairs are rejected.
    - The repaired Step 2 is executed normally.
    - The final experiment result is saved to JSON.

Important:
    This experiment does NOT claim that the LLM discovered a
    fundamentally different manipulation strategy.

    It demonstrates:
        multi-step planning
        partial execution
        execution failure detection
        world-state updating
        feedback generation
        LLM replanning
        deterministic plan validation
        successful recovery
"""

import json
import os
from pathlib import Path

import numpy as np
import pybullet


from .environment import PickPlaceEnv
from .world_state import WorldStateBuilder
from .llm_planner import LLMTaskPlanner
from .llm_replanner import LLMTaskReplanner
from .plan_validator import PlanValidator
from .plan_to_cliport import (
    extract_pick_place_pairs,
    pair_to_instruction,
)
from .execution_feedback import ExecutionFeedbackBuilder
from .failure_injector import FailureInjector
from .cliport_policy import CLIPortPolicy
from .config import CHECKPOINT_PATH


# ============================================================
# CONFIGURATION
# ============================================================

MAX_PLAN_REPAIR_ATTEMPTS = 3


# ------------------------------------------------------------
# Controlled failure experiment
# ------------------------------------------------------------
#
# Change ONLY FAILURE_MODE to run a different failure experiment:
#
#   "grasp_failure"
#   "movement_failure"
#   "placement_failure"
#   "none"
#
# The failure is injected only on CONTROLLED_FAILURE_STEP.

CONTROLLED_FAILURE_STEP = 2

FAILURE_MODE = "grasp_failure"

EXPERIMENT_SEED = int(
    os.environ.get(
        "EXPERIMENT_SEED",
        "100",
    )
)


VALID_FAILURE_MODES = {
    "none",
    "grasp_failure",
    "movement_failure",
    "placement_failure",
}


if FAILURE_MODE not in VALID_FAILURE_MODES:

    raise ValueError(
        f"Invalid FAILURE_MODE: {FAILURE_MODE!r}. "
        f"Choose one of: {sorted(VALID_FAILURE_MODES)}"
    )


FAILURE_INJECTOR = FailureInjector(
    mode=FAILURE_MODE,
    placement_offset=0.08,
    grasp_offset=0.12,
)


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
    "replanning_experiment_result.json"
)


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def make_json_safe(value):
    """
    Recursively convert NumPy/Python values into objects that
    can be serialized by json.dumps().

    This is needed because CLIPort predictions and PyBullet
    results can contain NumPy arrays or NumPy scalar values.
    """

    # NumPy arrays
    if hasattr(value, "tolist"):

        try:

            return make_json_safe(
                value.tolist()
            )

        except Exception:
            pass

    # NumPy scalar values
    if hasattr(value, "item") and callable(
        value.item
    ):

        try:

            return value.item()

        except Exception:
            pass

    # Dictionaries
    if isinstance(value, dict):

        return {
            key: make_json_safe(val)
            for key, val in value.items()
        }

    # Lists / tuples
    if isinstance(value, (list, tuple)):

        return [
            make_json_safe(item)
            for item in value
        ]

    # Primitive Python values
    return value


# ============================================================
# PRINTING HELPERS
# ============================================================

def print_section(title):

    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


def print_plan(title, plan):

    print(
        f"\n{title}"
    )

    print(
        json.dumps(
            make_json_safe(plan),
            indent=2,
        )
    )


def print_world_state(
    title,
    world_state,
):

    print(
        f"\n{title}"
    )

    for obj in world_state["objects"]:

        print(
            f"  {obj['name']}: "
            f"type={obj['type']}, "
            f"color={obj['color']}, "
            f"position={obj['position']}"
        )


def print_execution_result(
    task_result
):

    print(
        "\nTask result:"
    )

    for key, value in task_result.items():

        print(
            f"  {key}: {value}"
        )


# ============================================================
# PLAN VALIDATION
# ============================================================

def validate_plan(
    validator,
    plan,
    world_state,
    required_pairs=None,
):
    """
    Validate a symbolic plan against the current world state.

    required_pairs:
        Optional list of required manipulation pairs.

        Example:

            [
                ("red block", "blue block")
            ]

        When supplied, the deterministic validator rejects
        candidate manipulation pairs that are unrelated to
        the remaining task.
    """

    return validator.validate(
        plan=plan,
        world_state=world_state,
        required_pairs=required_pairs,
    )


# ============================================================
# REPLANNING / PLAN REPAIR
# ============================================================

def repair_plan_until_valid(
    replanner,
    validator,
    instruction,
    world_state,
    previous_plan,
    execution_feedback,
    required_pairs,
):
    """
    Ask the LLM to generate a revised symbolic plan.

    Every candidate plan is checked by the deterministic
    validator.

    If the LLM generates an invalid or unnecessary plan,
    the validator errors are fed back into the next
    replanning attempt.

    required_pairs contains the manipulation pairs that
    are still required after partial execution.

    Example:

        required_pairs = [
            ("red block", "blue block")
        ]

    If the LLM produces:

        green -> blue
        red -> blue

    the validator rejects the candidate because
    green -> blue is not required.

    Returns
    -------
    tuple
        (
            validated_plan,
            validation_result,
            number_of_repair_attempts
        )

    Raises
    ------
    RuntimeError
        If no valid plan is produced.
    """

    current_plan = previous_plan

    validation_errors = []

    for repair_attempt in range(
        1,
        MAX_PLAN_REPAIR_ATTEMPTS + 1,
    ):

        print(
            f"\nPlan repair attempt "
            f"{repair_attempt} / "
            f"{MAX_PLAN_REPAIR_ATTEMPTS}"
        )

        # ----------------------------------------------------
        # Ask LLM for revised plan
        # ----------------------------------------------------

        current_plan = replanner.replan(
            instruction=instruction,
            world_state=world_state,
            previous_plan=current_plan,
            execution_feedback=execution_feedback,
            validation_errors=validation_errors,
            required_pairs=required_pairs,
        )

        print_plan(
            "Candidate replanned task:",
            current_plan,
        )

        # ----------------------------------------------------
        # Deterministic validation
        # ----------------------------------------------------

        validation = validate_plan(
            validator=validator,
            plan=current_plan,
            world_state=world_state,
            required_pairs=required_pairs,
        )

        print(
            "\nValidation result:"
        )

        print(
            json.dumps(
                make_json_safe(validation),
                indent=2,
            )
        )

        # ----------------------------------------------------
        # VALID PLAN
        # ----------------------------------------------------

        if validation["valid"]:

            print(
                "\nREPLANNED PLAN PASSED VALIDATION"
            )

            validated_plan = {
                "goal": current_plan["goal"],
                "actions": validation[
                    "validated_actions"
                ],
            }

            return (
                validated_plan,
                validation,
                repair_attempt,
            )

        # ----------------------------------------------------
        # INVALID PLAN
        # ----------------------------------------------------

        print(
            "\nREPLANNED PLAN REJECTED"
        )

        validation_errors = validation[
            "errors"
        ]

        for error in validation_errors:

            print(
                f"  - {error}"
            )

        if repair_attempt < MAX_PLAN_REPAIR_ATTEMPTS:

            print(
                "\nSending validator errors back "
                "to the LLM for another repair."
            )

    raise RuntimeError(
        "The replanner failed to produce a valid "
        "plan after "
        f"{MAX_PLAN_REPAIR_ATTEMPTS} "
        "repair attempts."
    )


# ============================================================
# PLAN / PAIR HELPERS
# ============================================================

def extract_pairs(plan):
    """
    Convert a symbolic pick/place plan into physical
    manipulation pairs.

    Example:

        pick blue
        place blue on green
        pick red
        place red on blue

    becomes:

        [
            {
                "pick": "blue block",
                "place": "green block"
            },
            {
                "pick": "red block",
                "place": "blue block"
            }
        ]
    """

    pairs = extract_pick_place_pairs(
        plan
    )

    if not pairs:

        raise RuntimeError(
            "No valid pick/place pairs could be "
            "extracted from the plan."
        )

    return pairs


def pair_is_completed(
    pair,
    completed_pairs,
):
    """
    Check whether a semantic manipulation pair has
    already been completed.
    """

    for completed in completed_pairs:

        if (
            pair["pick"]
            == completed["pick"]
            and
            pair["place"]
            == completed["place"]
        ):

            return True

    return False


def find_next_uncompleted_pair(
    pairs,
    completed_pairs,
):
    """
    Return the first pair that has not yet been completed.
    """

    for pair in pairs:

        if not pair_is_completed(
            pair,
            completed_pairs,
        ):

            return pair

    return None


def build_remaining_instruction(
    pairs,
    completed_pairs,
):
    """
    Construct a natural-language instruction describing
    only the remaining manipulations.
    """

    remaining = []

    for pair in pairs:

        if pair_is_completed(
            pair,
            completed_pairs,
        ):

            continue

        remaining.append(
            pair_to_instruction(
                pair
            )
        )

    if not remaining:
        return None

    return " Then ".join(
        remaining
    )


def build_remaining_plan(
    pairs,
    completed_pairs,
):
    """
    Build a symbolic previous plan containing only the
    manipulations that still need to be completed.

    This gives the replanner a much cleaner context after
    partial execution.
    """

    actions = []

    for pair in pairs:

        if pair_is_completed(
            pair,
            completed_pairs,
        ):

            continue

        actions.append(
            {
                "type": "pick",
                "object": pair["pick"],
                "target": None,
            }
        )

        actions.append(
            {
                "type": "place",
                "object": pair["pick"],
                "target": pair["place"],
            }
        )

    return {
        "goal": (
            "Complete the remaining "
            "manipulation steps."
        ),
        "actions": actions,
    }


def build_required_pairs(
    pairs,
    completed_pairs,
):
    """
    Build the deterministic list of manipulation pairs
    that are still required.

    This is the critical bridge between the execution
    controller and the plan validator.

    Example:

        Initial pairs:
            blue -> green
            red -> blue

        Completed:
            blue -> green

        Result:
            [
                ("red block", "blue block")
            ]
    """

    required_pairs = []

    for pair in pairs:

        if pair_is_completed(
            pair,
            completed_pairs,
        ):

            continue

        required_pairs.append(
            (
                pair["pick"],
                pair["place"],
            )
        )

    return required_pairs


# ============================================================
# CURRENT OBJECT POSITIONS
# ============================================================

def get_current_object_positions(env):
    """
    Read the current XYZ position of every object in the
    PyBullet environment.

    This is used by the controlled grasp-failure injector
    to select an empty workspace location after partial
    execution.
    """

    object_positions = {}

    for name, body_id in (
        env.obj_name_to_id.items()
    ):

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

    return object_positions


# ============================================================
# PHYSICAL EXECUTION
# ============================================================

def execute_pair_with_obs(
    env,
    policy,
    obs,
    pair,
    step_number,
    controlled_failure=False,
):
    """
    Execute one semantic pick/place pair using the current
    observation.

    For grasp_failure, the failure injector receives the
    current object positions so the injected pick location
    can be selected away from all known objects.

    Returns
    -------
    tuple
        (
            new_obs,
            prediction,
            executed_action,
            injection_metadata,
            reward,
            done,
            task_result
        )
    """

    instruction = pair_to_instruction(
        pair
    )

    print_section(
        f"PHYSICAL STEP {step_number}"
    )

    print(
        f"Semantic instruction: "
        f"{instruction}"
    )

    # --------------------------------------------------------
    # CLIPort prediction
    # --------------------------------------------------------

    print(
        "\nRunning CLIPort..."
    )

    obs["_instruction"] = instruction

    prediction = policy.predict(
        obs
    )

    print(
        "\nCLIPort prediction:"
    )

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
    # Construct semantic + physical action
    # --------------------------------------------------------

    action = {
        "pick": prediction["pick_xyz"],
        "place": prediction["place_xyz"],
        "pick_name": pair["pick"],
        "place_name": pair["place"],
    }

    # --------------------------------------------------------
    # Default injection metadata
    # --------------------------------------------------------

    injection_metadata = {
        "enabled": False,
        "mode": "none",
        "injected": False,
        "description": None,
    }

    # --------------------------------------------------------
    # Controlled failure
    # --------------------------------------------------------

    if controlled_failure:

        print(
            "\nCONTROLLED FAILURE INJECTION"
        )

        print(
            f"Failure mode: {FAILURE_MODE}"
        )

        # ----------------------------------------------------
        # Build current object-position map.
        #
        # This is particularly important for grasp_failure:
        # the injector needs to know where the objects
        # currently are after partial task execution.
        # ----------------------------------------------------

        object_positions = (
            get_current_object_positions(
                env
            )
        )

        print(
            "\nCurrent object positions used "
            "for failure injection:"
        )

        for name, position in (
            object_positions.items()
        ):

            print(
                f"  {name}: {position}"
            )

        # ----------------------------------------------------
        # Apply controlled failure
        # ----------------------------------------------------

        (
            modified_action,
            injection_metadata,
        ) = FAILURE_INJECTOR.apply(
            action,
            enabled=True,
            object_positions=object_positions,
        )

        action = modified_action

        print(
            "\nFailure injection metadata:"
        )

        print(
            json.dumps(
                make_json_safe(
                    injection_metadata
                ),
                indent=2,
            )
        )

        print(
            "Injected pick XYZ:",
            action["pick"],
        )

        print(
            "Injected place XYZ:",
            action["place"],
        )

    else:

        print(
            "\nUsing normal CLIPort "
            "coordinates. No failure injected."
        )

    # --------------------------------------------------------
    # Execute in PyBullet
    # --------------------------------------------------------

    print(
        "\nExecuting robot action..."
    )

    (
        new_obs,
        reward,
        done,
        info,
    ) = env.step(
        action
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
    # Print result
    # --------------------------------------------------------

    print_execution_result(
        task_result
    )

    return (
        new_obs,
        prediction,
        action,
        injection_metadata,
        reward,
        done,
        task_result,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "MULTI-STEP LLM REPLANNING EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        "\nTask:"
    )

    print(
        f"  {TASK_INSTRUCTION}"
    )

    print(
        "\nControlled failure:"
    )

    print(
        f"  Physical step "
        f"{CONTROLLED_FAILURE_STEP}"
    )

    print(
        f"  Failure mode: "
        f"{FAILURE_MODE}"
    )

    # --------------------------------------------------------
    # Experiment log
    # --------------------------------------------------------

    experiment_log = {

        "experiment": (
            "multi_step_failure_recovery"
        ),

        "task_instruction": (
            TASK_INSTRUCTION
        ),

        "controlled_failure_step": (
            CONTROLLED_FAILURE_STEP
        ),

        "failure_mode": FAILURE_MODE,

        "experiment_seed": EXPERIMENT_SEED,

        "initial_world_state": None,

        "initial_plan": None,

        "initial_plan_valid": False,

        "initial_pairs": [],

        "completed_pairs": [],

        "steps": [],

        "replanning_triggered": False,

        "replan_valid": False,

        "replan_attempts": 0,

        "recovery_attempted": False,

        "recovery_success": False,

        "final_success": False,

        "completed_steps": 0,
    }

    # --------------------------------------------------------
    # Initialize environment
    # --------------------------------------------------------

    env = PickPlaceEnv(
        gui=True
    )

    try:

        # ----------------------------------------------------
        # Initialize components
        # ----------------------------------------------------

        world_state_builder = (
            WorldStateBuilder(env)
        )

        planner = LLMTaskPlanner()

        replanner = LLMTaskReplanner()

        validator = PlanValidator()

        feedback_builder = (
            ExecutionFeedbackBuilder()
        )

        # ----------------------------------------------------
        # Load CLIPort ONCE
        # ----------------------------------------------------

        print_section(
            "LOADING CLIPORT"
        )

        if not CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                f"Missing CLIPort checkpoint: "
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

        print_section(
            "INITIALIZING ENVIRONMENT"
        )

        np.random.seed(
            EXPERIMENT_SEED
        )

        print(
            f"Experiment seed: "
            f"{EXPERIMENT_SEED}"
        )

        obs = env.reset(
            CONFIG
        )

        print(
            "Environment initialized."
        )

        # ----------------------------------------------------
        # Initial world state
        # ----------------------------------------------------

        print_section(
            "BUILDING INITIAL WORLD STATE"
        )

        world_state = (
            world_state_builder.build()
        )

        experiment_log[
            "initial_world_state"
        ] = world_state

        print_world_state(
            "Initial world state:",
            world_state,
        )

        # ----------------------------------------------------
        # Initial LLM planning
        # ----------------------------------------------------

        print_section(
            "GENERATING INITIAL PLAN"
        )

        initial_plan = planner.plan(
            instruction=TASK_INSTRUCTION,
            world_state=world_state,
        )

        experiment_log[
            "initial_plan"
        ] = initial_plan

        print_plan(
            "Initial LLM plan:",
            initial_plan,
        )

        # ----------------------------------------------------
        # Validate initial plan
        # ----------------------------------------------------

        print_section(
            "VALIDATING INITIAL PLAN"
        )

        initial_validation = validate_plan(
            validator=validator,
            plan=initial_plan,
            world_state=world_state,
        )

        print(
            json.dumps(
                make_json_safe(
                    initial_validation
                ),
                indent=2,
            )
        )

        if not initial_validation[
            "valid"
        ]:

            print(
                "\nInitial plan is invalid."
            )

            return

        experiment_log[
            "initial_plan_valid"
        ] = True

        print(
            "\nInitial plan accepted."
        )

        # ----------------------------------------------------
        # Build validated initial plan
        # ----------------------------------------------------

        current_plan = {

            "goal": initial_plan[
                "goal"
            ],

            "actions": (
                initial_validation[
                    "validated_actions"
                ]
            ),
        }

        # ----------------------------------------------------
        # Extract physical pairs
        # ----------------------------------------------------

        current_pairs = extract_pairs(
            current_plan
        )

        experiment_log[
            "initial_pairs"
        ] = current_pairs

        print(
            "\nInitial manipulation pairs:"
        )

        for index, pair in enumerate(
            current_pairs,
            start=1,
        ):

            print(
                f"  Step {index}: "
                f"{pair['pick']} -> "
                f"{pair['place']}"
            )

        # ----------------------------------------------------
        # Completed symbolic manipulations
        # ----------------------------------------------------

        completed_pairs = []

        # ----------------------------------------------------
        # Step-level execution loop
        # ----------------------------------------------------

        step_number = 1

        while True:

            # ------------------------------------------------
            # Find next uncompleted pair
            # ------------------------------------------------

            next_pair = (
                find_next_uncompleted_pair(
                    current_pairs,
                    completed_pairs,
                )
            )

            if next_pair is None:

                print(
                    "\nNo remaining manipulation steps."
                )

                break

            # ------------------------------------------------
            # Determine controlled failure
            # ------------------------------------------------

            failure_injected = (
                step_number
                == CONTROLLED_FAILURE_STEP
                and not experiment_log[
                    "replanning_triggered"
                ]
            )

            # ------------------------------------------------
            # Execute current physical step
            # ------------------------------------------------

            (
                obs,
                prediction,
                action,
                injection_metadata,
                reward,
                done,
                task_result,
            ) = execute_pair_with_obs(
                env=env,
                policy=policy,
                obs=obs,
                pair=next_pair,
                step_number=step_number,
                controlled_failure=(
                    failure_injected
                ),
            )

            # ------------------------------------------------
            # Build structured feedback
            # ------------------------------------------------

            execution_feedback = (
                feedback_builder.build(
                    task_result
                )
            )

            print(
                "\nStructured execution feedback:"
            )

            print(
                json.dumps(
                    make_json_safe(
                        execution_feedback
                    ),
                    indent=2,
                )
            )

            # ------------------------------------------------
            # Log physical execution
            # ------------------------------------------------

            step_log = {

                "step_number": step_number,

                "planned_pair": next_pair,

                "instruction": (
                    pair_to_instruction(
                        next_pair
                    )
                ),

                "controlled_failure": (
                    failure_injected
                ),

                "failure_mode": (
                    FAILURE_MODE
                    if failure_injected
                    else "none"
                ),

                "failure_injection_metadata": (
                    injection_metadata
                ),

                "prediction": {

                    "pick_yx": prediction.get(
                        "pick_yx"
                    ),

                    "place_yx": prediction.get(
                        "place_yx"
                    ),

                    "pick_xyz": prediction.get(
                        "pick_xyz"
                    ),

                    "place_xyz": prediction.get(
                        "place_xyz"
                    ),
                },

                "action": {

                    "pick_name": action.get(
                        "pick_name"
                    ),

                    "place_name": action.get(
                        "place_name"
                    ),

                    "pick": action.get(
                        "pick"
                    ),

                    "place": action.get(
                        "place"
                    ),
                },

                "reward": reward,

                "done": done,

                "execution_result": (
                    task_result
                ),

                "execution_feedback": (
                    execution_feedback
                ),

                "success": bool(
                    task_result.get(
                        "success",
                        False,
                    )
                ),
            }

            experiment_log[
                "steps"
            ].append(
                step_log
            )

            # =================================================
            # SUCCESSFUL PHYSICAL STEP
            # =================================================

            if task_result.get(
                "success",
                False,
            ):

                print(
                    f"\nSTEP {step_number} "
                    "SUCCEEDED"
                )

                completed_pairs.append(
                    next_pair
                )

                experiment_log[
                    "completed_pairs"
                ] = completed_pairs.copy()

                experiment_log[
                    "completed_steps"
                ] = len(
                    completed_pairs
                )

                # --------------------------------------------
                # Update world state after success
                # --------------------------------------------

                world_state = (
                    world_state_builder.build()
                )

                print_world_state(
                    "World state after successful step:",
                    world_state,
                )

                # --------------------------------------------
                # Move to next physical step
                # --------------------------------------------

                step_number += 1

                continue

            # =================================================
            # FAILED PHYSICAL STEP
            # =================================================

            print(
                f"\nSTEP {step_number} "
                "FAILED"
            )

            # ------------------------------------------------
            # Replanning
            # ------------------------------------------------

            experiment_log[
                "replanning_triggered"
            ] = True

            # ------------------------------------------------
            # Update world state after failure
            # ------------------------------------------------

            print_section(
                "UPDATING WORLD STATE AFTER FAILURE"
            )

            world_state = (
                world_state_builder.build()
            )

            print_world_state(
                "Current world state:",
                world_state,
            )

            # ------------------------------------------------
            # Determine remaining task
            # ------------------------------------------------

            remaining_instruction = (
                build_remaining_instruction(
                    current_pairs,
                    completed_pairs,
                )
            )

            if remaining_instruction is None:

                print(
                    "\nNo remaining task exists."
                )

                break

            # ------------------------------------------------
            # Build previous remaining plan
            # ------------------------------------------------

            remaining_previous_plan = (
                build_remaining_plan(
                    current_pairs,
                    completed_pairs,
                )
            )

            # ------------------------------------------------
            # Build deterministic required pairs
            # ------------------------------------------------

            required_pairs = (
                build_required_pairs(
                    current_pairs,
                    completed_pairs,
                )
            )

            print(
                "\nRemaining task given to "
                "the replanner:"
            )

            print(
                f"  {remaining_instruction}"
            )

            print(
                "\nRequired manipulation pairs:"
            )

            for pair in required_pairs:

                print(
                    f"  {pair[0]} -> {pair[1]}"
                )

            print_plan(
                "Previous remaining plan:",
                remaining_previous_plan,
            )

            # ------------------------------------------------
            # LLM REPLANNING
            # ------------------------------------------------

            print_section(
                "LLM REPLANNING"
            )

            (
                replanned_plan,
                replan_validation,
                replan_attempts,
            ) = repair_plan_until_valid(

                replanner=replanner,

                validator=validator,

                instruction=remaining_instruction,

                world_state=world_state,

                previous_plan=(
                    remaining_previous_plan
                ),

                execution_feedback=(
                    execution_feedback
                ),

                required_pairs=required_pairs,
            )

            experiment_log[
                "replan_valid"
            ] = True

            experiment_log[
                "replan_attempts"
            ] += replan_attempts

            # ------------------------------------------------
            # Store replan information in failed step
            # ------------------------------------------------

            experiment_log[
                "steps"
            ][-1][
                "replan"
            ] = {

                "remaining_instruction": (
                    remaining_instruction
                ),

                "required_pairs": (
                    required_pairs
                ),

                "previous_remaining_plan": (
                    remaining_previous_plan
                ),

                "replanned_plan": (
                    replanned_plan
                ),

                "validation": (
                    replan_validation
                ),

                "attempts": (
                    replan_attempts
                ),
            }

            print_plan(
                "Final validated replan:",
                replanned_plan,
            )

            # ------------------------------------------------
            # Extract remaining physical pairs
            # ------------------------------------------------

            current_pairs = extract_pairs(
                replanned_plan
            )

            print(
                "\nReplanned physical pairs:"
            )

            for index, pair in enumerate(
                current_pairs,
                start=1,
            ):

                print(
                    f"  {index}: "
                    f"{pair['pick']} -> "
                    f"{pair['place']}"
                )

            # ------------------------------------------------
            # Find the next uncompleted pair
            # ------------------------------------------------

            next_pair = (
                find_next_uncompleted_pair(
                    current_pairs,
                    completed_pairs,
                )
            )

            if next_pair is None:

                raise RuntimeError(
                    "The replanner produced a plan "
                    "with no remaining executable "
                    "manipulation."
                )

            # ------------------------------------------------
            # Execute the replan normally.
            #
            # IMPORTANT:
            # Controlled failure is now OFF.
            # ------------------------------------------------

            print_section(
                "EXECUTING RECOVERY PLAN"
            )

            experiment_log[
                "recovery_attempted"
            ] = True

            (
                obs,
                recovery_prediction,
                recovery_action,
                recovery_injection_metadata,
                recovery_reward,
                recovery_done,
                recovery_result,
            ) = execute_pair_with_obs(

                env=env,

                policy=policy,

                obs=obs,

                pair=next_pair,

                step_number=step_number,

                controlled_failure=False,
            )

            recovery_feedback = (
                feedback_builder.build(
                    recovery_result
                )
            )

            print(
                "\nRecovery execution feedback:"
            )

            print(
                json.dumps(
                    make_json_safe(
                        recovery_feedback
                    ),
                    indent=2,
                )
            )

            # ------------------------------------------------
            # Log recovery execution
            # ------------------------------------------------

            recovery_log = {

                "step_number": step_number,

                "planned_pair": next_pair,

                "instruction": (
                    pair_to_instruction(
                        next_pair
                    )
                ),

                "controlled_failure": False,

                "recovery_execution": True,

                "failure_injection_metadata": (
                    recovery_injection_metadata
                ),

                "prediction": {

                    "pick_yx": (
                        recovery_prediction.get(
                            "pick_yx"
                        )
                    ),

                    "place_yx": (
                        recovery_prediction.get(
                            "place_yx"
                        )
                    ),

                    "pick_xyz": (
                        recovery_prediction.get(
                            "pick_xyz"
                        )
                    ),

                    "place_xyz": (
                        recovery_prediction.get(
                            "place_xyz"
                        )
                    ),
                },

                "action": {

                    "pick_name": (
                        recovery_action.get(
                            "pick_name"
                        )
                    ),

                    "place_name": (
                        recovery_action.get(
                            "place_name"
                        )
                    ),

                    "pick": (
                        recovery_action.get(
                            "pick"
                        )
                    ),

                    "place": (
                        recovery_action.get(
                            "place"
                        )
                    ),
                },

                "reward": recovery_reward,

                "done": recovery_done,

                "execution_result": (
                    recovery_result
                ),

                "execution_feedback": (
                    recovery_feedback
                ),

                "success": bool(
                    recovery_result.get(
                        "success",
                        False,
                    )
                ),
            }

            experiment_log[
                "steps"
            ].append(
                recovery_log
            )

            # ------------------------------------------------
            # Recovery success
            # ------------------------------------------------

            if recovery_result.get(
                "success",
                False,
            ):

                print_section(
                    "RECOVERY EXECUTION SUCCEEDED"
                )

                completed_pairs.append(
                    next_pair
                )

                experiment_log[
                    "completed_pairs"
                ] = completed_pairs.copy()

                experiment_log[
                    "completed_steps"
                ] = len(
                    completed_pairs
                )

                experiment_log[
                    "recovery_success"
                ] = True

                # --------------------------------------------
                # Update world state
                # --------------------------------------------

                world_state = (
                    world_state_builder.build()
                )

                print_world_state(
                    "Final world state after recovery:",
                    world_state,
                )

                step_number += 1

                continue

            # ------------------------------------------------
            # Recovery failure
            # ------------------------------------------------

            print(
                "\nRecovery execution failed."
            )

            break

        # ====================================================
        # FINAL EVALUATION
        # ====================================================

        expected_steps = len(
            experiment_log[
                "initial_pairs"
            ]
        )

        completed_steps = len(
            completed_pairs
        )

        final_success = (
            completed_steps
            == expected_steps
        )

        experiment_log[
            "completed_steps"
        ] = completed_steps

        experiment_log[
            "final_success"
        ] = final_success

        # ----------------------------------------------------
        # Final world state
        # ----------------------------------------------------

        final_world_state = (
            world_state_builder.build()
        )

        experiment_log[
            "final_world_state"
        ] = final_world_state

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        print(
            "\n"
        )

        print(
            "=" * 70
        )

        print(
            "MULTI-STEP EXPERIMENT SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Initial plan valid      : "
            f"{experiment_log['initial_plan_valid']}"
        )

        print(
            f"Initial steps           : "
            f"{expected_steps}"
        )

        print(
            f"Completed steps         : "
            f"{completed_steps} / "
            f"{expected_steps}"
        )

        print(
            f"Failure mode            : "
            f"{FAILURE_MODE}"
        )

        print(
            f"Replanning triggered    : "
            f"{experiment_log['replanning_triggered']}"
        )

        print(
            f"Replan valid            : "
            f"{experiment_log['replan_valid']}"
        )

        print(
            f"Replan attempts         : "
            f"{experiment_log['replan_attempts']}"
        )

        print(
            f"Recovery attempted      : "
            f"{experiment_log['recovery_attempted']}"
        )

        print(
            f"Recovery success        : "
            f"{experiment_log['recovery_success']}"
        )

        print(
            f"Final success           : "
            f"{final_success}"
        )

        # ----------------------------------------------------
        # Save experiment log
        # ----------------------------------------------------

        print(
            "\nConverting experiment log "
            "to JSON-safe format..."
        )

        json_safe_log = make_json_safe(
            experiment_log
        )

        RESULT_PATH.write_text(
            json.dumps(
                json_safe_log,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\nExperiment result saved to:"
        )

        print(
            f"  {RESULT_PATH}"
        )

    finally:

        env.close()


if __name__ == "__main__":

    main()
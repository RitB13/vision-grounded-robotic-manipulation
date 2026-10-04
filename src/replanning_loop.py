import json

from .environment import PickPlaceEnv
from .world_state import WorldStateBuilder
from .llm_planner import LLMTaskPlanner
from .llm_replanner import LLMTaskReplanner
from .plan_validator import PlanValidator
from .plan_to_cliport import plan_to_instruction
from .execution_feedback import ExecutionFeedbackBuilder
from .cliport_policy import CLIPortPolicy
from .config import CHECKPOINT_PATH


# Maximum number of physical execution attempts.
MAX_EXECUTION_ATTEMPTS = 3

# Maximum number of times the LLM is allowed to repair an invalid
# plan before we give up on that particular execution attempt.
MAX_PLAN_REPAIR_ATTEMPTS = 3

# -------------------------------------------------------------
# Controlled experiment setting
# -------------------------------------------------------------
#
# On the first physical execution attempt only, deliberately
# replace the predicted place position with the predicted pick
# position.
#
# This should cause:
#
#   grasp       -> success
#   lift        -> success
#   movement    -> failure
#   placement   -> failure
#
# The disturbance is removed on subsequent attempts.
#
CONTROLLED_FAILURE = True


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_plan(title, plan):
    print(f"\n{title}")
    print(json.dumps(plan, indent=2))


def validate_plan(
    validator,
    plan,
    world_state,
):
    """
    Validate a symbolic plan and return the validation result.
    """

    return validator.validate(
        plan=plan,
        world_state=world_state,
    )


def repair_plan_until_valid(
    replanner,
    validator,
    instruction,
    world_state,
    previous_plan,
    execution_feedback,
):
    """
    Ask the LLM replanner to generate a revised plan.

    If the revised plan is rejected by the deterministic validator,
    send the validation errors back to the LLM and allow it to repair
    the plan.

    Returns
    -------
    tuple
        (valid_plan, validation_result)

    Raises
    ------
    RuntimeError
        If no valid plan is produced within the repair limit.
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

        current_plan = replanner.replan(
            instruction=instruction,
            world_state=world_state,
            previous_plan=current_plan,
            execution_feedback=execution_feedback,
            validation_errors=validation_errors,
        )

        print_plan(
            "Candidate replanned task:",
            current_plan,
        )

        validation = validate_plan(
            validator=validator,
            plan=current_plan,
            world_state=world_state,
        )

        if validation["valid"]:

            print("\n✅ REPLANNED PLAN PASSED VALIDATION")

            validated_plan = {
                "goal": current_plan["goal"],
                "actions": validation["validated_actions"],
            }

            return validated_plan, validation

        # -----------------------------------------------------
        # Invalid candidate plan
        # -----------------------------------------------------

        print("\n❌ REPLANNED PLAN REJECTED")

        validation_errors = validation["errors"]

        for error in validation_errors:
            print(f"  - {error}")

        if repair_attempt < MAX_PLAN_REPAIR_ATTEMPTS:
            print(
                "\nSending validator errors back to "
                "the LLM for another repair."
            )

    raise RuntimeError(
        "The replanner failed to produce a valid plan after "
        f"{MAX_PLAN_REPAIR_ATTEMPTS} repair attempts."
    )


def main():

    instruction = (
        "Pick the blue block and place it on the green block."
    )

    config = {
        "pick": [
            "blue block",
            "green block",
            "red block",
        ],
        "place": [],
    }

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing CLIPort checkpoint: {CHECKPOINT_PATH}\n"
            "Download it before running the experiment."
        )

    # ---------------------------------------------------------
    # Initialize components
    # ---------------------------------------------------------

    env = PickPlaceEnv(gui=True)

    world_state_builder = WorldStateBuilder(env)
    planner = LLMTaskPlanner()
    replanner = LLMTaskReplanner()
    validator = PlanValidator()
    feedback_builder = ExecutionFeedbackBuilder()

    # Load CLIPort ONCE.
    #
    # We do not want to reload the neural network after every
    # replanning attempt.
    print("\nLoading CLIPort policy...")

    policy = CLIPortPolicy(CHECKPOINT_PATH)

    try:

        # -----------------------------------------------------
        # Initial environment
        # -----------------------------------------------------

        print_section("INITIALIZING EXPERIMENT")

        obs = env.reset(config)

        print("Environment initialized.")

        # -----------------------------------------------------
        # Initial world state
        # -----------------------------------------------------

        print_section("BUILDING INITIAL WORLD STATE")

        world_state = world_state_builder.build()

        for obj in world_state["objects"]:
            print(
                f"{obj['name']}: "
                f"type={obj['type']}, "
                f"color={obj['color']}, "
                f"position={obj['position']}"
            )

        # -----------------------------------------------------
        # Initial LLM planning
        # -----------------------------------------------------

        print_section("GENERATING INITIAL PLAN")

        current_plan = planner.plan(
            instruction=instruction,
            world_state=world_state,
        )

        print_plan(
            "Initial LLM plan:",
            current_plan,
        )

        # -----------------------------------------------------
        # Physical execution attempt loop
        # -----------------------------------------------------

        for attempt in range(
            1,
            MAX_EXECUTION_ATTEMPTS + 1,
        ):

            print_section(
                f"ATTEMPT {attempt} / "
                f"{MAX_EXECUTION_ATTEMPTS}"
            )

            # -------------------------------------------------
            # Validate current plan
            # -------------------------------------------------

            print("Validating current plan...")

            validation = validate_plan(
                validator=validator,
                plan=current_plan,
                world_state=world_state,
            )

            if not validation["valid"]:

                print("\n❌ CURRENT PLAN IS INVALID")

                for error in validation["errors"]:
                    print(f"  - {error}")

                print(
                    "\nThe current plan should have been "
                    "validated before reaching execution."
                )

                return

            print("✅ Plan accepted.")

            validated_plan = {
                "goal": current_plan["goal"],
                "actions": validation["validated_actions"],
            }

            print_plan(
                "Validated plan:",
                validated_plan,
            )

            # -------------------------------------------------
            # Convert symbolic plan to CLIPort instruction
            # -------------------------------------------------

            cliport_instruction = plan_to_instruction(
                validated_plan
            )

            print("\nCLIPort instruction:")
            print(f"  {cliport_instruction}")

            # -------------------------------------------------
            # Run CLIPort
            # -------------------------------------------------

            print("\nRunning CLIPort...")

            obs["_instruction"] = cliport_instruction

            prediction = policy.predict(obs)

            print("\nCLIPort prediction:")

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

            # -------------------------------------------------
            # Construct robot action
            # -------------------------------------------------

            action = {
                "pick": prediction["pick_xyz"],
                "place": prediction["place_xyz"],
            }

            # -------------------------------------------------
            # Controlled failure injection
            # -------------------------------------------------

            failure_injected = (
                CONTROLLED_FAILURE
                and attempt == 1
            )

            if failure_injected:

                print(
                    "\n⚠️ CONTROLLED FAILURE INJECTION"
                )

                print(
                    "Replacing the predicted place "
                    "position with the pick position."
                )

                action["place"] = prediction["pick_xyz"]

                print(
                    "Injected place XYZ:",
                    action["place"],
                )

            else:

                print(
                    "\nUsing normal CLIPort "
                    "place position."
                )

            # -------------------------------------------------
            # Execute robot action
            # -------------------------------------------------

            print("\nExecuting robot action...")

            obs, reward, done, info = env.step(action)

            task_result = info.get("task_result")

            print("\nExecution result:")

            print(f"  Reward: {reward}")
            print(f"  Task success: {done}")

            if task_result is None:
                raise RuntimeError(
                    "Environment did not return task_result."
                )

            print("\nTask result:")

            for key, value in task_result.items():
                print(f"  {key}: {value}")

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            if done:

                print_section(
                    "✅ TASK SUCCESSFULLY COMPLETED"
                )

                print(
                    f"Successful attempt: {attempt}"
                )

                print(
                    "\nThe closed-loop system completed "
                    "the task successfully."
                )

                return

            # -------------------------------------------------
            # FAILURE
            # -------------------------------------------------

            print_section(
                f"❌ ATTEMPT {attempt} FAILED"
            )

            # -------------------------------------------------
            # Build structured execution feedback
            # -------------------------------------------------

            execution_feedback = feedback_builder.build(
                task_result
            )

            print(
                "\nStructured execution feedback:"
            )

            print(
                json.dumps(
                    execution_feedback,
                    indent=2,
                )
            )

            # -------------------------------------------------
            # Stop if this was the final physical attempt
            # -------------------------------------------------

            if attempt >= MAX_EXECUTION_ATTEMPTS:

                print_section(
                    "❌ MAXIMUM EXECUTION ATTEMPTS REACHED"
                )

                print(
                    "The task could not be completed "
                    "within the allowed attempts."
                )

                return

            # -------------------------------------------------
            # Rebuild world state after failure
            # -------------------------------------------------

            print_section(
                "UPDATING WORLD STATE AFTER FAILURE"
            )

            world_state = world_state_builder.build()

            for obj in world_state["objects"]:
                print(
                    f"{obj['name']}: "
                    f"position={obj['position']}"
                )

            # -------------------------------------------------
            # LLM REPLANNING + VALIDATION LOOP
            # -------------------------------------------------

            print_section(
                "GENERATING REVISED PLAN"
            )

            try:

                current_plan, validation = (
                    repair_plan_until_valid(
                        replanner=replanner,
                        validator=validator,
                        instruction=instruction,
                        world_state=world_state,
                        previous_plan=current_plan,
                        execution_feedback=(
                            execution_feedback
                        ),
                    )
                )

            except RuntimeError as exc:

                print(
                    "\n❌ REPLANNING FAILED"
                )

                print(str(exc))

                return

            print_plan(
                "Final validated replan:",
                current_plan,
            )

            print(
                "\nReplanning and validation completed."
            )

    finally:

        env.close()


if __name__ == "__main__":
    main()
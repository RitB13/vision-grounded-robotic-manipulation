import json

from .environment import PickPlaceEnv
from .world_state import WorldStateBuilder
from .llm_planner import LLMTaskPlanner
from .plan_validator import PlanValidator
from .plan_to_cliport import (
    extract_pick_place_pairs,
    pair_to_instruction,
)
from .cliport_policy import CLIPortPolicy
from .config import CHECKPOINT_PATH


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():

    # ========================================================================
    # USER TASK
    # ========================================================================

    instruction = (
        "Pick the blue block and place it on the green block, "
        "then pick the red block and place it on the blue block."
    )

    config = {
        "pick": [
            "blue block",
            "green block",
            "red block",
        ],
        "place": [],
    }

    # ========================================================================
    # CHECKPOINT
    # ========================================================================

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing CLIPort checkpoint: {CHECKPOINT_PATH}\n"
            "Download it before running the experiment."
        )

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    print_section(
        "INITIALIZING MULTI-STEP EXPERIMENT"
    )

    env = PickPlaceEnv(
        gui=True
    )

    try:

        obs = env.reset(
            config
        )

        print(
            "Environment initialized."
        )

        # ====================================================================
        # WORLD STATE
        # ====================================================================

        print_section(
            "BUILDING INITIAL WORLD STATE"
        )

        world_state_builder = (
            WorldStateBuilder(env)
        )

        world_state = (
            world_state_builder.build()
        )

        for obj in world_state["objects"]:

            print(
                f"{obj['name']}: "
                f"type={obj['type']}, "
                f"color={obj['color']}, "
                f"position={obj['position']}"
            )

        # ====================================================================
        # LLM PLANNER
        # ====================================================================

        print_section(
            "GENERATING MULTI-STEP LLM PLAN"
        )

        planner = LLMTaskPlanner()

        plan = planner.plan(
            instruction=instruction,
            world_state=world_state,
        )

        print(
            "\nLLM-generated plan:"
        )

        print(
            json.dumps(
                plan,
                indent=2,
            )
        )

        # ====================================================================
        # VALIDATION
        # ====================================================================

        print_section(
            "VALIDATING COMPLETE PLAN"
        )

        validator = PlanValidator()

        validation = validator.validate(
            plan=plan,
            world_state=world_state,
        )

        print(
            json.dumps(
                validation,
                indent=2,
            )
        )

        if not validation["valid"]:

            print(
                "\n❌ PLAN REJECTED"
            )

            for error in validation["errors"]:
                print(
                    f"  - {error}"
                )

            return

        print(
            "\n✅ COMPLETE PLAN ACCEPTED"
        )

        validated_plan = {
            "goal": plan["goal"],
            "actions": validation[
                "validated_actions"
            ],
        }

        # ====================================================================
        # EXTRACT EXECUTION PAIRS
        # ====================================================================

        print_section(
            "EXTRACTING EXECUTION STEPS"
        )

        pairs = extract_pick_place_pairs(
            validated_plan
        )

        print(
            f"Number of physical manipulation steps: "
            f"{len(pairs)}"
        )

        for index, pair in enumerate(
            pairs,
            start=1,
        ):

            print(
                f"\nStep {index}:"
            )

            print(
                f"  Pick : {pair['pick']}"
            )

            print(
                f"  Place: {pair['place']}"
            )

        # ====================================================================
        # LOAD CLIPORT ONCE
        # ====================================================================

        print_section(
            "LOADING CLIPORT"
        )

        policy = CLIPortPolicy(
            CHECKPOINT_PATH
        )

        print(
            "CLIPort policy loaded."
        )

        # ====================================================================
        # EXECUTE EACH STEP
        # ====================================================================

        successful_steps = 0

        for step_index, pair in enumerate(
            pairs,
            start=1,
        ):

            print_section(
                f"EXECUTION STEP {step_index} / "
                f"{len(pairs)}"
            )

            cliport_instruction = (
                pair_to_instruction(
                    pair
                )
            )

            print(
                "\nCLIPort instruction:"
            )

            print(
                f"  {cliport_instruction}"
            )

            # ----------------------------------------------------------------
            # IMPORTANT:
            #
            # The observation is generated from the CURRENT simulated scene.
            # Therefore CLIPort sees the result of all previous manipulation
            # steps.
            # ----------------------------------------------------------------

            obs["_instruction"] = (
                cliport_instruction
            )

            print(
                "\nRunning CLIPort..."
            )

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

            # ----------------------------------------------------------------
            # ROBOT ACTION
            #
            # CLIPort is responsible for producing the geometric coordinates.
            #
            # The symbolic plan is responsible for specifying WHICH object
            # is picked and WHICH object is the intended target.
            #
            # We therefore pass both pieces of information to the environment.
            # This prevents semantic ambiguity when two objects occupy nearly
            # identical coordinates after stacking.
            # ----------------------------------------------------------------

            print(
                "\nIntended semantic action:"
            )

            print(
                f"  Pick : {pair['pick']}"
            )

            print(
                f"  Place: {pair['place']}"
            )

            action = {
                "pick": prediction[
                    "pick_xyz"
                ],
                "place": prediction[
                    "place_xyz"
                ],
                "pick_name": pair[
                    "pick"
                ],
                "place_name": pair[
                    "place"
                ],
            }

            print(
                "\nExecuting robot action..."
            )

            obs, reward, done, info = (
                env.step(
                    action
                )
            )

            task_result = info.get(
                "task_result"
            )

            print(
                "\nStep execution result:"
            )

            print(
                f"  Reward: {reward}"
            )

            print(
                f"  Success: {done}"
            )

            if task_result is not None:

                print(
                    "\nTask evaluation:"
                )

                print(
                    json.dumps(
                        task_result,
                        indent=2,
                    )
                )

            # =================================================================
            # FAILURE
            # =================================================================

            if not done:

                print_section(
                    f"❌ MULTI-STEP TASK FAILED "
                    f"AT STEP {step_index}"
                )

                print(
                    "Stopping execution because the "
                    "current step did not succeed."
                )

                return

            successful_steps += 1

            # =================================================================
            # UPDATE WORLD STATE
            # =================================================================

            world_state = (
                world_state_builder.build()
            )

            print(
                "\nUpdated world state:"
            )

            for obj in world_state[
                "objects"
            ]:

                print(
                    f"  {obj['name']}: "
                    f"{obj['position']}"
                )

            # =================================================================
            # CONTINUE
            # =================================================================

            if step_index < len(pairs):

                print(
                    "\nStep succeeded. "
                    "Continuing to next planned action."
                )

        # ====================================================================
        # COMPLETE
        # ====================================================================

        print_section(
            "✅ MULTI-STEP TASK COMPLETED"
        )

        print(
            f"Successful manipulation steps: "
            f"{successful_steps} / {len(pairs)}"
        )

        print(
            "\nThe symbolic multi-step plan was "
            "successfully executed sequentially."
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()
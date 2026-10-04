from pathlib import Path

import matplotlib.pyplot as plt

from .environment import PickPlaceEnv
from .world_state import WorldStateBuilder
from .llm_planner import LLMTaskPlanner
from .plan_validator import PlanValidator
from .plan_to_cliport import plan_to_instruction
from .cliport_policy import CLIPortPolicy
from .config import CHECKPOINT_PATH


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # ---------------------------------------------------------
    # USER TASK
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # CHECK CLIPORT CHECKPOINT
    # ---------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing CLIPort checkpoint: {CHECKPOINT_PATH}\n"
            "Download it before running the integration."
        )

    # ---------------------------------------------------------
    # INITIALIZE ENVIRONMENT
    # ---------------------------------------------------------
    env = PickPlaceEnv(gui=True)

    try:
        print_section("1. INITIALIZING ENVIRONMENT")

        obs = env.reset(config)

        print("Environment initialized.")

        # -----------------------------------------------------
        # BUILD WORLD STATE
        # -----------------------------------------------------
        print_section("2. BUILDING WORLD STATE")

        world_state_builder = WorldStateBuilder(env)
        world_state = world_state_builder.build()

        for obj in world_state["objects"]:
            print(
                f"{obj['name']}: "
                f"type={obj['type']}, "
                f"color={obj['color']}, "
                f"position={obj['position']}"
            )

        # -----------------------------------------------------
        # LLM PLANNER
        # -----------------------------------------------------
        print_section("3. GENERATING LLM PLAN")

        planner = LLMTaskPlanner()

        plan = planner.plan(
            instruction=instruction,
            world_state=world_state,
        )

        print("\nLLM-generated plan:")
        print(plan)

        # -----------------------------------------------------
        # PLAN VALIDATION
        # -----------------------------------------------------
        print_section("4. VALIDATING PLAN")

        validator = PlanValidator()

        validation = validator.validate(
            plan=plan,
            world_state=world_state,
        )

        print("\nValidation result:")
        print(validation)

        if not validation["valid"]:
            print("\n❌ PLAN REJECTED")

            for error in validation["errors"]:
                print(f"  - {error}")

            return

        print("\n✅ PLAN ACCEPTED")

        # -----------------------------------------------------
        # CONVERT PLAN → CLIPORT INSTRUCTION
        # -----------------------------------------------------
        print_section("5. CONVERTING PLAN FOR CLIPORT")

        validated_plan = {
            "goal": plan["goal"],
            "actions": validation["validated_actions"],
        }

        cliport_instruction = plan_to_instruction(validated_plan)

        print("CLIPort instruction:")
        print(f"  {cliport_instruction}")

        # -----------------------------------------------------
        # GIVE CLIPORT THE VALIDATED INSTRUCTION
        # -----------------------------------------------------
        print_section("6. RUNNING CLIPORT")

        obs["_instruction"] = cliport_instruction

        before = env.get_camera_image().copy()

        print("Loading CLIPort policy...")

        policy = CLIPortPolicy(CHECKPOINT_PATH)

        prediction = policy.predict(obs)

        print("\nCLIPort prediction:")
        print(f"  pick pixel : {prediction['pick_yx']}")
        print(f"  pick XYZ   : {prediction['pick_xyz']}")
        print(f"  place pixel: {prediction['place_yx']}")
        print(f"  place XYZ  : {prediction['place_xyz']}")

        # -----------------------------------------------------
        # EXECUTE ROBOT ACTION
        # -----------------------------------------------------
        print_section("7. EXECUTING ROBOT ACTION")

        action = {
            "pick": prediction["pick_xyz"],
            "place": prediction["place_xyz"],
        }

        obs, reward, done, info = env.step(action)

        after = env.get_camera_image().copy()

        print("\nExecution result:")
        print(f"  Reward: {reward}")
        print(f"  Task success: {done}")

        task_result = info.get("task_result")

        if task_result is not None:
            print("\nTask evaluation:")

            if isinstance(task_result, dict):
                for key, value in task_result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {task_result}")

        # -----------------------------------------------------
        # SAVE VISUAL RESULT
        # -----------------------------------------------------
        print_section("8. SAVING RESULT")

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
        )

        axes[0].imshow(before)
        axes[0].set_title("Before")
        axes[0].axis("off")

        axes[1].imshow(after)
        axes[1].set_title("After")
        axes[1].axis("off")

        plt.tight_layout()

        output = (
            Path(__file__).resolve().parents[1]
            / "llm_cliport_result.png"
        )

        plt.savefig(
            output,
            dpi=150,
        )

        print(f"Saved comparison image to: {output}")

        plt.show()

    finally:
        env.close()


if __name__ == "__main__":
    main()
from .environment import PickPlaceEnv
from .world_state import WorldStateBuilder
from .llm_planner import LLMTaskPlanner
from .plan_validator import PlanValidator


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    instruction = "Pick the blue block and place it on the green block."

    config = {
        "pick": ["blue block", "green block", "red block"],
        "place": [],
    }

    env = PickPlaceEnv(gui=False)

    try:
        # ---------------------------------------------------------
        # 1. Create simulation scene
        # ---------------------------------------------------------
        print_section("1. INITIALIZING ENVIRONMENT")

        env.reset(config)

        print("Environment initialized.")

        # ---------------------------------------------------------
        # 2. Build world state
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 3. Generate LLM plan
        # ---------------------------------------------------------
        print_section("3. GENERATING LLM PLAN")

        planner = LLMTaskPlanner()

        plan = planner.plan(
            instruction=instruction,
            world_state=world_state,
        )

        print("\nGenerated plan:")
        print(plan)

        # ---------------------------------------------------------
        # 4. Validate plan
        # ---------------------------------------------------------
        print_section("4. VALIDATING PLAN")

        validator = PlanValidator()

        validation = validator.validate(
            plan=plan,
            world_state=world_state,
        )

        print("\nValidation result:")
        print(validation)

        # ---------------------------------------------------------
        # 5. Final decision
        # ---------------------------------------------------------
        print_section("5. PLANNING RESULT")

        if validation["valid"]:
            print("✅ PLAN ACCEPTED")
            print("\nValidated actions:")

            for index, action in enumerate(
                validation["validated_actions"],
                start=1,
            ):
                print(f"  {index}. {action}")

        else:
            print("❌ PLAN REJECTED")

            print("\nErrors:")
            for error in validation["errors"]:
                print(f"  - {error}")

        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
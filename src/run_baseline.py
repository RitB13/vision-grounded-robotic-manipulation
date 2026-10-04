from pathlib import Path
import matplotlib.pyplot as plt

from .environment import PickPlaceEnv
from .cliport_policy import CLIPortPolicy
from .config import CHECKPOINT_PATH


def main():
    instruction = "Pick the blue block and place it on the green block."

    config = {
        "pick": ["blue block", "green block", "red block"],
        "place": [],
    }

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing CLIPort checkpoint: {CHECKPOINT_PATH}\n"
            "Download it before running the baseline."
        )

    env = PickPlaceEnv(gui=True)

    try:
        obs = env.reset(config)
        obs["_instruction"] = instruction

        before = env.get_camera_image().copy()

        print("Loading CLIPort policy...")
        policy = CLIPortPolicy(CHECKPOINT_PATH)

        print(f"Instruction: {instruction}")
        prediction = policy.predict(obs)

        print("\nPredicted action:")
        print("  pick pixel :", prediction["pick_yx"])
        print("  pick XYZ   :", prediction["pick_xyz"])
        print("  place pixel:", prediction["place_yx"])
        print("  place XYZ  :", prediction["place_xyz"])

        action = {
            "pick": prediction["pick_xyz"],
            "place": prediction["place_xyz"],
        }

        # Execute the manipulation and KEEP the evaluation results.
        obs, reward, done, info = env.step(action)

        after = env.get_camera_image().copy()

        print("\n" + "=" * 60)
        print("BASELINE EXECUTION RESULT")
        print("=" * 60)

        print(f"Reward: {reward}")
        print(f"Task success: {done}")

        task_result = info.get("task_result")

        if task_result is not None:
            print("\nTask evaluation:")

            if isinstance(task_result, dict):
                for key, value in task_result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {task_result}")

        print("\nFull info dictionary:")
        print(info)

        print("\nBaseline execution finished.")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(before)
        axes[0].set_title("Before")
        axes[0].axis("off")

        axes[1].imshow(after)
        axes[1].set_title("After")
        axes[1].axis("off")

        plt.tight_layout()

        output = (
            Path(__file__).resolve().parents[1]
            / "baseline_result.png"
        )

        plt.savefig(output, dpi=150)

        print(f"\nSaved comparison image to: {output}")

        plt.show()

    finally:
        env.close()


if __name__ == "__main__":
    main()
from typing import Any, Dict


class ExecutionFeedbackBuilder:
    """
    Convert the robot execution result into structured feedback
    that can later be supplied to the LLM replanner.

    This class does not perform replanning.
    It only interprets the objective execution result.
    """

    def build(
        self,
        task_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(task_result, dict):
            raise TypeError(
                "task_result must be a dictionary."
            )

        required_fields = [
            "success",
            "pick_name",
            "place_name",
            "grasp_success",
            "movement_success",
            "placement_success",
            "reason",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in task_result
        ]

        if missing_fields:
            raise ValueError(
                "task_result is missing required fields: "
                + ", ".join(missing_fields)
            )

        success = bool(task_result["success"])

        pick_name = task_result["pick_name"]
        place_name = task_result["place_name"]

        # ---------------------------------------------------------
        # Successful execution
        # ---------------------------------------------------------
        if success:
            return {
                "execution_success": True,
                "failure_type": None,
                "failed_action": None,
                "feedback": (
                    f"The robot successfully picked the "
                    f"{pick_name} and placed it on the "
                    f"{place_name}."
                ),
                "reason": task_result["reason"],
                "metrics": self._extract_metrics(task_result),
            }

        # ---------------------------------------------------------
        # Determine failure type
        # ---------------------------------------------------------
        if not task_result["grasp_success"]:
            failure_type = "grasp_failure"

            failed_action = {
                "type": "pick",
                "object": pick_name,
                "target": None,
            }

            feedback = (
                f"The robot failed to grasp the {pick_name}."
            )

        elif not task_result["movement_success"]:
            failure_type = "movement_failure"

            failed_action = {
                "type": "place",
                "object": pick_name,
                "target": place_name,
            }

            feedback = (
                f"The robot successfully grasped and lifted "
                f"the {pick_name}, but failed to move it "
                f"to the intended target, {place_name}."
            )

        elif not task_result["placement_success"]:
            failure_type = "placement_failure"

            failed_action = {
                "type": "place",
                "object": pick_name,
                "target": place_name,
            }

            feedback = (
                f"The robot successfully grasped and moved "
                f"the {pick_name}, but failed to place it "
                f"within the required tolerance of the "
                f"{place_name}."
            )

        else:
            failure_type = "execution_failure"

            failed_action = {
                "type": "place",
                "object": pick_name,
                "target": place_name,
            }

            feedback = (
                "The execution failed, but the specific "
                "failure stage could not be determined."
            )

        return {
            "execution_success": False,
            "failure_type": failure_type,
            "failed_action": failed_action,
            "feedback": feedback,
            "reason": task_result["reason"],
            "metrics": self._extract_metrics(task_result),
        }

    @staticmethod
    def _extract_metrics(
        task_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        metric_names = [
            "lift_height",
            "pick_displacement",
            "placement_xy_error",
            "vertical_error",
            "placement_tolerance",
        ]

        metrics = {}

        for name in metric_names:
            if name in task_result:
                metrics[name] = task_result[name]

        return metrics


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Test 1: successful execution
    # ---------------------------------------------------------

    successful_result = {
        "success": True,
        "pick_name": "blue block",
        "place_name": "green block",
        "grasp_success": True,
        "movement_success": True,
        "placement_success": True,
        "reason": (
            "Object was lifted, moved, and placed "
            "within the target tolerance."
        ),
        "lift_height": 0.201,
        "pick_displacement": 0.172,
        "placement_xy_error": 0.003,
        "vertical_error": 0.040,
        "placement_tolerance": 0.035,
    }

    builder = ExecutionFeedbackBuilder()

    feedback = builder.build(successful_result)

    print("\n" + "=" * 60)
    print("TEST 1 — SUCCESSFUL EXECUTION")
    print("=" * 60)

    for key, value in feedback.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # Test 2: placement failure
    # ---------------------------------------------------------

    failed_result = {
        "success": False,
        "pick_name": "blue block",
        "place_name": "green block",
        "grasp_success": True,
        "movement_success": False,
        "placement_success": False,
        "reason": (
            "The object did not move far enough "
            "from its initial position."
        ),
        "lift_height": 0.203,
        "pick_displacement": 0.001,
        "placement_xy_error": 0.239,
        "vertical_error": 0.000,
        "placement_tolerance": 0.035,
    }

    feedback = builder.build(failed_result)

    print("\n" + "=" * 60)
    print("TEST 2 — FAILED EXECUTION")
    print("=" * 60)

    for key, value in feedback.items():
        print(f"{key}: {value}")
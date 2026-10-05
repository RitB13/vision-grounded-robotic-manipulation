"""
Automated evaluation runner for the Vision-Grounded LLM
Robotic Manipulation project.

Runs:

    Baseline:
        grasp_failure
        placement_failure
        movement_failure

    Replanning:
        grasp_failure
        placement_failure
        movement_failure

Outputs:

    evaluation_results/
        evaluation_summary.json
        evaluation_summary.csv
        raw_logs/
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# CONFIGURATION
# ============================================================

FAILURE_MODES = [
    "grasp_failure",
    "placement_failure",
    "movement_failure",
]

FAILURE_STEP = 2
SEEDS = {"grasp_failure": 100, "placement_failure": 200, "movement_failure": 300}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
RAW_LOG_DIR = RESULTS_DIR / "raw_logs"

SUMMARY_JSON = RESULTS_DIR / "evaluation_summary.json"
SUMMARY_CSV = RESULTS_DIR / "evaluation_summary.csv"


# ============================================================
# DISPLAY
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SUBPROCESS
# ============================================================

def run_command(
    command: List[str],
    log_path: Path,
    seed: int,
) -> tuple[int, str]:

    print()
    print("Running:")
    print(" ".join(command))
    print()

    env = os.environ.copy()
    env["EXPERIMENT_SEED"] = str(seed)

    # Fix Windows CP1252 / Unicode issues.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    output = process.stdout + "\n" + process.stderr

    log_path.write_text(
        output,
        encoding="utf-8",
    )

    print(output)

    return process.returncode, output


# ============================================================
# EXTRACTION
# ============================================================

def extract_last_bool(
    text: str,
    label: str,
) -> bool | None:
    """
    Extract the LAST occurrence of:

        label : True
        label : False

    This is important because the logs contain intermediate
    step-level success values before the final experiment result.
    """

    pattern = (
        rf"{re.escape(label)}\s*:\s*(True|False)"
    )

    matches = re.findall(
        pattern,
        text,
    )

    if not matches:
        return None

    return matches[-1] == "True"


def extract_last_int(
    text: str,
    label: str,
) -> int | None:

    pattern = (
        rf"{re.escape(label)}\s*:\s*(\d+)"
    )

    matches = re.findall(
        pattern,
        text,
    )

    if not matches:
        return None

    return int(matches[-1])


def extract_last_float(
    text: str,
    label: str,
) -> float | None:

    pattern = (
        rf"{re.escape(label)}\s*:\s*"
        rf"(-?\d+(?:\.\d+)?)"
    )

    matches = re.findall(
        pattern,
        text,
    )

    if not matches:
        return None

    return float(matches[-1])


def extract_task_result(
    text: str,
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    for metric in [
        "grasp_success",
        "movement_success",
        "placement_success",
    ]:

        result[metric] = extract_last_bool(
            text,
            metric,
        )

    for metric in [
        "placement_xy_error",
        "lift_height",
        "pick_displacement",
    ]:

        result[metric] = extract_last_float(
            text,
            metric,
        )

    return result


def load_json_result(
    path: Path,
) -> Dict[str, Any] | None:

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        print(
            f"WARNING: Could not read {path}: {exc}"
        )

        return None


def clear_json_result(path: Path) -> None:
    """Remove a previous subprocess result so stale data cannot be reused."""

    if path.exists():
        path.unlink()


def make_process_failure_record(
    *,
    condition: str,
    failure_mode: str,
    return_code: int | None,
    output: str,
    error: str,
    seed: int,
) -> Dict[str, Any]:
    """Create a complete evaluation record when a subprocess fails."""

    task_result = extract_task_result(output)

    return {
        "condition": condition,
        "experiment_seed": seed,
        "initial_world_state": None,
        "failure_mode": failure_mode,
        "failure_step": FAILURE_STEP,
        "return_code": return_code,
        "process_success": False,
        "initial_plan_valid": extract_last_bool(output, "Initial plan valid"),
        "initial_steps": extract_last_int(output, "Initial steps"),
        "completed_steps": extract_last_int(output, "Completed steps"),
        "failure_occurred": extract_last_bool(output, "Failure occurred"),
        "replanning_used": (
            extract_last_bool(output, "Replanning triggered")
            if condition == "replanning"
            else extract_last_bool(output, "Replanning used")
        ),
        "replan_valid": (
            extract_last_bool(output, "Replan valid")
            if condition == "replanning"
            else None
        ),
        "recovery_attempted": (
            extract_last_bool(output, "Recovery attempted")
            if condition == "baseline"
            else extract_last_bool(output, "Recovery attempted")
        ),
        "recovery_success": (
            extract_last_bool(output, "Recovery success")
            if condition == "replanning"
            else None
        ),
        "task_success": False,
        "replanning_attempts": (
            extract_last_int(output, "Replan attempts")
            if condition == "replanning"
            else None
        ),
        "json_result_available": False,
        "error": error,
        **task_result,
    }


# ============================================================
# TEMPORARY FAILURE MODE
# ============================================================

def temporarily_set_failure_mode(
    module_path: Path,
    failure_mode: str,
) -> str:

    original_text = module_path.read_text(
        encoding="utf-8"
    )

    pattern = (
        r'FAILURE_MODE\s*=\s*["\'][^"\']+["\']'
    )

    replacement = (
        f'FAILURE_MODE = "{failure_mode}"'
    )

    modified_text, count = re.subn(
        pattern,
        replacement,
        original_text,
        count=1,
    )

    if count != 1:

        raise RuntimeError(
            f"Could not locate FAILURE_MODE in "
            f"{module_path}"
        )

    module_path.write_text(
        modified_text,
        encoding="utf-8",
    )

    return original_text


# ============================================================
# BASELINE
# ============================================================

def run_baseline(
    failure_mode: str,
) -> Dict[str, Any]:

    print_header(
        f"BASELINE EXPERIMENT: {failure_mode}"
    )

    module_path = (
        PROJECT_ROOT
        / "src"
        / "baseline_eval.py"
    )

    original_text = (
        temporarily_set_failure_mode(
            module_path,
            failure_mode,
        )
    )

    log_path = (
        RAW_LOG_DIR
        / f"baseline_{failure_mode}.txt"
    )

    result_path = PROJECT_ROOT / "baseline_experiment_result.json"
    clear_json_result(result_path)

    try:

        return_code, output = run_command(
            [
                sys.executable,
                "-m",
                "src.baseline_eval",
            ],
            log_path,
            SEEDS[failure_mode],
        )

    finally:

        module_path.write_text(
            original_text,
            encoding="utf-8",
        )

    if return_code != 0:
        return make_process_failure_record(
            condition="baseline",
            failure_mode=failure_mode,
            return_code=return_code,
            output=output,
            error=(
                f"Baseline subprocess failed for {failure_mode}. "
                f"Return code: {return_code}. See log: {log_path}"
            ),
            seed=SEEDS[failure_mode],
        )

    json_result = load_json_result(result_path)

    task_result = extract_task_result(
        output
    )

    record = {

        "condition":
            "baseline",

        "experiment_seed": SEEDS[failure_mode],
        "initial_world_state": json_result.get("initial_world_state") if json_result else None,

        "failure_mode":
            failure_mode,

        "failure_step":
            FAILURE_STEP,

        "return_code":
            return_code,

        "process_success":
            return_code == 0,

        "initial_plan_valid":
            extract_last_bool(
                output,
                "Initial plan valid",
            ),

        "initial_steps":
            extract_last_int(
                output,
                "Initial steps",
            ),

        "completed_steps":
            extract_last_int(
                output,
                "Completed steps",
            ),

        "failure_occurred":
            extract_last_bool(
                output,
                "Failure occurred",
            ),

        "replanning_used":
            extract_last_bool(
                output,
                "Replanning used",
            ),

        "recovery_attempted":
            extract_last_bool(
                output,
                "Recovery attempted",
            ),

        "task_success":
            extract_last_bool(
                output,
                "Task success",
            ),

        "json_result_available":
            json_result is not None,

        **task_result,
    }

    return record


# ============================================================
# REPLANNING
# ============================================================

def run_replanning(
    failure_mode: str,
) -> Dict[str, Any]:

    print_header(
        f"REPLANNING EXPERIMENT: {failure_mode}"
    )

    module_path = (
        PROJECT_ROOT
        / "src"
        / "replanning_loop.py"
    )

    original_text = (
        temporarily_set_failure_mode(
            module_path,
            failure_mode,
        )
    )

    log_path = (
        RAW_LOG_DIR
        / f"replanning_{failure_mode}.txt"
    )

    result_path = PROJECT_ROOT / "replanning_experiment_result.json"
    clear_json_result(result_path)

    try:

        return_code, output = run_command(
            [
                sys.executable,
                "-m",
                "src.replanning_loop",
            ],
            log_path,
            SEEDS[failure_mode],
        )

    finally:

        module_path.write_text(
            original_text,
            encoding="utf-8",
        )

    if return_code != 0:
        return make_process_failure_record(
            condition="replanning",
            failure_mode=failure_mode,
            return_code=return_code,
            output=output,
            error=(
                f"Replanning subprocess failed for {failure_mode}. "
                f"Return code: {return_code}. See log: {log_path}"
            ),
            seed=SEEDS[failure_mode],
        )

    json_result = load_json_result(result_path)

    task_result = extract_task_result(
        output
    )

    record = {

        "condition":
            "replanning",

        "experiment_seed": SEEDS[failure_mode],
        "initial_world_state": json_result.get("initial_world_state") if json_result else None,

        "failure_mode":
            failure_mode,

        "failure_step":
            FAILURE_STEP,

        "return_code":
            return_code,

        "process_success":
            return_code == 0,

        "initial_plan_valid":
            extract_last_bool(
                output,
                "Initial plan valid",
            ),

        "initial_steps":
            extract_last_int(
                output,
                "Initial steps",
            ),

        "completed_steps":
            extract_last_int(
                output,
                "Completed steps",
            ),

        "failure_occurred":
            True,

        # Replanning uses different final-summary labels.
        "replanning_used":
            extract_last_bool(
                output,
                "Replanning triggered",
            ),

        "replan_valid":
            extract_last_bool(
                output,
                "Replan valid",
            ),

        "recovery_attempted":
            extract_last_bool(
                output,
                "Recovery attempted",
            ),

        "recovery_success":
            extract_last_bool(
                output,
                "Recovery success",
            ),

        "task_success":
            extract_last_bool(
                output,
                "Final success",
            ),

        "replanning_attempts":
            extract_last_int(
                output,
                "Replan attempts",
            ),

        "json_result_available":
            json_result is not None,

        **task_result,
    }

    return record


# ============================================================
# STATISTICS
# ============================================================

def success_rate(
    rows: List[Dict[str, Any]],
    key: str,
) -> float | None:

    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return (
        sum(bool(value) for value in values)
        / len(values)
    )


def average(
    rows: List[Dict[str, Any]],
    key: str,
) -> float | None:

    values = [
        row[key]
        for row in rows
        if isinstance(
            row.get(key),
            (int, float),
        )
    ]

    if not values:
        return None

    return sum(values) / len(values)


def calculate_summary(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:

    baseline = [
        row
        for row in records
        if row["condition"] == "baseline"
    ]

    replanning = [
        row
        for row in records
        if row["condition"] == "replanning"
    ]

    scene_matches = {}
    for mode in FAILURE_MODES:
        b = next((r for r in baseline if r["failure_mode"] == mode), None)
        r = next((r for r in replanning if r["failure_mode"] == mode), None)
        if not b or not r or not b.get("initial_world_state") or not r.get("initial_world_state"):
            scene_matches[mode] = False
            continue
        def positions(record):
            return {obj["name"]: obj["position"] for obj in record["initial_world_state"]["objects"]}
        bp, rp = positions(b), positions(r)
        scene_matches[mode] = bp.keys() == rp.keys() and all(
            all(abs(x-y) <= 1e-5 for x,y in zip(bp[name],rp[name])) for name in bp
        )

    return {
        "paired_initial_scene_matches": scene_matches,
        "all_initial_scenes_match": all(scene_matches.values()),

        "generated_at":
            datetime.now().isoformat(),

        "number_of_experiments":
            len(records),

        "baseline_experiments":
            len(baseline),

        "replanning_experiments":
            len(replanning),

        "baseline_task_success_rate":
            success_rate(
                baseline,
                "task_success",
            ),

        "replanning_task_success_rate":
            success_rate(
                replanning,
                "task_success",
            ),

        "replanning_recovery_success_rate":
            success_rate(
                replanning,
                "recovery_success",
            ),

        "baseline_average_completed_steps":
            average(
                baseline,
                "completed_steps",
            ),

        "replanning_average_completed_steps":
            average(
                replanning,
                "completed_steps",
            ),

        "replanning_average_attempts":
            average(
                replanning,
                "replanning_attempts",
            ),
    }


# ============================================================
# OUTPUT
# ============================================================

def save_csv(
    records: List[Dict[str, Any]],
) -> None:

    if not records:
        return

    fieldnames: List[str] = []

    for record in records:

        for key in record:

            if key not in fieldnames:
                fieldnames.append(key)

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)


def save_json(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:

    output = {
        "summary": summary,
        "experiments": records,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "AUTOMATED FAILURE EVALUATION"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        "Failure modes: "
        + ", ".join(FAILURE_MODES)
    )

    print(
        "Conditions   : baseline + replanning"
    )

    print(
        f"Total runs   : "
        f"{len(FAILURE_MODES) * 2}"
    )

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    RAW_LOG_DIR.mkdir(
        exist_ok=True
    )

    records: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    for failure_mode in FAILURE_MODES:

        try:
            records.append(
                run_baseline(failure_mode)
            )
        except Exception as exc:
            error = (
                f"Unhandled baseline evaluation error for "
                f"{failure_mode}: {type(exc).__name__}: {exc}"
            )
            print(f"ERROR: {error}")
            records.append(
                make_process_failure_record(
                    condition="baseline",
                    failure_mode=failure_mode,
                    return_code=None,
                    output="",
                    error=error,
                    seed=SEEDS[failure_mode],
                )
            )

    # --------------------------------------------------------
    # REPLANNING
    # --------------------------------------------------------

    for failure_mode in FAILURE_MODES:

        try:
            records.append(
                run_replanning(failure_mode)
            )
        except Exception as exc:
            error = (
                f"Unhandled replanning evaluation error for "
                f"{failure_mode}: {type(exc).__name__}: {exc}"
            )
            print(f"ERROR: {error}")
            records.append(
                make_process_failure_record(
                    condition="replanning",
                    failure_mode=failure_mode,
                    return_code=None,
                    output="",
                    error=error,
                    seed=SEEDS[failure_mode],
                )
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = calculate_summary(
        records
    )

    print("Paired initial scene matches:", summary["paired_initial_scene_matches"])
    if not summary["all_initial_scenes_match"]:
        print(
            "WARNING: One or more paired initial scenes could not be "
            "verified. Results are still saved, but the affected pair(s) "
            "should not be used for paired comparison."
        )

    save_json(
        records,
        summary,
    )

    save_csv(
        records
    )

    # --------------------------------------------------------
    # FINAL DISPLAY
    # --------------------------------------------------------

    print_header(
        "AUTOMATED EVALUATION COMPLETE"
    )

    print(
        f"Experiments completed : "
        f"{len(records)}"
    )

    print()

    print(
        "Baseline task success rate   : "
        f"{summary['baseline_task_success_rate']}"
    )

    print(
        "Replanning task success rate : "
        f"{summary['replanning_task_success_rate']}"
    )

    print(
        "Recovery success rate        : "
        f"{summary['replanning_recovery_success_rate']}"
    )

    print()

    print(
        "Baseline average steps       : "
        f"{summary['baseline_average_completed_steps']}"
    )

    print(
        "Replanning average steps     : "
        f"{summary['replanning_average_completed_steps']}"
    )

    print(
        "Average replan attempts      : "
        f"{summary['replanning_average_attempts']}"
    )

    print()

    print(
        "Results directory:"
    )

    print(
        f"  {RESULTS_DIR}"
    )

    print()

    print(
        "JSON:"
    )

    print(
        f"  {SUMMARY_JSON}"
    )

    print()

    print(
        "CSV:"
    )

    print(
        f"  {SUMMARY_CSV}"
    )

    print()

    print(
        "Raw logs:"
    )

    print(
        f"  {RAW_LOG_DIR}"
    )


if __name__ == "__main__":
    main()
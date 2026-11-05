"""Shadow-only planning contract above the stable single-step execution agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reconstruction_observer import build_observation, execute_observation_tasks, observe_workspace


PLAN_SCHEMA_VERSION = "reconstruction-experiment-plan-0.1"


def build_planning_prompt(observation: dict[str, Any], max_new_jobs: int = 8) -> dict[str, Any]:
    """Build the bounded input/output contract for a future PlanningAgent call."""
    return {
        "instruction": (
            "You are a reconstruction planning agent. Do not create CryoSPARC jobs and do not "
            "invent Job IDs, tool names, or measurements. Review the full workspace observation, "
            "identify the highest-value unresolved quality bottleneck, and return one bounded "
            "experiment plan. Prefer comparison or curation before repeating unchanged refinement."
        ),
        "observation": observation,
        "output_contract": {
            "schema_version": PLAN_SCHEMA_VERSION,
            "decision_type": "request_observations | plan_experiments | wait | request_input | finalize",
            "observation_tasks": [{"task_id": "string", "tool": "allow-listed read-only tool", "arguments": {}}],
            "objective": "The immediate quality uncertainty to resolve.",
            "hypotheses": [{"id": "string", "claim": "string", "evidence": ["observation reference"]}],
            "experiments": [{
                "experiment_id": "string",
                "template": "compare_class2d_branches | reselect_2d | reextract_and_classify | repick_and_curate | refine_selected_branch",
                "source_job_uids": ["existing CryoSPARC Job IDs from observation"],
                "parameter_bounds": {},
                "expected_information_gain": "string",
                "comparison_metrics": ["string"],
            }],
            "resource_budget": {"max_new_jobs": max_new_jobs, "max_parallel_gpu_jobs": 2},
            "stop_conditions": ["string"],
            "evidence_gaps": ["string"],
        },
    }


def build_shadow_plan(observation: dict[str, Any], max_new_jobs: int = 8) -> dict[str, Any]:
    """Produce a conservative non-LLM reference plan for pilot validation."""
    issues = observation.get("observed_issues") or []
    comparisons = observation.get("class2d_comparisons") or []
    experiments = []
    hypotheses = []
    issue_types = {issue.get("issue_type") for issue in issues}

    if "permissive_select2d" in issue_types:
        issue = next(item for item in issues if item.get("issue_type") == "permissive_select2d")
        evidence = issue.get("evidence") or {}
        hypotheses.append({
            "id": "H1_select2d_overretention",
            "claim": "The selected particle set needs quality review before further refinement.",
            "evidence": [evidence],
        })
        experiments.append({
            "experiment_id": "E1_compare_and_reselect_class2d",
            "template": "compare_class2d_branches" if comparisons else "reselect_2d",
            "source_job_uids": [str(evidence.get("class2d_job_uid"))] if evidence.get("class2d_job_uid") else [],
            "parameter_bounds": {"retain_fraction_requires_visual_review": True},
            "expected_information_gain": "Determine whether poor class averages or permissive class selection caused low-quality particles to persist.",
            "comparison_metrics": ["class_average_visual_quality", "cryosift_score", "selected_particle_count"],
        })

    if comparisons:
        source_uids = [
            branch.get("class2d_job_uid")
            for group in comparisons for branch in group.get("branches") or []
            if branch.get("class2d_job_uid")
        ]
        experiments.append({
            "experiment_id": "E2_rank_box_size_branches",
            "template": "compare_class2d_branches",
            "source_job_uids": source_uids,
            "parameter_bounds": {},
            "expected_information_gain": "Select one extraction scale using class evidence before new 3D work.",
            "comparison_metrics": ["box_size_pix", "class_average_visual_quality", "cryosift_score", "class_statistics"],
        })

    if "target_resolution_unmet" in issue_types:
        hypotheses.append({
            "id": "H2_upstream_quality_limits_resolution",
            "claim": "An upstream particle-selection intervention is more informative than unchanged refinement.",
            "evidence": [next(item.get("evidence") for item in issues if item.get("issue_type") == "target_resolution_unmet")],
        })

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "mode": "shadow_only",
        "decision_type": "plan_experiments" if experiments else "wait",
        "objective": "Resolve the highest-priority observed quality bottleneck before executing more reconstruction jobs.",
        "hypotheses": hypotheses,
        "experiments": experiments[:max_new_jobs],
        "resource_budget": {"max_new_jobs": max_new_jobs, "max_parallel_gpu_jobs": 2},
        "stop_conditions": ["No experiment may be executed by this pilot.", "A future execution gateway must validate all live actions."],
        "evidence_gaps": ["Particle diameter and box occupancy require Class 2D visual measurement."],
    }


def validate_planning_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Validate the narrow PlanningAgent contract before any observer call."""
    decision_type = decision.get("decision_type")
    allowed = {"request_observations", "plan_experiments", "wait", "request_input", "finalize"}
    if decision_type not in allowed:
        raise ValueError("unsupported planning decision type")
    tasks = decision.get("observation_tasks") or []
    if decision_type == "request_observations" and not tasks:
        raise ValueError("request_observations requires at least one observation task")
    if decision_type != "request_observations" and tasks:
        raise ValueError("only request_observations may include observation tasks")
    if len(tasks) > 4:
        raise ValueError("at most four observation tasks are allowed")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only reconstruction planning pilot.")
    parser.add_argument("--project")
    parser.add_argument("--workspace")
    parser.add_argument("--workflow-state-file")
    parser.add_argument("--dataset-json-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-jobs", type=int, default=8)
    parser.add_argument("--observation-request-file", help="Execute a validated read-only PlanningAgent request JSON.")
    args = parser.parse_args()
    if bool(args.project) != bool(args.workspace):
        parser.error("--project and --workspace must be supplied together")
    if not args.workflow_state_file and not args.project:
        parser.error("supply a live project/workspace or --workflow-state-file")
    dataset = _read_json(args.dataset_json_file) if args.dataset_json_file else {}
    if args.workflow_state_file:
        observation = build_observation(_read_json(args.workflow_state_file), dataset)
    else:
        observation = observe_workspace(args.project, args.workspace, dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "observation.json", observation)
    _write_json(output_dir / "planning_prompt.json", build_planning_prompt(observation, args.max_new_jobs))
    _write_json(output_dir / "shadow_plan.json", build_shadow_plan(observation, args.max_new_jobs))
    if args.observation_request_file:
        if args.workflow_state_file:
            workflow_state = _read_json(args.workflow_state_file)
        else:
            from workflow_state import extract_workflow_state
            workflow_state = extract_workflow_state(args.project, args.workspace)
        decision = validate_planning_decision(_read_json(args.observation_request_file))
        _write_json(
            output_dir / "requested_observations.json",
            {"decision": decision, "results": execute_observation_tasks(workflow_state, dataset, decision["observation_tasks"])},
        )


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

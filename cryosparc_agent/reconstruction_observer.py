"""Build a bounded whole-workspace observation for reconstruction planning."""
from __future__ import annotations

from collections import defaultdict
import copy
import math
from typing import Any

from class2d_scale import build_class2d_scale_context
from abinitio_class2d_consistency import (
    build_abinitio_class2d_consistency_context,
    build_template_matching_consistency_context,
)
from quality_policy import assess_node
from workflow_state import extract_workflow_state, job_uid_number


OBSERVATION_SCHEMA_VERSION = "reconstruction-observation-0.1"
READ_ONLY_OBSERVATION_TOOLS = {
    "get_workspace_overview",
    "get_job_detail",
    "trace_upstream_lineage",
    "get_branch_comparison",
    "get_refinement_quality_context",
    "get_class2d_scale_context",
    "get_abinitio_class2d_consistency_context",
    "get_abinitio_class2d_template_matching_context",
}


def normalize_observation_tasks(tasks, *, project_uid=None, workspace_uid=None):
    """Canonicalize known read-only aliases without accepting new tools or scopes."""
    if tasks is None:
        tasks = []
    if not isinstance(tasks, list) or len(tasks) > 4:
        raise ValueError('observation_tasks must be a list of at most four tasks')
    result = []
    for task in tasks:
        if not isinstance(task, dict) or task.get('tool') not in READ_ONLY_OBSERVATION_TOOLS:
            raise ValueError('unsupported read-only observation tool')
        arguments = task.get('arguments', {})
        if not isinstance(arguments, dict):
            raise ValueError('observation arguments must be an object')
        arguments = copy.deepcopy(arguments)
        for key, expected in (('project_uid', project_uid), ('workspace_uid', workspace_uid)):
            if key in arguments:
                if expected is not None and arguments[key] != expected:
                    raise ValueError('observation request leaves the assigned ' + key)
                if expected is not None:
                    arguments.pop(key)
        aliases = []
        if task['tool'] in {'get_abinitio_class2d_consistency_context',
                            'get_abinitio_class2d_template_matching_context'}:
            aliases.append('abinitio_job_uid')
        if task['tool'] == 'get_class2d_scale_context':
            aliases.append('class2d_job_uid')
        for alias in aliases:
            if alias not in arguments:
                continue
            value = arguments.pop(alias)
            if 'job_uid' in arguments and arguments['job_uid'] != value:
                raise ValueError('conflicting observation job aliases')
            arguments['job_uid'] = value
        result.append({**copy.deepcopy(task), 'arguments': arguments})
    return result


def observe_workspace(
    project_uid: str,
    workspace_uid: str,
    dataset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read a live workspace and return the planner's compact global view."""
    state = extract_workflow_state(project_uid, workspace_uid)
    return build_observation(state, dataset_info)


def build_observation(
    workflow_state: dict[str, Any],
    dataset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a normalized DAG into facts without proposing or executing jobs."""
    nodes = sorted(
        workflow_state.get("nodes") or [],
        key=lambda item: job_uid_number(item.get("cryosparc_job_uid", "")),
    )
    summaries = [_node_summary(node) for node in nodes]
    by_uid = {node["cryosparc_job_uid"]: node for node in nodes}
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "project_uid": workflow_state.get("project_uid"),
        "workspace_uid": workflow_state.get("workspace_uid"),
        "generated_at": workflow_state.get("generated_at"),
        "dataset_info": _dataset_facts(dataset_info or {}),
        "workflow_status": workflow_state.get("workflow_status"),
        "dag": {
            "root_nodes": workflow_state.get("root_nodes") or [],
            "terminal_nodes": workflow_state.get("terminal_nodes") or [],
            "running_nodes": workflow_state.get("running_nodes") or [],
            "failed_nodes": workflow_state.get("failed_nodes") or [],
            "nodes": summaries,
        },
        "branch_points": _branch_points(nodes),
        "class2d_comparisons": _class2d_comparisons(nodes, by_uid),
        "selection_outcomes": _selection_outcomes(nodes, by_uid),
        "resolution_observations": _resolution_observations(nodes),
        "observed_issues": _observed_issues(nodes, by_uid, dataset_info or {}),
        "limitations": [
            "This is a read-only summary of CryoSPARC-exposed state.",
            "Particle diameter and CTF-aware extraction scale are estimated by the dedicated Class 2D scale observer when requested.",
            "Observed correlations are not treated as causal conclusions.",
        ],
    }


def execute_observation_tasks(
    workflow_state: dict[str, Any],
    dataset_info: dict[str, Any] | None,
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run allow-listed read-only requests selected by a PlanningAgent."""
    tasks = normalize_observation_tasks(tasks,
        project_uid=workflow_state.get('project_uid'), workspace_uid=workflow_state.get('workspace_uid'))
    observation = build_observation(workflow_state, dataset_info)
    nodes = {
        node.get("cryosparc_job_uid"): node
        for node in workflow_state.get("nodes") or []
    }
    results = []
    for task in tasks:
        tool = task.get("tool")
        arguments = task.get("arguments") or {}
        requested = [arguments[key] for key in ('job_uid', 'start_job_uid') if arguments.get(key)]
        for key in ('job_uids', 'class2d_job_uids'):
            values = arguments.get(key) or []
            if not isinstance(values, list):
                raise ValueError('observation job IDs must be lists')
            requested.extend(values)
        if any(not isinstance(uid, str) or uid not in nodes for uid in requested):
            raise ValueError('unknown workspace job in observation request')
        if tool not in READ_ONLY_OBSERVATION_TOOLS:
            raise ValueError(f"unsupported read-only observation tool: {tool}")
        if tool == "get_workspace_overview":
            payload = {
                "workflow_status": observation["workflow_status"],
                "branch_points": observation["branch_points"],
                "observed_issues": observation["observed_issues"],
                "resolution_observations": observation["resolution_observations"],
            }
        elif tool == "get_job_detail":
            job_uid = str(arguments.get("job_uid") or "")
            if job_uid not in nodes:
                raise ValueError(f"unknown workspace job: {job_uid}")
            payload = nodes[job_uid]
        elif tool == "trace_upstream_lineage":
            payload = _trace_upstream(nodes, str(arguments.get("start_job_uid") or ""), arguments.get("stop_at_job_types") or [])
        elif tool == "get_branch_comparison":
            requested = set(arguments.get("class2d_job_uids") or [])
            payload = [
                group for group in observation["class2d_comparisons"]
                if not requested or requested.intersection({branch["class2d_job_uid"] for branch in group["branches"]})
            ]
        elif tool == "get_refinement_quality_context":
            payload = [
                item for item in observation["resolution_observations"]
                if item.get("job_type") in {"homo_abinit", "homo_refine_new", "nonuniform_refine_new", "validation"}
            ]
        elif tool == "get_abinitio_class2d_consistency_context":
            job_uid = str(arguments.get("job_uid") or "")
            if job_uid not in nodes:
                raise ValueError(f"unknown workspace job: {job_uid}")
            context = build_abinitio_class2d_consistency_context(
                workflow_state.get("project_uid"), job_uid,
                int(arguments.get("max_selected_classes", 30)),
            )
            # Planning prompts consume a concise observation; the visual sub-observer
            # receives the image separately and returns a verdict for the planner.
            payload = {key: value for key, value in context.items() if key != "contact_sheet"}
            payload["contact_sheet"] = {
                key: value for key, value in context["contact_sheet"].items() if key != "data_url"
            }
        elif tool == "get_abinitio_class2d_template_matching_context":
            job_uid = str(arguments.get("job_uid") or "")
            if job_uid not in nodes:
                raise ValueError(f"unknown workspace job: {job_uid}")
            payload = build_template_matching_consistency_context(
                workflow_state.get("project_uid"), job_uid,
                int(arguments.get("max_selected_classes", 30)),
            )
        else:
            job_uids = arguments.get("job_uids") or [arguments.get("job_uid")]
            contexts = [
                build_class2d_scale_context(
                    workflow_state.get("project_uid"),
                    str(job_uid),
                    int(arguments.get("max_classes", 50)),
                    target_resolution_A=(dataset_info or {}).get("target_resolution_A")
                    or (dataset_info or {}).get("resolution"),
                )
                for job_uid in job_uids if job_uid
            ]
            payload = {
                "contexts": contexts,
                "cross_branch_consistency": _cross_branch_scale_consistency(contexts),
            }
        results.append({
            "task_id": task.get("task_id"),
            "tool": tool,
            "arguments": arguments,
            "result": payload,
        })
    return results


def _dataset_facts(dataset_info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "empiar_id", "emdb_id", "target_resolution_A", "resolution", "pixel_size_A",
        "accelerating_voltage_kv", "spherical_aberration_mm",
        "total_exposure_dose_e_per_A2", "symmetry", "dataset_parameter_facts",
    )
    return {key: dataset_info.get(key) for key in keys if dataset_info.get(key) is not None}


def _node_summary(node: dict[str, Any]) -> dict[str, Any]:
    quality = assess_node(node)
    output_counts = {
        name: output.get("num_items")
        for name, output in (node.get("outputs") or {}).items()
    }
    return {
        "job_uid": node.get("cryosparc_job_uid"),
        "job_type": node.get("job_type"),
        "status": node.get("status"),
        "parents": node.get("parent_job_uids") or [],
        "children": node.get("child_job_uids") or [],
        "parameters": node.get("key_parameters") or {},
        "output_counts": output_counts,
        "quality": {
            "status": quality.get("quality_status"),
            "resolution_A": quality.get("resolution_evidence", {}).get("values", {}),
            "metrics": quality.get("observed_metrics", {}),
        },
        "has_warning": bool(node.get("has_warning")),
        "has_error": bool(node.get("has_error")),
    }


def _branch_points(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_uid": node.get("cryosparc_job_uid"),
            "job_type": node.get("job_type"),
            "children": node.get("child_job_uids") or [],
        }
        for node in nodes
        if len(node.get("child_job_uids") or []) > 1
    ]


def _class2d_comparisons(
    nodes: list[dict[str, Any]], by_uid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("job_type") != "class_2D_new" or node.get("status") != "completed":
            continue
        inputs = (node.get("inputs") or {}).get("particles") or []
        extract_uid = (inputs[0] if inputs else {}).get("source_job_uid")
        extract = by_uid.get(extract_uid, {})
        # Compare branches by the source feeding their extraction, not by the
        # extraction Job UID itself: each box size naturally has a distinct UID.
        sources = tuple(sorted(
            (str(item.get("source_job_uid")), str(item.get("source_output")))
            for connections in (extract.get("inputs") or {}).values()
            for item in connections
        ))
        if not sources:
            continue
        groups[sources].append(node)

    comparisons = []
    for source, members in groups.items():
        branches = []
        for node in members:
            particle_input = ((node.get("inputs") or {}).get("particles") or [{}])[0]
            extract = by_uid.get(particle_input.get("source_job_uid"), {})
            branches.append({
                "class2d_job_uid": node.get("cryosparc_job_uid"),
                "extract_job_uid": extract.get("cryosparc_job_uid"),
                "box_size_pix": (extract.get("key_parameters") or {}).get("box_size_pix"),
                "input_particles": _output_count(extract, "particles"),
                "class_metrics": assess_node(node).get("observed_metrics", {}),
            })
        if len(branches) > 1:
            comparisons.append({"source": source, "branches": branches})
    return comparisons


def _selection_outcomes(
    nodes: list[dict[str, Any]], by_uid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    outcomes = []
    for node in nodes:
        if node.get("job_type") != "select_2D" or node.get("status") != "completed":
            continue
        particles = ((node.get("inputs") or {}).get("particles") or [{}])[0]
        source = by_uid.get(particles.get("source_job_uid"), {})
        input_count = _output_count(source, "particles")
        selected_count = _output_count(node, "particles_selected")
        retention = selected_count / input_count if input_count else None
        outcomes.append({
            "select2d_job_uid": node.get("cryosparc_job_uid"),
            "class2d_job_uid": source.get("cryosparc_job_uid"),
            "input_particles": input_count,
            "selected_particles": selected_count,
            "retention_fraction": retention,
        })
    return outcomes


def _resolution_observations(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    for node in nodes:
        values = assess_node(node).get("resolution_evidence", {}).get("values", {})
        for metric, value in values.items():
            observations.append({
                "job_uid": node.get("cryosparc_job_uid"),
                "job_type": node.get("job_type"),
                "metric": metric,
                "resolution_A": value,
            })
    return observations


def _observed_issues(
    nodes: list[dict[str, Any]], by_uid: dict[str, dict[str, Any]], dataset_info: dict[str, Any],
) -> list[dict[str, Any]]:
    issues = []
    for outcome in _selection_outcomes(nodes, by_uid):
        retention = outcome.get("retention_fraction")
        if retention is not None and retention > 0.95 and (outcome.get("input_particles") or 0) >= 100_000:
            issues.append({
                "issue_type": "permissive_select2d",
                "severity": "high",
                "evidence": outcome,
                "interpretation": "High retention is an observed review trigger, not proof that every retained class is poor.",
            })
    try:
        target = float(dataset_info.get("target_resolution_A"))
    except (TypeError, ValueError):
        target = None
    resolutions = _resolution_observations(nodes)
    values = [item["resolution_A"] for item in resolutions if isinstance(item.get("resolution_A"), (int, float))]
    if target and values and min(values) > target:
        issues.append({
            "issue_type": "target_resolution_unmet",
            "severity": "high",
            "evidence": {"target_resolution_A": target, "best_observed_resolution_A": min(values)},
            "interpretation": "The best observed resolution remains above the stated target.",
        })
    return issues


def _output_count(node: dict[str, Any], output_name: str) -> int | None:
    value = ((node.get("outputs") or {}).get(output_name) or {}).get("num_items")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _trace_upstream(
    nodes: dict[str, dict[str, Any]], start_job_uid: str, stop_at_job_types: list[str],
) -> list[dict[str, Any]]:
    if start_job_uid not in nodes:
        raise ValueError(f"unknown workspace job: {start_job_uid}")
    lineage = []
    queue = [start_job_uid]
    visited = set()
    while queue:
        job_uid = queue.pop(0)
        if job_uid in visited:
            continue
        visited.add(job_uid)
        node = nodes[job_uid]
        lineage.append(_node_summary(node))
        if node.get("job_type") in stop_at_job_types:
            continue
        queue.extend(parent for parent in node.get("parent_job_uids") or [] if parent in nodes)
    return lineage


def _cross_branch_scale_consistency(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Prevent a box-dependent footprint from being mistaken for particle diameter."""
    estimates = [
        item.get("aggregate", {}).get("recommended_particle_diameter_A")
        or item.get("aggregate", {}).get("particle_diameter_A_p80")
        for item in contexts
    ]
    values = [float(value) for value in estimates if isinstance(value, (int, float)) and value > 0]
    if len(values) < 2:
        return {"status": "insufficient_branches", "usable_for_blob_picker_update": False}
    mean = sum(values) / len(values)
    coefficient_of_variation = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean
    consistent = coefficient_of_variation <= 0.15
    return {
        "status": "consistent" if consistent else "inconsistent",
        "diameter_estimates_A": values,
        "coefficient_of_variation": coefficient_of_variation,
        "usable_for_blob_picker_update": consistent,
        "interpretation": (
            "The estimated footprint is stable across box sizes."
            if consistent else
            "The footprint changes materially with box size; obtain stronger visual or model-based segmentation evidence before changing Blob Picker diameter."
        ),
    }

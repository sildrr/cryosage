# Plans and eventually executes CryoSPARC jobs through a generic job API adapter.
from typing import Any
import os

from cryosparc_client import cryosparc_client
from job_specs import get_job_spec
from resource_strategy import select_available_lane

CORE_REQUIRED_INPUTS = {
    'select_2D': {'templates': 1},
    'class_2D_new': {'particles': 1},
    'homo_refine_new': {'particles': 1, 'volume': 1},
    'nonuniform_refine_new': {'particles': 1, 'volume': 1},
    'template_picker_gpu': {'micrographs': 1, 'templates': 1},
}


def validate_input_connections(job_type, connections, required_counts=None):
    counts = required_counts if required_counts is not None else CORE_REQUIRED_INPUTS.get(job_type, {})
    missing = []
    for name, minimum in counts.items():
        value = (connections or {}).get(name)
        count = len(value) if isinstance(value, list) else int(bool(value))
        if count < minimum:
            missing.append(name)
    if missing:
        raise ValueError('Missing required input connections: ' + ', '.join(sorted(missing)))


def plan_job_action(
    action: dict[str, Any],
    candidate_action: dict[str, Any] | None = None,
    lane: str | None = None,
) -> dict[str, Any]:
    """Build one dry-run job plan from a validated action and candidate context."""
    job_type = action["job_type"]
    parameters = action["resolved_parameters"]
    if job_type == 'inspect_picks_v2':
        from inspect_score_contract import raw_parameters
        parameters = raw_parameters(parameters)
    spec = get_job_spec(job_type)
    metadata = (candidate_action or {}).get("job_spec_metadata") or {}
    if metadata:
        spec.update({
            key: value
            for key, value in metadata.items()
            if value is not None
        })
    connections = build_explicit_connections(action.get("connections"))
    if connections is None:
        connections = build_connections(
            (candidate_action or {}).get("required_inputs", {})
        )
    from job_input_contracts import normalize_input_parameters, normalize_volume_connections
    parameters = normalize_input_parameters(job_type, parameters, connections)
    connections = normalize_volume_connections(job_type, parameters, connections,
        repeat_allowed='volume' in metadata.get('repeat_allowed_inputs', []))
    input_counts = metadata.get('required_input_counts')
    validate_input_connections(job_type, connections, input_counts)
    queue = build_queue_plan(spec, lane)
    approval_reasons = approval_reasons_for(action, spec)

    return {
        "plan_step": action["plan_step"],
        "action_id": action["action_id"],
        "action_type": action["action_type"],
        "workflow_node_id": action["workflow_node_id"],
        "job_type": job_type,
        "job_category": spec["category"],
        "execution_mode": action["execution_mode"],
        "mcp_tool_name": action.get("mcp_tool_name"),
        "connections": connections,
        "repeat_allowed_inputs": metadata.get('repeat_allowed_inputs', []),
        "required_input_counts": input_counts if input_counts is not None else CORE_REQUIRED_INPUTS.get(job_type, {}),
        "resolved_parameters": parameters,
        "queue": queue,
        "requires_gpu": spec["requires_gpu"],
        "interactive": spec["interactive"],
        "approval_required": bool(
            action.get("approval_required")
            or spec["requires_approval"]
            or approval_reasons
        ),
        "approval_reasons": sorted(
            set(action.get("approval_reasons", []) + approval_reasons)
        ),
        "rollback_target": action.get("rollback_target"),
        "status": "planned",
    }


def build_connections(
    required_inputs: dict[str, list[dict[str, Any]]],
) -> dict[str, tuple[str, str] | list[tuple[str, str]]]:
    """Convert registry input metadata into cryosparc-tools connections."""
    connections: dict[str, tuple[str, str] | list[tuple[str, str]]] = {}
    for input_name, input_connections in required_inputs.items():
        values = [
            (
                connection["source_job_uid"],
                connection["source_output"],
            )
            for connection in input_connections
        ]
        if not values:
            continue
        connections[input_name] = values[0] if len(values) == 1 else values
    return connections


def build_explicit_connections(
    raw_connections: dict[str, Any] | None,
) -> dict[str, tuple[str, str] | list[tuple[str, str]]] | None:
    """Normalize model-supplied CryoSPARC connections, if present."""
    if raw_connections is None or raw_connections == {}:
        return None
    if not isinstance(raw_connections, dict):
        raise ValueError('connections must be an object or null')
    connections: dict[str, tuple[str, str] | list[tuple[str, str]]] = {}
    for input_name, raw_value in raw_connections.items():
        values = normalize_connection_values(raw_value)
        if not values:
            raise ValueError(f'Invalid explicit connection for {input_name}')
        connections[input_name] = values[0] if len(values) == 1 else values
    return connections


def normalize_connection_values(raw_value: Any) -> list[tuple[str, str]]:
    """Accept compact or object-shaped model connection values."""
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    normalized = []
    for value in values:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            normalized.append((str(value[0]), str(value[1])))
            continue
        if not isinstance(value, dict):
            continue
        source_job = (
            value.get("source_job_uid")
            or value.get("source_job")
            or value.get("job_uid")
        )
        source_output = value.get("source_output") or value.get("output")
        if source_job and source_output:
            normalized.append((str(source_job), str(source_output)))
    return normalized


def build_queue_plan(spec: dict[str, Any], lane: str | None) -> dict[str, Any]:
    """Prepare queue settings without submitting anything to CryoSPARC."""
    configured_lane = os.getenv("CRYOAGENT_GPU_LANE")
    auto_select = os.getenv("CRYOAGENT_AUTO_SELECT_LANE", "1") == "1"
    # A static registry default is only a fallback.  In live mode, leaving
    # the lane open lets execute_job_action query current scheduler capacity.
    # Explicit function/env configuration still wins for reproducible runs.
    # This CryoSPARC deployment requires a lane for every non-interactive Job,
    # including CPU-only jobs such as Create Templates and Inspect Picks. A
    # CPU-only spec merely means no GPU devices are requested from that lane.
    # Interactive jobs are not auto-queued and must retain no lane selection.
    queue_without_lane = bool(spec.get("queue_without_lane"))
    selected_lane = None if (spec["interactive"] or queue_without_lane) else (lane or configured_lane)
    if not selected_lane and not auto_select:
        selected_lane = spec.get("default_lane")
    return {
        "lane": selected_lane,
        "hostname": None,
        "gpus": [],
        "cluster_vars": {},
        "queue_without_lane": queue_without_lane,
        "will_queue": not spec["interactive"] or queue_without_lane,
    }


def approval_reasons_for(
    action: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    """Explain why a planned action needs human approval."""
    reasons: list[str] = []
    if spec["requires_approval"]:
        reasons.append("job_spec_requires_approval")
    if spec["interactive"]:
        reasons.append("interactive_job")
    # A Select 2D candidate may be labelled ``branch`` only because earlier
    # failed submissions left multiple child shells. With an explicit class
    # list it is still one autonomous selection, not a parallel experiment.
    if action["action_type"] == "branch" and not spec.get("queue_without_lane"):
        reasons.append("branch_decision")
    gpu_count = action["resolved_parameters"].get("compute_num_gpus")
    if (
        spec["requires_gpu"]
        and isinstance(gpu_count, int)
        and not isinstance(gpu_count, bool)
        and gpu_count > spec["max_auto_gpus"]
    ):
        reasons.append("high_gpu_count")
    return reasons


def execute_job_action(
    project_uid: str,
    workspace_uid: str,
    planned_action: dict[str, Any],
    dry_run: bool = True,
    allow_approval_required_create: bool = False,
) -> dict[str, Any]:
    """
    Execute one planned job action.

    Dry-run mode is the default and only returns the plan. Live mode is kept
    behind approval gates and should be enabled only after policy is added.
    """
    if planned_action.get('job_type') == 'inspect_picks_v2':
        from inspect_score_contract import raw_parameters, SCORE_SPACE
        planned_action = {**planned_action,
            'resolved_parameters': raw_parameters(planned_action.get('resolved_parameters') or {}),
            'score_space': SCORE_SPACE}
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "status": "planned",
            "planned_action": planned_action,
            "message": "Dry run only; no CryoSPARC job was created or queued.",
        }

    if planned_action["approval_required"] and not allow_approval_required_create:
        return {
            "success": False,
            "dry_run": False,
            "status": "approval_required",
            "planned_action": planned_action,
            "issues": [
                {
                    "severity": "error",
                    "code": "approval_required",
                    "message": "Live execution requires human approval.",
                    "path": None,
                }
            ],
        }

    # Reconstruct execution-local state. ``spec`` and ``queue`` are local to
    # plan_job_action, so live execution must not rely on those names leaking
    # across function boundaries.
    spec = get_job_spec(planned_action["job_type"])
    # Candidate construction may use the live CryoSPARC registry for job
    # types that are not listed in the local metadata table. Preserve those
    # execution-critical flags from the validated plan.
    for key in (
        "category",
        "requires_gpu",
        "requires_approval",
        "interactive",
        "default_lane",
        "max_auto_gpus",
    ):
        if key in planned_action and planned_action[key] is not None:
            spec[key] = planned_action[key]
    queue = planned_action.get("queue") or build_queue_plan(spec, None)
    planned_action["queue"] = queue

    job = None
    try:
        cs = cryosparc_client()
        from refinement_optimization import PARTICLE_OPERATIONS, execute_particle_operation
        if planned_action['job_type'] in PARTICLE_OPERATIONS:
            return execute_particle_operation(cs, project_uid, workspace_uid, planned_action)
        counts = planned_action.get('required_input_counts')
        native = None
        specs = getattr(getattr(cs, 'job_register', None), 'specs', None)
        if isinstance(specs, (list, tuple)):
            native = next((s for s in specs if s.type == planned_action['job_type']), None)
            if native is not None:
                from dynamic_candidates import obsolete_registry_job, registry_required_input_counts
                if obsolete_registry_job(native):
                    return {"success": False, "dry_run": False, "status": "obsolete_job_type",
                        "error": "CryoSPARC marks this job type obsolete; no Job was created.",
                        "replacement_job_type": "orientation_diagnostics" if native.type == "fsc3D" else None,
                        "planned_action": planned_action}
                counts = registry_required_input_counts(native, planned_action.get('connections') or {})
        validate_input_connections(planned_action['job_type'], planned_action.get('connections'), counts)
        from job_input_contracts import normalize_input_parameters
        planned_action['resolved_parameters'] = normalize_input_parameters(
            planned_action['job_type'], planned_action.get('resolved_parameters') or {},
            planned_action.get('connections'))
        from job_input_contracts import normalize_volume_connections
        native_volume = getattr(getattr(native, 'inputs', None), 'root', {}).get('volume')
        planned_action['connections'] = normalize_volume_connections(planned_action['job_type'],
            planned_action['resolved_parameters'], planned_action.get('connections'),
            repeat_allowed=bool(getattr(native_volume, 'repeat_allowed', False)))
        workspace = cs.find_workspace(project_uid, workspace_uid)
        if queue["will_queue"] and spec.get("requires_gpu") and not queue.get("lane"):
            capacity = select_available_lane()
            if not capacity.get("success"):
                return {"success": False, "dry_run": False, "status": "resource_unavailable", "resource_check": capacity, "planned_action": planned_action}
            queue["lane"] = capacity["lane"]
            planned_action["queue"] = queue
            planned_action["resource_check"] = capacity
        # Create first and connect inputs explicitly so a rejected connection
        # can be rolled back instead of leaving an unqueued orphan job.
        job = workspace.create_job(
            planned_action["job_type"],
            connections={},
            params=planned_action["resolved_parameters"],
            title=f"Agent {planned_action['action_id']}",
            desc="Created by cryosparc_agent execute_model_decision.",
        )
        try:
            for input_name, connections in (planned_action["connections"] or {}).items():
                connection_list = connections if isinstance(connections, list) else [connections]
                for source_job_uid, source_output in connection_list:
                    job.connect(input_name, source_job_uid, source_output)
            if planned_action['job_type'] == 'class_3D' and planned_action['connections'].get('volume'):
                actual_count = len(job.inputs['volume'].connections)
                if actual_count != planned_action['resolved_parameters']['class3D_N_K']:
                    raise ValueError('Connected native initial-volume count does not match class3D_N_K; not queued')
        except Exception as connection_error:
            cleanup = {"attempted": True, "success": False}
            try:
                job.kill()
                cleanup["success"] = True
            except Exception as cleanup_error:
                cleanup["error"] = str(cleanup_error)
            return {
                "success": False,
                "dry_run": False,
                "status": "input_connection_failed",
                "error": str(connection_error),
                "error_type": type(connection_error).__name__,
                "job_uid": job.uid,
                "cleanup": cleanup,
                "planned_action": planned_action,
            }
        queue = planned_action["queue"]
        queued = False
        if queue["will_queue"]:
            # CryoSPARC rejects any lane/hostname for interactive queue jobs
            # (notably Inspect Particle Picks). This must take precedence over
            # a globally configured GPU lane carried by an earlier plan.
            if queue.get("queue_without_lane"):
                job.queue()
            elif queue["lane"]:
                job.queue(
                    lane=queue["lane"],
                    gpus=queue["gpus"],
                    cluster_vars=queue["cluster_vars"],
                )
            else:
                # A live GPU registry candidate can be unknown to the local
                # job table. Never submit such a job without routing it.
                if spec.get("requires_gpu"):
                    capacity = select_available_lane()
                    if not capacity.get("success"):
                        return {
                            "success": False,
                            "dry_run": False,
                            "status": "resource_unavailable",
                            "resource_check": capacity,
                            "job_uid": job.uid,
                            "planned_action": planned_action,
                        }
                    queue["lane"] = capacity["lane"]
                    planned_action["queue"] = queue
                    planned_action["resource_check"] = capacity
                    job.queue(
                        lane=queue["lane"],
                        gpus=queue["gpus"],
                        cluster_vars=queue["cluster_vars"],
                    )
                else:
                    # build_queue_plan supplies a lane for all non-interactive
                    # local specs. Keep this fallback for unknown registry jobs.
                    fallback_lane = os.getenv("CRYOAGENT_GPU_LANE") or spec.get("default_lane")
                    if not fallback_lane:
                        return {
                            "success": False,
                            "dry_run": False,
                            "status": "queue_lane_required",
                            "job_uid": job.uid,
                            "planned_action": planned_action,
                        }
                    queue["lane"] = fallback_lane
                    planned_action["queue"] = queue
                    job.queue(lane=fallback_lane)
            queued = True

        return {
            "success": True,
            "dry_run": False,
            "status": "queued" if queued else job.status,
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": job.uid,
            "job_type": planned_action["job_type"],
            "queued": queued,
            "approval_required": planned_action["approval_required"],
            "approval_reasons": planned_action.get("approval_reasons", []),
            "approval_required_bypassed_for_creation": bool(
                planned_action["approval_required"] and allow_approval_required_create
            ),
            "human_action_required": bool(planned_action["approval_required"]),
            "planned_action": planned_action,
        }
    except Exception as exc:
        # Creation and input wiring can succeed before CryoSPARC rejects the
        # enqueue request. Do not leave an indistinguishable ``building`` job
        # behind: it cannot contain useful output and otherwise becomes a
        # duplicate child candidate on the next autonomous round.
        cleanup = {"attempted": False, "success": False}
        if job is not None:
            cleanup["attempted"] = True
            try:
                job.kill()
                cleanup["success"] = True
            except Exception as cleanup_error:
                cleanup["error"] = str(cleanup_error)
        return {
            "success": False,
            "dry_run": False,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "job_uid": getattr(job, "uid", None),
            "cleanup": cleanup,
            "planned_action": planned_action,
        }

"""Dynamic candidate discovery from the live CryoSPARC Job Registry."""
from typing import Any, Dict, List, Optional, Set
import os

from cryosparc_client import cryosparc_client
from job_specs import DEFAULT_GPU_LANE, get_job_spec, annotate_extraction_parameters


def obsolete_registry_job(spec: Any) -> bool:
    stability = getattr(spec, "stability", None)
    return str(getattr(stability, "value", stability)).lower() == "obsolete"


def registry_required_input_counts(spec: Any, connections: Dict[str, Any]) -> Dict[str, int]:
    inputs = getattr(getattr(spec, 'inputs', None), 'root', {})
    counts = {name: value.count_min for name, value in inputs.items() if value.count_min}
    if spec.type == 'orientation_diagnostics':
        # The native schema lists both volume modes as required, but this job
        # accepts exactly one of them (single-map or multi-class analysis).
        modes = [name for name in ('volume', 'volumes_all_classes') if connections.get(name)]
        if len(modes) > 1:
            raise ValueError('Orientation Diagnostics needs one volume input mode, not both')
        counts.pop('volume', None)
        counts.pop('volumes_all_classes', None)
        counts[modes[0] if modes else 'volume'] = 1
    return counts


def build_registry_next_actions(
    workflow_state: Dict[str, Any],
    current_node: Dict[str, Any],
    candidate_job_types: Optional[Set[str]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Discover safe next jobs from CryoSPARC's live Job Registry."""
    if current_node["status"] != "completed":
        return [], []
    try:
        cs = cryosparc_client()
        specs = list(getattr(cs.job_register, "specs", []))
    except Exception as exc:
        return [], [{
            "action_id": "blocked_registry_unavailable",
            "available": False,
            "blocked_by": [f"CryoSPARC Job Registry unavailable: {exc}"],
        }]

    sources = available_output_sources(
        workflow_state,
        max_job_uid=current_node.get("cryosparc_job_uid"),
    )
    actions: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for registry_spec in specs:
        job_type = getattr(registry_spec, "type", None)
        if not job_type or job_type == current_node["job_type"]:
            continue
        if getattr(registry_spec, "hidden", False):
            continue
        if obsolete_registry_job(registry_spec):
            blocked.append({
                "action_id": f"blocked_obsolete_{current_node['cryosparc_job_uid']}_{job_type}",
                "job_type": job_type, "available": False,
                "blocked_by": ["CryoSPARC marks this task obsolete; it cannot be created or parameterized."],
                "replacement_job_type": "orientation_diagnostics" if job_type == "fsc3D" else None,
            })
            continue
        required_inputs = resolve_registry_connections(registry_spec, sources)
        if required_inputs is None:
            if candidate_job_types and job_type in candidate_job_types:
                blocked.append({
                    "action_id": f"blocked_{current_node['cryosparc_job_uid']}_{job_type}",
                    "action_type": "forward",
                    "workflow_node_id": f"{current_node['workflow_node_id']}:{job_type}",
                    "reference_job_uid": None,
                    "reference_status": None,
                    "job_type": job_type,
                    "available": False,
                    "blocked_by": [
                        "CryoSPARC Job Registry required input slots are not "
                        "available from completed outputs."
                    ],
                })
            continue
        actions.append(build_registry_candidate(current_node, registry_spec, required_inputs))
    return actions, blocked


def available_output_sources(
    workflow_state: Dict[str, Any],
    max_job_uid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Flatten prior completed node outputs into connection candidates.

    The upper bound prevents a recovery run from accidentally connecting to
    jobs created after its requested current node.
    """
    max_job_number = _job_uid_number(max_job_uid) if max_job_uid else None
    sources = []
    for node in workflow_state["nodes"]:
        if node["status"] != "completed":
            continue
        if max_job_number is not None and _job_uid_number(node["cryosparc_job_uid"]) > max_job_number:
            continue
        for output_name, output in node["outputs"].items():
            if not output.get("available"):
                continue
            sources.append({
                "source_workflow_node_id": node["workflow_node_id"],
                "source_logical_node_id": node["logical_node_id"],
                "source_job_uid": node["cryosparc_job_uid"],
                "source_output": output_name,
                "source_type": output.get("type"),
                "num_items": output.get("num_items") or 0,
                "result_names": output.get("result_names") or [],
                "source_job_type": node["job_type"],
            })
    return sources


def resolve_registry_connections(
    registry_spec: Any,
    sources: List[Dict[str, Any]],
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Resolve required Registry input groups against completed outputs."""
    inputs = getattr(getattr(registry_spec, "inputs", None), "root", {})
    if not inputs:
        return None
    resolved: Dict[str, List[Dict[str, Any]]] = {}
    alternative_volumes = getattr(registry_spec, 'type', None) == 'orientation_diagnostics'
    for input_name, input_spec in inputs.items():
        count_min = getattr(input_spec, "count_min", 0) or 0
        if alternative_volumes and input_name in {'volume', 'volumes_all_classes'}:
            count_min = 0
        expected_type = getattr(input_spec, "type", None)
        required_slots: Set[str] = set()
        for slot in getattr(input_spec, "slots", []) or []:
            if isinstance(slot, str):
                if slot.startswith("?"):
                    continue
                required_slots.add(slot)
            elif getattr(slot, "required", False):
                required_slots.add(getattr(slot, "name", ""))
        matches = [
            source for source in sources
            if source_matches_input(source, expected_type, required_slots)
        ]
        if count_min and not matches:
            return None
        if matches:
            resolved[input_name] = [select_preferred_source(registry_spec, input_name, matches)]
    if alternative_volumes:
        if 'volume' in resolved:
            resolved.pop('volumes_all_classes', None)
        elif 'volumes_all_classes' not in resolved:
            return None
    return resolved or None


def select_preferred_source(
    registry_spec: Any,
    input_name: str,
    matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Prefer primary particle sets over explicit unused/rejected branches.

    CryoSPARC exposes these branches with compatible particle schemas, but
    ``particles_unused`` and ``particles_rejected`` are usually side outputs,
    not the population intended for the next reconstruction stage. Keep them
    as a fallback for jobs that have no primary particle source available.
    """
    primary = [source for source in matches if not is_excluded_data_output(source)]
    return max(
        primary or matches,
        key=lambda source: (
            _job_uid_number(source.get("source_job_uid")),
            int(source.get("num_items") or 0),
        ),
    )


def is_excluded_data_output(source: Dict[str, Any]) -> bool:
    """Exclude incomplete/rejected side outputs when a primary output exists."""
    output_name = str(source.get("source_output") or "").lower()
    return any(
        token in output_name
        for token in ("_incomplete", "_failed", "_rejected", "_unused")
    )


def is_excluded_particle_output(source: Dict[str, Any]) -> bool:
    """Return whether a particle output is an explicit side/rejection branch."""
    output_name = str(source.get("source_output") or "").lower()
    return output_name.endswith("_unused") or output_name.endswith("_rejected")


def _job_uid_number(job_uid: Optional[str]) -> int:
    """Return the numeric part of a CryoSPARC job UID for chronological scoping."""
    try:
        return int(str(job_uid or "J0").lstrip("J"))
    except ValueError:
        return 0


def source_matches_input(
    source: Dict[str, Any],
    expected_type: Optional[str],
    required_slots: Set[str],
) -> bool:
    """Check type and required result slots without guessing missing data."""
    source_type = infer_source_type(source)
    output_name = str(source.get("source_output") or "").lower()
    # ``volumes_all_classes`` is a CryoSPARC ``volume_multi`` output. It is
    # not connectable to a single-volume input such as Homogeneous Refinement.
    if expected_type == "volume" and (
        output_name.startswith("volumes_")
        or output_name.endswith("_all_classes")
    ):
        return False
    # CryoSPARC uses the broad ``exposure`` dtype for both movie and
    # micrograph inputs. Required result slots still distinguish them: a
    # movie can satisfy Motion Correction, but not CTF estimation.
    movie_exposure = (
        expected_type == "exposure"
        and source_type == "movie"
        and "movie_blob" in required_slots
    )
    if expected_type and source_type and expected_type != source_type and not movie_exposure:
        return False
    return required_slots.issubset(set(source.get("result_names") or []))


def infer_source_type(source: Dict[str, Any]) -> Optional[str]:
    """Infer the Registry data type from normalized output names/results."""
    # Native output types are authoritative; imported particles need not carry
    # location/pick_stats, and class averages also contain a blob slot.
    if source.get("source_type"):
        return source["source_type"]
    names = set(source.get("result_names") or [])
    output_name = source.get("source_output", "")
    if "blob" in names and ("template" in output_name or output_name == "class_averages"):
        return "template"
    if "blob" in names and (output_name.startswith("particles") or output_name == "imported_particles"):
        return "particle"
    if "micrograph_blob" in names or "micrograph_blob_non_dw" in names:
        return "exposure"
    if "movie_blob" in names:
        return "movie"
    if "location" in names or "pick_stats" in names:
        return "particle"
    if "blob" in names and "template" in output_name:
        return "template"
    if "volume" in output_name or "volume_blob" in names:
        return "volume"
    return None


def registry_parameter_template(registry_spec: Any) -> Dict[str, Dict[str, Any]]:
    """Expose Registry types, defaults, enums, and numeric constraints."""
    template = {}
    for name, param in getattr(registry_spec, "params", {}).items():
        if getattr(param, "hidden", False) and registry_spec.type != "inspect_picks_v2":
            continue
        value_type = registry_param_type(param)
        item = {"type": value_type}
        if getattr(param, "required_param", False):
            item["required"] = True
        if getattr(param, "default", None) is not None:
            item["default"] = param.default
        enum = getattr(param, "enum", None)
        if enum:
            item["enum"] = list(enum)
        lower = getattr(param, "ge", None)
        upper = getattr(param, "le", None)
        if lower is not None:
            item["minimum"] = lower
        if upper is not None:
            item["maximum"] = upper
        template[name] = item
    if registry_spec.type == 'inspect_picks_v2':
        from inspect_score_contract import raw_parameter_template
        template = raw_parameter_template(template)
    return annotate_extraction_parameters(registry_spec.type, template)


def registry_param_type(param: Any) -> str:
    """Resolve optional Registry unions such as number|null to JSON types."""
    direct = getattr(param, "type", None)
    if direct:
        return direct
    for option in getattr(param, "anyOf", []) or []:
        option_type = getattr(option, "type", None)
        if option_type and option_type != "null":
            return option_type
    return "string"


def build_registry_candidate(
    current_node: Dict[str, Any],
    registry_spec: Any,
    required_inputs: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Convert a live Registry spec into the MCP candidate contract."""
    job_type = registry_spec.type
    local_spec = get_job_spec(job_type)
    tags = set(getattr(registry_spec, "tags", []) or [])
    requires_gpu = "gpuEnabled" in tags
    interactive = bool(getattr(registry_spec, "interactive", False))
    requires_approval = interactive
    # CryoSPARC labels these jobs interactive for UI use, but the agent has
    # explicit non-interactive modes: score thresholds for Inspect Picks and
    # a model-selected class list for Select 2D. Preserve the local queue
    # policy instead of overwriting it with registry metadata.
    if job_type in {"inspect_picks_v2", "select_2D"}:
        interactive = local_spec["interactive"]
        requires_approval = local_spec["requires_approval"]
    default_lane = os.getenv("CRYOAGENT_GPU_LANE", DEFAULT_GPU_LANE) if requires_gpu else None
    defaults = {name: value.default
                for name, value in getattr(registry_spec, "params", {}).items()
                if getattr(value, "default", None) is not None}
    if job_type == 'inspect_picks_v2':
        defaults.update(calibrate_ncc=False, calibrate_pow=False)
        defaults.pop('lpower_thresh_max',None)
    from job_input_contracts import normalize_input_parameters
    defaults = normalize_input_parameters(job_type, defaults, required_inputs)
    template = registry_parameter_template(registry_spec)
    if job_type == 'class_3D' and required_inputs.get('volume'):
        template['class3D_init_mode'] = {**template.get('class3D_init_mode', {}),
            'type': 'string', 'default': 'input', 'const': 'input'}
    return {
        "action_id": f"registry_{current_node['cryosparc_job_uid']}_{job_type}",
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:{job_type}",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": getattr(registry_spec, "title", None)
        or f"Create {job_type} using compatible completed outputs.",
        "execution_mode": "create_job",
        "available": True,
        "blocked_by": [],
        "required_inputs": required_inputs,
        "missing_required_inputs": [],
        "parameter_template": template,
        "default_parameters": defaults,
        "registry_source": "cryosparc_job_register",
        "job_spec_metadata": {
            "repeat_allowed_inputs": [name for name, value in
                getattr(getattr(registry_spec, 'inputs', None), 'root', {}).items()
                if getattr(value, 'repeat_allowed', False)],
            "required_input_counts": registry_required_input_counts(registry_spec, required_inputs),
            "category": getattr(registry_spec, "category", None) or local_spec["category"],
            "requires_gpu": requires_gpu,
            "multi_gpu": "multiGpu" in tags,
            "requires_approval": requires_approval,
            "interactive": interactive,
            "default_lane": default_lane,
            "max_auto_gpus": local_spec["max_auto_gpus"],
            "queue_without_lane": local_spec["queue_without_lane"],
        },
    }

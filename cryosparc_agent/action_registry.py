# Builds candidate CryoSPARC actions, validates model decisions, and plans safe execution.
from typing import Any
import os
import re

from schemas import (
    Action,
    ModelDecision,
    ResolvedAction,
    ValidationIssue,
    ValidationResult,
    parse_model_decision,
)
from cryosparc_client import cryosparc_client
from job_executor import execute_job_action, plan_job_action
from job_specs import get_parameter_template
from dynamic_candidates import build_registry_next_actions
from workflow_policy import (
    annotate_candidates,
    build_decision_guidance,
    infer_current_stage,
    POLICY_VERSION,
    quality_rollback_enabled,
    box_size_sweep_enabled,
    fixed_initial_funnel_enabled,
    initial_scale_calibration_enabled,
    initial_scale_calibration_micrographs,
)
from quality_policy import assess_node, build_retry_candidate
from workflow_state import content_hash, extract_workflow_state, find_node
from dataset_info import normalize_dataset_info
from abinitio_class2d_consistency import build_template_matching_consistency_context
from class2d_scale import build_class2d_scale_context


REGISTRY_VERSION = "workflow_v1"
TEMPLATE_REPICK_MAX_TARGET_GAP_A = 0.5
TEMPLATE_REPICK_MIN_ABINITIO_CLASS2D_NCC = 0.75
ABINITIO_CLASS2D_MISMATCH_NCC = 0.75
REPEATED_ABINITIO_CLASS2D_MISMATCHES = 2
MAX_EXTRACT_BOX_SIZE_PIX = 720
INITIAL_SCALE_PILOT_BOX_SIZE_PIX = 384

def get_candidate_actions(
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the current workflow and return selectable next actions."""
    dataset_info = normalize_dataset_info(dataset_info or {})
    workflow_state = extract_workflow_state(project_uid, workspace_uid)
    from extract_review import without_review_pilots
    workflow_state = without_review_pilots(workflow_state)
    current_node = (
        find_node(workflow_state, current_node_id)
        if current_node_id
        else None
    )
    canonical_current_node_id = (
        current_node["workflow_node_id"]
        if current_node
        else current_node_id
    )
    candidate_actions, blocked_actions = generate_candidate_actions(
        workflow_state,
        canonical_current_node_id,
        project_uid=project_uid,
        include_registry=True,
        dataset_info=dataset_info,
    )
    if canonical_current_node_id is None:
        candidate_actions.extend(build_initial_import_candidates(dataset_info or {}))
    if current_node and current_node["status"] in {"failed", "killed"}:
        retry = build_retry_candidate(current_node, workflow_state)
        client = cryosparc_client()
        specs = getattr(getattr(client, 'job_register', None), 'specs', None)
        if isinstance(specs, (list, tuple)):
            native = next((spec for spec in specs if spec.type == current_node['job_type']), None)
            if native is not None:
                params = client.find_project(project_uid).find_job(current_node['cryosparc_job_uid']).params.model_dump()
                retry = build_retry_candidate(current_node, workflow_state, native, params)
        (candidate_actions if retry.get('available') else blocked_actions).append(retry)
    quality_assessment = (
        assess_node(current_node)
        if current_node
        else {"policy_version": "cryoem_quality_v1", "quality_status": "not_started"}
    )
    if quality_rollback_enabled() and current_node:
        rollback_actions = build_refinement_rollback_candidates(
            workflow_state,
            current_node,
            quality_assessment,
            dataset_info,
            project_uid=project_uid,
        )
        if rollback_actions:
            # A refinement checkpoint is the only point where the planner chooses
            # among rollback experiments. Do not dilute this menu with arbitrary
            # registry actions from a downstream refinement node.
            candidate_actions = [action for action in rollback_actions if action.get("available")]
            blocked_actions.extend(
                action for action in rollback_actions if not action.get("available")
            )
    if current_node and not box_size_sweep_enabled():
        candidate_actions = [
            apply_default_extract_box_size(action)
            for action in candidate_actions
        ]
    if unknown_dataset_symmetry(dataset_info):
        candidate_actions = [allow_unknown_symmetry(action) for action in candidate_actions]
    from inspect_review import review_candidate
    inspection_revision = review_candidate(workflow_state, current_node, candidate_actions)
    if inspection_revision:
        candidate_actions.append(inspection_revision)
    workflow_policy = infer_current_stage(
        workflow_state["nodes"],
        current_node_id=canonical_current_node_id,
    )
    workflow_guidance = build_decision_guidance(
        dataset_info=dataset_info,
        quality_assessment=quality_assessment,
        current_node=current_node,
        candidates=candidate_actions,
    )
    return {
        "schema_version": "1.0",
        "registry_version": REGISTRY_VERSION,
        "generated_at": workflow_state["generated_at"],
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "current_node_id": canonical_current_node_id,
        "requested_current_node_id": current_node_id,
        "workflow_status": workflow_state["workflow_status"],
        "candidate_actions": candidate_actions,
        "inspect_review_required": bool(current_node and current_node.get('job_type') == 'inspect_picks_v2'
            and any(c.get('job_type') == 'extract_micrographs_multi' for c in candidate_actions)),
        "blocked_actions": blocked_actions,
        "decision_hint": "stop" if not candidate_actions else None,
        "workflow_policy": workflow_policy,
        "workflow_policy_version": POLICY_VERSION,
        "quality_assessment": quality_assessment,
        "workflow_guidance": workflow_guidance,
    }


def build_quality_rollback_candidates(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    quality_assessment: dict[str, Any],
    dataset_info: dict[str, Any],
    project_uid: str | None = None,
) -> list[dict[str, Any]]:
    """Offer an executable Select 2D recovery when FSC is below target."""
    if current_node.get("job_type") not in {
        "homo_refine_new", "nonuniform_refine_new", "refine_3D_new", "validation"
    }:
        return []
    target = dataset_info.get("target_resolution_A")
    if target is None:
        target = dataset_info.get("resolution")
    if isinstance(target, (list, tuple)):
        target = target[0] if len(target) == 1 else None
    try:
        target = float(target) if target is not None else None
    except (TypeError, ValueError):
        target = None
    observed = quality_assessment.get("resolution_evidence", {}).get("values", {})
    values = []
    for value in observed.values():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if target is None or not values or min(values) <= target:
        return []

    select_nodes = [
        node for node in workflow_state.get("nodes", [])
        if node.get("job_type") == "select_2D" and node.get("status") == "completed"
    ]
    if not select_nodes:
        return []
    select_node = sorted(
        select_nodes,
        key=lambda item: int(str(item.get("cryosparc_job_uid", "J0"))[1:])
        if str(item.get("cryosparc_job_uid", "J0"))[1:].isdigit()
        else -1,
    )[-1]
    # Recover the Class 2D branch that fed the historical Select 2D, then expose
    # every completed sibling branch sharing the same micrograph lineage.
    select_particles = (select_node.get("inputs") or {}).get("particles") or []
    source_class_uid = (
        select_particles[0].get("source_job_uid") if select_particles else None
    )
    source_class = next(
        (
            node for node in workflow_state.get("nodes", [])
            if node.get("cryosparc_job_uid") == source_class_uid
            and node.get("job_type") == "class_2D_new"
            and node.get("status") == "completed"
        ),
        None,
    )
    if source_class is None:
        return []
    comparison_actions = build_class2d_comparison_candidates(
        workflow_state,
        source_class,
        project_uid=project_uid,
    )
    if not comparison_actions:
        return []

    actions = []
    def action_source_class_uid(action: dict[str, Any]) -> str | None:
        particles = (action.get("required_inputs") or {}).get("particles") or []
        particle_source = particles[0].get("source_job_uid") if particles else None
        return action.get("reference_job_uid") or particle_source

    comparison_uids = [
        action_source_class_uid(item) or item.get("action_id")
        for item in comparison_actions
    ]
    for action in comparison_actions:
        action = dict(action)
        branch_uid = action_source_class_uid(action) or source_class_uid
        branch_node = next(
            (
                node for node in workflow_state.get("nodes", [])
                if node.get("cryosparc_job_uid") == branch_uid
                and node.get("job_type") == "class_2D_new"
            ),
            None,
        )
        branch_box_size = None
        branch_extract_uid = None
        if branch_node:
            branch_particles = (branch_node.get("inputs") or {}).get("particles") or []
            branch_extract_uid = (
                branch_particles[0].get("source_job_uid")
                if branch_particles else None
            )
            branch_extract = next(
                (
                    node for node in workflow_state.get("nodes", [])
                    if node.get("cryosparc_job_uid") == branch_extract_uid
                ),
                None,
            )
            branch_box_size = (branch_extract or {}).get("key_parameters", {}).get(
                "box_size_pix"
            )
        action["action_id"] = (
            f"quality_rollback_{current_node['cryosparc_job_uid']}_select_2D_{branch_uid}"
        )
        action["workflow_node_id"] = (
            f"{current_node['workflow_node_id']}:quality_rollback_select_2D:{branch_uid}"
        )
        action["quality_rollback"] = True
        action["workflow_policy_recommendation"] = "quality_rollback"
        action["description"] = (
            "Revisit Select 2D from this completed Class 2D box-size branch after "
            "below-target refinement quality; use visual evidence to revise particle selection."
        )
        action["rollback_evidence"] = {
            "target_resolution_A": target,
            "observed_best_resolution_A": min(values),
            "source_select_2D_job_uid": select_node["cryosparc_job_uid"],
            "source_class2d_job_uid": branch_uid,
            "source_extract_job_uid": branch_extract_uid,
            "box_size_pix": branch_box_size,
            "available_class2d_branches": comparison_uids,
        }
        action["quality_rollback_branch"] = {
            "class2d_job_uid": branch_uid,
            "extract_job_uid": branch_extract_uid,
            "box_size_pix": branch_box_size,
            "visual_evidence_preferred": True,
            "structured_evidence_acceptable": True,
        }
        action["comparison_instruction"] = (
            "Compare this branch with the other available Class 2D box-size branches "
            "using visual evidence before selecting it for reprocessing."
        )
        actions.append(action)
    return actions


def build_refinement_rollback_candidates(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    quality_assessment: dict[str, Any],
    dataset_info: dict[str, Any],
    project_uid: str | None = None,
) -> list[dict[str, Any]]:
    """Build the bounded rollback menu for one below-target refinement branch."""
    if current_node.get("job_type") not in {
        "homo_refine_new", "nonuniform_refine_new", "refine_3D_new"
    } or not refinement_target_unmet(quality_assessment, dataset_info):
        return []

    select_node = find_ancestor_node(workflow_state, current_node, "select_2D")
    class_node = find_ancestor_node(workflow_state, current_node, "class_2D_new")
    extract_node = find_ancestor_node(workflow_state, current_node, "extract_micrographs_multi")
    inspect_node = find_ancestor_node(workflow_state, current_node, "inspect_picks_v2")
    blob_node = find_ancestor_node(workflow_state, current_node, "blob_picker_gpu")
    selection_progress = reselection_progress(workflow_state, class_node)
    actions: list[dict[str, Any]] = []
    prefix = f"refinement_rollback_{current_node['cryosparc_job_uid']}"

    consistency_history = build_abinitio_consistency_history(
        workflow_state, extract_node, project_uid,
    )
    from symmetry_policy import build_symmetry_comparison
    comparison = build_symmetry_comparison(workflow_state, current_node, dataset_info)
    if comparison:
        actions.append(comparison)

    from refinement_optimization import build_candidates as build_optimization_candidates
    actions.extend(build_optimization_candidates(workflow_state, current_node, project_uid))

    alternate_type = {
        "homo_refine_new": "nonuniform_refine_new",
        "nonuniform_refine_new": "homo_refine_new",
    }.get(current_node["job_type"])
    if alternate_type:
        current_uid = current_node["cryosparc_job_uid"]
        prior_alternate = find_ancestor_node(workflow_state, current_node, alternate_type)
        existing_alternate = any(
            node.get("job_type") == alternate_type
            and (
                current_uid in (node.get("parent_job_uids") or [])
                or (
                    select_node is not None
                    and (branch_select := find_ancestor_node(workflow_state, node, "select_2D")) is not None
                    and branch_select["cryosparc_job_uid"] == select_node["cryosparc_job_uid"]
                )
            )
            for node in workflow_state.get("nodes", [])
        )
        particles = output_source(current_node, "particles")
        volume = output_source(current_node, "volume")
        blocked_by = []
        if prior_alternate or existing_alternate:
            blocked_by.append("This refinement type was already tried on this branch.")
        if particles is None or volume is None:
            blocked_by.append("Current refinement must expose both particles and volume outputs.")
        source_parameters = current_node.get("key_parameters") or {}
        parameter_template = get_parameter_template(alternate_type)
        defaults = {
            name: source_parameters[name]
            for name in ("refine_symmetry", "refine_defocus_refine", "refine_ctf_global_refine")
            if name in source_parameters and name in parameter_template
        }
        actions.append({
            "action_id": f"{prefix}_alternate_refinement",
            "action_type": "forward",
            "workflow_node_id": f"{current_node['workflow_node_id']}:alternate_refinement",
            "reference_job_uid": current_uid,
            "reference_status": current_node["status"],
            "job_type": alternate_type,
            "description": (
                f"Compare {alternate_type} once on the current refined particles and volume. "
                "Keep symmetry and CTF settings comparable; do not repeat either refinement "
                "type on this branch or use this to avoid an upstream particle-quality fix."
            ),
            "execution_mode": "create_job",
            "available": not blocked_by,
            "blocked_by": blocked_by,
            "required_inputs": {
                "particles": [particles] if particles else [],
                "volume": [volume] if volume else [],
            },
            "missing_required_inputs": [name for name, source in (("particles", particles), ("volume", volume)) if source is None],
            "parameter_template": parameter_template,
            "default_parameters": defaults,
            "refinement_rollback": True,
            "refinement_strategy_switch": True,
            "rollback_kind": "alternate_refinement",
            "rollback_source": {"refine_job_uid": current_uid},
        })

    if class_node is not None:
        particles = output_source(class_node, "particles")
        templates = output_source(class_node, "class_averages")
        if particles and templates:
            previous_selection = str(
                (select_node or {}).get("key_parameters", {}).get("selected_templates") or ""
            )
            previous_class_ids = [
                int(item) for item in previous_selection.split(",")
                if item.strip().isdigit()
            ]
            action = apply_auto_select_2d_policy({
                "action_id": f"{prefix}_reselect_2d",
                "action_type": "forward",
                "workflow_node_id": f"{current_node['workflow_node_id']}:reselect_2d",
                "reference_job_uid": class_node["cryosparc_job_uid"],
                "reference_status": class_node["status"],
                "job_type": "select_2D",
                "execution_mode": "create_job",
                "available": True,
                "blocked_by": [],
                "required_inputs": {"particles": [particles], "templates": [templates]},
                "missing_required_inputs": [],
                "parameter_template": get_parameter_template("select_2D"),
                "default_parameters": {},
            }, workflow_state)
            action.update({
                "quality_rollback": True,
                "refinement_rollback": True,
                "rollback_kind": "reselect_2d",
                "rollback_source": {"class2d_job_uid": class_node["cryosparc_job_uid"]},
                "reselection_constraint": {
                    "mode": "strict_subset_of_previous_selection",
                    "previous_select2d_job_uid": (select_node or {}).get("cryosparc_job_uid"),
                    "previous_selected_templates": previous_selection,
                    "previous_selected_count": len(previous_class_ids),
                    "instruction": (
                        "Retain only a strict subset of the previously selected Class 2D IDs. "
                        "Exclude lower-quality, incomplete, off-center, or weak-signal classes; "
                        "do not add previously excluded classes."
                    ),
                },
                "description": (
                    "Re-select classes from the Class 2D run that produced this refinement "
                    "branch, using its contact sheet and CryoSift scores. Exclude off-center "
                    "or density-shifted classes when they are inconsistent with the Ab-initio views."
                ),
            })
            action["selection_iteration_guard"] = selection_progress
            if selection_progress["stagnant"] or len(previous_class_ids) <= 1:
                action["available"] = False
                action["blocked_by"] = [
                    "Reselection did not improve corrected final FSC; change upstream route."
                    if selection_progress["stagnant"] else
                    "A nonempty strict subset of a singleton/empty selection is impossible."]
            actions.append(action)

    if select_node is not None and output_source(select_node, "particles_selected"):
        source = output_source(select_node, "particles_selected")
        actions.append({
            "action_id": f"{prefix}_reclassify_selected_particles",
            "action_type": "forward",
            "workflow_node_id": f"{current_node['workflow_node_id']}:reclassify_selected_particles",
            "reference_job_uid": select_node["cryosparc_job_uid"],
            "reference_status": select_node["status"],
            "job_type": "class_2D_new",
            "description": (
                "Run another Class 2D pass on the particles already selected for this branch, "
                "then re-select classes to improve particle purity before reconstruction."
            ),
            "execution_mode": "create_job",
            "available": not selection_progress["stagnant"],
            "blocked_by": (["Reselection stagnated; do not recycle the same particles through Class 2D."]
                           if selection_progress["stagnant"] else []),
            "required_inputs": {"particles": [source]},
            "missing_required_inputs": [],
            "parameter_template": get_parameter_template("class_2D_new"),
            "default_parameters": {"class2D_K": 50},
            "refinement_rollback": True,
            "rollback_kind": "reclassify_selected_particles",
            "selection_iteration_guard": selection_progress,
            "rollback_source": {"select2d_job_uid": select_node["cryosparc_job_uid"]},
        })

    if extract_node is not None:
        inputs = relevant_inputs(extract_node, {"micrographs", "particles"})
        if inputs:
            template = build_parameter_template(extract_node)
            box_spec = dict(template.get("box_size_pix") or {"type": "integer", "minimum": 32})
            box_spec["required"] = True
            template["box_size_pix"] = box_spec
            current_box_size = (extract_node.get("key_parameters") or {}).get("box_size_pix")
            next_box_size = next_larger_box_size(current_box_size)
            if next_box_size is not None:
                actions.append({
                "action_id": f"{prefix}_new_box_size",
                "action_type": "forward",
                "workflow_node_id": f"{current_node['workflow_node_id']}:new_box_size",
                "reference_job_uid": extract_node["cryosparc_job_uid"],
                "reference_status": extract_node["status"],
                "job_type": "extract_micrographs_multi",
                "description": (
                    "Create one alternative extraction box-size branch from the same inspected "
                    "picks. Choose box_size_pix explicitly; downstream Class 2D, Select 2D, "
                    "Ab-initio, and Homo-refine will then run through the fixed funnel. Prefer "
                    "a larger box when an Ab-initio view lacks a corresponding selected Class 2D "
                    "orientation, which can indicate particle clipping."
                ),
                "execution_mode": "create_job",
                "available": True,
                "blocked_by": [],
                "required_inputs": inputs,
                "missing_required_inputs": [],
                "parameter_template": template,
                # A scale rollback must be a real expansion, not another 384-pixel run.
                "default_parameters": {"box_size_pix": next_box_size},
                "refinement_rollback": True,
                "rollback_kind": "new_box_size",
                "rollback_source": {"extract_job_uid": extract_node["cryosparc_job_uid"]},
                })

    if inspect_node is not None:
        inputs = relevant_inputs(inspect_node, {"micrographs", "particles"})
        if inputs:
            actions.append({
                "action_id": f"{prefix}_reinspect_picks",
                "action_type": "forward",
                "workflow_node_id": f"{current_node['workflow_node_id']}:reinspect_picks",
                "reference_job_uid": inspect_node["cryosparc_job_uid"],
                "reference_status": inspect_node["status"],
                "job_type": "inspect_picks_v2",
                "description": (
                    "Re-run Inspect Picks on this branch's pre-extraction picks. Choose explicit "
                    "NCC/Power thresholds from the overlay and score scatter before the fixed funnel."
                ),
                "execution_mode": "create_job",
                "available": True,
                "blocked_by": [],
                "required_inputs": inputs,
                "missing_required_inputs": [],
                "parameter_template": get_parameter_template("inspect_picks_v2"),
                "default_parameters": {},
                "refinement_rollback": True,
                "rollback_kind": "reinspect_picks",
                "rollback_source": {"inspect_job_uid": inspect_node["cryosparc_job_uid"]},
            })

    if blob_node is not None:
        inputs = relevant_inputs(blob_node, {"micrographs"})
        if inputs:
            template = get_parameter_template("blob_picker_gpu")
            for name in ("diameter", "diameter_max"):
                template[name] = {**template[name], "required": True}
            actions.append({
                "action_id": f"{prefix}_repick_diameter",
                "action_type": "forward",
                "workflow_node_id": f"{current_node['workflow_node_id']}:repick_diameter",
                "reference_job_uid": blob_node["cryosparc_job_uid"],
                "reference_status": blob_node["status"],
                "job_type": "blob_picker_gpu",
                "description": (
                    "Re-run Blob Picker on the original CTF-estimated micrographs. Choose an explicit "
                    "particle diameter range; Junk Detector, Inspect Picks, and the fixed funnel follow."
                ),
                "execution_mode": "create_job",
                "available": True,
                "blocked_by": [],
                "required_inputs": inputs,
                "missing_required_inputs": [],
                "parameter_template": template,
                "default_parameters": {},
                "refinement_rollback": True,
                "rollback_kind": "repick_diameter",
                "rollback_source": {
                    "blob_picker_job_uid": blob_node["cryosparc_job_uid"],
                    "class2d_job_uid": class_node["cryosparc_job_uid"] if class_node else None,
                    "extract_job_uid": extract_node["cryosparc_job_uid"] if extract_node else None,
                },
            })
    volume = output_source(current_node, "volume")
    if volume is not None:
        template_gate = build_template_repick_gate(
            workflow_state, current_node, quality_assessment, dataset_info, project_uid,
        )
        actions.append({
            "action_id": f"{prefix}_template_repick",
            "action_type": "forward",
            "workflow_node_id": f"{current_node['workflow_node_id']}:template_repick",
            "reference_job_uid": current_node["cryosparc_job_uid"],
            "reference_status": current_node["status"],
            "job_type": "create_templates",
            "description": (
                "Create 2D templates from this refinement volume, then automatically run Template "
                "Picker on the branch's CTF-estimated micrographs before particle curation and the "
                "fixed reconstruction funnel."
            ),
            "execution_mode": "create_job",
            "available": template_gate["allowed"],
            "blocked_by": template_gate["blocked_by"],
            "required_inputs": {"volume": [volume]},
            "missing_required_inputs": [],
            "parameter_template": get_parameter_template("create_templates"),
            "default_parameters": {"n_templates": 50},
            "refinement_rollback": True,
            "rollback_kind": "template_repick",
            "rollback_source": {"refine_job_uid": current_node["cryosparc_job_uid"]},
            "template_repick_gate": template_gate,
        })
    if class_node is not None and project_uid:
        scale_context = _build_scale_rollback_evidence(
            project_uid,
            class_node["cryosparc_job_uid"],
            dataset_info.get("target_resolution_A") or dataset_info.get("resolution"),
        )
        if scale_context:
            _apply_scale_rollback_defaults(actions, scale_context, extract_node)

    if consistency_history["assessed_abinitio_count"]:
        for action in actions:
            action["abinitio_consistency_history"] = consistency_history
    if consistency_history["repeated_mismatch"]:
        # Keep every rollback available. This is a planner-facing evidence signal,
        # not a hard gate: repeated mismatch can also originate in Select 2D.
        for action in actions:
            if action.get("rollback_kind") == "new_box_size":
                action["rollback_priority"] = "recommended_after_repeated_abinitio_mismatch"
                action["description"] = action.get("description", "") + (
                    " This is recommended because multiple reconstructions from the same extraction "
                    "scale have Ab-initio/Class 2D mismatches."
                )
    return actions


def _build_scale_rollback_evidence(
    project_uid: str,
    class2d_job_uid: str,
    target_resolution_A: Any,
) -> dict[str, Any] | None:
    """Build compact, measured scale evidence for the refinement rollback menu."""
    try:
        context = build_class2d_scale_context(
            project_uid,
            class2d_job_uid,
            target_resolution_A=target_resolution_A,
            max_box_size_pix=MAX_EXTRACT_BOX_SIZE_PIX,
        )
    except Exception:
        return None
    return {
        "source_class2d_job_uid": class2d_job_uid,
        "measurement_method": context.get("measurement_method"),
        "aggregate": context.get("aggregate") or {},
        "ctf_evidence": context.get("ctf_evidence") or {},
        "box_size_recommendation": context.get("box_size_recommendation") or {},
    }


def _apply_scale_rollback_defaults(
    actions: list[dict[str, Any]],
    evidence: dict[str, Any],
    extract_node: dict[str, Any] | None,
) -> None:
    """Attach measured particle-diameter and box-size defaults to rollback actions."""
    aggregate = evidence.get("aggregate") or {}
    box = evidence.get("box_size_recommendation") or {}
    diameter_range = aggregate.get("particle_diameter_range_A")
    if not aggregate.get("usable_for_picker") or not isinstance(diameter_range, list) or len(diameter_range) != 2:
        return
    try:
        diameter_min, diameter_max = sorted(float(value) for value in diameter_range)
    except (TypeError, ValueError):
        return
    if diameter_min <= 0 or diameter_max <= 0:
        return

    for action in actions:
        kind = action.get("rollback_kind")
        if kind == "repick_diameter":
            defaults = dict(action.get("default_parameters") or {})
            defaults.update({
                "diameter": round(diameter_min, 1),
                "diameter_max": round(diameter_max, 1),
            })
            action["default_parameters"] = defaults
            action["scale_observer_evidence"] = evidence
            action["description"] = (
                f"Re-run Blob Picker using the measured Class 2D diameter range "
                f"{diameter_min:.1f}-{diameter_max:.1f} A, then inspect the picks and continue the fixed funnel."
            )
        elif kind == "new_box_size" and box.get("usable_for_extraction"):
            try:
                proposed = int(box["recommended_box_size_pix"])
                current = int((extract_node or {}).get("key_parameters", {}).get("box_size_pix"))
            except (KeyError, TypeError, ValueError):
                continue
            if current < proposed <= MAX_EXTRACT_BOX_SIZE_PIX:
                defaults = dict(action.get("default_parameters") or {})
                defaults["box_size_pix"] = proposed
                action["default_parameters"] = defaults
                template = dict(action.get("parameter_template") or {})
                box_spec = dict(template.get("box_size_pix") or {"type": "integer", "minimum": 32})
                box_spec["maximum"] = MAX_EXTRACT_BOX_SIZE_PIX
                template["box_size_pix"] = box_spec
                action["parameter_template"] = template
                action["scale_observer_evidence"] = evidence
                action["description"] = action.get("description", "") + (
                    f" Class 2D radial-profile and quality-filtered CTF evidence recommend "
                    f"box_size_pix={proposed} (median + 1 sigma defocus)."
                )


def next_larger_box_size(current_box_size: Any) -> int | None:
    """Choose the next configured scale without exceeding the safe box-size cap."""
    try:
        current = int(current_box_size)
    except (TypeError, ValueError):
        current = default_extract_box_size()
    for size in configured_box_size_trials():
        if size > current:
            return size
    return None


def build_abinitio_consistency_history(
    workflow_state: dict[str, Any],
    extract_node: dict[str, Any] | None,
    project_uid: str | None,
    template_matcher: Any = build_template_matching_consistency_context,
) -> dict[str, Any]:
    """Summarize calibrated mismatch evidence for one extraction scale."""
    history = {
        "threshold_min_best_ncc": ABINITIO_CLASS2D_MISMATCH_NCC,
        "same_extract_job_uid": extract_node.get("cryosparc_job_uid") if extract_node else None,
        "assessed_abinitio_count": 0,
        "mismatch_count": 0,
        "repeated_mismatch": False,
        "attempts": [],
    }
    if extract_node is None or not project_uid:
        return history
    extract_uid = extract_node.get("cryosparc_job_uid")
    for node in workflow_state.get("nodes", []):
        if node.get("job_type") != "homo_abinit" or node.get("status") != "completed":
            continue
        source_extract = find_ancestor_node(workflow_state, node, "extract_micrographs_multi")
        if not source_extract or source_extract.get("cryosparc_job_uid") != extract_uid:
            continue
        try:
            matching = template_matcher(project_uid, node["cryosparc_job_uid"])
            min_ncc = (matching.get("summary") or {}).get("min_best_ncc")
            mismatch = isinstance(min_ncc, (int, float)) and min_ncc < ABINITIO_CLASS2D_MISMATCH_NCC
            history["assessed_abinitio_count"] += 1
            history["mismatch_count"] += int(mismatch)
            history["attempts"].append({
                "abinitio_job_uid": node["cryosparc_job_uid"],
                "min_best_ncc": min_ncc,
                "mismatch": mismatch,
            })
        except Exception as exc:
            history["attempts"].append({
                "abinitio_job_uid": node["cryosparc_job_uid"],
                "error": str(exc),
            })
    history["repeated_mismatch"] = (
        history["mismatch_count"] >= REPEATED_ABINITIO_CLASS2D_MISMATCHES
    )
    return history


def build_template_repick_gate(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    quality_assessment: dict[str, Any],
    dataset_info: dict[str, Any],
    project_uid: str | None,
    template_matcher: Any = build_template_matching_consistency_context,
) -> dict[str, Any]:
    """Require near-target FSC and supported Ab-initio views before templating.

    Template picking can amplify an incorrect reconstruction, so absence of live
    matching evidence is deliberately fail-closed rather than a soft prompt cue.
    """
    target = _target_resolution_A(dataset_info)
    best_resolution = _best_resolution_A(quality_assessment)
    evidence: dict[str, Any] = {
        "max_target_gap_A": TEMPLATE_REPICK_MAX_TARGET_GAP_A,
        "min_abinitio_class2d_ncc": TEMPLATE_REPICK_MIN_ABINITIO_CLASS2D_NCC,
        "target_resolution_A": target,
        "best_resolution_A": best_resolution,
        "abinitio_job_uid": None,
        "template_matching": None,
        "allowed": False,
        "blocked_by": [],
    }
    if target is None or best_resolution is None:
        evidence["blocked_by"].append(
            "Create Template gate: target and observed FSC resolution are both required."
        )
        return evidence
    target_gap = best_resolution - target
    evidence["target_gap_A"] = target_gap
    if target_gap > TEMPLATE_REPICK_MAX_TARGET_GAP_A:
        evidence["blocked_by"].append(
            "Create Template gate: best resolution must be within "
            f"{TEMPLATE_REPICK_MAX_TARGET_GAP_A:.1f} A of target "
            f"(observed gap {target_gap:.3f} A)."
        )
        return evidence
    abinitio_node = find_ancestor_node(workflow_state, current_node, "homo_abinit")
    if abinitio_node is None or not project_uid:
        evidence["blocked_by"].append(
            "Create Template gate: a traceable completed Ab-initio source is required for view-consistency validation."
        )
        return evidence
    abinitio_job_uid = abinitio_node["cryosparc_job_uid"]
    evidence["abinitio_job_uid"] = abinitio_job_uid
    try:
        matching = template_matcher(project_uid, abinitio_job_uid)
    except Exception as exc:
        evidence["blocked_by"].append(
            "Create Template gate: Ab-initio/Class 2D template matching could not be obtained "
            f"({type(exc).__name__}: {exc})."
        )
        return evidence
    min_ncc = ((matching.get("summary") or {}).get("min_best_ncc"))
    evidence["template_matching"] = {
        "source": matching.get("source"),
        "min_best_ncc": min_ncc,
        "best_ncc_by_projection": (matching.get("summary") or {}).get("best_ncc_by_projection"),
        "selected_minus_excluded_best_ncc": (
            (matching.get("summary") or {}).get("selected_minus_excluded_best_ncc")
        ),
    }
    try:
        min_ncc = float(min_ncc)
    except (TypeError, ValueError):
        evidence["blocked_by"].append(
            "Create Template gate: template matching did not expose a valid min_best_ncc."
        )
        return evidence
    if min_ncc < TEMPLATE_REPICK_MIN_ABINITIO_CLASS2D_NCC:
        evidence["blocked_by"].append(
            "Create Template gate: all Ab-initio canonical views require selected Class 2D support "
            f"with min_best_ncc >= {TEMPLATE_REPICK_MIN_ABINITIO_CLASS2D_NCC:.2f} "
            f"(observed {min_ncc:.3f})."
        )
        return evidence
    evidence["allowed"] = True
    return evidence


def reselection_progress(workflow_state, class_node):
    """Compare distinct selections using corrected final FSC, not loose-mask minima."""
    grouped = {}
    source_uid = (class_node or {}).get("cryosparc_job_uid")
    extraction = find_ancestor_node(workflow_state, class_node, "extract_micrographs_multi") if class_node else None
    extraction_uid = (extraction or {}).get("cryosparc_job_uid")
    for node in workflow_state.get("nodes", []):
        if not source_uid or node.get("status") != "completed" or node.get("job_type") not in {
                "homo_refine_new", "nonuniform_refine_new", "refine_3D_new"}:
            continue
        source = find_ancestor_node(workflow_state, node, "class_2D_new")
        selection = find_ancestor_node(workflow_state, node, "select_2D")
        source_extraction = find_ancestor_node(workflow_state, source, "extract_micrographs_multi") if source else None
        same_population = ((source_extraction or {}).get("cryosparc_job_uid") == extraction_uid
                           if extraction_uid else (source or {}).get("cryosparc_job_uid") == source_uid)
        if not source or not same_population or not selection:
            continue
        values = []
        for output in node.get("outputs", {}).values():
            for field in ("latest_summary_stat_values", "summary_values"):
                fsc = (output.get(field) or {}).get("fsc_info") or {}
                value = fsc.get("radwn_final_A", fsc.get("radwn_autotight_noisesub_A"))
                if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < float("inf"):
                    values.append(float(value))
        uid = selection["cryosparc_job_uid"]
        grouped.setdefault(uid, None)
        if values:
            grouped[uid] = min(grouped[uid] if grouped[uid] is not None else float("inf"), min(values))
    ordered = sorted(grouped, key=lambda uid: int(uid[1:]))
    prior = [grouped[u] for u in ordered[:-1] if grouped[u] is not None]
    stale = len(ordered) >= 2 and (not prior or grouped[ordered[-1]] is None
                                  or grouped[ordered[-1]] >= min(prior) - 0.05)
    return {"source_class2d_job_uid": source_uid, "stagnant": stale,
            "source_extract_job_uid": extraction_uid,
            "minimum_improvement_A": 0.05,
            "selection_results": [{"select2d_job_uid": u, "corrected_resolution_A": grouped[u]} for u in ordered]}


def _target_resolution_A(dataset_info: dict[str, Any]) -> float | None:
    value = dataset_info.get("target_resolution_A", dataset_info.get("resolution"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _best_resolution_A(quality_assessment: dict[str, Any]) -> float | None:
    values = []
    for value in (quality_assessment.get("resolution_evidence", {}).get("values", {}) or {}).values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            values.append(number)
    return min(values) if values else None


def refinement_target_unmet(
    quality_assessment: dict[str, Any], dataset_info: dict[str, Any],
) -> bool:
    """Return whether the current refinement has a valid resolution above target."""
    target = dataset_info.get("target_resolution_A", dataset_info.get("resolution"))
    try:
        target = float(target)
    except (TypeError, ValueError):
        return False
    values = []
    for value in (quality_assessment.get("resolution_evidence", {}).get("values", {}) or {}).values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            values.append(number)
    return bool(values) and min(values) > target


def find_ancestor_node(
    workflow_state: dict[str, Any], start_node: dict[str, Any], job_type: str,
) -> dict[str, Any] | None:
    """Find the nearest completed ancestor of a given CryoSPARC job type."""
    by_uid = {node.get("cryosparc_job_uid"): node for node in workflow_state.get("nodes", [])}
    pending = list(start_node.get("parent_job_uids") or [])
    visited: set[str] = set()
    while pending:
        uid = pending.pop(0)
        if uid in visited:
            continue
        visited.add(uid)
        node = by_uid.get(uid)
        if node is None:
            continue
        if node.get("job_type") == job_type and node.get("status") == "completed":
            return node
        pending.extend(node.get("parent_job_uids") or [])
    return None


def find_ancestor_node_by_types(
    workflow_state: dict[str, Any], start_node: dict[str, Any], job_types: set[str],
) -> dict[str, Any] | None:
    """Find the nearest completed ancestor matching any compatible job type."""
    by_uid = {node.get("cryosparc_job_uid"): node for node in workflow_state.get("nodes", [])}
    pending = list(start_node.get("parent_job_uids") or [])
    visited: set[str] = set()
    while pending:
        uid = pending.pop(0)
        if uid in visited:
            continue
        visited.add(uid)
        node = by_uid.get(uid)
        if node is None:
            continue
        if node.get("job_type") in job_types and node.get("status") == "completed":
            return node
        pending.extend(node.get("parent_job_uids") or [])
    return None


def find_template_repick_refinement_source(
    workflow_state: dict[str, Any], start_node: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the refinement volume that seeded a Template RePick branch."""
    templates_node = find_ancestor_node(workflow_state, start_node, "create_templates")
    if templates_node is None:
        return None
    return find_ancestor_node_by_types(
        workflow_state,
        templates_node,
        {"homo_refine_new", "nonuniform_refine_new", "refine_3D_new"},
    )


def output_source(node: dict[str, Any], output_name: str) -> dict[str, Any] | None:
    """Convert a completed node output into an executor-compatible input source."""
    output = (node.get("outputs") or {}).get(output_name)
    if not output or not output.get("available"):
        return None
    return {
        "source_workflow_node_id": node["workflow_node_id"],
        "source_logical_node_id": node["logical_node_id"],
        "source_job_uid": node["cryosparc_job_uid"],
        "source_output": output_name,
        "result_names": output.get("result_names") or [],
    }


def output_item_count(node: dict[str, Any], output_name: str) -> int | None:
    """Return a positive item count for an observed workflow output."""
    output = (node.get("outputs") or {}).get(output_name) or {}
    try:
        count = int(output.get("num_items"))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def relevant_inputs(node: dict[str, Any], names: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Copy only executor-relevant historical input connections."""
    return {
        name: connections
        for name, connections in (node.get("inputs") or {}).items()
        if name in names and connections
    }


def build_initial_import_candidates(
    dataset_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build import candidates from explicit dataset file facts at startup."""
    dataset_info = normalize_dataset_info(dataset_info)
    files = dataset_info.get("available_input_files") or {}
    candidates = []
    sources = [
        ("micrograph_blob_paths", "import_micrographs", "micrograph_blob_paths"),
        ("movie_blob_paths", "import_movies", "movie_blob_paths"),
        ("volume_blob_path", "import_volumes", "volume_blob_path"),
    ]
    for file_key, job_type, parameter_name in sources:
        value = files.get(file_key)
        if not value:
            continue
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else value[0]
        template = get_parameter_template(job_type)
        defaults = {parameter_name if parameter_name in template else "blob_paths": value}
        for name in ("psize_A", "accel_kv", "cs_mm", "total_dose_e_per_A2"):
            if dataset_info.get(name) is not None and name in template:
                defaults[name] = dataset_info[name]
        fact_names = [
            name for name in ("psize_A", "accel_kv", "cs_mm", "total_dose_e_per_A2")
            if defaults.get(name) is not None
        ]
        fact_note = (
            " Known dataset facts are supplied as default parameters and must be "
            "preserved unless explicit evidence supports changing them: "
            + ", ".join(fact_names) + "."
            if fact_names
            else ""
        )
        candidates.append({
            "action_id": f"initial_{job_type}",
            "action_type": "forward",
            "workflow_node_id": f"initial:{job_type}",
            "reference_job_uid": None,
            "reference_status": None,
            "job_type": job_type,
            "description": f"Import dataset input using {job_type}.{fact_note}",
            "execution_mode": "create_job",
            "available": True,
            "blocked_by": [],
            "required_inputs": {},
            "missing_required_inputs": [],
            "parameter_template": template,
            "default_parameters": defaults,
            "workflow_stage": "import",
            "workflow_policy_recommendation": "preferred_next_stage",
        })
    return candidates


def generate_candidate_actions(
    workflow_state: dict[str, Any],
    current_node_id: str | None,
    project_uid: str | None = None,
    include_registry: bool = False,
    dataset_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn the selected node's child jobs into forward or branch candidates."""
    if not current_node_id:
        return [], []

    current_node = find_node(workflow_state, current_node_id)
    if current_node is None:
        return [], [
            {
                "action_id": None,
                "available": False,
                "blocked_by": [f"Unknown current_node_id: {current_node_id}"],
            }
        ]

    child_uids = current_node["child_job_uids"]
    child_nodes = [
        find_node(workflow_state, child_uid)
        for child_uid in child_uids
    ]
    child_nodes = [
        node
        for node in child_nodes
        if node is not None and node["status"] not in {"failed", "killed"}
    ]

    is_branch = len(child_nodes) > 1
    candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for child in child_nodes:
        action_type = "branch" if is_branch else "forward"
        action = build_candidate_action(
            workflow_state,
            child,
            action_type,
        )
        if action["available"]:
            candidates.append(action)
        else:
            blocked.append(action)

    synthetic_actions, synthetic_blocked = build_synthetic_next_actions(
        workflow_state,
        current_node,
        project_uid=project_uid,
        dataset_info=dataset_info,
    )
    candidates.extend(synthetic_actions)
    blocked.extend(synthetic_blocked)

    if include_registry:
        registry_actions, registry_blocked = build_registry_next_actions(
            workflow_state,
            current_node,
            candidate_job_types={node["job_type"] for node in child_nodes},
        )
        blocked_job_types = {action.get("job_type") for action in registry_blocked}
        candidates = [
            action for action in candidates
            if action.get("job_type") not in blocked_job_types
        ]
        candidates = merge_candidates(candidates, registry_actions)
        blocked = merge_candidates(blocked, registry_blocked)

    if current_node.get("job_type") == "class_2D_new":
        comparison_actions = build_class2d_comparison_candidates(
            workflow_state,
            current_node,
            project_uid=project_uid,
        )
        if len(comparison_actions) > 1:
            candidates = [
                action for action in candidates
                if action.get("job_type") != "select_2D"
            ]
            group = [
                action.get("reference_job_uid")
                or str(action.get("action_id", "")).split("_")[1]
                for action in comparison_actions
            ]
            for action in comparison_actions:
                action["class2d_comparison_group"] = group
                action["comparison_instruction"] = (
                    "This is one of several completed Class 2D box-size trials. "
                    "Compare all group members using their visual evidence before selecting one."
                )
            candidates.extend(comparison_actions)

    candidates = [
        apply_auto_select_2d_policy(action, workflow_state)
        for action in candidates
    ]
    blocked = [
        apply_auto_select_2d_policy(action, workflow_state)
        for action in blocked
    ]
    candidates, gated = apply_hard_workflow_gates(
        candidates, blocked, workflow_state, current_node,
        project_uid=project_uid,
        dataset_info=dataset_info or {},
    )
    blocked.extend(gated)
    policy = infer_current_stage(
        workflow_state["nodes"],
        current_node_id=current_node["workflow_node_id"],
    )
    candidates = annotate_candidates(candidates, policy["current_stage"])
    blocked = annotate_candidates(blocked, policy["current_stage"])
    candidates.sort(key=lambda action: action["action_id"])
    blocked.sort(key=lambda action: action["action_id"])
    return candidates, blocked


def apply_hard_workflow_gates(
    candidates: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    project_uid: str | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enforce the fixed reconstruction funnel outside refinement checkpoints."""
    from inspect_score_contract import raw_candidate
    candidates = [raw_candidate(action) for action in candidates]
    gated: list[dict[str, Any]] = []
    job_type = current_node.get("job_type")
    dataset_info = dataset_info or {}

    from refinement_optimization import continuation as optimization_continuation
    optimization_actions = optimization_continuation(workflow_state, current_node)
    if optimization_actions is not None:
        return optimization_actions, [
            {**action, 'available': False,
             'blocked_by': list(action.get('blocked_by') or []) + [
                 'Complete the selected bounded refinement intervention before another experiment.']}
            for action in candidates
        ]

    from diameter_trial import trial_candidates
    diameter_actions = trial_candidates(workflow_state, current_node, candidates)
    if diameter_actions is not None:
        selected_ids = {action['action_id'] for action in diameter_actions}
        return diameter_actions, [
            {**action, 'available': False,
             'blocked_by': list(action.get('blocked_by') or []) + [
                 'Uncertain diameter trial: review pilot evidence before full-data reconstruction.']}
            for action in candidates if action.get('action_id') not in selected_ids
        ]

    if initial_scale_calibration_enabled():
        # Sampling/pilot actions are deliberately constrained; normal model
        # choice resumes as soon as the two calibration passes are complete.
        calibration_actions = build_initial_scale_calibration_actions(
            workflow_state, current_node, candidates, project_uid, dataset_info,
        )
        if calibration_actions is not None:
            selected_ids = {action.get("action_id") for action in calibration_actions}
            for action in candidates:
                if action.get("action_id") not in selected_ids:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Initial scale calibration is active: complete its bounded two-pass pilot before the full-data funnel."
                        ],
                    })
            return calibration_actions, gated
    if fixed_initial_funnel_enabled() and job_type in {
        "patch_ctf", "patch_ctf_estimation", "patch_ctf_estimation_multi",
        "ctf_estimation", "ctf_estimation_gctf",
    }:
        blob_picker = [
            action for action in candidates
            if action.get("job_type") in {"blob_picker_gpu", "auto_blob_picker_gpu"}
        ]
        if blob_picker:
            if initial_scale_calibration_enabled():
                sample_count = initial_scale_calibration_micrographs()
                for action in blob_picker:
                    action["action_id"] = (
                        f"initial_scale_pilot_1_{action['job_type']}_{current_node['cryosparc_job_uid']}"
                    )
                    _set_candidate_parameter_default(action, "num_process", sample_count, "integer", 1)
                    action["parameter_template"]["num_process"]["const"] = sample_count
                    action["calibration_phase"] = "pilot_1"
                    action["description"] = (
                        f"Scale calibration pilot 1/2: process {sample_count} micrographs only; "
                        "a second pilot will refine the particle diameter and extraction box before full processing."
                    )
            else:
                full_count = output_item_count(current_node, "exposures")
                if full_count:
                    for action in blob_picker:
                        _set_candidate_parameter_default(
                            action, "num_process", full_count, "integer", 1,
                        )
                        action["parameter_template"]["num_process"]["const"] = full_count
                        action["description"] = (
                            f"Process all {full_count} input micrographs; num_process is the micrograph "
                            "count limit for this CryoSPARC Blob Picker job."
                        )
            for action in candidates:
                if action not in blob_picker:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Fixed reconstruction funnel: run Blob Picker after CTF estimation."
                        ],
                    })
            from diameter_trial import enabled as diameter_trial_enabled, initial_trial
            if (not initial_scale_calibration_enabled() and diameter_trial_enabled()
                    and full_count and full_count > 20):
                # Historical child jobs can expose more Blob Picker candidates.
                # Offer one fresh trial, never one trial per observed child.
                fresh_id = f"forward_{current_node['cryosparc_job_uid']}_blob_picker_gpu"
                trial_base = next((a for a in blob_picker if a['action_id'] == fresh_id), blob_picker[0])
                blob_picker = blob_picker + [initial_trial(trial_base, full_count)]
            return blob_picker, gated
    # Junk Detector is an optional curation branch.  The executor receives the
    # picker visual evidence and can choose it when carbon, ice, or contamination
    # is apparent; clean fields proceed directly to Inspect Picks.

    if job_type == "create_templates":
        template_source = output_source(current_node, "templates")
        ctf_node = find_ancestor_node_by_types(
            workflow_state,
            current_node,
            {"patch_ctf_estimation_multi", "patch_ctf_estimation", "patch_ctf", "ctf_estimation"},
        )
        micrographs = output_source(ctf_node, "exposures") if ctf_node else None
        if template_source and micrographs:
            action = build_synthetic_template_picker_candidate(
                current_node, micrographs, template_source,
            )
            action["description"] = (
                "Fixed template-repicking funnel: use the newly generated templates on the "
                "ancestor CTF-estimated micrographs."
            )
            return [action], gated

    if job_type == "junk_detector_v1":
        inspection = [
            action for action in candidates
            if action.get("job_type") == "inspect_picks_v2"
        ]
        if inspection:
            for action in candidates:
                if action not in inspection:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Mandatory Inspect Picks gate: complete inspect_picks_v2 using "
                            "Junk Detector accepted particles before extraction or downstream actions."
                        ],
                    })
            return inspection, gated

    if fixed_initial_funnel_enabled() and job_type == "inspect_picks_v2":
        extraction = [
            action for action in candidates
            if action.get("job_type") == "extract_micrographs_multi"
        ]
        has_calibrated_box = any(
            action.get("calibration_phase") in {
                "full_initial_extract", "full_initial_extract_fallback",
            }
            for action in extraction
        )
        if extraction and not box_size_sweep_enabled() and not has_calibrated_box:
            # Historical workspaces can expose old extraction branches. The first
            # pass always begins from one fresh 384 px extraction instead.
            selected = next(
                (
                    action for action in extraction
                    if (action.get("default_parameters") or {}).get("box_size_pix")
                    == default_extract_box_size()
                ),
                extraction[0],
            )
            selected = dict(selected)
            selected["default_parameters"] = {
                **(selected.get("default_parameters") or {}),
                "box_size_pix": default_extract_box_size(),
            }
            extraction = [selected]
        if extraction:
            for action in candidates:
                if action not in extraction:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Fixed reconstruction funnel: extract inspected particles before downstream actions."
                        ],
                    })
            return extraction, gated

    if job_type == "extract_micrographs_multi" and box_size_sweep_enabled():
        trials = build_box_size_trial_candidates(workflow_state, current_node)
        if trials:
            for action in candidates:
                gated.append({
                    **action,
                    "available": False,
                    "blocked_by": list(action.get("blocked_by") or []) + [
                        "Mandatory box-size sweep: complete all configured extraction trials before Class 2D."
                    ],
                })
            return trials, gated
        class_trials = build_class_2d_trial_candidates(workflow_state, current_node)
        if len(class_trials) > 1:
            for action in candidates:
                gated.append({
                    **action,
                    "available": False,
                    "blocked_by": list(action.get("blocked_by") or []) + [
                        "Mandatory Class 2D comparison: run Class 2D for every completed box-size trial first."
                    ],
                })
            return class_trials, gated
    if job_type == "extract_micrographs_multi":
        class2d = [
            action for action in candidates
            if action.get("job_type") == "class_2D_new"
        ]
        if class2d:
            for action in candidates:
                if action not in class2d:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Fixed reconstruction funnel: run Class 2D after extraction."
                        ],
                    })
            return class2d, gated
    if job_type == "class_2D_new" and box_size_sweep_enabled():
        pending_trials = build_pending_class_2d_trial_candidates(workflow_state, current_node)
        if pending_trials:
            for action in candidates:
                gated.append({
                    **action,
                    "available": False,
                    "blocked_by": list(action.get("blocked_by") or []) + [
                        "Mandatory Class 2D comparison: complete the remaining box-size "
                        "Class 2D trial(s) before Select 2D or downstream reconstruction."
                    ],
                })
            return pending_trials, gated
    if job_type == "class_2D_new":
        selection = [
            action for action in candidates
            if action.get("job_type") == "select_2D"
        ]
        current_branch = [
            action for action in selection
            if action_references_node(action, current_node["cryosparc_job_uid"])
        ]
        if current_branch:
            selection = current_branch
        if selection:
            for action in candidates:
                if action not in selection:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Fixed reconstruction funnel: submit Select 2D after Class 2D."
                        ],
                    })
            return selection, gated
    if job_type == "select_2D":
        template_refine = find_template_repick_refinement_source(
            workflow_state, current_node,
        )
        if template_refine is not None:
            template_refines = [
                action for action in candidates
                if action.get("job_type") == "homo_refine_new"
                and action.get("template_repick_refinement")
            ]
            if template_refines:
                for action in candidates:
                    if action not in template_refines:
                        gated.append({
                            **action,
                            "available": False,
                            "blocked_by": list(action.get("blocked_by") or []) + [
                                "Template RePick transition: refine the newly selected particles "
                                "against the source refinement volume; do not re-run Ab-initio."
                            ],
                        })
                return template_refines, gated
        abinitio = [
            action for action in candidates
            if action.get("job_type") == "homo_abinit"
        ]
        current_branch = [
            action for action in abinitio
            if action_references_node(action, current_node["cryosparc_job_uid"])
        ]
        if current_branch:
            abinitio = current_branch
        if abinitio:
            for action in candidates:
                if action not in abinitio:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Mandatory reconstruction transition: run Homo Ab-initio from "
                            "Select 2D particles before refinement or upstream re-picking."
                        ],
                    })
            return abinitio, gated
    if job_type == "homo_abinit":
        baseline = [
            action for action in candidates
            if action.get("job_type") == "homo_refine_new"
        ]
        current_branch = [
            action for action in baseline
            if action_references_node(action, current_node["cryosparc_job_uid"])
        ]
        if current_branch:
            baseline = current_branch
        if baseline:
            for action in candidates:
                if action not in baseline:
                    gated.append({
                        **action,
                        "available": False,
                        "blocked_by": list(action.get("blocked_by") or []) + [
                            "Fixed reconstruction funnel: establish the first Homo-refine baseline."
                        ],
                    })
            return baseline, gated
    return candidates, gated


def build_initial_scale_calibration_actions(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    candidates: list[dict[str, Any]],
    project_uid: str | None,
    dataset_info: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Drive two small Blob Picker/Extract/Class 2D pilots, then full data."""
    sample_count = initial_scale_calibration_micrographs()
    job_type = current_node.get("job_type")
    target = dataset_info.get("target_resolution_A", dataset_info.get("resolution"))
    pilot_classes = _initial_scale_pilot_class2d_nodes(workflow_state, sample_count)
    latest_scale = None
    is_pilot_class = job_type == "class_2D_new" and _is_scale_pilot_class2d(
        workflow_state, current_node, sample_count,
    )
    current_picker_is_pilot = (
        job_type in {"blob_picker_gpu", "auto_blob_picker_gpu"}
        and _is_initial_scale_pilot_picker(current_node, sample_count)
    )
    current_inspect_is_pilot = False
    if job_type == "inspect_picks_v2":
        ancestor_picker = find_ancestor_node_by_types(
            workflow_state, current_node, {"blob_picker_gpu", "auto_blob_picker_gpu"},
        )
        current_inspect_is_pilot = bool(
            ancestor_picker and _is_initial_scale_pilot_picker(ancestor_picker, sample_count)
        )
    needs_scale = bool(pilot_classes) and (
        is_pilot_class
        or (job_type in {"blob_picker_gpu", "auto_blob_picker_gpu"} and (current_picker_is_pilot or len(pilot_classes) >= 2))
        or (job_type == "inspect_picks_v2" and (current_inspect_is_pilot or len(pilot_classes) >= 2))
    )
    if needs_scale and project_uid:
        try:
            latest_scale = build_class2d_scale_context(
                project_uid,
                pilot_classes[-1]["cryosparc_job_uid"],
                target_resolution_A=target,
                max_box_size_pix=MAX_EXTRACT_BOX_SIZE_PIX,
            )
        except Exception:
            latest_scale = None

    if is_pilot_class:
        if len(pilot_classes) < 2:
            next_pilot = len(pilot_classes) + 1
            source_ctf = find_ancestor_node_by_types(
                workflow_state, current_node,
                {"patch_ctf_estimation_multi", "patch_ctf_estimation", "patch_ctf", "ctf_estimation"},
            )
            micrographs = output_source(source_ctf, "exposures") if source_ctf else None
            aggregate = (latest_scale or {}).get("aggregate") or {}
            box = (latest_scale or {}).get("box_size_recommendation") or {}
            if micrographs:
                candidate = build_synthetic_blob_picker_candidate(current_node, micrographs)
                candidate["action_id"] = f"initial_scale_pilot_{next_pilot}_{current_node['cryosparc_job_uid']}"
                candidate["calibration_phase"] = f"pilot_{next_pilot}"
                candidate["description"] = (
                    f"Scale calibration pilot {next_pilot}/2: sample {sample_count} micrographs "
                    "using the preceding measured particle-diameter range."
                )
                if next_pilot == 2 and not box.get("usable_for_extraction"):
                    box = {"recommended_box_size_pix": INITIAL_SCALE_PILOT_BOX_SIZE_PIX}
                    candidate["description"] += (
                        f" The first-pass CTF-aware box estimate was unavailable, so this bounded pilot "
                        f"uses the balanced {INITIAL_SCALE_PILOT_BOX_SIZE_PIX}-pixel fallback; "
                        "the final full-data box still requires review."
                    )
                _set_candidate_parameter_default(
                    candidate, "num_process", sample_count, "integer", 1,
                )
                candidate["parameter_template"]["num_process"]["const"] = sample_count
                diameter_range = aggregate.get("particle_diameter_range_A")
                if isinstance(diameter_range, list) and len(diameter_range) == 2:
                    low, high = sorted(float(value) for value in diameter_range)
                    _set_candidate_parameter_default(candidate, "diameter", round(low, 1), "number", 0)
                    _set_candidate_parameter_default(candidate, "diameter_max", round(high, 1), "number", 0)
                    if next_pilot == 2:
                        candidate["parameter_template"]["diameter"]["const"] = round(low, 1)
                        candidate["parameter_template"]["diameter_max"]["const"] = round(high, 1)
                if next_pilot == 2 and box.get("recommended_box_size_pix"):
                    candidate["calibration_box_size_pix"] = int(box["recommended_box_size_pix"])
                candidate["scale_observer_evidence"] = latest_scale
                return [candidate]
        else:
            source_ctf = find_ancestor_node_by_types(
                workflow_state, current_node,
                {"patch_ctf_estimation_multi", "patch_ctf_estimation", "patch_ctf", "ctf_estimation"},
            )
            micrographs = output_source(source_ctf, "exposures") if source_ctf else None
            full_count = output_item_count(source_ctf, "exposures") if source_ctf else None
            aggregate = (latest_scale or {}).get("aggregate") or {}
            if micrographs and full_count:
                candidate = build_synthetic_blob_picker_candidate(current_node, micrographs)
                candidate["action_id"] = f"initial_scale_full_picker_{current_node['cryosparc_job_uid']}"
                candidate["calibration_phase"] = "full_initial_picker"
                _set_candidate_parameter_default(
                    candidate, "num_process", full_count, "integer", 1,
                )
                candidate["parameter_template"]["num_process"]["const"] = full_count
                candidate["description"] = (
                    f"Both small-batch calibration passes are complete. Process all {full_count} "
                    "Patch CTF micrographs (num_process is the per-job micrograph count limit) "
                    "using the final measured particle-diameter range."
                )
                diameter_range = aggregate.get("particle_diameter_range_A")
                if isinstance(diameter_range, list) and len(diameter_range) == 2:
                    low, high = sorted(float(value) for value in diameter_range)
                    _set_candidate_parameter_default(candidate, "diameter", round(low, 1), "number", 0)
                    _set_candidate_parameter_default(candidate, "diameter_max", round(high, 1), "number", 0)
                    if aggregate.get("usable_for_picker"):
                        candidate["parameter_template"]["diameter"]["const"] = round(low, 1)
                        candidate["parameter_template"]["diameter_max"]["const"] = round(high, 1)
                candidate["scale_observer_evidence"] = latest_scale
                if not aggregate.get("usable_for_picker"):
                    candidate["description"] += (
                        " The final pilot did not pass all scale quality checks; no measured diameter "
                        "default is applied, and pilot particles are not reused as full-data particles."
                    )
                return [candidate]
        return []

    if job_type in {"blob_picker_gpu", "auto_blob_picker_gpu"}:
        pilot_picker = _is_initial_scale_pilot_picker(current_node, sample_count)
        if pilot_picker:
            extraction = [action for action in candidates if action.get("job_type") == "extract_micrographs_multi"]
            if extraction:
                box_size = INITIAL_SCALE_PILOT_BOX_SIZE_PIX
                if pilot_classes:
                    box = (latest_scale or {}).get("box_size_recommendation") or {}
                    if box.get("usable_for_extraction"):
                        box_size = int(box["recommended_box_size_pix"])
                    else:
                        box_size = INITIAL_SCALE_PILOT_BOX_SIZE_PIX
                result = []
                for action in extraction:
                    item = dict(action)
                    round_index = min(2, len(pilot_classes) + 1)
                    item["calibration_phase"] = f"pilot_{round_index}_extract"
                    item["description"] = (
                        f"Extract only {sample_count} micrographs for scale-calibration pilot "
                        f"{round_index}/2 at box_size_pix={box_size}."
                    )
                    _set_candidate_parameter_default(item, "num_extract", sample_count, "integer", 1)
                    item["parameter_template"]["num_extract"]["const"] = sample_count
                    _set_candidate_parameter_default(
                        item, "box_size_pix", box_size, "integer", 32,
                        maximum=MAX_EXTRACT_BOX_SIZE_PIX,
                    )
                    item["parameter_template"]["box_size_pix"]["const"] = box_size
                    result.append(item)
                return result
        elif len(pilot_classes) >= 2 and latest_scale:
            box = latest_scale.get("box_size_recommendation") or {}
            proposed_box = box.get("recommended_box_size_pix")
            if proposed_box is not None and box.get("usable_for_extraction"):
                for action in candidates:
                    if action.get("job_type") == "extract_micrographs_multi":
                        _set_candidate_parameter_default(
                            action, "box_size_pix", int(proposed_box),
                            "integer", 32, maximum=MAX_EXTRACT_BOX_SIZE_PIX,
                        )
                        action["parameter_template"]["box_size_pix"]["const"] = int(proposed_box)
                        action["calibration_phase"] = "full_initial_extract"
                        action["scale_observer_evidence"] = latest_scale
                        action["description"] = (
                            "Full-data extraction using the final box estimate from both calibration passes. "
                            "Review the observer's 2.5D criterion flag before submission."
                        )
            else:
                for action in candidates:
                    if action.get("job_type") == "extract_micrographs_multi":
                        _set_candidate_parameter_default(
                            action, "box_size_pix", INITIAL_SCALE_PILOT_BOX_SIZE_PIX,
                            "integer", 32, maximum=MAX_EXTRACT_BOX_SIZE_PIX,
                        )
                        action["parameter_template"]["box_size_pix"]["const"] = INITIAL_SCALE_PILOT_BOX_SIZE_PIX
                        action["calibration_phase"] = "full_initial_extract_fallback"
                        action["description"] = (
                            "Full-data extraction: the scale observer could not produce a numerical box estimate; "
                            f"use the balanced {INITIAL_SCALE_PILOT_BOX_SIZE_PIX} px fallback and review "
                            "Class 2D/CTF evidence before increasing it."
                        )

    if job_type == "inspect_picks_v2":
        blob = find_ancestor_node_by_types(
            workflow_state, current_node, {"blob_picker_gpu", "auto_blob_picker_gpu"},
        )
        if blob and _is_initial_scale_pilot_picker(blob, sample_count):
            extraction = [action for action in candidates if action.get("job_type") == "extract_micrographs_multi"]
            if extraction:
                box = (latest_scale or {}).get("box_size_recommendation") or {}
                box_size = (
                    int(box["recommended_box_size_pix"])
                    if box.get("usable_for_extraction") else INITIAL_SCALE_PILOT_BOX_SIZE_PIX
                )
                result = []
                for action in extraction:
                    item = dict(action)
                    round_index = min(2, len(pilot_classes) + 1)
                    item["calibration_phase"] = f"pilot_{round_index}_extract"
                    _set_candidate_parameter_default(item, "num_extract", sample_count, "integer", 1)
                    item["parameter_template"]["num_extract"]["const"] = sample_count
                    _set_candidate_parameter_default(item, "box_size_pix", box_size, "integer", 32, maximum=MAX_EXTRACT_BOX_SIZE_PIX)
                    item["parameter_template"]["box_size_pix"]["const"] = box_size
                    result.append(item)
                return result
        if blob and len(pilot_classes) >= 2 and latest_scale:
            box = latest_scale.get("box_size_recommendation") or {}
            proposed_box = box.get("recommended_box_size_pix")
            if proposed_box is not None and box.get("usable_for_extraction"):
                for action in candidates:
                    if action.get("job_type") == "extract_micrographs_multi":
                        _set_candidate_parameter_default(
                            action, "box_size_pix", int(proposed_box),
                            "integer", 32, maximum=MAX_EXTRACT_BOX_SIZE_PIX,
                        )
                        action["parameter_template"]["box_size_pix"]["const"] = int(proposed_box)
                        action["calibration_phase"] = "full_initial_extract"
                        action["scale_observer_evidence"] = latest_scale
                        action["description"] = (
                            "Full-data extraction using the final box estimate from both calibration passes. "
                            "Review the observer's 2.5D criterion flag before submission."
                        )
            else:
                for action in candidates:
                    if action.get("job_type") == "extract_micrographs_multi":
                        _set_candidate_parameter_default(
                            action, "box_size_pix", INITIAL_SCALE_PILOT_BOX_SIZE_PIX,
                            "integer", 32, maximum=MAX_EXTRACT_BOX_SIZE_PIX,
                        )
                        action["parameter_template"]["box_size_pix"]["const"] = INITIAL_SCALE_PILOT_BOX_SIZE_PIX
                        action["calibration_phase"] = "full_initial_extract_fallback"
                        action["description"] = (
                            "Full-data extraction: the scale observer could not produce a numerical box estimate; "
                            f"use the balanced {INITIAL_SCALE_PILOT_BOX_SIZE_PIX} px fallback and review "
                            "Class 2D/CTF evidence before increasing it."
                        )

    return None


def _initial_scale_pilot_class2d_nodes(
    workflow_state: dict[str, Any], sample_count: int,
) -> list[dict[str, Any]]:
    nodes = []
    for node in workflow_state.get("nodes", []):
        if node.get("job_type") != "class_2D_new" or node.get("status") != "completed":
            continue
        if _is_scale_pilot_class2d(workflow_state, node, sample_count):
            nodes.append(node)
    return sorted(nodes, key=lambda node: _calibration_job_number(node.get("cryosparc_job_uid")))


def _calibration_job_number(job_uid: Any) -> int:
    try:
        return int(str(job_uid or "J0").lstrip("J"))
    except ValueError:
        return 0


def _is_scale_pilot_class2d(
    workflow_state: dict[str, Any], node: dict[str, Any], sample_count: int,
) -> bool:
    extract = find_ancestor_node(workflow_state, node, "extract_micrographs_multi")
    if extract is None or _parameter(extract, "num_extract") != sample_count:
        return False
    picker = find_ancestor_node_by_types(
        workflow_state, extract,
        {"blob_picker_gpu", "auto_blob_picker_gpu"},
    )
    return bool(picker and _is_initial_scale_pilot_picker(picker, sample_count))


def _parameter(node: dict[str, Any], name: str) -> Any:
    return (node.get("key_parameters") or {}).get(name)


def _is_initial_scale_pilot_picker(node: dict[str, Any], sample_count: int) -> bool:
    title = str(node.get("title") or "").lower()
    if "initial_scale_full_picker" in title:
        return False
    return "initial_scale_pilot" in title or _parameter(node, "num_process") == sample_count


def _set_candidate_parameter_default(
    action: dict[str, Any], name: str, value: Any, value_type: str,
    minimum: int | float, maximum: int | None = None,
) -> None:
    template = dict(action.get("parameter_template") or {})
    spec = dict(template.get(name) or {"type": value_type})
    spec.update({"type": value_type, "default": value, "minimum": minimum})
    if maximum is not None:
        spec["maximum"] = maximum
    template[name] = spec
    action["parameter_template"] = template
    defaults = dict(action.get("default_parameters") or {})
    defaults[name] = value
    action["default_parameters"] = defaults


def action_references_node(action: dict[str, Any], job_uid: str) -> bool:
    """Return whether an action consumes output from the current workflow node."""
    if action.get("reference_job_uid") == job_uid:
        return True
    for connections in (action.get("required_inputs") or {}).values():
        for connection in connections or []:
            if connection.get("source_job_uid") == job_uid:
                return True
    return False


def configured_box_size_trials() -> list[int]:
    """Return the finite extraction sizes used by the mandatory sweep."""
    raw = os.getenv(
        "CRYOAGENT_BOX_SIZE_TRIALS",
        "320,384,416,448,480,504,544,576,608,640,672,704,720",
    )
    values: list[int] = []
    for item in raw.split(","):
        try:
            value = int(item.strip())
        except ValueError:
            continue
        if 32 <= value <= MAX_EXTRACT_BOX_SIZE_PIX and value not in values:
            values.append(value)
    return values or [320, 384, 416, 448, 480, 504, 544, 576, 608, 640, 672, 704, 720]


def default_extract_box_size() -> int:
    """Return the single-path extraction box size used before the first refine."""
    try:
        value = int(os.getenv("CRYOAGENT_DEFAULT_BOX_SIZE_PIX", "384"))
    except ValueError:
        value = 384
    return max(32, value)


def apply_default_extract_box_size(action: dict[str, Any]) -> dict[str, Any]:
    """Supply the standard 384-pixel extraction default without locking later trials."""
    if action.get("job_type") != "extract_micrographs_multi":
        return action
    defaults = dict(action.get("default_parameters") or {})
    if defaults.get("box_size_pix") is not None or action.get("box_size_trial"):
        return action
    enriched = dict(action)
    defaults["box_size_pix"] = default_extract_box_size()
    enriched["default_parameters"] = defaults
    template = dict(enriched.get("parameter_template") or {})
    box_spec = dict(template.get("box_size_pix") or {"type": "integer", "minimum": 32})
    box_spec.setdefault("default", defaults["box_size_pix"])
    template["box_size_pix"] = box_spec
    enriched["parameter_template"] = template
    enriched["description"] = (
        f"{action.get('description', '')} Default fixed-funnel extraction uses "
        f"box_size_pix={defaults['box_size_pix']}."
    ).strip()
    return enriched


def extraction_input_signature(inputs: dict[str, Any]) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Identify an extraction branch by the actual micrograph and particle sources."""
    return tuple(
        (
            input_name,
            tuple(
                (str(item.get("source_job_uid") or ""), str(item.get("source_output") or ""))
                for item in inputs.get(input_name, [])
                if isinstance(item, dict)
            ),
        )
        for input_name in ("micrographs", "particles")
    )


def build_box_size_trial_candidates(
    workflow_state: dict[str, Any],
    extract_node: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build available branch candidates for missing standard box sizes."""
    sizes = configured_box_size_trials()
    # A 384-pixel extraction from another Picker/Inspect branch is not a
    # substitute for testing 384 pixels on this branch's particles.
    branch_signature = extraction_input_signature(extract_node.get("inputs") or {})
    existing = set()
    for node in workflow_state.get("nodes", []):
        if node.get("job_type") != "extract_micrographs_multi":
            continue
        if extraction_input_signature(node.get("inputs") or {}) != branch_signature:
            continue
        value = (node.get("key_parameters") or {}).get("box_size_pix")
        try:
            existing.add(int(value))
        except (TypeError, ValueError):
            continue
    required_inputs = {
        name: connections
        for name, connections in (extract_node.get("inputs") or {}).items()
        if name in {"micrographs", "particles"} and connections
    }
    if not required_inputs:
        return []
    template = build_parameter_template(extract_node)
    actions = []
    for size in sizes:
        if size in existing:
            continue
        defaults = {
            name: value
            for name, value in (extract_node.get("key_parameters") or {}).items()
            if name == "compute_num_gpus"
        }
        defaults["box_size_pix"] = size
        actions.append({
            "action_id": f"box_trial_{extract_node['cryosparc_job_uid']}_{size}",
            "action_type": "forward",
            "workflow_node_id": f"{extract_node['workflow_node_id']}:box_trial_{size}",
            "reference_job_uid": extract_node["cryosparc_job_uid"],
            "reference_status": extract_node["status"],
            "job_type": "extract_micrographs_multi",
            "description": (
                f"Mandatory box-size trial extraction with box_size_pix={size}; run it before downstream Class 2D."
            ),
            "execution_mode": "create_job",
            "available": True,
            "blocked_by": [],
            "required_inputs": required_inputs,
            "missing_required_inputs": [],
            "parameter_template": template,
            "default_parameters": defaults,
            "box_size_trial": True,
        })
    return actions


def build_class_2d_trial_candidates(
    workflow_state: dict[str, Any],
    current_extract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one Class 2D candidate for each completed extraction trial."""
    current_inputs = current_extract.get("inputs") or {}
    source_key = tuple(
        (item.get("source_job_uid"), item.get("source_output"))
        for item in current_inputs.get("micrographs", [])
    )
    if not source_key:
        return []
    extracts = []
    for node in workflow_state.get("nodes", []):
        if node.get("job_type") != "extract_micrographs_multi" or node.get("status") != "completed":
            continue
        inputs = node.get("inputs") or {}
        key = tuple(
            (item.get("source_job_uid"), item.get("source_output"))
            for item in inputs.get("micrographs", [])
        )
        if key == source_key and node.get("outputs", {}).get("particles", {}).get("available"):
            extracts.append(node)
    if len(extracts) < 2:
        return []
    template = get_parameter_template("class_2D_new")
    actions = []
    for node in sorted(extracts, key=lambda item: int(str(item["cryosparc_job_uid"])[1:])):
        particles = node["outputs"]["particles"]
        box_size = (node.get("key_parameters") or {}).get("box_size_pix")
        actions.append({
            "action_id": f"class2d_trial_{node['cryosparc_job_uid']}",
            "action_type": "forward",
            "workflow_node_id": f"{node['workflow_node_id']}:class_2d_trial",
            "reference_job_uid": node["cryosparc_job_uid"],
            "reference_status": node["status"],
            "job_type": "class_2D_new",
            "description": f"Class 2D comparison for extraction box_size_pix={box_size}.",
            "execution_mode": "create_job",
            "available": True,
            "blocked_by": [],
            "required_inputs": {
                "particles": [{
                    "source_workflow_node_id": node["workflow_node_id"],
                    "source_logical_node_id": node["logical_node_id"],
                    "source_job_uid": node["cryosparc_job_uid"],
                    "source_output": "particles",
                    "result_names": particles.get("result_names") or [],
                }]
            },
            "missing_required_inputs": [],
            "parameter_template": template,
            "default_parameters": {},
            "class_2d_trial": True,
            "box_size_pix": box_size,
        })
    return actions


def build_pending_class_2d_trial_candidates(
    workflow_state: dict[str, Any],
    class2d_node: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep unmatched Class 2D box-size trials visible after one branch completes."""
    extract_node = find_previous_job_node(
        workflow_state, class2d_node, job_type="extract_micrographs_multi",
    )
    if extract_node is None:
        return []
    trials = build_class_2d_trial_candidates(workflow_state, extract_node)
    completed_extract_sources = {
        ((node.get("inputs") or {}).get("particles") or [{}])[0].get("source_job_uid")
        for node in workflow_state.get("nodes", [])
        if node.get("job_type") == "class_2D_new" and node.get("status") == "completed"
    }
    return [
        action for action in trials
        if action.get("reference_job_uid") not in completed_extract_sources
    ]


def build_class2d_comparison_candidates(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    project_uid: str | None,
) -> list[dict[str, Any]]:
    """Expose Select 2D candidates for every comparable Class 2D branch."""
    current_particles = (current_node.get("inputs") or {}).get("particles") or []
    current_extract_uid = current_particles[0].get("source_job_uid") if current_particles else None
    if not current_extract_uid:
        return []
    extract_nodes = {
        node.get("cryosparc_job_uid"): node
        for node in workflow_state.get("nodes", [])
        if node.get("job_type") == "extract_micrographs_multi"
        and node.get("status") == "completed"
    }
    current_extract = extract_nodes.get(current_extract_uid)
    current_mic_inputs = (current_extract or {}).get("inputs", {}).get("micrographs") or []
    current_mic_source = current_mic_inputs[0].get("source_job_uid") if current_mic_inputs else None
    if not current_mic_source:
        return []

    class_nodes = []
    for node in workflow_state.get("nodes", []):
        if node.get("job_type") != "class_2D_new" or node.get("status") != "completed":
            continue
        particles = (node.get("inputs") or {}).get("particles") or []
        extract = extract_nodes.get(particles[0].get("source_job_uid")) if particles else None
        mic_inputs = (extract or {}).get("inputs", {}).get("micrographs") or []
        if mic_inputs and mic_inputs[0].get("source_job_uid") == current_mic_source:
            class_nodes.append(node)
    if len(class_nodes) <= 1 or project_uid is None:
        return []

    actions = []
    for node in sorted(class_nodes, key=lambda item: int(str(item["cryosparc_job_uid"])[1:])):
        registry_actions, _ = build_registry_next_actions(
            workflow_state,
            node,
            candidate_job_types={"select_2D"},
        )
        actions.extend(
            action for action in registry_actions
            if action.get("job_type") == "select_2D"
        )
    return actions


def apply_auto_select_2d_policy(
    action: dict[str, Any],
    workflow_state: dict[str, Any],
) -> dict[str, Any]:
    """Make Select 2D autonomous while requiring an explicit model class list."""
    if action.get("job_type") != "select_2D":
        return action
    class_count = class_average_count(action, workflow_state)
    if class_count is None:
        return action
    enriched = dict(action)
    template = dict(enriched.get("parameter_template") or {})
    template["selected_templates"] = {
        "type": "string",
        "required": True,
        "description": (
            "Comma-separated Class 2D class IDs explicitly selected by the model from "
            "the labelled contact sheet and CryoSift evidence."
        ),
    }
    enriched["parameter_template"] = template
    defaults = dict(enriched.get("default_parameters") or {})
    # Do not synthesize a class list from retention fractions or resolution ranks.
    # The decision model must supply the visual/CryoSift-informed selection itself.
    defaults.pop("selected_templates", None)
    enriched["default_parameters"] = defaults
    enriched["execution_mode"] = "create_job"
    enriched["approval_required"] = False
    enriched["approval_reasons"] = []
    enriched["auto_policy"] = {
        "name": "model_selected_templates",
        "class_count": class_count,
        "selection_source": "model_visual_and_cryosift_evidence",
        "note": (
            "No retention fraction, class-resolution ranking, or class-index fallback is "
            "applied. The model must explicitly provide selected_templates."
        ),
    }
    metadata = dict(enriched.get("job_spec_metadata") or {})
    metadata.update({
        "interactive": False,
        "requires_approval": False,
        "queue_without_lane": True,
    })
    enriched["job_spec_metadata"] = metadata
    enriched["description"] = (
        f"Select 2D classes from {class_count} labelled class averages. The model must "
        "explicitly provide selected_templates using visual and CryoSift evidence."
    )
    return enriched


def class_average_count(
    action: dict[str, Any],
    workflow_state: dict[str, Any],
) -> int | None:
    """Find the available class-average count for a select_2D candidate."""
    template_sources = (action.get("required_inputs") or {}).get("templates") or []
    for source in template_sources:
        node = find_node(workflow_state, source.get("source_job_uid", ""))
        if node is None:
            continue
        output = node.get("outputs", {}).get(source.get("source_output")) or {}
        count = output.get("num_items")
        if isinstance(count, int) and count > 0:
            return count
    return None


def build_synthetic_next_actions(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    project_uid: str | None,
    dataset_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create safe next-step candidates when no child job exists yet."""
    if current_node["status"] != "completed":
        return [], []
    if current_node["job_type"] == "homo_abinit":
        return build_abinitio_baseline_refine_action(current_node, workflow_state)
    if current_node["job_type"] == "homo_refine_new":
        return build_parallel_homo_refine_actions(current_node)
    if current_node["job_type"] == "class_2D_new":
        pending_trials = build_pending_class_2d_trial_candidates(workflow_state, current_node)
        if pending_trials:
            return pending_trials, []
        fallback = build_extract_fallback_candidate(workflow_state, current_node)
        if fallback["available"]:
            return [fallback], []
        return [], [fallback]
    micrograph_source = micrograph_source_from_node(current_node)
    if micrograph_source:
        template_source = find_available_template_source(
            workflow_state,
            current_node,
        )
        actions = [
            build_synthetic_blob_picker_candidate(
                current_node,
                micrograph_source,
            )
        ]
        template_picker = build_synthetic_template_picker_candidate(
            current_node,
            micrograph_source,
            template_source,
        )
        blocked = []
        if template_picker["available"]:
            actions.append(template_picker)
        else:
            blocked.append(template_picker)
        return actions, blocked
    if current_node["job_type"] != "select_2D":
        return [], []
    particles = current_node["outputs"].get("particles_selected")
    if not particles or not particles["available"]:
        return [], [
            build_synthetic_homo_refine_candidate(
                current_node,
                required_inputs={},
                blocked_by=["Current select_2D job has no particles_selected output."],
            )
        ]

    required_inputs = {
        "particles": [
            {
                "source_workflow_node_id": current_node["workflow_node_id"],
                "source_logical_node_id": current_node["logical_node_id"],
                "source_job_uid": current_node["cryosparc_job_uid"],
                "source_output": "particles_selected",
                "result_names": particles["result_names"],
            }
        ],
    }
    template_refine = find_template_repick_refinement_source(workflow_state, current_node)
    if template_refine is not None:
        volume = output_source(template_refine, "volume")
        if volume is not None:
            action = build_synthetic_homo_refine_candidate(
                current_node,
                required_inputs={**required_inputs, "volume": [volume]},
                blocked_by=[],
                description=(
                    "Template RePick continuation: refine the newly selected particles against "
                    "the completed source refinement volume. Do not run Ab-initio because this "
                    "branch already has a validated 3D reference."
                ),
                default_parameters={
                    "refine_symmetry": resolve_dataset_symmetry(dataset_info) or "C1",
                },
            )
            action["template_repick_refinement"] = True
            action["template_repick_source_refine_job_uid"] = template_refine["cryosparc_job_uid"]
            return ([action], []) if action["available"] else ([], [action])
    action = build_synthetic_homo_abinit_candidate(
        current_node,
        required_inputs=required_inputs,
        blocked_by=[],
        symmetry=resolve_dataset_symmetry(dataset_info),
    )
    return ([action], []) if action["available"] else ([], [action])


def build_abinitio_baseline_refine_action(
    current_node: dict[str, Any],
    workflow_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Offer the ab-initio subset and the selected full stack for baseline refinement."""
    particles = current_node.get("outputs", {}).get("particles_class_0")
    volume = current_node.get("outputs", {}).get("volume_class_0")
    required_inputs: dict[str, list[dict[str, Any]]] = {}
    blocked_by: list[str] = []
    if particles and particles.get("available"):
        required_inputs["particles"] = [{
            "source_workflow_node_id": current_node["workflow_node_id"],
            "source_logical_node_id": current_node["logical_node_id"],
            "source_job_uid": current_node["cryosparc_job_uid"],
            "source_output": "particles_class_0",
            "result_names": particles.get("result_names") or [],
        }]
    else:
        blocked_by.append("Ab-initio job has no available particles_class_0 output.")
    if volume and volume.get("available"):
        required_inputs["volume"] = [{
            "source_workflow_node_id": current_node["workflow_node_id"],
            "source_logical_node_id": current_node["logical_node_id"],
            "source_job_uid": current_node["cryosparc_job_uid"],
            "source_output": "volume_class_0",
            "result_names": volume.get("result_names") or [],
        }]
    else:
        blocked_by.append("Ab-initio job has no available volume_class_0 output.")
    action = build_synthetic_homo_refine_candidate(
        current_node,
        required_inputs=required_inputs,
        blocked_by=blocked_by,
        description=(
            "Baseline homogeneous refinement using only the Ab-initio class-0 "
            "particle subset and its volume. This excludes any Select 2D particles "
            "left unused by single-class Ab-initio."
        ),
        default_parameters={
            "refine_symmetry": current_node.get("key_parameters", {}).get(
                "abinit_symmetry", "C1"
            )
        },
    )
    actions = [action] if action["available"] else []
    blocked = [] if action["available"] else [action]

    selected_node = find_ancestor_node(workflow_state, current_node, "select_2D") if workflow_state else None
    selected = (selected_node or {}).get("outputs", {}).get("particles_selected") or {}
    selected_count = selected.get("num_items")
    subset_count = (particles or {}).get("num_items")
    if selected.get("available") and (
        not isinstance(selected_count, int)
        or not isinstance(subset_count, int)
        or selected_count > subset_count
    ) and "volume" in required_inputs:
        full_source = {
            "source_workflow_node_id": selected_node["workflow_node_id"],
            "source_logical_node_id": selected_node["logical_node_id"],
            "source_job_uid": selected_node["cryosparc_job_uid"],
            "source_output": "particles_selected",
            "result_names": selected.get("result_names") or [],
        }
        full_action = build_synthetic_homo_refine_candidate(
            current_node,
            required_inputs={"particles": [full_source], "volume": required_inputs["volume"]},
            blocked_by=[],
            description=(
                "Refine all particles selected by upstream Select 2D using the class-0 "
                "Ab-initio volume as the initial reference. Select 2D has "
                f"{selected_count if isinstance(selected_count, int) else 'unknown'} particles; "
                f"Ab-initio used {subset_count if isinstance(subset_count, int) else 'unknown'}. "
                "Prefer this when the omitted selected particles are likely useful; the "
                "full set may also contain lower-quality particles, so compare the evidence."
            ),
            default_parameters=action["default_parameters"],
        )
        full_action["action_id"] = f"forward_{current_node['cryosparc_job_uid']}_homo_refine_all_selected"
        full_action["workflow_node_id"] = f"{current_node['workflow_node_id']}:homo_refine_all_selected"
        full_action["particle_source_comparison"] = {
            "selected_2d_count": selected_count,
            "abinitio_used_count": subset_count,
            "abinitio_unused_count": (
                selected_count - subset_count
                if isinstance(selected_count, int) and isinstance(subset_count, int)
                else None
            ),
        }
        actions.append(full_action)
    return actions, blocked



def build_extract_fallback_candidate(
    workflow_state: dict[str, Any],
    class2d_node: dict[str, Any],
) -> dict[str, Any]:
    """Offer re-extraction when 2D evidence suggests particles may be clipped."""
    extract_node = find_previous_job_node(
        workflow_state,
        class2d_node,
        job_type="extract_micrographs_multi",
    )
    action_id = f"fallback_{class2d_node['cryosparc_job_uid']}_extract_micrographs_multi"
    if extract_node is None:
        return {
            "action_id": action_id,
            "action_type": "forward",
            "workflow_node_id": f"{class2d_node['workflow_node_id']}:extract_fallback",
            "reference_job_uid": None,
            "reference_status": None,
            "job_type": "extract_micrographs_multi",
            "description": "Re-extract particles with a larger box after reviewing 2D classes.",
            "execution_mode": "create_job",
            "available": False,
            "blocked_by": ["No previous extract_micrographs_multi source was found."],
            "required_inputs": {},
            "missing_required_inputs": ["micrographs", "particles"],
            "parameter_template": get_parameter_template("extract_micrographs_multi"),
            "default_parameters": {},
        }

    required_inputs = {
        name: connections
        for name, connections in (extract_node.get("inputs") or {}).items()
        if name in {"micrographs", "particles"} and connections
    }
    missing = [name for name in ("micrographs", "particles") if name not in required_inputs]
    defaults = {
        name: value
        for name, value in (extract_node.get("key_parameters") or {}).items()
        if name in {"box_size_pix", "compute_num_gpus"}
    }
    blocked_by = []
    if missing:
        blocked_by.append(
            "Previous extraction is missing required input(s): " + ", ".join(missing)
        )
    return {
        "action_id": action_id,
        "action_type": "forward",
        "workflow_node_id": f"{class2d_node['workflow_node_id']}:extract_fallback",
        "reference_job_uid": extract_node["cryosparc_job_uid"],
        "reference_status": extract_node["status"],
        "job_type": "extract_micrographs_multi",
        "description": (
            "Fallback re-extraction after Class 2D review. If particles appear clipped "
            "or truncated, increase box_size_pix before rerunning 2D classification."
        ),
        "execution_mode": "create_job",
        "available": not blocked_by,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "missing_required_inputs": missing,
        "parameter_template": build_parameter_template(extract_node),
        "default_parameters": defaults,
        "fallback_policy": {
            "trigger_stage": "classification_2d",
            "reason": "2D class images may reveal particle clipping caused by an undersized extraction box.",
            "model_instruction": (
                "Use this fallback only when 2D visual evidence indicates clipping; "
                "increase box_size_pix enough to include the full particle and rerun classification."
            ),
        },
    }


def find_previous_job_node(
    workflow_state: dict[str, Any],
    node: dict[str, Any],
    job_type: str,
) -> dict[str, Any] | None:
    """Find the nearest earlier job of a requested type in the workspace DAG."""
    parents = set(node.get("parent_job_uids") or [])
    nodes = sorted(
        workflow_state.get("nodes") or [],
        key=lambda item: int(str(item.get("cryosparc_job_uid", "J"))[1:])
        if str(item.get("cryosparc_job_uid", "J"))[1:].isdigit()
        else -1,
        reverse=True,
    )
    for candidate in nodes:
        if candidate.get("job_type") != job_type:
            continue
        if candidate.get("cryosparc_job_uid") in parents:
            return candidate
    for candidate in nodes:
        if candidate.get("job_type") == job_type and candidate.get("status") == "completed":
            return candidate
    return None

def merge_candidates(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate reference/synthetic and Registry candidates by job type."""
    seen = {action.get("job_type") for action in existing}
    return existing + [
        action for action in additions if action.get("job_type") not in seen
    ]


def micrograph_source_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return a usable micrograph-like output from a completed workflow node."""
    if node["status"] != "completed":
        return None
    output_names = ("exposures", "micrographs", "imported_micrographs")
    # Exposure Sets Tool split outputs are separate datasets. Prefer split_0
    # over its full-set/remainder siblings so fixed-subset trials cannot
    # accidentally expand back to the full acquisition set.
    split_output = node["outputs"].get("split_0")
    if split_output and split_output.get("available") and "micrograph_blob" in (split_output.get("result_names") or []):
        output_names = ("split_0", *output_names)
    for output_name in output_names:
        output = node["outputs"].get(output_name)
        if not output or not output["available"]:
            continue
        result_names = output.get("result_names") or []
        if "micrograph_blob" not in result_names:
            continue
        return {
            "source_workflow_node_id": node["workflow_node_id"],
            "source_logical_node_id": node["logical_node_id"],
            "source_job_uid": node["cryosparc_job_uid"],
            "source_output": output_name,
            "result_names": result_names,
        }
    return None


def find_available_template_source(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a usable template-like output from the current workflow."""
    current_source = template_source_from_node(current_node)
    if current_source:
        return current_source
    for node in workflow_state["nodes"]:
        if node["cryosparc_job_uid"] == current_node["cryosparc_job_uid"]:
            continue
        source = template_source_from_node(node)
        if source:
            return source
    return None


def template_source_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return a template-like output from a completed workflow node."""
    if node["status"] != "completed":
        return None
    for output_name, output in node["outputs"].items():
        if not output or not output["available"]:
            continue
        if output_name not in {"templates", "templates_selected"}:
            continue
        result_names = output.get("result_names") or []
        return {
            "source_workflow_node_id": node["workflow_node_id"],
            "source_logical_node_id": node["logical_node_id"],
            "source_job_uid": node["cryosparc_job_uid"],
            "source_output": output_name,
            "result_names": result_names,
        }
    return None


def build_synthetic_blob_picker_candidate(
    current_node: dict[str, Any],
    micrograph_source: dict[str, Any],
) -> dict[str, Any]:
    """Build a generic blob_picker_gpu candidate from available micrographs."""
    job_type = "blob_picker_gpu"
    return {
        "action_id": f"forward_{current_node['cryosparc_job_uid']}_blob_picker_gpu",
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:blob_picker_gpu",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": (
            "Create a Blob Picker job from the current available micrographs."
        ),
        "execution_mode": "create_job",
        "available": True,
        "blocked_by": [],
        "required_inputs": {"micrographs": [micrograph_source]},
        "missing_required_inputs": [],
        "parameter_template": get_parameter_template(job_type),
        "default_parameters": {},
    }


def build_synthetic_template_picker_candidate(
    current_node: dict[str, Any],
    micrograph_source: dict[str, Any],
    template_source: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a Template Picker candidate when templates are available."""
    job_type = "template_picker_gpu"
    required_inputs = {
        "micrographs": [micrograph_source],
        "templates": [template_source] if template_source else [],
    }
    blocked_by = []
    missing_required_inputs = []
    if not template_source:
        blocked_by.append(
            "Required input templates has no available source in the current branch."
        )
        missing_required_inputs.append("templates")
    return {
        "action_id": (
            f"forward_{current_node['cryosparc_job_uid']}_template_picker_gpu"
            if template_source
            else f"blocked_{current_node['cryosparc_job_uid']}_template_picker_gpu"
        ),
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:template_picker_gpu",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": (
            "Template Picker requires templates in addition to micrographs. "
            "Its min_distance parameter is a unitless multiple of particle diameter "
            "(normally 0.8), not a distance in Angstroms."
        ),
        "execution_mode": "create_job",
        "available": template_source is not None,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "missing_required_inputs": missing_required_inputs,
        "parameter_template": get_parameter_template(job_type),
        # Leave picker controls unset so CryoSPARC applies its own defaults.
        "default_parameters": {},
    }


def build_parallel_homo_refine_actions(
    current_node: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Allow another homogeneous refinement using the same upstream inputs."""
    required_inputs = {
        name: connections
        for name, connections in current_node["inputs"].items()
        if name in {"particles", "volume", "mask"} and connections
    }
    blocked_by = []
    if "particles" not in required_inputs:
        blocked_by.append("Current homo_refine_new job has no particles input.")
    if "volume" not in required_inputs:
        blocked_by.append("Current homo_refine_new job has no volume input.")

    action = build_synthetic_homo_refine_candidate(
        current_node,
        required_inputs=required_inputs,
        blocked_by=blocked_by,
        description=(
            "Create another homogeneous refinement job in parallel using the "
            "same upstream particles and volume inputs."
        ),
    )
    return ([action], []) if action["available"] else ([], [action])


def build_synthetic_homo_refine_candidate(
    current_node: dict[str, Any],
    required_inputs: dict[str, list[dict[str, Any]]],
    blocked_by: list[str],
    description: str | None = None,
    default_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a synthetic homo_refine_new creation candidate."""
    job_type = "homo_refine_new"
    parameter_template = get_parameter_template(job_type)
    known_symmetry = (default_parameters or {}).get("refine_symmetry")
    if known_symmetry:
        parameter_template["refine_symmetry"]["default"] = known_symmetry
        parameter_template["refine_symmetry"]["enum"] = [known_symmetry]
    return {
        "action_id": f"forward_{current_node['cryosparc_job_uid']}_homo_refine_new",
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:homo_refine_new",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": description or (
            "Create a new homogeneous refinement job from selected particles "
            "and an available imported volume."
        ),
        "execution_mode": "create_job",
        "available": not blocked_by,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "parameter_template": parameter_template,
        "default_parameters": default_parameters or {},
    }


def build_synthetic_homo_abinit_candidate(
    current_node: dict[str, Any],
    required_inputs: dict[str, list[dict[str, Any]]],
    blocked_by: list[str],
    symmetry: str | None = None,
) -> dict[str, Any]:
    """Build an ab-initio candidate from curated Select 2D particles."""
    job_type = "homo_abinit"
    parameter_template = get_parameter_template(job_type)
    if symmetry:
        parameter_template["abinit_symmetry"]["default"] = symmetry
        parameter_template["abinit_symmetry"]["enum"] = [symmetry]
    return {
        "action_id": f"forward_{current_node['cryosparc_job_uid']}_homo_abinit",
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:homo_abinit",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": (
            f"Create one {symmetry or 'C1'} ab-initio reconstruction from all selected "
            "particles before baseline homogeneous refinement."
        ),
        "execution_mode": "create_job",
        "available": not blocked_by,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "parameter_template": parameter_template,
        "default_parameters": {"abinit_symmetry": symmetry} if symmetry else {},
    }


def resolve_dataset_symmetry(dataset_info: dict[str, Any] | None) -> str | None:
    """Return a user-supplied point group, never an inferred free-text label."""
    normalized = normalize_dataset_info(dataset_info or {})
    metadata = normalized.get("dataset_metadata") or {}
    values = [
        normalized.get("symmetry"),
        normalized.get("point_group"),
        metadata.get("symmetry") if isinstance(metadata, dict) else None,
        metadata.get("point_group") if isinstance(metadata, dict) else None,
    ]
    for value in values:
        if isinstance(value, str):
            point_group = value.strip().upper()
            if re.fullmatch(r"(?:C[1-9][0-9]*|D[2-9][0-9]*|T|O|I)", point_group):
                return point_group
    return None


def unknown_dataset_symmetry(dataset_info: dict[str, Any] | None) -> bool:
    normalized = normalize_dataset_info(dataset_info or {})
    return normalized.get("symmetry_status") == "unknown" and resolve_dataset_symmetry(normalized) is None


def allow_unknown_symmetry(action: dict[str, Any]) -> dict[str, Any]:
    """An inherited point group is a starting hypothesis, not a user constraint."""
    import copy
    action = copy.deepcopy(action)
    if action.get("symmetry_comparison"):
        return action
    for name in ("abinit_symmetry", "refine_symmetry"):
        spec = (action.get("parameter_template") or {}).get(name)
        if not isinstance(spec, dict):
            continue
        spec.pop("enum", None)
        spec.pop("const", None)
        spec["pattern"] = r"(?:C[1-9][0-9]*|D[2-9][0-9]*|T|O|I)"
        spec["description"] = (
            "Dataset symmetry is unknown. C1 or the inherited symmetry is only a starting "
            "hypothesis. You may choose another valid point group with map/ProSHADE evidence; "
            "preserve the original branch, compare structure and corrected FSC, and explain."
        )
        action["symmetry_status"] = "unknown"
    if action.get("symmetry_status") == "unknown":
        action["description"] = action.get("description", "").replace(
            "Keep symmetry and CTF settings comparable;",
            "Keep CTF settings comparable; inherited symmetry may be reconsidered with evidence;",
        )
        action["description"] = action.get("description", "") + (
            " Symmetry is unknown, not known C1. C1 is a conservative start; inherited "
            "symmetry is not fixed. An evidence-backed symmetry comparison is allowed, "
            "but do not repeat unchanged refinement or assume higher symmetry is better."
        )
    return action


def find_completed_volume_source(
    workflow_state: dict[str, Any],
    project_uid: str | None,
) -> dict[str, Any] | None:
    """Find a completed imported volume in the workspace or parent project."""
    for node in workflow_state["nodes"]:
        source = volume_source_from_node(node)
        if source:
            return source
    if not project_uid or os.getenv("CRYOAGENT_BENCHMARK_BUDGET_FILE"):
        return None
    try:
        cs = cryosparc_client()
        jobs = sorted(
            cs.find_jobs(project_uid=project_uid),
            key=lambda job: int(job.uid[1:]) if job.uid.startswith("J") and job.uid[1:].isdigit() else 10**12,
        )
    except Exception:
        return None
    for job in jobs:
        if job.status != "completed":
            continue
        for output_name, output in job.outputs.items():
            if output_name.startswith("imported_volume") and output.num_items > 0:
                return {
                    "source_workflow_node_id": job.uid,
                    "source_logical_node_id": None,
                    "source_job_uid": job.uid,
                    "source_output": output_name,
                    "result_names": sorted(
                        result.name
                        for result in output.results
                    ),
                }
    return None


def volume_source_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return a volume output from a normalized completed job node."""
    if node["status"] != "completed":
        return None
    for output_name, output in node["outputs"].items():
        if output_name.startswith("imported_volume") and output["available"]:
            return {
                "source_workflow_node_id": node["workflow_node_id"],
                "source_logical_node_id": node["logical_node_id"],
                "source_job_uid": node["cryosparc_job_uid"],
                "source_output": output_name,
                "result_names": output["result_names"],
            }
    return None


def build_candidate_action(
    workflow_state: dict[str, Any],
    target_node: dict[str, Any],
    action_type: str,
) -> dict[str, Any]:
    """Build one candidate action and record why it is unavailable, if blocked."""
    required_inputs: dict[str, list[dict[str, Any]]] = {}
    blocked_by: list[str] = []

    for input_name, connections in target_node["inputs"].items():
        required_inputs[input_name] = connections
        if not connections:
            blocked_by.append(
                f"Required input {input_name} has no available source."
            )
        for connection in connections:
            source_node = find_node(
                workflow_state,
                connection["source_job_uid"],
            )
            if source_node is None:
                blocked_by.append(
                    f"Missing source job {connection['source_job_uid']}"
                )
                continue
            if source_node["status"] != "completed":
                blocked_by.append(
                    f"Source job {source_node['cryosparc_job_uid']} is "
                    f"{source_node['status']}, not completed"
                )
            source_output = source_node["outputs"].get(
                connection["source_output"]
            )
            if not source_output or not source_output["available"]:
                blocked_by.append(
                    f"Output {connection['source_job_uid']}."
                    f"{connection['source_output']} is unavailable"
                )

    parameter_template = build_parameter_template(target_node)

    return {
        "action_id": f"{action_type}_{target_node['cryosparc_job_uid']}",
        "action_type": action_type,
        "workflow_node_id": target_node["workflow_node_id"],
        "reference_job_uid": target_node["cryosparc_job_uid"],
        "reference_status": target_node["status"],
        "job_type": target_node["job_type"],
        "description": (
            f"Reproduce the validated P2-W3 workflow node "
            f"{target_node['cryosparc_job_uid']}."
        ),
        "execution_mode": "dry_run_only",
        "available": not blocked_by,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "parameter_template": parameter_template,
        "default_parameters": target_node["key_parameters"],
    }


def build_parameter_template(
    target_node: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Combine static parameter rules with defaults copied from the reference job."""
    template = get_parameter_template(target_node["job_type"])
    for name, value in target_node["key_parameters"].items():
        template.setdefault(name, {"type": infer_json_type(value)})
        template[name]["default"] = value
    return template


def validate_model_decision_payload(
    payload: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a model decision without creating or enqueueing CryoSPARC jobs."""
    decision, schema_issues = parse_model_decision(payload)
    if decision is None:
        result = ValidationResult(
            success=False,
            valid_schema=False,
            valid_actions=False,
            issues=schema_issues,
        )
        return result.model_dump()

    result = validate_decision_against_registry(
        decision,
        candidate_actions=candidate_actions,
    )

    return result.model_dump()


def execute_model_decision_payload(
    payload: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
    dry_run: bool = True,
    project_uid: str | None = None,
    workspace_uid: str | None = None,
    allow_approval_required_create: bool = False,
) -> dict[str, Any]:
    """
    Convert a validated decision into an execution plan.

    The current implementation is intentionally dry-run first. It verifies the
    decision and returns the jobs that would be executed, but it does not create
    or queue CryoSPARC jobs.
    """
    validation = validate_model_decision_payload(
        payload,
        candidate_actions=candidate_actions,
    )
    if not validation["success"]:
        return {
            "success": False,
            "dry_run": dry_run,
            "execution_mode": "validation_failed",
            "decision_type": validation.get("decision_type"),
            "validation": validation,
            "execution_plan": None,
            "execution_results": [],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
        }

    try:
        execution_plan = build_execution_plan(payload, validation, candidate_actions)
    except (ValueError, TypeError, KeyError) as exc:
        return dict(success=False, dry_run=dry_run, execution_mode='invalid_input_connections',
                    validation=validation, execution_results=[], issues=[dict(
                        severity='error', code='invalid_input_connections', message=str(exc), path='connections')])
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "execution_mode": "dry_run",
            "decision_type": validation["decision_type"],
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [],
            "warnings": validation["warnings"],
            "message": "Dry run only; no CryoSPARC jobs were created or queued.",
        }

    if validation["decision_type"] == "request_input":
        return {
            "success": True,
            "dry_run": dry_run,
            "execution_mode": "awaiting_input",
            "decision_type": "request_input",
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [],
            "warnings": validation["warnings"],
            "message": "Additional input is required; no CryoSPARC job was created or queued.",
        }

    if not project_uid or not workspace_uid:
        return {
            "success": False,
            "dry_run": False,
            "execution_mode": "missing_execution_context",
            "decision_type": validation["decision_type"],
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [
                {
                    "severity": "error",
                    "code": "missing_execution_context",
                    "message": "project_uid and workspace_uid are required for live execution.",
                    "path": None,
                }
            ],
            "warnings": validation["warnings"],
        }

    execution_results = [
        execute_job_action(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            planned_action=action,
            dry_run=False,
            allow_approval_required_create=allow_approval_required_create,
        )
        for action in execution_plan["actions"]
    ]
    return {
        "success": all(result["success"] for result in execution_results),
        "dry_run": False,
        "execution_mode": "live_execution",
        "model_visible": False,
        "next_model_input": None,
        "decision_type": validation["decision_type"],
        "validation": validation,
        "execution_plan": execution_plan,
        "execution_results": execution_results,
        "issues": [
            issue
            for result in execution_results
            for issue in result.get("issues", [])
        ],
        "warnings": validation["warnings"],
        "message": (
            "Live execution status is MCP-internal. Wait for the created job to "
            "complete, then call get_job_result_package before asking the model "
            "for the next decision."
        ),
    }


def build_execution_plan(
    payload: dict[str, Any],
    validation: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize validated decisions into a future execution-ready plan."""
    decision_type = validation["decision_type"]
    candidates_by_id = {
        action["action_id"]: action
        for action in candidate_actions or []
    }

    if decision_type in {"forward", "branch"}:
        actions = []
        for idx, action in enumerate(validation["resolved_actions"]):
            base_action = {
                "plan_step": idx + 1,
                "action_id": action["action_id"],
                "action_type": action["action_type"],
                "workflow_node_id": action["workflow_node_id"],
                "job_type": action["job_type"],
                "execution_mode": action["execution_mode"],
                "mcp_tool_name": action["mcp_tool_name"],
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": action["resolved_parameters"],
                "connections": action.get("connections"),
                "rollback_target": None,
                "status": "planned",
            }
            actions.append(
                plan_job_action(
                    base_action,
                    candidate_action=candidates_by_id.get(action["action_id"]),
                )
            )
        return make_execution_plan(payload, validation, actions)

    if decision_type == "rollback":
        rollback_target = payload.get("rollback_target")
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "rollback",
                "workflow_node_id": (
                    rollback_target or {}
                ).get("workflow_node_id"),
                "job_type": (rollback_target or {}).get("job_type"),
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": True,
                "approval_reasons": ["rollback_decision"],
                "resolved_parameters": {},
                "rollback_target": payload.get("rollback_target"),
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    if decision_type == "stop":
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "stop",
                "workflow_node_id": None,
                "job_type": None,
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": {},
                "rollback_target": None,
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    if decision_type == "request_input":
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "request_input",
                "workflow_node_id": None,
                "job_type": None,
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": {},
                "rollback_target": None,
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    return make_execution_plan(payload, validation, [])


def make_execution_plan(
    payload: dict[str, Any],
    validation: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the stable top-level ExecutionPlan object."""
    approval_reasons = sorted(
        {
            reason
            for action in actions
            for reason in action["approval_reasons"]
        }
    )
    plan_payload = {
        "decision_type": validation["decision_type"],
        "actions": actions,
    }
    return {
        "plan_id": content_hash("plan", plan_payload),
        "plan_version": "1.0",
        "status": "planned",
        "dry_run_only": True,
        "decision_type": validation["decision_type"],
        "action_count": len(actions),
        "approval_required": any(
            action["approval_required"]
            for action in actions
        ),
        "approval_reasons": approval_reasons,
        "actions": actions,
        "execution_results": [],
    }


def validate_decision_against_registry(
    decision: ModelDecision,
    candidate_actions: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    """Validate actions, using candidates as optional helpers."""
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    resolved_actions: list[ResolvedAction] = []

    if decision.decision_type in {"rollback", "stop", "request_input"}:
        return ValidationResult(
            success=True,
            valid_schema=True,
            valid_actions=True,
            decision_type=decision.decision_type,
            dry_run=True,
        )

    candidates_by_id = {
        action["action_id"]: action
        for action in candidate_actions or []
    }

    for idx, action in enumerate(decision.selected_actions):
        action_issues, action_warnings, resolved_action = (
            validate_action_against_candidates(
                action,
                candidates_by_id,
                idx,
            )
        )
        issues.extend(action_issues)
        warnings.extend(action_warnings)
        if resolved_action:
            resolved_actions.append(resolved_action)

    return ValidationResult(
        success=not issues,
        valid_schema=True,
        valid_actions=not issues,
        decision_type=decision.decision_type,
        dry_run=True,
        issues=issues,
        warnings=warnings,
        resolved_actions=resolved_actions,
    )


def validate_action_against_candidates(
    action: Action,
    candidates_by_id: dict[str, dict[str, Any]],
    index: int,
) -> tuple[list[ValidationIssue], list[ValidationIssue], ResolvedAction | None]:
    """Validate one selected action and resolve its final parameters."""
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    path = f"selected_actions.{index}"
    candidate = candidates_by_id.get(action.action_id or "")

    if candidate is not None:
        if candidate.get('inspect_review'):
            from inspect_review import threshold_error
            error = threshold_error(action.parameters, candidate)
            if error:
                issues.append(ValidationIssue(code='invalid_inspect_review', message=error, path=path))
            if action.connections is not None:
                from job_executor import build_connections, build_explicit_connections
                try:
                    same = build_explicit_connections(action.connections) == build_connections(candidate['required_inputs'])
                except (TypeError, ValueError, KeyError):
                    same = False
                if not same:
                    issues.append(ValidationIssue(code='inspect_review_inputs_changed',
                        message='Re-inspect the original picker candidates, not the accepted subset.', path=path))
        if candidate.get("symmetry_comparison") or candidate.get('optimization_contract'):
            from job_executor import build_connections, build_explicit_connections
            if action.connections is not None:
                try:
                    same = build_explicit_connections(action.connections) == build_connections(candidate["required_inputs"])
                except (TypeError, ValueError, KeyError):
                    same = False
                if not same:
                    issues.append(ValidationIssue(code="symmetry_comparison_inputs_changed",
                        message="Keep this controlled intervention's exact source inputs.", path=path))
            allowed = candidate.get("default_parameters") or {}
            for name, value in action.parameters.items():
                # Cache placement is operational; its declared schema is checked below.
                if name == "compute_use_ssd" and name in candidate.get("parameter_template", {}):
                    continue
                if (candidate.get('optimization_contract') or name not in {"compute_num_gpus", "compute_num_cpus", "compute_num_threads"}) and (
                        name not in allowed or value != allowed[name]):
                    issues.append(ValidationIssue(code="symmetry_comparison_parameters_changed",
                        message=f"Controlled intervention must retain the specified setting for {name}.", path=path))
        if candidate.get("available") is False:
            issues.append(
                ValidationIssue(
                    code="action_unavailable",
                    message=(
                        f"action_id {action.action_id!r} is blocked: "
                        + "; ".join(candidate.get("blocked_by") or ["not available"])
                    ),
                    path=path,
                )
            )
        for field_name in ("action_type", "workflow_node_id", "job_type"):
            expected = candidate[field_name]
            actual = getattr(action, field_name)
            if actual != expected:
                issues.append(
                    ValidationIssue(
                        code=f"{field_name}_mismatch",
                        message=(
                            f"{field_name} must be {expected!r} for "
                            f"action_id {action.action_id!r}"
                        ),
                        path=f"{path}.{field_name}",
                    )
                )
        from job_executor import build_connections, build_explicit_connections, validate_input_connections
        try:
            connections = build_explicit_connections(action.connections)
            if connections is None:
                connections = build_connections(candidate.get('required_inputs') or {})
            validate_input_connections(action.job_type, connections,
                (candidate.get('job_spec_metadata') or {}).get('required_input_counts'))
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(ValidationIssue(code='invalid_input_connections', message=str(exc), path=path + '.connections'))
        parameter_template = candidate["parameter_template"]
        execution_mode = candidate["execution_mode"]
    else:
        if action.job_type in {'agent_reference_union', 'agent_scale_cleanup'}:
            issues.append(ValidationIssue(code='particle_operation_requires_bounded_candidate',
                message='Particle operations require an explicit source-scoped optimization candidate.', path=path))
        if action.job_type == "create_templates":
            issues.append(
                ValidationIssue(
                    code="create_templates_requires_gated_candidate",
                    message=(
                        "create_templates must use an available, evidence-gated Template RePick candidate; "
                        "generic Create Template actions are not permitted."
                    ),
                    path=path,
                )
            )
        parameter_template = get_parameter_template(action.job_type)
        execution_mode = "create_job"
        warnings.append(
            ValidationIssue(
                severity="warning",
                code="generic_action_without_candidate",
                message=(
                    f"job_type {action.job_type!r} did not match an internal "
                    "candidate; MCP will plan it as a generic CryoSPARC job."
                ),
                path=path,
            )
        )

    parameter_issues, resolved_parameters = validate_parameters(
        action.parameters,
        parameter_template,
        path=f"{path}.parameters",
    )
    gpu_cap_raw = os.getenv("CRYOAGENT_GPU_COUNT_CAP")
    if gpu_cap_raw and "compute_num_gpus" in resolved_parameters:
        try:
            gpu_cap = max(1, int(gpu_cap_raw))
        except ValueError:
            gpu_cap = None
        if gpu_cap is not None and resolved_parameters["compute_num_gpus"] > gpu_cap:
            resolved_parameters["compute_num_gpus"] = gpu_cap
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="gpu_count_capped",
                    message=f"compute_num_gpus capped to {gpu_cap} by CRYOAGENT_GPU_COUNT_CAP",
                    path=f"{path}.parameters.compute_num_gpus",
                )
            )
    for parameter_issue in parameter_issues:
        if parameter_issue.code == "unknown_parameter":
            parameter_issue.severity = "warning"
            warnings.append(parameter_issue)
        else:
            issues.append(parameter_issue)

    if issues:
        return issues, warnings, None

    return (
        issues,
        warnings,
        ResolvedAction(
            action_id=action.action_id or f"generic_{index}_{action.job_type}",
            action_type=action.action_type,
            workflow_node_id=action.workflow_node_id or f"generic:{action.job_type}",
            job_type=action.job_type,
            execution_mode=execution_mode,
            resolved_parameters=resolved_parameters,
            connections=action.connections,
            mcp_tool_name=None,
        ),
    )


def validate_parameters(
    parameters: dict[str, Any],
    template: dict[str, dict[str, Any]],
    path: str,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Merge defaults with model-supplied parameters and validate the result."""
    issues: list[ValidationIssue] = []
    resolved = {
        name: spec["default"]
        for name, spec in template.items()
        if "default" in spec
    }
    resolved.update(parameters)

    for name, spec in template.items():
        if spec.get("required") and resolved.get(name) is None:
            issues.append(
                ValidationIssue(
                    code="missing_required_parameter",
                    message=f"Missing required parameter {name!r}",
                    path=f"{path}.{name}",
                )
            )

    for name, value in parameters.items():
        if value is None:
            continue
        spec = template.get(name)
        if spec is None:
            issues.append(
                ValidationIssue(
                    code="unknown_parameter",
                    message=f"Parameter {name!r} is not allowed for this action",
                    path=f"{path}.{name}",
                )
            )
            continue

        issues.extend(
            validate_parameter_value(
                name,
                value,
                spec,
                f"{path}.{name}",
            )
        )

    return issues, resolved


def validate_parameter_value(
    name: str,
    value: Any,
    spec: dict[str, Any],
    path: str,
) -> list[ValidationIssue]:
    """Validate one parameter's type and simple numeric constraints."""
    issues: list[ValidationIssue] = []
    expected_type = spec.get("type")

    if "const" in spec and value != spec["const"]:
        issues.append(ValidationIssue(
            code="parameter_const_mismatch",
            message=f"Parameter {name!r} is fixed to {spec['const']} for this workflow phase",
            path=path,
        ))

    if expected_type and not matches_type(value, expected_type):
        return [
            ValidationIssue(
                code="parameter_type_mismatch",
                message=f"Parameter {name!r} must be {expected_type}",
                path=path,
            )
        ]

    if "enum" in spec and value not in spec["enum"]:
        issues.append(
            ValidationIssue(
                code="parameter_enum_mismatch",
                message=f"Parameter {name!r} must be one of {spec['enum']}",
                path=path,
            )
        )

    if isinstance(value, str) and spec.get("pattern") and not re.fullmatch(spec["pattern"], value):
        issues.append(ValidationIssue(
            code="parameter_pattern_mismatch",
            message=f"Parameter {name!r} does not match the permitted format",
            path=path,
        ))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and value < minimum:
            issues.append(
                ValidationIssue(
                    code="parameter_below_minimum",
                    message=f"Parameter {name!r} must be >= {minimum}",
                    path=path,
                )
            )
        if maximum is not None and value > maximum:
            issues.append(
                ValidationIssue(
                    code="parameter_above_maximum",
                    message=f"Parameter {name!r} must be <= {maximum}",
                    path=path,
                )
            )

    return issues


def matches_type(value: Any, expected_type: str) -> bool:
    """Map JSON schema-style type names to Python runtime checks."""
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def infer_json_type(value: Any) -> str:
    """Infer a compact JSON type name from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"

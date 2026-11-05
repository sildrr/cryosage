"""Bounded experiment planning for the autonomous reconstruction runner.

The planner never creates CryoSPARC jobs.  It converts whole-workspace facts
into a small, safe set of actions that the existing V2 executor may consider.
"""
from __future__ import annotations

import json
from typing import Any
from stage_guidance import SupplementalGuidance, last_completed_from_observation

from cryosift_adapter import CRYOSIFT_SCORE_GUIDANCE, CRYOSIFT_EMPTY_GUIDANCE
from selection_safety import SELECTION_SAFETY_GUIDANCE
from visual_decision_contract import (
    EXTRACT_BOX_PREFERENCE_GUIDANCE, REFINEMENT_PREFERENCE_GUIDANCE,
)
from reconstruction_observer import normalize_observation_tasks, READ_ONLY_OBSERVATION_TOOLS
from refinement_optimization import GUIDANCE as OPTIMIZATION_GUIDANCE


PLAN_SCHEMA_VERSION = "reconstruction-experiment-plan-0.2"
UPSTREAM_JOB_TYPES = {
    "blob_picker_gpu", "auto_blob_picker_gpu", "inspect_picks_v2",
    "extract_micrographs_multi", "select_2D", "class_2D_new",
    "micrograph_junk_detector",
}
REFINEMENT_JOB_TYPES = {
    "homo_abinit", "homo_refine_new", "nonuniform_refine_new", "refine_3D_new",
}


def build_planning_prompt(
    observation: dict[str, Any], candidate_actions: list[dict[str, Any]],
    experiment_history: list[dict[str, Any]], max_new_jobs: int,
    *, stage_guidance: SupplementalGuidance | None = None,
    guidance_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return an API-ready, strictly bounded PlanningAgent request."""
    contract = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "decision_type": "request_observations | plan_experiments | wait | request_input | finalize",
        "observation_tasks": [{"task_id": "string", "tool": "allow-listed read-only tool", "arguments": {}}],
        "objective": "string",
        "hypotheses": [{"id": "string", "claim": "string", "evidence": ["workspace fact"]}],
        "approved_action_ids": ["action_id from candidate_actions only"],
        "comparison_metrics": ["string"],
        "stop_conditions": ["string"],
    }
    instruction = (
        "You are a CryoSPARC reconstruction PlanningAgent. Return exactly one JSON object, "
        "without markdown or hidden reasoning. You may only approve action_id values appearing "
        "in candidate_actions. You do not create jobs. When the target resolution is unmet, do "
        "not approve repeated refinement of unchanged particles and unchanged optical settings while any upstream candidate is "
        "available. If a completed Class 2D branch has no attached scale_observer_evidence and "
        "particle diameter or extraction size is relevant, request get_class2d_scale_context for "
        "that branch before planning. Compare its particle-count-weighted 5%/10% radial diameter "
        "range and CTF box estimate with the calibrated full-field image evidence. "
        "During resolution optimization prefer preserving native particle sampling. "
        "bin_size_pix means the absolute Fourier-cropped output edge length in pixels, "
        "not a downsampling factor; setting it to 2 produces 2x2 images. Avoid Fourier "
        "downsampling that discards high-frequency information needed for the target. "
        "At a refinement checkpoint candidate_actions is a bounded intervention menu: "
        "choose exactly one experiment, not a multi-step plan. "
        "To retrieve evidence, return decision_type=request_observations with observation_tasks "
        "and an empty approved_action_ids list, not request_input. Use arguments.job_uid "
        "for an Ab-initio observation; project/workspace scope is supplied by the controller. "
        "After the observation round, choose one experiment using the returned evidence; "
        "do not request a second observation turn. "
        "If dataset symmetry is unknown, do not treat inherited C1 as a user-specified "
        "fact. Use symmetry_evidence when available to consider an allowed, evidence-backed "
        "symmetry comparison; the executor may set the point group. Preserve previous "
        "results, compare corrected FSC and map plausibility, and never repeat unchanged "
        "refinement or impose symmetry merely because it improves FSC. Failed symmetry "
        "analysis is uncertainty, not evidence for C1. "
        "When a symmetry_comparison candidate is available after the C1 baseline, run its "
        "one-time same-input comparison before changing the particle population. This is "
        "a controlled test, not repetition of unchanged refinement. Persistent evidence "
        "is tied to its source job and must not be described as absent merely because the "
        "last step is Select 2D. "
        + CRYOSIFT_SCORE_GUIDANCE + " " + CRYOSIFT_EMPTY_GUIDANCE + " " + SELECTION_SAFETY_GUIDANCE
        + REFINEMENT_PREFERENCE_GUIDANCE
        + OPTIMIZATION_GUIDANCE
        + "When those preferred options are ineligible or contradicted by direct evidence, "
        "consider reclassify_selected_particles for residual heterogeneity. For a poor result, prefer "
        "reselect_2d to remove weak high-score signal classes before escalating to new_box_size, "
        "reinspect_picks, repick_diameter, or template_repick when the evidence supports that "
        "hypothesis. Compare the three Ab-initio views against the views represented by the "
        "Select 2D class set whenever that visual evidence is available. "
        "Do not repeat the same refinement type on the same branch except an explicit "
        "one-time changed-parameter optics_refinement experiment; prioritize "
        "upstream picking or class-quality fixes when that evidence is poor. "
        "For every reselect_2d candidate, its reselection_constraint identifies the previous "
        "selection. Require the execution model to choose a nonempty strict subset of those IDs: "
        "remove lower-quality selected classes and never add previously excluded classes. "
        "If an Ab-initio view "
        "has no plausible corresponding selected 2D class, treat the reconstruction as failed "
        "rather than evidence for another unchanged refine. You may request "
        "get_abinitio_class2d_template_matching_context for a completed Ab-initio job. A negative "
        "selected-minus-excluded NCC means an excluded Class 2D image better matches that view, "
        "which supports reselect_2d; do not use a fixed threshold or treat one ambiguous side view "
        "as conclusive. For the first confirmed Ab-initio/Class 2D mismatch, choose reselect_2d "
        "and remove clearly incomplete, off-center, or density-shifted classes from that existing "
        "Class 2D run. Do not choose reclassify_selected_particles first, because it can repeat "
        "the same incomplete-class selection. If two or more reconstruction attempts at the same "
        "extraction scale show this mismatch, treat new_box_size with a larger box as the leading "
        "hypothesis, while retaining reselect_2d, reinspect_picks, and repick_diameter as valid "
        "alternatives when their evidence is stronger. Prefer "
        "reselect_2d when the "
        "selected 2D classes show off-center particles or density shifts; exclude those classes "
        "more strictly before rebuilding the volume. Choose template_repick only when the current resolution is within 0.5 A "
        "of the target, Ab-initio/Class 2D template matching reports min_best_ncc >= 0.75, and the volume has a "
        "plausible, correct overall shape; otherwise do not "
        "use it because an incorrect reference can reinforce a wrong picking population. When "
        "eligible, prefer it as described above, particularly when upstream picking likely "
        "missed or contaminated particles; it creates "
        "templates first, then after Select 2D directly refines against that same source volume "
        "without re-running Ab-initio. Treat "
        "CryoSift as quality-ranking evidence, not a particle-diameter measurement. "
        + EXTRACT_BOX_PREFERENCE_GUIDANCE
        + "When scale_observer_evidence is attached, cross-check its weighted radial-profile "
        "diameter range and CTF-aware box recommendation, including the 2.5D empirical check. "
        "The CTF box formula uses the CTF-fit resolution "
        "distribution peak for d (never the target/final FSC resolution), and median-plus-one-sigma "
        "quality-filtered defocus for deltaF from the complete Patch CTF set, not the pilot subset. "
        "The first calibration extraction starts at 384 px; 720 px is only a hard cap. Treat the estimate as a "
        "measured starting point and compare the resulting Class 2D evidence. Never approve "
        "box_size_pix above 720. At the 720-pixel cap, prioritize particle-selection or upstream "
        "picking alternatives rather than expanding extraction further. Keep experiments bounded "
        f"to at most {max_new_jobs} new jobs."
    )
    payload = {
        "workspace_observation": observation,
        "candidate_actions": candidate_actions,
        "previous_experiments": experiment_history[-12:],
        "output_contract": contract,
        "read_only_observation_tools": sorted(READ_ONLY_OBSERVATION_TOOLS),
    }
    from coverage_context import planner_context
    coverage = planner_context(observation)
    if coverage is not None:
        payload['coverage_rollout_context'] = coverage
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
    if stage_guidance is not None:
        return stage_guidance.inject(
            messages, last_completed_job=last_completed_from_observation(observation),
            candidate_actions=candidate_actions, workflow_state=observation,
            audience="planner", audit=guidance_audit,
        )
    return messages


def validate_plan(plan: dict[str, Any], candidate_actions: list[dict[str, Any]], max_actions: int) -> dict[str, Any]:
    """Validate model output and return only fields safe for runner use."""
    allowed_types = {"request_observations", "plan_experiments", "wait", "request_input", "finalize"}
    decision_type = plan.get("decision_type")
    if decision_type not in allowed_types:
        raise ValueError("unsupported planning decision type")
    tasks = normalize_observation_tasks(plan.get('observation_tasks', []))
    normalized_request = decision_type == 'request_input' and bool(tasks)
    if normalized_request:
        if plan.get('approved_action_ids'):
            raise ValueError('observation requests cannot also approve jobs')
        decision_type = 'request_observations'
    if decision_type == "request_observations" and not tasks:
        raise ValueError("request_observations requires at least one task")
    if decision_type != "request_observations" and tasks:
        raise ValueError("only request_observations may include tasks")
    if len(tasks) > 4:
        raise ValueError("at most four observation tasks are allowed")
    if tasks and plan.get('approved_action_ids'):
        raise ValueError('observation requests cannot also approve jobs')
    available_ids = {str(item.get("action_id")) for item in candidate_actions if item.get("available")}
    requested = [str(value) for value in (plan.get("approved_action_ids") or [])]
    if len(requested) > max_actions:
        raise ValueError("plan exceeds its action budget")
    unknown = sorted(set(requested) - available_ids)
    if unknown:
        raise ValueError(f"plan contains unavailable action IDs: {unknown}")
    result = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "decision_type": decision_type,
        "objective": str(plan.get("objective") or ""),
        "hypotheses": plan.get("hypotheses") or [],
        "observation_tasks": tasks,
        "approved_action_ids": requested,
        "comparison_metrics": plan.get("comparison_metrics") or [],
        "stop_conditions": plan.get("stop_conditions") or [],
    }
    if normalized_request:
        result['request_normalization'] = 'request_input_with_tasks_to_request_observations'
    return result


def build_policy_plan(
    observation: dict[str, Any], candidate_actions: list[dict[str, Any]],
    experiment_history: list[dict[str, Any]], max_actions: int,
) -> dict[str, Any]:
    """Conservative deterministic baseline, also used after planner API failures."""
    available = [item for item in candidate_actions if item.get("available")]
    issue_types = {item.get("issue_type") for item in observation.get("observed_issues") or []}
    target_unmet = "target_resolution_unmet" in issue_types
    upstream = [item for item in available if _is_upstream_intervention(item)]
    mandatory = [item for item in available if item.get("box_size_trial") or item.get("class_2d_trial")]
    rollback_menu = [item for item in available if item.get("refinement_rollback")]
    rollback = [item for item in available if item.get("quality_rollback")]
    cryosift_empty = any(
        item.get("success") is True and item.get("selected_class_ids") == []
        and bool(item.get("class_scores"))
        for item in observation.get("cryosift_evidence") or []
    )
    select = [item for item in available if item.get("job_type") == "select_2D"]
    inspect = [item for item in available if item.get("job_type") == "inspect_picks_v2"]
    if mandatory:
        chosen, objective = mandatory, "Complete the finite box-size/Class 2D comparison before 3D refinement."
    elif rollback_menu:
        # API outages must not collapse the bounded menu back to another 2D
        # classification pass. Re-selecting the existing classes is the safer
        # default because it can directly remove incomplete or off-center views.
        reclassify = [
            item for item in rollback_menu
            if item.get("rollback_kind") == "reclassify_selected_particles"
        ]
        reselect = [
            item for item in rollback_menu
            if item.get("rollback_kind") == "reselect_2d"
        ]
        picker_review = [
            item for item in rollback_menu
            if item.get("rollback_kind") in {"reinspect_picks", "repick_diameter"}
        ]
        templates = [item for item in rollback_menu
            if item.get('rollback_kind') == 'template_repick'
            and item.get('job_type') == 'create_templates'
            and (item.get('template_repick_gate') or {}).get('allowed') is True]
        comparisons = [item for item in rollback_menu if item.get('symmetry_comparison')]
        # Keep the prompt's gated template preference during real API failures;
        # never invent eligibility or re-enable a blocked candidate.
        preferred = templates
        if templates:
            gap = templates[0]['template_repick_gate'].get('target_gap_A')
            switches = [item for item in rollback_menu
                if item.get('job_type') == 'nonuniform_refine_new'
                and item.get('refinement_strategy_switch')
                and (item['job_type'], _input_signature(item)) not in {
                    (entry.get('job_type'), entry.get('input_signature')) for entry in experiment_history}]
            if isinstance(gap, (int, float)) and 0 < gap < .1 and switches:
                preferred = switches
        optics = [item for item in rollback_menu if item.get('rollback_kind') == 'optics_refinement']
        nu = [item for item in rollback_menu if item.get('refinement_strategy_switch')
              and item.get('job_type') == 'nonuniform_refine_new'
              and item.get('rollback_kind') == 'alternate_refinement']
        reference_expansion = [item for item in rollback_menu if item.get('rollback_kind') == 'reference_class_expansion']
        scale_cleanup = [item for item in rollback_menu if item.get('rollback_kind') == 'low_scale_cleanup']
        chosen = comparisons or optics or nu or reference_expansion or scale_cleanup or preferred or (picker_review if cryosift_empty else []) or reselect or reclassify or rollback_menu
        objective = (
            'Compare the detected point group once against the preserved same-input baseline.' if comparisons else
            'Test defocus plus global tilt/trefoil/Cs once on the preserved best population.' if optics else
            'Compare one modern NU refinement before repeated upstream changes.' if nu else
            'Expand the original class selection using a lowpassed own reference, then refine.' if reference_expansion else
            'Test one mild source-specific low-scale cut and preserve the unfiltered best.' if scale_cleanup else
            'Compare one eligible NU-refine near target.' if preferred and preferred != templates else
            'Use the safety-gated refinement volume for Create Templates and Template Picker.' if templates else
            "Investigate upstream picking: no Class 2D score met the CryoSift threshold."
            if cryosift_empty and picker_review else
            "Run one bounded upstream particle-quality experiment after below-target refinement."
        )
    elif rollback:
        chosen, objective = rollback, "Revisit upstream particle selection after below-target refinement quality."
    elif target_unmet and select:
        chosen, objective = select, "Use Class 2D evidence to revise particle selection before another refinement."
    elif target_unmet and inspect:
        chosen, objective = inspect, "Inspect Blob Picker outputs before extraction and downstream reconstruction."
    elif target_unmet and upstream:
        chosen, objective = upstream, "Resolve the highest-value upstream particle-quality bottleneck."
    else:
        chosen, objective = available, "Proceed with the only currently safe workflow stage."
    chosen = _cap_actions(chosen, max_actions, preserve_mandatory=bool(mandatory))
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "decision_type": "plan_experiments" if chosen else "wait",
        "objective": objective,
        "hypotheses": [{
            "id": "H_upstream_first",
            "claim": "Bounded changed-input or changed-optics comparisons are more informative than repeating unchanged refinement.",
            "evidence": sorted(issue_types),
        }],
        "approved_action_ids": [str(item.get("action_id")) for item in chosen],
        "comparison_metrics": ["class_average_visual_quality", "cryosift_score", "selected_particle_count", "FSC_resolution_A"],
        "stop_conditions": ["Do not repeat refinement with unchanged particle inputs."],
    }
    return apply_anti_refinement_loop(plan, observation, candidate_actions, experiment_history, max_actions)


def apply_anti_refinement_loop(
    plan: dict[str, Any], observation: dict[str, Any], candidate_actions: list[dict[str, Any]],
    experiment_history: list[dict[str, Any]], max_actions: int,
) -> dict[str, Any]:
    """Hard guard: upstream evidence wins over repeated, unchanged refinement."""
    available = {str(item.get("action_id")): item for item in candidate_actions if item.get("available")}
    issue_types = {item.get("issue_type") for item in observation.get("observed_issues") or []}
    target_unmet = "target_resolution_unmet" in issue_types
    comparisons = [item for item in available.values() if item.get("symmetry_comparison")
                   and (item.get("job_type"), _input_signature(item)) not in {
                       (entry.get("job_type"), entry.get("input_signature")) for entry in experiment_history}]
    if target_unmet and comparisons:
        result = dict(plan)
        result.update(decision_type="plan_experiments",
                      approved_action_ids=[str(comparisons[0]["action_id"])],
                      objective="Test symmetry with the preserved baseline's particles and initial volume.",
                      symmetry_comparison_guard={"one_time_same_input_comparison": True})
        return result
    upstream = [item for item in available.values() if _is_upstream_intervention(item)]
    completed_job_types = {
        item.get("job_type")
        for item in (observation.get("dag") or {}).get("nodes") or []
        if item.get("status") == "completed"
    }
    has_abinitio = "homo_abinit" in completed_job_types
    baseline_refines = [
        item for item in available.values()
        if item.get("job_type") == "homo_refine_new"
    ]
    prior_refinement_inputs = {
        (entry.get("job_type"), entry.get("input_signature")) for entry in experiment_history
        if entry.get("job_type") in REFINEMENT_JOB_TYPES and entry.get("input_signature")
    }
    new_branch_baselines = [
        item for item in baseline_refines
        if (item.get("job_type"), _input_signature(item)) not in prior_refinement_inputs
    ]
    # Ab-initio establishes an initial volume but usually does not expose the
    # FSC resolution required to judge target attainment. Always obtain one
    # baseline refinement for each distinct particle/volume branch before
    # using upstream rollback as a quality fix.
    if has_abinitio and new_branch_baselines:
        result = dict(plan)
        result["approved_action_ids"] = [
            str(item.get("action_id")) for item in _cap_actions(new_branch_baselines, 1)
        ]
        result["anti_refinement_guard"] = {
            "baseline_refinement_required": True,
            "target_unmet": target_unmet,
            "upstream_candidates_available": bool(upstream),
            "repeated_refinement_blocked": False,
        }
        return result
    approved = [available[item_id] for item_id in plan.get("approved_action_ids") or [] if item_id in available]
    if target_unmet and upstream:
        approved = [
            item for item in approved
            if item.get("job_type") not in REFINEMENT_JOB_TYPES
            or item.get("refinement_strategy_switch")
        ]
        if not approved:
            approved = _cap_actions(upstream, max_actions)
    filtered = []
    for item in approved:
        if item.get("job_type") in REFINEMENT_JOB_TYPES and (
            item.get("job_type"), _input_signature(item)
        ) in prior_refinement_inputs:
            continue
        filtered.append(item)
    result = dict(plan)
    result["approved_action_ids"] = [str(item.get("action_id")) for item in _cap_actions(filtered, max_actions)]
    result["anti_refinement_guard"] = {
        "baseline_refinement_required": False,
        "target_unmet": target_unmet,
        "upstream_candidates_available": bool(upstream),
        "repeated_refinement_blocked": len(filtered) != len(approved),
    }
    return result


def restrict_candidate_context(candidate_context: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Expose only planned actions to the execution model; MCP validates again later."""
    approved = set(plan.get("approved_action_ids") or [])
    result = dict(candidate_context)
    visible, blocked = [], list(candidate_context.get("blocked_actions") or [])
    for action in candidate_context.get("candidate_actions") or []:
        if str(action.get("action_id")) in approved:
            visible.append(action)
        elif action.get("available"):
            blocked.append({**action, "available": False, "blocked_by": list(action.get("blocked_by") or []) + ["Not approved by the bounded experiment plan."]})
    result["candidate_actions"] = visible
    result["blocked_actions"] = blocked
    result["planning_directive"] = plan
    return result


def decision_respects_plan(decision: dict[str, Any], plan: dict[str, Any]) -> tuple[bool, str | None]:
    """Require explicit action IDs so execution cannot escape a planned branch."""
    if decision.get("decision_type") in {"stop", "request_input", "rollback"}:
        return True, None
    approved = set(plan.get("approved_action_ids") or [])
    from v2_decision_adapter import normalize_requested_actions
    requested = normalize_requested_actions(decision)
    action_ids = [item.get("action_id") for item in requested if isinstance(item, dict)]
    action_ids = [action_id or (item.get('action') if isinstance(item.get('action'), str) and item.get('action') in approved else None)
                  for action_id, item in zip(action_ids, requested)]
    if not action_ids or any(not action_id for action_id in action_ids):
        return False, "Planning mode requires action_id on every submitted action."
    outside = [str(action_id) for action_id in action_ids if str(action_id) not in approved]
    if outside:
        return False, f"Action IDs are outside the approved experiment plan: {outside}"
    return True, None


def _is_upstream_intervention(action: dict[str, Any]) -> bool:
    return bool(action.get("quality_rollback")) or action.get("job_type") in UPSTREAM_JOB_TYPES


def _cap_actions(actions: list[dict[str, Any]], max_actions: int, preserve_mandatory: bool = False) -> list[dict[str, Any]]:
    if preserve_mandatory:
        return actions
    return actions[:max(0, max_actions)]


def _input_signature(action: dict[str, Any]) -> str:
    inputs = action.get("required_inputs") or {}
    if action.get("symmetry_comparison"):
        inputs = {"inputs": inputs, "refine_symmetry": action["default_parameters"]["refine_symmetry"]}
    return json.dumps(inputs, ensure_ascii=False, sort_keys=True, default=str)

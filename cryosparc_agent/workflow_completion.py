"""Deterministic target completion, separate from scientific planning preferences."""

import math


REFINE_TYPES = {"homo_refine_new", "nonuniform_refine_new", "refine_3D_new", "homo_refine"}


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def corrected_resolution(info):
    values = (info.get("quality_assessment") or {}).get("resolution_evidence", {}).get("values", {})
    keys = ("fsc_resolution",) if info.get("job_type") == "validation" else (
        "fsc_info.radwn_final_A", "fsc_info.radwn_autotight_noisesub_A",
    )
    for key in keys:
        value = values.get(key)
        if positive_number(value):
            return {"resolution_A": float(value), "metric": key}
    return None


def volume_parent(info):
    sources = (info.get("inputs") or {}).get("volume")
    if sources and isinstance(sources, list):
        return sources[0].get("source_job_uid") if len(sources) == 1 else None
    for group in (info.get("inputs") or {}).get("groups", []):
        if group.get("name") == "volume":
            ids = group.get("connected_job_uids") or []
            return ids[0] if len(ids) == 1 else None
    return None


async def resolve_completion(info, target, candidates, fetch_job, observe):
    parent = None
    if info.get("job_type") == "validation" and volume_parent(info):
        parent = await fetch_job(volume_parent(info))
    result = completion_check(info, target, candidates, parent)
    if (info.get("job_type") == "validation" and parent
            and parent.get("job_uid") == volume_parent(info)
            and completion_check(parent, target, [])["action"] == "stop"
            and result["action"] != "stop"):
        if info.get("status") not in {"completed", "failed", "killed"}:
            return {"action": "wait_validation", "validation_job_uid": info["job_uid"]}
        return {"action": "hold", "stop_reason": "target_validation_needs_attention",
                "refine_job_uid": parent["job_uid"], "validation_job_uid": info["job_uid"],
                "validation_status": info.get("status"),
                "reason": "Validation does not confirm the target-reaching refinement; review before further optimization."}
    if result["action"] != "validate":
        return result
    # Explicit recovery from the refine node must reuse its existing validation,
    # rather than create another validation after every restart.
    observation = await observe()
    existing = [node for node in (observation.get("dag") or {}).get("nodes", [])
                if node.get("job_type") == "validation" and info.get("job_uid") in node.get("parents", [])]
    if existing:
        node = existing[-1]
        package = await fetch_job(node["job_uid"])
        checked = completion_check(package, target, [], info)
        if checked["action"] == "stop":
            return checked
        if package.get("status") not in {"completed", "failed", "killed"}:
            return {"action": "wait_validation", "validation_job_uid": node["job_uid"]}
        return {"action": "hold", "stop_reason": "target_validation_needs_attention",
                "refine_job_uid": info["job_uid"], "validation_job_uid": node["job_uid"],
                "validation_status": package.get("status"),
                "reason": "Existing validation is active, failed or disagrees with target attainment; do not duplicate it or restart picking."}
    return result


def completion_check(info, target, candidates, parent=None):
    """Only stop on completed, error-free, branch-specific corrected FSC evidence."""
    if not positive_number(target) or info.get("status") != "completed":
        return {"action": "continue"}
    if (info.get("metrics") or {}).get("has_error") or info.get("has_error"):
        return {"action": "continue"}
    measured = corrected_resolution(info)
    if not measured or measured["resolution_A"] > target:
        return {"action": "continue"}
    uid = info.get("job_uid")
    if info.get("job_type") == "validation":
        if not parent or parent.get("job_uid") != volume_parent(info):
            return {"action": "continue"}
        if parent.get("status") != "completed" or parent.get("job_type") not in REFINE_TYPES:
            return {"action": "continue"}
        if parent.get("has_error") or (parent.get("metrics") or {}).get("has_error"):
            return {"action": "continue"}
        parent_resolution = corrected_resolution(parent)
        if not parent_resolution or parent_resolution["resolution_A"] > target:
            return {"action": "continue"}
        return {"action": "stop", "stop_reason": "target_resolution_met", "validated": True,
                "refine_job_uid": parent["job_uid"], "validation_job_uid": uid,
                "target_resolution_A": target, **measured}
    if info.get("job_type") not in REFINE_TYPES:
        return {"action": "continue"}
    for candidate in candidates:
        sources = (candidate.get("required_inputs") or {}).get("volume") or []
        if (candidate.get("available") and candidate.get("job_type") == "validation"
                and len(sources) == 1 and sources[0].get("source_job_uid") == uid):
            return {"action": "validate", "candidate": candidate, "refine_job_uid": uid,
                    "target_resolution_A": target, **measured}
    return {"action": "stop", "stop_reason": "target_resolution_met", "validated": False,
            "refine_job_uid": uid, "target_resolution_A": target, **measured}


def validation_decision(completion):
    candidate = completion["candidate"]
    parameters = {}
    if "validate_do_noisesub" in (candidate.get("parameter_template") or {}):
        parameters["validate_do_noisesub"] = True
    return {
        "schema_version": "2.0", "decision_type": "forward", "action": "validation",
        "job_type": "validation", "action_id": candidate["action_id"], "parameters": parameters,
        "connections": {"volume": {"source_job_uid": completion["refine_job_uid"], "source_output": "volume"}},
        "reason": "Corrected final FSC meets the target. Run one branch-matched validation before finalizing; do not restart upstream optimization.",
    }

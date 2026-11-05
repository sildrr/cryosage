"""Evidence-based quality assessment and conservative recovery advice."""
from __future__ import annotations

from typing import Any, Dict, List
import math
from movie_metadata import MOVIE_SUMMARY_FIELDS

QUALITY_POLICY_VERSION = "cryoem_quality_v1"


def assess_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Assess only signals available from the normalized CryoSPARC state."""
    status = node.get("status")
    outputs = node.get("outputs") or {}
    total_items = sum(int(output.get("num_items") or 0) for output in outputs.values())
    available_outputs = [name for name, output in outputs.items() if output.get("available")]
    warnings = list(node.get("run_errors", {}).get("errors_run") or [])
    errors = []
    for key in ("errors_build_inputs", "errors_build_params"):
        value = node.get("run_errors", {}).get(key)
        if value:
            errors.append({"source": key, "value": value})
    observed_metrics = extract_quality_metrics(node)
    resolution_evidence = find_resolution_evidence(node, observed_metrics)

    if status in {"failed", "killed"} or node.get("has_error"):
        quality_status = "failed"
        recommendation = "recover_or_review"
    elif status != "completed":
        quality_status = "not_terminal"
        recommendation = "wait"
    elif total_items == 0 and outputs:
        quality_status = "completed_without_outputs"
        recommendation = "recover_or_review"
    elif node.get("has_warning"):
        quality_status = "completed_with_warnings"
        recommendation = "continue_with_caution"
    else:
        quality_status = "completed_observed"
        recommendation = "continue_if_next_inputs_are_compatible"

    return {
        "policy_version": QUALITY_POLICY_VERSION,
        "quality_status": quality_status,
        "recommendation": recommendation,
        "evidence_level": "observed_state_only",
        "output_item_count": total_items,
        "available_outputs": available_outputs,
        "run_error_count": len(errors),
        "run_warning_count": len(warnings),
        "resolution_evidence": resolution_evidence,
        "observed_metrics": observed_metrics,
        "recovery_options": recovery_options(node, quality_status),
        "limitations": [
            "FSC/resolution values are not inferred unless exposed by CryoSPARC summary data.",
            "Image-based quality requires a separate image artifact extractor.",
        ],
    }


def extract_quality_metrics(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact set of numeric quality/acquisition values from summaries."""
    selected = {}
    tokens = (
        "fsc", "resolution", "ctf", "defocus", "astig", "num_particles",
        "particle", "pick", "motion", "psize", "accel_kv", "cs_mm",
        "total_dose",
    )
    for output_name, output in (node.get("outputs") or {}).items():
        for field_name in ("summary_values", "latest_summary_stat_values"):
            for key, value in (output.get(field_name) or {}).items():
                key_text = str(key)
                if key_text not in MOVIE_SUMMARY_FIELDS and not any(token in key_text.lower() for token in tokens):
                    continue
                selected[key_text] = {
                    "value": value,
                    "source": f"outputs.{output_name}.{field_name}.{key_text}",
                }
    return selected


def find_resolution_evidence(
    node: Dict[str, Any],
    observed_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Report resolution evidence and its value when CryoSPARC exposes it."""
    keys = set()
    for output in (node.get("outputs") or {}).values():
        keys.update(output.get("summary_keys") or [])
        keys.update(output.get("latest_summary_stat_keys") or [])
    matches = sorted(
        key for key in keys
        if any(token in key.lower() for token in ("fsc", "resolution", "res_"))
        and "psize" not in key.lower()
        and "shape" not in key.lower()
    )
    values = {}
    for key, item in (observed_metrics or {}).items():
        if key not in matches:
            continue
        value = item.get("value")
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            values[key] = float(value)
        elif isinstance(value, dict) and key.lower() in {"fsc_info", "fsc_info_best"}:
            for nested_key, nested_value in value.items():
                if not str(nested_key).endswith("_A"):
                    continue
                if isinstance(nested_value, (int, float)) and math.isfinite(float(nested_value)) and float(nested_value) > 0:
                    values[f"{key}.{nested_key}"] = float(nested_value)
    return {
        "available": bool(values),
        "keys": matches,
        "values": values,
        "note": "Only explicit FSC/resolution values are reported; pixel size and volume shape are excluded.",
    }


def recovery_options(node: Dict[str, Any], quality_status: str) -> List[str]:
    if quality_status == "failed":
        from unattended import enabled
        if enabled():
            return ['automatic_failure_diagnostics', 'approved_bounded_retry']
        return ["retry_same_job_with_adjusted_parameters", "request_human_review"]
    if quality_status == "completed_without_outputs":
        return ["inspect_input_and_parameters", "retry_same_job"]
    if quality_status == "completed_with_warnings":
        return ["inspect_warnings", "continue_or_branch"]
    return []


def build_retry_candidate(node: Dict[str, Any], workflow_state=None, registry_spec=None,
                          original_parameters=None) -> Dict[str, Any]:
    """Create an approval-gated retry candidate for a failed terminal Job."""
    candidate = {
        "action_id": f"retry_{node['cryosparc_job_uid']}",
        "action_type": "forward",
        "workflow_node_id": f"{node['workflow_node_id']}:retry",
        "reference_job_uid": node["cryosparc_job_uid"],
        "reference_status": node["status"],
        "job_type": node["job_type"],
        "description": f"Retry failed {node['job_type']} after reviewing its errors.",
        "execution_mode": "create_job",
        "available": True,
        "blocked_by": [],
        "required_inputs": node.get("inputs") or {},
        "missing_required_inputs": [],
        "parameter_template": {},
        "default_parameters": node.get("key_parameters") or {},
        "approval_required": True,
        "approval_reasons": ["failed_job_retry_requires_human_review"],
        "recovery_policy": "retry_failed_job",
    }
    if registry_spec is not None:
        from dynamic_candidates import build_registry_candidate, obsolete_registry_job
        if obsolete_registry_job(registry_spec):
            candidate.update(available=False, blocked_by=['obsolete_job_type'])
        else:
            native = build_registry_candidate(node, registry_spec, candidate['required_inputs'])
            candidate['job_spec_metadata'] = native['job_spec_metadata']
            candidate['parameter_template'] = native['parameter_template']
            params = original_parameters if original_parameters is not None else node.get('key_parameters', {})
            from job_input_contracts import normalize_input_parameters
            candidate['default_parameters'] = normalize_input_parameters(node['job_type'],
                {k: v for k, v in params.items() if v is not None}, candidate['required_inputs'])
            from job_input_contracts import normalize_volume_connections
            candidate['required_inputs'] = normalize_volume_connections(node['job_type'],
                candidate['default_parameters'], candidate['required_inputs'],
                repeat_allowed='volume' in candidate['job_spec_metadata'].get('repeat_allowed_inputs', []))
            if node['job_type'] == 'class_3D' and candidate['required_inputs'].get('volume'):
                candidate['description'] += (' Input initialization needs one volume connection per class. '
                    'The native Registry permits repeating the single source volume; these are '
                    'identical starting references, not independently reconstructed structures.')
            candidate['failure_diagnostics'] = node.get('run_errors') or {}
    from unattended import authorize_retry
    return authorize_retry(candidate, node, workflow_state)

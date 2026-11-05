"""Scoped symmetry evidence memory and bounded same-input comparisons."""
import json
import math
import os
from pathlib import Path
import re
import uuid


def read_memory(path, project, workspace):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if (data.get("project_uid"), data.get("workspace_uid")) != (project, workspace):
        return {}
    return data.get("evidence_by_job") or {}


def remember_evidence(path, project, workspace, evidence):
    records = read_memory(path, project, workspace)
    if evidence.get("success") is True and evidence.get("project_uid") == project and evidence.get("job_uid"):
        fields = ("success", "tool", "project_uid", "job_uid", "job_type", "recommended_symmetry",
                  "average_fsc", "confidence", "axis_height", "primary_axis", "resolution_A",
                  "find_symmetry_center", "map_source", "evidence_policy")
        records[evidence["job_uid"]] = {k: evidence.get(k) for k in fields}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        temp.write_text(json.dumps(dict(project_uid=project, workspace_uid=workspace,
                                       evidence_by_job=records), indent=2))
        temp.replace(destination)
    return records


def _inputs(node):
    return {slot: values for slot, values in (node.get("inputs") or {}).items()
            if slot in {"particles", "volume"}}


def _signature(node):
    return sorted((slot, item.get("source_job_uid"), item.get("source_output"))
                  for slot, values in _inputs(node).items() for item in values)


def _resolution(node):
    for output in node.get("outputs", {}).values():
        for field in ("latest_summary_stat_values", "summary_values"):
            info = (output.get(field) or {}).get("fsc_info") or {}
            value = info.get("radwn_final_A", info.get("radwn_autotight_noisesub_A"))
            if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                return value
    return float('inf')


def _source_parameters(project, uid):
    from cryosparc_client import cryosparc_client
    return cryosparc_client().find_project(project).find_job(uid).params.model_dump()


def build_symmetry_comparison(workflow, current, dataset):
    """Offer one untried point-group test on a preserved ancestral C1 baseline."""
    if dataset.get("symmetry") or dataset.get("symmetry_status") != "unknown":
        return None
    path = os.getenv("CRYOAGENT_SYMMETRY_EVIDENCE_FILE")
    if not path:
        return None
    records = read_memory(path, workflow.get("project_uid"), workflow.get("workspace_uid"))
    nodes = {n["cryosparc_job_uid"]: n for n in workflow.get("nodes", [])}
    lineage, pending = set(), [current["cryosparc_job_uid"]]
    while pending:
        uid = pending.pop()
        if uid in lineage or uid not in nodes:
            continue
        lineage.add(uid)
        pending.extend(nodes[uid].get("parent_job_uids") or [])
    options = []
    for uid, evidence in records.items():
        source = nodes.get(uid)
        target = str(evidence.get("recommended_symmetry") or "").upper()
        match = re.fullmatch(r"([CD])(\d+)", target)
        valid_group = target in {"T", "O", "I"} or (match is not None and int(match[2]) >= 2)
        score = evidence.get("average_fsc")
        if (uid not in lineage or not source or source.get("status") != "completed"
                or source.get("job_type") not in {"homo_refine_new", "nonuniform_refine_new", "refine_3D_new"}
                or not valid_group or not isinstance(score, (int, float)) or not math.isfinite(score) or score < .75
                or not math.isfinite(_resolution(source))):
            continue
        if source.get("key_parameters", {}).get("refine_symmetry", "C1") == target:
            continue
        inputs = _inputs(source)
        if not inputs.get("particles") or not inputs.get("volume"):
            continue
        tried = any(n.get("job_type") == source["job_type"]
                    and n.get("key_parameters", {}).get("refine_symmetry") == target
                    and _signature(n) == _signature(source) for n in nodes.values())
        if not tried:
            options.append((source, evidence, target))
    if not options:
        return None
    source, evidence, target = min(options, key=lambda item: _resolution(item[0]))
    from job_specs import get_parameter_template
    template = get_parameter_template(source["job_type"])
    template["refine_symmetry"] = {"type": "string", "required": True, "const": target}
    defaults = {k: v for k, v in _source_parameters(workflow["project_uid"], source["cryosparc_job_uid"]).items()
                if v is not None}
    defaults["refine_symmetry"] = target
    uid = source["cryosparc_job_uid"]
    return dict(action_id=f"symmetry_compare_{uid}_{target}", action_type="forward",
        workflow_node_id=f"{uid}:symmetry:{target}", job_type=source["job_type"],
        reference_job_uid=uid, reference_status="completed", execution_mode="create_job",
        available=True, blocked_by=[], missing_required_inputs=[], required_inputs=_inputs(source),
        parameter_template=template, default_parameters=defaults, refinement_rollback=True,
        rollback_kind="symmetry_comparison", symmetry_comparison=True,
        symmetry_evidence=evidence, comparison_reference=dict(job_uid=uid,
            corrected_resolution_A=_resolution(source), input_policy="same_particles_and_initial_volume"),
        description=f"Compare {target} once against preserved {uid}, with the same particles, initial volume "
                    "and refinement type. Change only symmetry; compare corrected FSC and map plausibility. "
                    "This tests a hypothesis, not an established symmetry. Do not replace the baseline.")

"""Bounded pre-extraction visual review with durable pilot-job state."""

import copy
import hashlib
import json


class ReviewBlocked(RuntimeError):
    pass


PILOT_TITLE_PREFIX = "Extract review pilot:"


def resolve_review_connections(decision, candidate):
    """Use the executor's accepted connection dialects without guessing sources."""
    from job_executor import normalize_connection_values

    connections = copy.deepcopy(decision.get("connections") or {})
    if not isinstance(connections, dict):
        raise ReviewBlocked("Extraction review connections must be an object")
    for name, sources in (candidate.get("required_inputs") or {}).items():
        if name not in connections and len(sources) == 1:
            connections[name] = sources[0]
    resolved = {}
    for name, raw in connections.items():
        values = normalize_connection_values(raw)
        if not values or (name == "particles" and len(values) != 1):
            raise ReviewBlocked("Extraction review requires an unambiguous explicit particle source")
        refs = [{"source_job_uid": uid, "source_output": output} for uid, output in values]
        resolved[name] = refs[0] if len(refs) == 1 else refs
    if "particles" not in resolved:
        raise ReviewBlocked("Extraction review requires an unambiguous explicit particle source")
    return resolved


def without_review_pilots(state):
    return {**state, "nodes": [n for n in state.get("nodes", [])
        if not str(n.get("title", "")).startswith(PILOT_TITLE_PREFIX)]}


def review_key(current_node, decision):
    return hashlib.sha256(json.dumps([current_node, decision.get("action_id"),
        decision.get("connections"), decision.get("parameters")], sort_keys=True).encode()).hexdigest()[:20]


def is_full_extraction(decision, candidate):
    params = {**(candidate.get("default_parameters") or {}), **(decision.get("parameters") or {})}
    sources = (candidate.get("required_inputs") or {}).get("micrographs") or []
    if len(sources) == 1 and sources[0].get("num_items") == 1:
        return False  # A single-micrograph smoke case cannot support this observer.
    return (decision.get("decision_type") == "forward"
            and decision.get("action", decision.get("job_type")) == "extract_micrographs_multi"
            and not candidate.get("calibration_phase") and not candidate.get("box_size_trial")
            and not params.get("num_extract") and not params.get("bin_size_pix"))


def validate_review(response, current_size, alternative_size=None):
    action = response.get("action")
    allowed = {"keep", "insufficient_evidence", "use_alternative" if alternative_size else "compare"}
    if action not in allowed or not isinstance(response.get("reason"), str):
        raise ReviewBlocked("Invalid extraction review action/reason")
    if action == "compare":
        size = response.get("box_size_pix")
        if isinstance(size, bool) or not isinstance(size, int) or size < 64 or size > 720 or size % 8 or size == current_size:
            raise ReviewBlocked("Comparison box must be a different supported multiple of 8 within 64..720")
    if action == "use_alternative":
        if response.get("box_size_pix") != alternative_size or response.get("central_target_complete") is not True:
            raise ReviewBlocked("Size change requires the actual tested alternative and positive completeness evidence")
        if not response.get("supporting_sample_ids"):
            raise ReviewBlocked("Size change requires sample-specific visual evidence")
    return response


async def review_extraction(state, decision, submit, wait, observe, persist,
                            remaining_jobs, micrographs=6):
    """Callbacks own MCP/API access. At most two small pilots precede one full job."""
    if state.get("status") == "ready":
        return copy.deepcopy(state["amended_decision"])
    if "jobs" not in state:
        state.update(status="starting", decision=copy.deepcopy(decision), jobs=[])
        persist()
    original = state["decision"]
    current_size = original["parameters"]["box_size_pix"]

    async def pilot(label, size):
        record = next((item for item in state["jobs"] if item["label"] == label), None)
        if record is None:
            # Reserve one remaining submission for the full extraction.
            if remaining_jobs - len(state["jobs"]) < 2:
                return None
            record = {"label": label, "box_size_pix": size, "status": "submitting"}
            state["jobs"].append(record)
            persist()
            pilot_decision = copy.deepcopy(original)
            pilot_decision["parameters"].update(box_size_pix=size, num_extract=micrographs, compute_num_gpus=1)
            record["job_uid"] = await submit(pilot_decision, label)
            record["status"] = "waiting"
            persist()
        if not record.get("job_uid"):
            raise ReviewBlocked("Ambiguous pilot submission; inspect existing jobs before resuming (no automatic duplicate)")
        if record["status"] != "completed":
            result = await wait(record["job_uid"])
            record["result"] = result
            if result.get("status") != "completed":
                persist()
                raise ReviewBlocked("Extraction review pilot failed or remains active; full extraction was not submitted")
            record["status"] = "completed"
            persist()
        return record["job_uid"]

    current = await pilot("original", current_size)
    amended = copy.deepcopy(original)
    if current is None:
        state["review_note"] = "Insufficient job budget for pilot plus full extraction"
        persist()
        raise ReviewBlocked(state["review_note"])
    else:
        if "initial_review" not in state:
            state["initial_review"] = validate_review(await observe(current, None), current_size)
            persist()
        if state["initial_review"]["action"] == "compare":
            alternative_size = state["initial_review"]["box_size_pix"]
            alternative = await pilot("alternative", alternative_size)
            if alternative:
                if "comparison_review" not in state:
                    state["comparison_review"] = validate_review(
                        await observe(current, alternative), current_size, alternative_size,
                    )
                    persist()
                if state["comparison_review"]["action"] == "use_alternative":
                    amended["parameters"]["box_size_pix"] = alternative_size
                    amended["reason"] = (original.get("reason", "") + "\nAutomatic matched-pilot review: "
                                         + state["comparison_review"]["reason"])
            else:
                state["review_note"] = "No budget for the requested comparison plus full extraction"
                persist()
                raise ReviewBlocked(state["review_note"])
    state.update(status="ready", amended_decision=amended)
    persist()
    return copy.deepcopy(amended)


def historical_extract_results(observation):
    nodes = {n["job_uid"]: n for n in (observation.get("dag") or {}).get("nodes", [])}
    results = []
    for node in nodes.values():
        if node.get("job_type") not in {"homo_refine_new", "nonuniform_refine_new"} or node.get("status") != "completed":
            continue
        values = (node.get("quality") or {}).get("resolution_A") or {}
        resolution = values.get("fsc_info.radwn_final_A")
        if not isinstance(resolution, (float, int)):
            continue
        parents = list(node.get("parents") or [])
        particle_parents = [uid for uid in parents if nodes.get(uid, {}).get("job_type") not in {"homo_refine_new", "nonuniform_refine_new"}]
        seen, pending = set(), particle_parents or parents
        extracts = []
        while pending:
            uid = pending.pop(0)
            if uid in seen:
                continue
            seen.add(uid)
            parent = nodes.get(uid, {})
            if parent.get("job_type") == "extract_micrographs_multi":
                extracts.append(parent)
                continue
            pending.extend(parent.get("parents") or [])
        for extract in extracts:
            if (extract.get("parameters") or {}).get("num_extract"):
                continue
            results.append({"extract_job_uid": extract["job_uid"],
                            "box_size_pix": (extract.get("parameters") or {}).get("box_size_pix"),
                            "refine_job_uid": node["job_uid"], "corrected_FSC_A": resolution})
    return sorted(results, key=lambda r: r["corrected_FSC_A"])[:8]


REVIEW_INSTRUCTIONS = (
    "A few peripheral neighbor picks in a large box are acceptable. Do not demand zero "
    "neighbors or shrink just to remove a handful of automatic candidate centers. "
    "Only frequent clearly resolved multi-particle contamination supports that rationale; "
    "central-target completeness and physical diameter margin have priority. "
    "Review a bounded cryo-EM extraction pilot before full-data extraction. Use full uncropped "
    "micrographs with calibrated scales, marked target centers, actual particle boxes, and "
    "neighbor-coordinate statistics. Automatic pick centers are not verified molecules; fewer "
    "neighbors alone does not justify a smaller box. Check central-target completeness and margin, "
    "off-centering, truncation and peripheral interference separately. Historical corrected FSC "
    "is a baseline from potentially different particles, not proof of an optimal box. Picker "
    "diameter may include mask allowance; do not multiply that allowance again blindly. "
    "Prefer a physical width around 1.5-2 times the maximum credible molecular extent as advice, "
    "not a hard constraint. When a change is plausible but cannot be confirmed from one box size, "
    "you may request ONE small matched comparison instead of pretending the choice is proven. "
    "Return JSON only: action (keep|compare|use_alternative|insufficient_evidence), box_size_pix "
    "(integer or null), central_target_complete (true|false|null), supporting_sample_ids (array), "
    "reason, uncertainty. In the initial review only keep, compare or insufficient_evidence "
    "are allowed. In a paired review only keep, use_alternative or insufficient_evidence are "
    "allowed; a size change must select the ACTUALLY tested alternative with positive, "
    "sample-specific completeness evidence. If a comparison is needed, propose a different "
    "multiple of 8 in 64..720 px and explain why. Do not request repeated comparisons. "
    "Follow authenticated operator guidance where safe, without bypassing input compatibility, "
    "the compute budget or a required completeness check. "
    "All descriptions should be in Chinese; do not invent measurements or private reasoning."
)

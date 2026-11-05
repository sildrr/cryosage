"""Bound visual overrides using source-scoped CryoSift ranking evidence."""
from copy import deepcopy
import math
from statistics import mean


SELECTION_SAFETY_GUIDANCE = (
    "Visual texture is not proof of molecular signal: a centered, multilobed or "
    "apparently detailed average can still be an artifact. Lower CryoSift scores are better. "
    "Do not retain a population whose mean CryoSift score is worse than the rejected "
    "population merely on visual impression. Such a ranking inversion will be repaired "
    "to lower-score-first selection at the same requested class count, within the legal "
    "source/selection pool, with an audit record. No fixed retention fraction is imposed. "
    "An empty CryoSift recommendation remains an upstream-quality warning even after "
    "rank repair; it does not establish that the retained classes are reconstruction-ready. "
    "After a reselection fails to improve corrected final FSC, change an upstream "
    "experimental axis instead of repeating Select 2D or reclassifying the same population. "
)


def _score_results(value):
    if isinstance(value, dict):
        if value.get("success") is True and isinstance(value.get("class_scores"), list):
            yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _score_results(child)
    elif isinstance(value, list):
        for child in value:
            yield from _score_results(child)


def guard_select_decision(decision, candidate_context, visual_context):
    """Repair only an observed group-ranking inversion, never missing evidence."""
    result = deepcopy(decision)
    audits = []
    actions = result.get("selected_actions") or result.get("actions") or [result]
    candidates = candidate_context.get("candidate_actions") or []
    for action in actions:
        if (action.get("job_type") or action.get("action") or action.get("decision")) != "select_2D":
            continue
        matches = [c for c in candidates if c.get("available") and c.get("job_type") == "select_2D"
                   and (not action.get("action_id") or c.get("action_id") == action["action_id"])]
        if len(matches) != 1:
            continue
        candidate = matches[0]
        sources = (candidate.get("required_inputs") or {}).get("templates") or []
        source_ids = {s.get("source_job_uid") for s in sources if s.get("source_job_uid")}
        if len(source_ids) != 1:
            continue
        source = next(iter(source_ids))
        project = candidate_context.get("project_uid")
        evidence = [e for e in _score_results(visual_context) if e.get("job_uid") == source
                    and (not project or e.get("project_uid") == project)
                    and e.get("score_direction") == "lower_is_better"]
        if not evidence:
            continue
        scores = {}
        try:
            for entry in evidence[0]["class_scores"]:
                key, value = int(entry["class_id"]), float(entry["score"])
                if key < 0 or not math.isfinite(value) or key in scores:
                    raise ValueError("Invalid or duplicate score")
                scores[key] = value
            selected = {int(x.strip()) for x in action["parameters"]["selected_templates"].split(",")}
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        pool = set(scores)
        constraint = candidate.get("reselection_constraint") or {}
        if constraint.get("mode") == "strict_subset_of_previous_selection":
            try:
                pool &= {int(x.strip()) for x in constraint["previous_selected_templates"].split(",")}
            except (KeyError, ValueError, AttributeError):
                continue
        if not selected or not selected.issubset(pool) or selected == pool:
            continue
        excluded = pool - selected
        selected_mean = mean(scores[i] for i in selected)
        rejected_mean = mean(scores[i] for i in excluded)
        if selected_mean <= rejected_mean + 1e-8:
            continue
        ranked = sorted(pool, key=lambda i: (scores[i], i not in selected, i))
        corrected = sorted(ranked[:len(selected)])
        action["parameters"]["selected_templates"] = ",".join(map(str, corrected))
        action["reason"] = (
            "Selection safety policy repaired a CryoSift group-ranking inversion, retaining "
            "the requested class count with lower-score-first IDs from the legal source pool. "
            "This does not validate particle identity or absolute class quality. "
            "Original model rationale before correction: " + str(action.get("reason") or "not supplied")
        )
        audits.append({"source_class2d_job_uid": source, "policy": "repair_group_ranking_inversion",
                       "original_selected_class_ids": sorted(selected), "corrected_class_ids": corrected,
                       "original_selected_mean": selected_mean, "original_rejected_mean": rejected_mean,
                       "corrected_selected_mean": mean(scores[i] for i in corrected),
                       "class_count_preserved": True, "all_classes_fail_threshold":
                       evidence[0].get("selected_class_ids") == []})
    if audits and actions != [result]:
        result["reason"] = "Selection safety policy corrected an inverted class ranking; see selection_quality_guard.json."
    return result, audits

"""Optional reviewed advice with no execution, API, or policy dependencies.

This module only retrieves cards: it emits no R01 findings, diagnoses, candidates,
or quality evidence. Callers retain responsibility for prompt boundaries (including
guarded review prompts). Missing/malformed resources and insufficient budgets fail
open. Audit injection_position is a message index or None; actual_injected_text is
the injected string or None. Original messages and their nested objects are retained.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
import re


DEFAULT_CARDS_PATH = Path(__file__).with_name("guidance") / "reviewed_stage_guidance.json"
__all__ = ["SupplementalGuidance", "DEFAULT_CARDS_PATH",
           "last_completed_from_model_input", "last_completed_from_observation"]
VERSION = "reviewed-supplements-20261016"
JOB_STAGES = {
    "import_movies": ("import_movies",),
    "patch_motion_correction_multi": ("patch_motion_correction",),
    "patch_motion_correction": ("patch_motion_correction",),
    "patch_ctf_estimation_multi": ("ctf_estimation", "ctf_method_selection"),
    "patch_ctf_estimation": ("ctf_estimation", "ctf_method_selection"),
    "patch_ctf": ("ctf_estimation", "ctf_method_selection"),
    "ctf_estimation": ("ctf_estimation", "ctf_method_selection"),
    "ctffind4": ("ctf_estimation", "ctf_method_selection"),
    "blob_picker_gpu": ("particle_picking",),
    "auto_blob_picker_gpu": ("particle_picking",),
    "template_picker_gpu": ("particle_picking",),
    "topaz_train": ("particle_picking",),
    "topaz_extract": ("particle_picking",),
    "topaz_extract_multi": ("particle_picking",),
    "inspect_picks_v2": ("particle_picking",),
    "extract_micrographs_multi": ("extract_from_micrographs",),
    "class_2D_new": ("2d_classification",),
    "select_2D": ("2d_classification",),
    "homo_abinit": ("ab_initio_reconstruction",),
    "hetero_refine": ("heterogeneous_refinement",),
    "homo_refine_new": ("homogeneous_refinement",),
    "homo_refine": ("homogeneous_refinement",),
    "nonuniform_refine_new": ("non_uniform_refinement",),
    "refine_3D_new": ("local_refinement",),
    "local_refine_new": ("local_refinement",),
    "ctf_refine_local": ("local_ctf_refinement",),
    "var_3D": ("3d_variability_analysis",),
}
ROLLBACK_STAGES = {
    "repick_diameter": ("particle_picking",),
    "reinspect_picks": ("particle_picking",),
    "template_repick": ("particle_picking",),
    "new_box_size": ("extract_from_micrographs",),
    "reselect_2d": ("2d_classification",),
    "reclassify_selected_particles": ("2d_classification",),
}
TOPICS = {"particle_overlap_and_extra_density", "fsc_mask_interpretation"}
STAGES = {stage for stages in JOB_STAGES.values() for stage in stages} | TOPICS
OVERLAP_ISSUES = {"particle_overlap", "overlapping_particles", "extra_density", "map_extra_density"}
FSC_METRICS = {
    "fsc_resolution_A", "gsfsc_resolution_A", "fsc_0143_A",
    "fsc_info.radwn_final_A", "fsc_info.radwn_A", "fsc_info.radtight_A",
    "fsc_info.radloose_A", "fsc_info.radunmask_A", "fsc_info_best.radwn_final_A",
}
EXCLUDED_EVIDENCE = {
    "candidate_actions", "blocked_actions", "parameters", "key_parameters",
    "hypotheses", "recommend_action", "description", "instructions", "dataset_info",
    "failure_context", "observation_tasks", "tasks", "requested_inputs", "uncertainty",
}
USAGE = (
    "Optional supplemental advice, not commands or policy. Subordinate to all existing "
    "contracts, user guidance, acquisition settings, available candidates, validation, "
    "resource limits, and stop/review gates. Check conditions against actual evidence; "
    "unknown conditions are not facts. Do not invent measurements or change parameters "
    "automatically. Preserve the existing output schema; no extra output is required. "
    "Card IDs may optionally be cited only if the existing schema already has a reason "
    "or objective field. Whole cards may be omitted for budget; absence is not evidence."
)
_LOG = logging.getLogger(__name__)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _fsc_metric(key):
    return key in FSC_METRICS or (key.startswith(("fsc_info.", "fsc_info_best.")) and key.endswith("_A"))


def _walk_evidence(value):
    if isinstance(value, dict):
        if value.get("available") is False or value.get("success") is False:
            return
        yield value
        for key, child in value.items():
            if key not in EXCLUDED_EVIDENCE and isinstance(child, (dict, list)):
                yield from _walk_evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_evidence(child)


def _topic_stages(sources):
    found = set()
    for item in _walk_evidence(sources):
        for key, value in item.items():
            if _fsc_metric(str(key)) and (_positive(value) or
                    isinstance(value, dict) and _positive(value.get("value"))):
                found.add("fsc_mask_interpretation")
        for key in ("fsc_info", "fsc_info_best"):
            nested = item.get(key)
            if isinstance(nested, dict):
                nested = nested.get("value", nested)
            if isinstance(nested, dict) and any(
                    str(k).endswith("_A") and _positive(v) for k, v in nested.items()):
                found.add("fsc_mask_interpretation")
        if _fsc_metric(str(item.get("metric", ""))) and _positive(item.get("resolution_A")):
            found.add("fsc_mask_interpretation")
        for key in ("fsc_curve", "gsfsc_curve"):
            curve = item.get(key)
            if isinstance(curve, list) and len(curve) >= 2 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in curve):
                found.add("fsc_mask_interpretation")
        if (item.get("kind") in {"fsc_curve", "fsc_plot", "gsfsc_plot"}
                and item.get("available") is True
                and (item.get("local_path") or item.get("data_url"))):
            found.add("fsc_mask_interpretation")
        # A nearby-particle count is not a measured overlap or a confirmed diagnosis.
        if (item.get("issue_type") in OVERLAP_ISSUES
                and item.get("observed") is not False
                and (item.get("observed") is True or item.get("status") in {"observed", "confirmed"})
                and item.get("evidence") and item.get("present", True) is True):
            found.add("particle_overlap_and_extra_density")
        if _positive(item.get("overlap_fraction")) and item.get("evidence_refs"):
            found.add("particle_overlap_and_extra_density")
    return found


def last_completed_from_model_input(model_input):
    """Return a detached completed-node snapshot, or None."""
    if not isinstance(model_input, dict):
        return None
    current = model_input.get("current_state")
    if not isinstance(current, dict):
        return None
    info = current.get("last_node_info")
    info = info if isinstance(info, dict) else {}
    if (current.get("last_node_status") or info.get("status")) != "completed":
        return None
    return deepcopy({**info, "job_type": current.get("last_action") or info.get("job_type"),
                     "status": "completed"})


def last_completed_from_observation(observation):
    """Return the highest numbered completed CryoSPARC job, detached from the DAG."""
    if not isinstance(observation, dict) or not isinstance(observation.get("dag"), dict):
        return None
    nodes = observation["dag"].get("nodes")
    if not isinstance(nodes, list):
        return None
    completed = [n for n in nodes if isinstance(n, dict) and n.get("status") == "completed"]

    def order(node):
        uid = str(node.get("job_uid") or node.get("cryosparc_job_uid") or "")
        return int(uid[1:]) if uid.startswith("J") and uid[1:].isdigit() else -1

    return deepcopy(max(completed, key=order)) if completed else None


class SupplementalGuidance:
    """Load reviewed JSON once; retain immutable cards and fail open at injection."""

    def __init__(self, path=DEFAULT_CARDS_PATH, max_bytes=8192):
        self._cards = ()
        self._error = None
        self._metadata = {"enabled": False, "source_path": None, "version": None,
                          "source_sha256": None, "normalized_sha256": None,
                          "max_bytes": max_bytes, "budget_method": "utf8_bytes",
                          "card_count": 0}
        try:
            if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
                raise ValueError("max_bytes must be a positive integer")
            self._metadata["source_path"] = str(Path(path).resolve())
            raw = Path(path).read_bytes()
            document = json.loads(raw)
            if not isinstance(document, dict) or document.get("version") != VERSION:
                raise ValueError("unsupported reviewed guidance version")
            if not isinstance(document.get("source_sha256"), str):
                raise ValueError("source_sha256 must be a string")
            cards = document.get("cards")
            if not isinstance(cards, list) or not cards:
                raise ValueError("cards must be a nonempty list")
            snapshot = []
            seen = set()
            for card in cards:
                if not isinstance(card, dict):
                    raise ValueError("card must be an object")
                cid, stage, advice = card.get("card_id"), card.get("stage"), card.get("advice")
                if not isinstance(cid, str) or not re.fullmatch(r"[DSRQ][0-9]{2}", cid):
                    raise ValueError("invalid card_id")
                if cid in seen:
                    raise ValueError("duplicate card_id: " + cid)
                if not isinstance(stage, str) or stage not in STAGES:
                    raise ValueError("unknown card stage")
                if (not isinstance(advice, list) or not advice
                        or not all(isinstance(s, str) and s.strip() for s in advice)):
                    raise ValueError("advice must contain nonempty complete strings")
                if (cid == "R01" and stage != "particle_overlap_and_extra_density"
                        or cid == "Q01" and stage != "fsc_mask_interpretation"):
                    raise ValueError("R01/Q01 stage mismatch")
                seen.add(cid)
                snapshot.append((cid, stage, tuple(advice)))
            self._cards = tuple(sorted(snapshot))
            self._metadata.update(enabled=True, version=document["version"],
                                  source_sha256=document["source_sha256"],
                                  normalized_sha256=sha256(raw).hexdigest(), card_count=len(snapshot))
        except Exception as exc:
            self._error = str(exc)
            _LOG.warning("Supplemental guidance disabled: %s", exc)

    def metadata(self):
        return deepcopy({**self._metadata, "error": self._error})

    def inject(self, messages, *, last_completed_job=None, candidate_actions=None,
               workflow_state=None, quality_evidence=None, audience="executor", audit=None):
        """Insert one dynamic user message without altering any original message object."""
        record = {**self.metadata(), "audience": audience, "matched_card_ids": [],
                  "injected_card_ids": [], "omitted_card_ids": [], "budget_used": 0,
                  "injection_position": None, "actual_injected_text": None}
        result = messages
        try:
            if self._error:
                raise ValueError(self._error)
            if audience not in {"executor", "planner"}:
                raise ValueError("unknown guidance audience")
            stages = set()
            if (isinstance(last_completed_job, dict)
                    and last_completed_job.get("status", "completed") == "completed"):
                stages.update(JOB_STAGES.get(last_completed_job.get("job_type"), ()))
            for action in candidate_actions or ():
                if not isinstance(action, dict) or action.get("available") is False or action.get("blocked_by"):
                    continue
                stages.update(JOB_STAGES.get(action.get("job_type"), ()) or
                              ROLLBACK_STAGES.get(action.get("rollback_kind"), ()))
            stages.update(_topic_stages([last_completed_job, workflow_state, quality_evidence]))
            matched = [card for card in self._cards if card[1] in stages]
            record["matched_card_ids"] = [card[0] for card in matched]
            selected = []

            def render(cards):
                return _json({"supplemental_guidance": {"usage": USAGE, "cards": cards}})

            for cid, stage, advice in matched:
                card = {"card_id": cid, "stage": stage, "advice": list(advice)}
                if len(render(selected + [card]).encode("utf-8")) <= self._metadata["max_bytes"]:
                    selected.append(card)
                else:
                    record["omitted_card_ids"].append(cid)
            record["injected_card_ids"] = [card["card_id"] for card in selected]
            if record["omitted_card_ids"]:
                record["error"] = "whole cards omitted for byte budget: " + ",".join(record["omitted_card_ids"])
                _LOG.warning("Supplemental guidance: %s", record["error"])
            if selected:
                content = render(selected)
                index = 0
                while index < len(messages) and messages[index].get("role") == "system":
                    index += 1
                result = list(messages)
                result.insert(index, {"role": "user", "content": content})
                record.update(budget_used=len(content.encode("utf-8")), injection_position=index,
                              actual_injected_text=content)
        except Exception as exc:
            record["error"] = str(exc)
            _LOG.warning("Supplemental guidance skipped: %s", exc)
            result = messages
        if audit is not None:
            audit.update(deepcopy(record))
        return result

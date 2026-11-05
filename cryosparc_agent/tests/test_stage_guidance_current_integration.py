"""Reviewed supplements must not replace the current execution contracts."""
import copy
import importlib
import json
import sys
import types

import pytest

from experiment_planner import build_planning_prompt, validate_plan
from stage_guidance import SupplementalGuidance


@pytest.fixture
def runner(monkeypatch):
    try:
        import mcp
    except ImportError:
        for name in ("mcp", "mcp.client", "mcp.client.stdio"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        sys.modules["mcp"].ClientSession = object
        sys.modules["mcp"].StdioServerParameters = object
        sys.modules["mcp.client.stdio"].stdio_client = object
    return importlib.import_module("autonomous_mcp_closed_loop")


def without_advice(messages, audit):
    result = list(messages)
    assert result.pop(audit["injection_position"])["content"] == audit["actual_injected_text"]
    return result


@pytest.mark.parametrize("stage,kind,job", [
    ("blob_picker_diameter", "micrograph_preview", "blob_picker_gpu"),
    ("inspect_picks_threshold", "pick_inspection", "inspect_picks_v2"),
    ("extract_box_size", "extract_box_comparison", "extract_micrographs_multi"),
])
def test_visual_early_return_receives_advice_without_changing_contract(runner, stage, kind, job):
    model_input = {"dataset_info": {"pixel_size_A": .6575}, "current_state": {}}
    candidates = {"candidate_actions": [{"job_type": job, "available": True, "action_id": "next"}]}
    visual = {"decision_stage": stage, "kind": kind, "visual_evidence_version": "full_field_v4",
              "evidence_images": [{"data_url": "data:image/png;base64,example"}]}
    before = copy.deepcopy((model_input, candidates, visual))
    baseline = runner.build_autonomous_prompt(model_input, candidates, 1, visual)
    audit = {}
    messages = runner.build_autonomous_prompt(model_input, candidates, 1, visual,
        stage_guidance=SupplementalGuidance(), guidance_audit=audit)
    assert audit["injected_card_ids"]
    assert without_advice(messages, audit) == baseline
    assert messages[0] == baseline[0]
    assert (model_input, candidates, visual) == before
    serialized = json.dumps(messages)
    if stage == "extract_box_size":
        assert "do not Fourier-crop" in serialized
        assert "1.5-2" in serialized
    if stage == "inspect_picks_threshold":
        assert "raw" in serialized and "not normalized scores" in serialized


def test_general_executor_system_cache_and_visual_evidence_remain_identical(runner):
    context = {"candidate_actions": [{"job_type": "select_2D", "action_id": "s", "available": True}]}
    visual = {"kind": "class_average", "contact_sheet": {"data_url": "data:image/png;base64,class"}}
    kwargs = {"mark_static_cache_breakpoint": True, "kb_tool_policy": "disabled"}
    baseline = runner.build_autonomous_prompt({}, context, 2, visual, **kwargs)
    audit = {}
    messages = runner.build_autonomous_prompt({}, context, 2, visual,
        stage_guidance=SupplementalGuidance(), guidance_audit=audit, **kwargs)
    assert without_advice(messages, audit) == baseline
    assert messages[:2] == baseline[:2]
    assert "D05" in audit["injected_card_ids"]


def test_local_compact_visual_prompt_is_not_changed(runner):
    context = {"candidate_actions": [{"job_type": "blob_picker_gpu", "available": True}]}
    visual = {"decision_stage": "blob_picker_diameter"}
    baseline = runner.build_autonomous_prompt({}, context, 1, visual, visual_sft_mode=True)
    messages = runner.build_autonomous_prompt({}, context, 1, visual, visual_sft_mode=True,
        stage_guidance=SupplementalGuidance())
    assert messages == baseline


def test_planner_preserves_payload_contract_candidates_and_validation():
    candidates = [{"action_id": "extract", "job_type": "extract_micrographs_multi", "available": True,
                   "rollback_kind": "new_box_size"}]
    observation = {"dag": {"nodes": [{"job_uid": "J8", "job_type": "homo_refine_new", "status": "completed"}]}}
    before = copy.deepcopy((candidates, observation))
    baseline = build_planning_prompt(observation, candidates, [], 1)
    audit = {}
    messages = build_planning_prompt(observation, candidates, [], 1,
        stage_guidance=SupplementalGuidance(), guidance_audit=audit)
    assert without_advice(messages, audit) == baseline
    assert (candidates, observation) == before
    assert {"D04", "D08"} <= set(audit["injected_card_ids"])
    plan = {"decision_type": "plan_experiments", "approved_action_ids": ["extract"]}
    assert validate_plan(plan, candidates, 1)["approved_action_ids"] == ["extract"]
    # No new requested_inputs requirement or recovery-family budget is added.
    assert validate_plan({"decision_type": "request_input"}, candidates, 1)["decision_type"] == "request_input"


def test_guidance_cli_disable_and_default_byte_budget(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner", "--project", "P1", "--workspace", "W1"])
    args = runner.parse_args()
    assert not args.disable_stage_guidance and args.guidance_max_bytes == 8192
    monkeypatch.setattr(sys, "argv", sys.argv + ["--disable-stage-guidance"])
    assert runner.parse_args().disable_stage_guidance

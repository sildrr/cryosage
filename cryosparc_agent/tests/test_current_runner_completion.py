import asyncio
from contextlib import asynccontextmanager
import importlib
import json
import sys
import types
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    try:
        import mcp
    except ImportError:
        for name in ("mcp", "mcp.client", "mcp.client.stdio"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        sys.modules["mcp"].ClientSession = object
        sys.modules["mcp"].StdioServerParameters = object
        sys.modules["mcp.client.stdio"].stdio_client = object
    module = importlib.import_module("autonomous_mcp_closed_loop")

    @asynccontextmanager
    async def transport(_):
        yield (None, None)

    class Session:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def initialize(self):
            pass

    monkeypatch.setattr(module, "stdio_client", transport)
    monkeypatch.setattr(module, "ClientSession", Session)
    monkeypatch.setattr(module, "StdioServerParameters", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "collect_job_logs", lambda *_: {})
    monkeypatch.setattr(module, "run_openai_compatible_model", lambda *a, **k: pytest.fail("No model call allowed after confirmed attainment"))
    monkeypatch.setattr(sys, "argv", ["runner", "--project", "P6", "--workspace", "W34",
        "--current-node", "J611", "--backend", "api", "--dataset-json", '{"target_resolution_A":2.8}',
        "--max-rounds", "4", "--vision", "--output-dir", str(tmp_path)])
    return module, tmp_path


def refine():
    return {"job_uid": "J610", "job_type": "homo_refine_new", "status": "completed",
        "quality_assessment": {"resolution_evidence": {"values": {"fsc_info.radwn_final_A": 2.6456}}}}


def validated():
    return {"job_uid": "J611", "job_type": "validation", "status": "completed", "ready_for_model": True,
        "inputs": {"groups": [{"name": "volume", "connected_job_uids": ["J610"]}]},
        "quality_assessment": {"resolution_evidence": {"values": {"fsc_resolution": 2.6529}}}}


def candidate():
    return {"action_id": "validate_J610", "job_type": "validation", "available": True,
        "required_inputs": {"volume": [{"source_job_uid": "J610", "source_output": "volume"}]},
        "parameter_template": {"validate_do_noisesub": {"type": "boolean"}}}


def test_actual_runner_stops_j611_before_visual_routing_or_submission(runner, monkeypatch):
    module, root = runner
    calls = []

    async def tool(_, name, arguments):
        calls.append(name)
        if name == "get_workflow_decision_context":
            return {"current_state": {"last_action": "validation", "last_node_info": validated()}}
        if name == "get_candidate_actions":
            return {"candidate_actions": [{"job_type": "inspect_picks_v2", "available": True}]}
        if name == "get_job_result_package":
            return refine()
        pytest.fail(f"Unexpected tool after target met: {name}")

    monkeypatch.setattr(module, "call_tool_json", tool)
    asyncio.run(module.main_async())
    summary = json.loads(next(root.glob("run_*/summary.json")).read_text())
    assert summary["stop_reason"] == "target_resolution_met"
    assert summary["success"] is True
    assert "execute_v2_model_decision" not in calls
    assert "get_pick_inspection_visual_context" not in calls


def test_actual_runner_validates_once_then_stops_without_model_or_upstream_jobs(runner, monkeypatch):
    module, root = runner
    monkeypatch.setattr(sys, "argv", ["runner", "--project", "P6", "--workspace", "W34",
        "--current-node", "J610", "--backend", "api", "--dataset-json", '{"target_resolution_A":2.8}',
        "--max-rounds", "4", "--vision", "--output-dir", str(root)])
    submitted = []

    async def tool(_, name, arguments):
        if name == "get_workflow_decision_context":
            info = refine() if arguments["current_job_uid"] == "J610" else validated()
            return {"current_state": {"last_action": info["job_type"], "last_node_info": info}}
        if name == "get_candidate_actions":
            return {"candidate_actions": [candidate()], "workflow_guidance": {}}
        if name == "get_reconstruction_observation":
            return {"dag": {"nodes": []}}
        if name == "get_job_result_package":
            return refine()
        if name == "validate_v2_model_decision":
            return {"success": True}
        if name == "execute_v2_model_decision":
            submitted.append(arguments["decision"])
            return {"execution_result": {"execution_results": [{"success": True, "job_uid": "J611", "job_type": "validation", "status": "queued", "queued": True}]}}
        if name == "wait_for_job_result_package":
            return validated()
        pytest.fail(f"Unexpected tool: {name}")

    monkeypatch.setattr(module, "call_tool_json", tool)
    asyncio.run(module.main_async())
    assert len(submitted) == 1 and submitted[0]["action"] == "validation"
    assert submitted[0]["parameters"]["validate_do_noisesub"] is True
    summary = json.loads(next(root.glob("run_*/summary.json")).read_text())
    assert summary["stop_reason"] == "target_resolution_met"


def test_actual_preflight_wrapper_reviews_then_revalidates_full_decision(runner, monkeypatch):
    module, root = runner
    args = module.parse_args()
    args.extract_review_micrographs = 6
    args.api_key = None
    args.max_rounds = 4
    current = "J605"
    initial = {"schema_version": "2.0", "decision_type": "forward", "action": "extract_micrographs_multi",
               "action_id": "extract", "parameters": {"box_size_pix": 640}, "reason": "initial"}
    spec = {"action_id": "extract", "default_parameters": {"box_size_pix": 384}, "required_inputs": {
        "particles": [{"source_job_uid": "J605", "source_output": "particles"}],
        "micrographs": [{"source_job_uid": "J605", "source_output": "micrographs"}],
    }}
    summary = {"extract_reviews": {}, "completion": {}, "experiment_history": [], "kb_escalation": {}}
    submitted, validated_decisions = [], []

    async def tool(_, name, arguments):
        if name == "validate_v2_model_decision":
            validated_decisions.append(arguments["decision"])
            return {"success": True}
        if name == "execute_v2_model_decision":
            submitted.append(arguments["decision"])
            return {"execution_result": {"execution_results": [{"success": True, "job_uid": f"P{len(submitted)}", "job_type": "extract_micrographs_multi", "status": "queued", "queued": True}]}}
        if name == "wait_for_job_result_package":
            return {"status": "completed", "ready_for_model": True}
        if name == "get_reconstruction_observation":
            return {"dag": {"nodes": []}}
        pytest.fail(f"Unexpected preflight tool: {name}")

    import extract_review_evidence
    import cryosparc_client
    job = Mock()
    client = Mock()
    client.find_project.return_value.find_job.return_value = job
    monkeypatch.setattr(cryosparc_client, "cryosparc_client", lambda: client)
    monkeypatch.setattr(extract_review_evidence, "build_evidence", lambda *a: ([], {"samples": [{"sample_id": "S01"}]}))
    responses = iter([
        {"action": "compare", "box_size_pix": 440, "reason": "test a smaller box"},
        {"action": "use_alternative", "box_size_pix": 440, "central_target_complete": True,
         "supporting_sample_ids": ["S01"], "reason": "center retained, less peripheral signal"},
    ])

    def model(*a, **kwargs):
        return {"raw_text": json.dumps(next(responses)), "raw_response": {"choices": [{"finish_reason": "stop"}]}}

    monkeypatch.setattr(module, "run_openai_compatible_model", model)
    monkeypatch.setattr(module, "resolve_api_key", lambda *_: "fake")
    monkeypatch.setattr(module, "call_tool_json", tool)
    round_dir = root / "round_01"
    round_dir.mkdir()
    result = asyncio.run(module.run_extraction_preflight(args, None, initial, spec, current,
        {}, summary, root, round_dir, 1, {}))
    assert result["parameters"] == {"box_size_pix": 440}
    assert result["connections"]["particles"]["source_job_uid"] == current
    assert len(submitted) == 2
    assert all(d["parameters"]["num_extract"] == 6 for d in submitted)
    assert validated_decisions[-1] == result
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    assert checkpoint["current_node"] == current and checkpoint["pending_jobs"] == []
    restored = {**summary, "extract_reviews": checkpoint["extract_reviews"]}
    again = asyncio.run(module.run_extraction_preflight(args, None, initial, spec, current,
        {}, restored, root, round_dir, 1, {}))
    assert again == result and len(submitted) == 2

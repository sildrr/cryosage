import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from workflow_completion import completion_check, resolve_completion, validation_decision


def info(uid="J610", kind="homo_refine_new", resolution=2.6456):
    key = "fsc_resolution" if kind == "validation" else "fsc_info.radwn_final_A"
    return {"job_uid": uid, "job_type": kind, "status": "completed", "metrics": {},
            "quality_assessment": {"resolution_evidence": {"values": {key: resolution}}}}


def candidate():
    return {"action_id": "v", "job_type": "validation", "available": True,
            "required_inputs": {"volume": [{"source_job_uid": "J610", "source_output": "volume"}]},
            "parameter_template": {"validate_do_noisesub": {"type": "boolean"}}}


def validation(resolution=2.6529):
    node = info("J611", "validation", resolution)
    node["inputs"] = {"groups": [{"name": "volume", "connected_job_uids": ["J610"]}]}
    return node


def test_j610_gets_only_one_matched_validation():
    result = completion_check(info(), 2.8, [candidate()])
    assert result["action"] == "validate"
    decision = validation_decision(result)
    assert decision["action"] == "validation"
    assert decision["parameters"] == {"validate_do_noisesub": True}
    assert decision["connections"]["volume"]["source_job_uid"] == "J610"


def test_j611_stops_instead_of_reentering_picking():
    result = completion_check(validation(), 2.8, [], info())
    assert result["action"] == "stop"
    assert result["stop_reason"] == "target_resolution_met"
    assert result["validated"] is True


def test_uncorrected_better_value_does_not_trigger_early_stop():
    node = info(resolution=2.8078)
    node["quality_assessment"]["resolution_evidence"]["values"]["fsc_info.radwn_autotight_A"] = 2.7997
    assert completion_check(node, 2.8, [candidate()])["action"] == "continue"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0, -1, None])
def test_invalid_measurements_do_not_stop(value):
    assert completion_check(info(resolution=value), 2.8, [])["action"] == "continue"


def test_other_branch_validation_cannot_mark_this_refine_complete():
    parent = info("J999")
    assert completion_check(validation(), 2.8, [], parent)["action"] == "continue"
    node = validation(3.1)
    assert completion_check(node, 2.8, [], info())["action"] == "continue"


@pytest.mark.parametrize("status", ["running", "failed", "killed"])
def test_noncompleted_jobs_do_not_stop(status):
    node = info()
    node["status"] = status
    assert completion_check(node, 2.8, [])["action"] == "continue"


def test_existing_validation_is_reused_after_refine_resume():
    fetch = AsyncMock(return_value=validation())
    observe = AsyncMock(return_value={"dag": {"nodes": [{"job_uid": "J611", "job_type": "validation", "parents": ["J610"]}]}})
    result = asyncio.run(resolve_completion(info(), 2.8, [candidate()], fetch, observe))
    assert result["action"] == "stop"
    fetch.assert_awaited_once_with("J611")


def test_running_validation_is_waited_not_duplicated():
    node = validation()
    node["status"] = "running"
    fetch = AsyncMock(return_value=node)
    observe = AsyncMock(return_value={"dag": {"nodes": [{"job_uid": "J611", "job_type": "validation", "parents": ["J610"]}]}})
    result = asyncio.run(resolve_completion(info(), 2.8, [candidate()], fetch, observe))
    assert result["action"] == "wait_validation"


def test_failed_or_disagreeing_existing_validation_requires_attention():
    observe = AsyncMock(return_value={"dag": {"nodes": [{"job_uid": "J611", "job_type": "validation", "parents": ["J610"]}]}})
    result = asyncio.run(resolve_completion(info(), 2.8, [candidate()], AsyncMock(return_value=validation(3.1)), observe))
    assert result["action"] == "hold"


@pytest.mark.parametrize("status", ["completed", "failed", "killed"])
def test_resume_at_disagreeing_validation_also_requires_attention(status):
    node = validation(3.1)
    node["status"] = status
    result = asyncio.run(resolve_completion(
        node, 2.8, [], AsyncMock(return_value=info()), AsyncMock()))
    assert result["action"] == "hold"
    assert result["stop_reason"] == "target_validation_needs_attention"


def test_exact_target_is_met_and_absent_validation_can_finalize():
    result = completion_check(info(resolution=2.8), 2.8, [])
    assert result["action"] == "stop" and result["validated"] is False

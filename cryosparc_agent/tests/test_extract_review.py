import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from extract_review import (review_extraction, validate_review, ReviewBlocked,
                           historical_extract_results, is_full_extraction, without_review_pilots,
                           resolve_review_connections)


@pytest.mark.parametrize("source", [
    {"job_uid": "J5559", "output": "particles"},
    {"source_job_uid": "J5559", "source_output": "particles"},
    {"source_job": "J5559", "source_output": "particles"},
    ("J5559", "particles"),
    [{"job_uid": "J5559", "output": "particles"}],
])
def test_review_accepts_executor_connection_dialects(source):
    original = {"connections": {"particles": source}}
    before = copy.deepcopy(original)
    assert resolve_review_connections(original, {})["particles"] == {
        "source_job_uid": "J5559", "source_output": "particles"}
    assert original == before


def test_review_only_infers_a_single_candidate_source():
    source = {"source_job_uid": "J5559", "source_output": "particles"}
    assert resolve_review_connections({}, {"required_inputs": {"particles": [source]}})["particles"] == source
    for decision, candidate in [({}, {}), ({}, {"required_inputs": {"particles": [source, source]}}),
            ({"connections": {"particles": [source, source]}}, {}),
            ({"connections": {"particles": {"job_uid": "J5559"}}}, {})]:
        with pytest.raises(ReviewBlocked):
            resolve_review_connections(decision, candidate)


def decision():
    return {"action": "extract_micrographs_multi", "decision_type": "forward", "action_id": "extract",
            "parameters": {"box_size_pix": 640}, "reason": "Initial visual estimate",
            "connections": {"particles": {"source_job_uid": "J605", "source_output": "particles"}}}


def keep():
    return {"action": "keep", "reason": "adequate"}


def compare():
    return {"action": "compare", "box_size_pix": 440, "reason": "test less peripheral signal"}


def use_alternative():
    return {"action": "use_alternative", "box_size_pix": 440, "central_target_complete": True,
            "supporting_sample_ids": ["S01"], "reason": "Matched target intact in smaller box"}


def harness(state, responses, remaining=10, submit=None, wait=None):
    submit = submit or AsyncMock(side_effect=["Jpilot1", "Jpilot2"])
    wait = wait or AsyncMock(return_value={"status": "completed"})
    observe = AsyncMock(side_effect=responses)
    snapshots = []
    result = asyncio.run(review_extraction(state, decision(), submit, wait, observe,
        lambda: snapshots.append(copy.deepcopy(state)), remaining))
    return result, submit, wait, observe, snapshots


def test_keep_submits_one_small_pilot_and_preserves_full_input():
    initial = decision()
    result, submit, _, _, snapshots = harness({}, [keep()])
    assert result == initial
    sent = submit.await_args.args[0]
    assert sent["parameters"]["num_extract"] == 6
    assert sent["parameters"]["compute_num_gpus"] == 1
    assert sent["connections"] == initial["connections"]
    assert "num_extract" not in result["parameters"]
    assert snapshots[1]["jobs"][0]["status"] == "submitting"


def test_compare_then_apply_only_verified_alternative():
    state = {}
    result, submit, _, observe, _ = harness(state, [compare(), use_alternative()])
    assert result["parameters"]["box_size_pix"] == 440
    assert "num_extract" not in result["parameters"]
    assert submit.await_count == 2
    assert observe.await_args_list[-1].args == ("Jpilot1", "Jpilot2")
    assert state["status"] == "ready"
    again, submit2, wait2, observe2, _ = harness(state, [])
    assert again == result
    assert submit2.await_count == wait2.await_count == observe2.await_count == 0


def test_uncertain_comparison_keeps_original():
    result, submit, _, _, _ = harness({}, [compare(), {"action": "insufficient_evidence", "reason": "weak boundaries"}])
    assert result["parameters"]["box_size_pix"] == 640
    assert submit.await_count == 2


def test_restart_after_api_error_reuses_completed_pilot():
    state = {}
    with pytest.raises(RuntimeError):
        harness(state, [RuntimeError("API unavailable")])
    result, submit, wait, _, _ = harness(state, [keep()])
    assert result["parameters"]["box_size_pix"] == 640
    assert submit.await_count == wait.await_count == 0


def test_ambiguous_submission_is_not_automatically_duplicated():
    state = {}
    with pytest.raises(RuntimeError):
        harness(state, [], submit=AsyncMock(side_effect=RuntimeError("connection lost")))
    submit = AsyncMock()
    with pytest.raises(ReviewBlocked, match="Ambiguous"):
        harness(state, [], submit=submit)
    assert submit.await_count == 0


def test_failed_pilot_blocks_full_submission():
    with pytest.raises(ReviewBlocked, match="pilot failed"):
        harness({}, [], wait=AsyncMock(return_value={"status": "failed"}))


@pytest.mark.parametrize("size", [True, 0, 32, 441, 728, 640])
def test_invalid_or_same_comparison_size_rejected(size):
    with pytest.raises(ReviewBlocked):
        validate_review({"action": "compare", "box_size_pix": size, "reason": "test"}, 640)


def test_no_untested_or_incomplete_alternative_can_be_applied():
    for field, value in [("box_size_pix", 384), ("central_target_complete", False), ("supporting_sample_ids", [])]:
        review = use_alternative()
        review[field] = value
        with pytest.raises(ReviewBlocked):
            validate_review(review, 640, 440)


def test_budget_reserves_full_extraction_and_does_not_force_comparison():
    submit = AsyncMock(side_effect=["Jpilot1"])
    with pytest.raises(ReviewBlocked, match="budget"):
        harness({}, [], remaining=1, submit=submit)
    assert submit.await_count == 0
    with pytest.raises(ReviewBlocked, match="budget"):
        harness({}, [compare()], remaining=2, submit=submit)
    assert submit.await_count == 1


def test_partial_or_binned_extractions_skip_preflight():
    assert is_full_extraction(decision(), {})
    for params in [{"num_extract": 6}, {"bin_size_pix": 128}]:
        candidate = {"default_parameters": params}
        assert not is_full_extraction(decision(), candidate)
    assert not is_full_extraction(decision(), {"required_inputs": {"micrographs": [{"num_items": 1}]}})


def test_pilot_nodes_do_not_pollute_normal_candidates():
    state = {"nodes": [{"title": "Extract review pilot: x", "cryosparc_job_uid": "P"},
                       {"title": "Agent full extract", "cryosparc_job_uid": "E"}]}
    assert [n["cryosparc_job_uid"] for n in without_review_pilots(state)["nodes"]] == ["E"]
    assert len(state["nodes"]) == 2


def test_filtered_pilot_children_do_not_turn_registry_extract_into_branch(monkeypatch):
    import action_registry as registry

    state = {"nodes": [
        {"workflow_node_id": "J1", "cryosparc_job_uid": "J1", "job_type": "inspect_picks_v2",
         "child_job_uids": ["J2", "J3"], "status": "completed"},
    ] + [
        {"workflow_node_id": uid, "cryosparc_job_uid": uid, "job_type": "extract_micrographs_multi",
         "title": "Extract review pilot: test", "status": "completed"}
        for uid in ("J2", "J3")
    ]}
    forward = {"action_id": "registry_extract", "action_type": "forward",
               "job_type": "extract_micrographs_multi"}
    def registry_actions(state, current, candidate_job_types):
        assert candidate_job_types == set()
        return [forward], []

    monkeypatch.setattr(registry, "build_synthetic_next_actions", lambda *a, **kw: ([], []))
    monkeypatch.setattr(registry, "build_registry_next_actions", registry_actions)
    monkeypatch.setattr(registry, "build_candidate_action",
                        lambda *a: pytest.fail("Technical pilot became a normal candidate"))
    monkeypatch.setattr(registry, "apply_auto_select_2d_policy", lambda action, state: action)
    monkeypatch.setattr(registry, "apply_hard_workflow_gates", lambda actions, *a, **kw: (actions, []))
    monkeypatch.setattr(registry, "infer_current_stage", lambda *a, **kw: {"current_stage": "extract"})
    monkeypatch.setattr(registry, "annotate_candidates", lambda actions, stage: actions)
    actions, blocked = registry.generate_candidate_actions(
        without_review_pilots(state), "J1", include_registry=True)
    assert actions == [forward]
    assert not blocked


def test_history_does_not_attribute_reference_volume_branch_to_particle_box():
    observation = {"dag": {"nodes": [
        {"job_uid": "Eold", "job_type": "extract_micrographs_multi", "parameters": {"box_size_pix": 320}},
        {"job_uid": "V", "job_type": "homo_refine_new", "parents": ["Eold"]},
        {"job_uid": "E", "job_type": "extract_micrographs_multi", "parameters": {"box_size_pix": 440}},
        {"job_uid": "S", "job_type": "select_2D", "parents": ["E"]},
        {"job_uid": "R", "job_type": "homo_refine_new", "status": "completed", "parents": ["V", "S"],
         "quality": {"resolution_A": {"fsc_info.radwn_final_A": 2.65}}},
    ]}}
    result = historical_extract_results(observation)
    assert len(result) == 1 and result[0]["box_size_pix"] == 440

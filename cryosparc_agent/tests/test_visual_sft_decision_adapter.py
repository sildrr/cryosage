import pytest

from autonomous_mcp_closed_loop import (
    build_autonomous_prompt,
    build_visual_sft_prompt,
    normalize_visual_sft_decision,
)
from scripts.model_generate_vision_once import merge_system_messages


def test_blob_sft_decision_maps_schema_and_processes_all_micrographs():
    result = normalize_visual_sft_decision(
        {
            "decision": "blob_picker_gpu",
            "action_id": "initial_scale_pilot_1_blob_picker_gpu_J470",
            "parameters": {"diameter": 137, "diameter_max": 220, "min_distance": 1},
        },
        {
            "candidate_actions": [{
                "action_id": "forward_J470_blob_picker_gpu",
                "action_type": "forward",
                "job_type": "blob_picker_gpu",
                "available": True,
            }]
        },
        "J470",
        {"current_state": {"last_action": "patch_ctf_estimation_multi"}},
        {"micrograph_count": 196},
    )
    assert result["decision_type"] == "forward"
    assert result["action"] == "blob_picker_gpu"
    assert result["action_id"] == "forward_J470_blob_picker_gpu"
    assert result["parameters"] == {
        "diameter": 137,
        "diameter_max": 220,
        "num_process": 196,
    }


def test_blob_sft_accepts_min_max_aliases_and_rejects_inverted_range():
    candidate = {"candidate_actions": [{
        "action_id": "forward_J80_blob_picker_gpu",
        "action_type": "forward",
        "job_type": "blob_picker_gpu",
        "available": True,
    }]}
    result = normalize_visual_sft_decision(
        {"decision": "blob_picker_gpu", "parameters": {
            "diameter_min_A": 120, "diameter_max_A": 240,
        }},
        candidate,
        "J80",
        {"current_state": {"last_action": "patch_ctf_estimation_multi"}},
        None,
    )
    assert result["parameters"] == {"diameter": 120, "diameter_max": 240}
    with pytest.raises(ValueError, match="must not exceed"):
        normalize_visual_sft_decision(
            {"decision": "blob_picker_gpu", "parameters": {
                "diameter": 250, "diameter_max": 180,
            }},
            candidate,
            "J80",
            {"current_state": {"last_action": "patch_ctf_estimation_multi"}},
            None,
        )


def test_inspect_sft_parameter_names_map_to_cryosparc_registry():
    result = normalize_visual_sft_decision(
        {
            "decision": "inspect_picks_v2",
            "parameters": {
                "ncc_score_min": 0.2,
                "local_power_min": 12,
                "local_power_max": 24,
            },
        },
        {
            "candidate_actions": [{
                "action_id": "forward_J80_inspect_picks_v2",
                "action_type": "forward",
                "job_type": "inspect_picks_v2",
                "available": True,
            }]
        },
        "J80",
        {"current_state": {"last_action": "blob_picker_gpu"}},
        None,
    )
    assert result["action_id"] == "forward_J80_inspect_picks_v2"
    assert result["parameters"] == {
        "ncc_score_thresh": 0.2,
        "lpower_thresh_min": 12,
        "lpower_thresh_max": 24,
    }


def test_visual_sft_refuses_ambiguous_current_job_candidates():
    candidate = {
        "action_type": "forward",
        "job_type": "extract_micrographs_multi",
        "available": True,
    }
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_visual_sft_decision(
            {"decision": "extract_micrographs_multi", "parameters": {"box_size_pix": 384}},
            {"candidate_actions": [
                {**candidate, "action_id": "forward_J90_extract_micrographs_multi_a"},
                {**candidate, "action_id": "forward_J90_extract_micrographs_multi_b"},
            ]},
            "J90",
            {"current_state": {"last_action": "inspect_picks_v2"}},
            None,
        )


def test_qwen_worker_merges_multiple_system_messages_into_one_leading_message():
    messages = merge_system_messages([
        {"role": "system", "content": "Base instructions."},
        {"role": "system", "content": [{"type": "text", "text": "Workflow rules."}]},
        {"role": "user", "content": "Decide."},
    ])
    assert messages[0] == {
        "role": "system",
        "content": "Base instructions.\n\nWorkflow rules.",
    }
    assert messages[1] == {"role": "user", "content": "Decide."}
    assert sum(message["role"] == "system" for message in messages) == 1


def test_visual_sft_prompt_uses_stage_contract_without_candidate_ids():
    messages = build_visual_sft_prompt(
        {
            "dataset_info": {
                "empiar_id": "EMPIAR-10025",
                "pixel_size_A": 0.6575,
                "accelerating_voltage_kv": 300,
            },
            "current_state": {"last_node_id": "J470", "last_node_status": "completed"},
        },
        {
            "decision_stage": "blob_picker_diameter",
            "kind": "micrograph_preview",
            "micrograph_count": 196,
            "contact_sheet": {"data_url": "data:image/png;base64,AA=="},
        },
    )
    assert messages[0]["role"] == "system"
    assert len(messages) == 2
    assert '"diameter_max":"optional number"' in messages[1]["content"][0]["text"]
    assert all("action_id" not in str(message) for message in messages)


def test_api_visual_stage_adds_reason_without_changing_local_evidence():
    model_input = {
        "dataset_info": {"empiar_id": "10025", "pixel_size_A": 0.6575},
        "current_state": {"last_node_id": "J478", "last_node_status": "completed"},
    }
    candidates = {"candidate_actions": [{
        "action_id": "forward_J478_inspect_picks_v2",
        "action_type": "forward",
        "job_type": "inspect_picks_v2",
        "available": True,
        "parameter_template": {"ncc_score_thresh": {"type": "number"}},
    }]}
    visual = {
        "decision_stage": "inspect_picks_threshold",
        "kind": "pick_inspection",
        "score_statistics": {"ncc_score": {"p50": 0.15}},
        "contact_sheet": {"data_url": "data:image/png;base64,AA=="},
        "score_scatter": {"data_url": "data:image/png;base64,BB=="},
        "evidence_image_descriptions": ["score-quadrant crop 1", "score-quadrant crop 2"],
        "evidence_images": [
            {"role": "score_crop", "description": f"quadrant {i}",
             "data_url": f"data:image/png;base64,EVIDENCE{i}=="}
            for i in range(2)
        ],
    }
    api_messages = build_autonomous_prompt(
        model_input, candidates, 1, visual_context=visual, visual_sft_mode=False
    )
    local_messages = build_autonomous_prompt(
        model_input, candidates, 1, visual_context=visual, visual_sft_mode=True
    )
    assert api_messages[0] == local_messages[0]
    assert api_messages[1]["content"][1:] == local_messages[1]["content"][1:]
    assert '"reason"' in api_messages[1]["content"][0]["text"]
    assert '"reason"' not in local_messages[1]["content"][0]["text"]
    assert len(api_messages[1]["content"]) == 5
    assert "candidate_actions" in api_messages[1]["content"][0]["text"]
    assert "data:image/png;base64,EVIDENCE0==" in str(api_messages)
    assert "data:image/png;base64,EVIDENCE1==" in str(api_messages)
    assert "EVIDENCE0" not in api_messages[1]["content"][0]["text"]

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import action_registry as registry
import autonomous_mcp_closed_loop as runner
from model_input_builder import build_dataset_info


def candidate():
    return {
        "action_id": "forward_J1_homo_refine_new", "job_type": "homo_refine_new",
        "available": True, "description": "Baseline refinement.",
        "default_parameters": {"refine_symmetry": "C1"},
        "parameter_template": {"refine_symmetry": {
            "type": "string", "default": "C1", "enum": ["C1"],
        }},
    }


@pytest.mark.parametrize("dataset,unlocked", [
    ({"symmetry_status": "unknown"}, True),
    ({"symmetry": "C1"}, False),
    ({"symmetry": "D2", "symmetry_status": "unknown"}, False),
    ({}, False),
])
def test_unknown_policy_only_applies_to_explicit_unknown_runs(dataset, unlocked):
    state = {"nodes": [], "generated_at": "test", "workflow_status": "not_started"}
    original = candidate()
    with patch.object(registry, "extract_workflow_state", return_value=state), \
         patch.object(registry, "generate_candidate_actions", return_value=([original], [])), \
         patch.object(registry, "build_initial_import_candidates", return_value=[]), \
         patch.object(registry, "infer_current_stage", return_value={}), \
         patch.object(registry, "build_decision_guidance", return_value={}):
        result = registry.get_candidate_actions("P1", "W1", dataset_info=dataset)
    spec = result["candidate_actions"][0]["parameter_template"]["refine_symmetry"]
    assert ("enum" not in spec) == unlocked
    assert original["parameter_template"]["refine_symmetry"]["enum"] == ["C1"]
    if unlocked:
        assert not registry.validate_parameter_value("refine_symmetry", "D2", spec, "parameters")
        assert registry.validate_parameter_value("refine_symmetry", "invalid", spec, "parameters")


def test_abinitio_and_refinement_defaults_are_not_hard_constraints_when_unknown():
    for name in ("abinit_symmetry", "refine_symmetry"):
        action = {"parameter_template": {name: {"type": "string", "default": "C1",
                                                  "enum": ["C1"], "const": "C1"}}}
        result = registry.allow_unknown_symmetry(action)
        spec = result["parameter_template"][name]
        assert spec["default"] == "C1"
        assert "enum" not in spec and "const" not in spec
        assert not registry.validate_parameter_value(name, "D2", spec, "parameters")


def test_unknown_status_survives_model_payload():
    dataset = build_dataset_info({"symmetry_status": "unknown"}, known_workflow_dirs=[])
    assert dataset["symmetry"] is None
    assert dataset["symmetry_status"] == "unknown"


def test_symmetry_evidence_reaches_planner_before_decision():
    evidence = {"success": True, "recommended_symmetry": "D2"}
    args = SimpleNamespace(project="P1", workspace="W1", cryosift=False, planning_mode="policy")
    with patch.object(runner, "call_tool_json", AsyncMock(return_value={})), \
         patch.object(runner, "build_policy_plan", return_value={"decision_type": "wait"}) as policy, \
         patch.object(runner, "apply_anti_refinement_loop", side_effect=lambda p, *a: p):
        observation, _, _ = asyncio.run(runner.build_round_plan(
            session=None, args=args, dataset_info={"symmetry_status": "unknown"},
            candidate_context={"candidate_actions": []}, experiment_history=[],
            symmetry_evidence=evidence))
    assert observation["symmetry_evidence"] == evidence
    assert policy.call_args.args[0]["symmetry_evidence"] == evidence

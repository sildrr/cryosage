import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from action_registry import allow_unknown_symmetry, validate_action_against_candidates
from experiment_planner import _input_signature, apply_anti_refinement_loop, build_policy_plan
from schemas import Action
from symmetry_policy import build_symmetry_comparison, read_memory, remember_evidence


def evidence(uid="J10", symmetry="D2", score=.92):
    return dict(success=True, project_uid="P6", job_uid=uid,
                recommended_symmetry=symmetry, average_fsc=score)


def baseline(uid="J10", resolution=3.25, parents=()):
    return dict(cryosparc_job_uid=uid, job_type="homo_refine_new", status="completed",
                parent_job_uids=list(parents), key_parameters={"refine_symmetry": "C1"},
                inputs={"particles": [{"source_job_uid":"J3", "source_output":"particles_selected"}],
                        "volume": [{"source_job_uid":"J4", "source_output":"volume_class_0"}]},
                outputs={"volume":{"latest_summary_stat_values":{"fsc_info":{"radwn_final_A":resolution}}}})


@pytest.fixture
def setup(monkeypatch, tmp_path):
    path = tmp_path / "memory.json"
    monkeypatch.setenv("CRYOAGENT_SYMMETRY_EVIDENCE_FILE", str(path))
    monkeypatch.setattr("symmetry_policy._source_parameters", lambda *_: {
        "refine_symmetry":"C1", "refine_ctf_global_refine":False, "compute_num_gpus":1})
    remember_evidence(path, "P6", "W1", evidence())
    node = baseline()
    return path, dict(project_uid="P6", workspace_uid="W1", nodes=[node]), node


def test_memory_persists_across_restart_without_promoting_symmetry_to_dataset_fact(setup):
    path, _, _ = setup
    assert read_memory(path, "P6", "W1")["J10"]["recommended_symmetry"] == "D2"
    assert read_memory(path, "P6", "W2") == {}
    remember_evidence(path, "P6", "W1", {"success":False, "project_uid":"P6", "job_uid":"J10"})
    assert read_memory(path, "P6", "W1")["J10"]["recommended_symmetry"] == "D2"


def test_same_input_comparison_preserves_baseline_and_other_parameters(setup):
    _, workflow, node = setup
    before = copy.deepcopy(node)
    candidate = build_symmetry_comparison(workflow, node, {"symmetry_status":"unknown"})
    assert candidate["required_inputs"] == before["inputs"]
    assert node == before
    assert candidate["default_parameters"]["refine_symmetry"] == "D2"
    assert candidate["default_parameters"]["refine_ctf_global_refine"] is False
    assert allow_unknown_symmetry(candidate)["parameter_template"]["refine_symmetry"]["const"] == "D2"


@pytest.mark.parametrize("status", ["building", "queued", "running", "completed", "failed"])
def test_already_created_comparison_is_never_created_again(setup, status):
    _, workflow, node = setup
    tried = baseline("J11")
    tried["status"] = status
    tried["key_parameters"]["refine_symmetry"] = "D2"
    workflow["nodes"].append(tried)
    assert build_symmetry_comparison(workflow, node, {"symmetry_status":"unknown"}) is None


@pytest.mark.parametrize("dataset", [{"symmetry":"C1"}, {"symmetry":"D2"}, {}])
def test_explicit_user_symmetry_is_not_overridden(setup, dataset):
    _, workflow, node = setup
    assert build_symmetry_comparison(workflow, node, dataset) is None


def test_foreign_workspace_and_unrelated_branch_evidence_are_not_applied(setup):
    _, workflow, node = setup
    foreign = dict(workflow, workspace_uid="W2")
    assert build_symmetry_comparison(foreign, node, {"symmetry_status":"unknown"}) is None
    unrelated = baseline("J12", parents=())
    workflow["nodes"].append(unrelated)
    assert build_symmetry_comparison(workflow, unrelated, {"symmetry_status":"unknown"}) is None


def test_preserved_ancestral_baseline_is_available_after_template_repick(setup):
    _, workflow, node = setup
    current = baseline("J12", resolution=6, parents=["J10"])
    current["inputs"]["particles"][0]["source_job_uid"] = "J11"
    workflow["nodes"].append(current)
    c = build_symmetry_comparison(workflow, current, {"symmetry_status":"unknown"})
    assert c["reference_job_uid"] == "J10"
    assert c["required_inputs"] == node["inputs"]


def test_planner_and_fallback_schedule_comparison_not_template_or_unchanged_refine(setup):
    _, workflow, node = setup
    c = build_symmetry_comparison(workflow, node, {"symmetry_status":"unknown"})
    template = dict(action_id="template",job_type="create_templates", available=True,
                    refinement_rollback=True, rollback_kind="template_repick",
                    template_repick_gate={"allowed":True})
    obs = {"observed_issues":[{"issue_type":"target_resolution_unmet"}]}
    plan = {"decision_type":"plan_experiments", "approved_action_ids":["template"]}
    assert apply_anti_refinement_loop(plan, obs, [c, template], [], 1)["approved_action_ids"] == [c["action_id"]]
    assert build_policy_plan(obs, [c, template], [], 1)["approved_action_ids"] == [c["action_id"]]
    assert _input_signature(c) != _input_signature({"required_inputs":c["required_inputs"]})


def test_comparison_validator_rejects_changed_particles_or_scientific_parameters(setup):
    _, workflow, node = setup
    c = build_symmetry_comparison(workflow, node, {"symmetry_status":"unknown"})
    action = Action(action_id=c["action_id"], action_type="forward", workflow_node_id=c["workflow_node_id"],
                    job_type=c["job_type"], parameters=c["default_parameters"],
                    connections={"particles":{"source_job_uid":"J99","source_output":"particles"}})
    issues, _, _ = validate_action_against_candidates(action, {c["action_id"]:c}, 0)
    assert any(i.code == "symmetry_comparison_inputs_changed" for i in issues)
    action.connections = None
    action.parameters = {**c["default_parameters"], "refine_ctf_global_refine":True}
    issues, _, _ = validate_action_against_candidates(action, {c["action_id"]:c}, 0)
    assert any(i.code == "symmetry_comparison_parameters_changed" for i in issues)


@pytest.mark.parametrize("cache", [True, False])
def test_controlled_optimization_accepts_declared_cache_without_default(cache):
    c = dict(action_id="opt_optics_J3135", action_type="forward",
        workflow_node_id="opt_optics_J3135", job_type="nonuniform_refine_new",
        execution_mode="create_job", available=True, optimization_contract=True,
        required_inputs=dict(particles=[dict(source_job_uid="J3135", source_output="particles")],
            volume=[dict(source_job_uid="J3135", source_output="volume")]),
        default_parameters=dict(refine_symmetry="C2", refine_defocus_refine=True),
        parameter_template=dict(compute_use_ssd=dict(type="boolean"),
            refine_symmetry=dict(type="string", default="C2", const="C2"),
            refine_defocus_refine=dict(type="boolean", default=True, const=True)))
    a = Action(action_id=c["action_id"], action_type="forward",
        workflow_node_id=c["workflow_node_id"], job_type=c["job_type"],
        parameters=dict(c["default_parameters"], compute_use_ssd=cache))
    issues, _, resolved = validate_action_against_candidates(a, {c["action_id"]: c}, 0)
    assert not issues and resolved.resolved_parameters["compute_use_ssd"] is cache
    a.parameters["compute_use_ssd"] = "true"
    issues, _, _ = validate_action_against_candidates(a, {c["action_id"]: c}, 0)
    assert issues
    a.parameters = dict(c["default_parameters"], compute_use_ssd=cache, refine_defocus_refine=False)
    issues, _, _ = validate_action_against_candidates(a, {c["action_id"]: c}, 0)
    assert any(i.code == "symmetry_comparison_parameters_changed" for i in issues)
    a.parameters = dict(c["default_parameters"], compute_use_ssd=cache)
    c["parameter_template"]["compute_use_ssd"]["const"] = not cache
    issues, _, _ = validate_action_against_candidates(a, {c["action_id"]: c}, 0)
    assert issues
    del c["parameter_template"]["compute_use_ssd"]
    issues, _, _ = validate_action_against_candidates(a, {c["action_id"]: c}, 0)
    assert any(i.code == "symmetry_comparison_parameters_changed" for i in issues)


def test_visual_consistency_context_does_not_reference_undefined_numeric_scores(tmp_path):
    import abinitio_class2d_consistency as module
    abinit = SimpleNamespace(type="homo_abinit",uid="J4")
    selected = SimpleNamespace(uid="J3",params=SimpleNamespace(selected_templates="0,1"))
    classes = SimpleNamespace(uid="J2")
    project = SimpleNamespace(find_job=lambda _: abinit)
    with patch.object(module,"cryosparc_client",return_value=SimpleNamespace(find_project=lambda _:project)), \
         patch.object(module,"_ancestor_by_type",side_effect=[selected,classes]), \
         patch.object(module,"_load_volume",return_value=np.zeros((8,8,8))), \
         patch.object(module,"_load_class_stack",return_value=np.zeros((2,8,8))), \
         patch.object(module,"_compose_comparison",return_value=Image.new("L",(16,16))), \
         patch.object(module,"vision_cache_dir",return_value=tmp_path):
        result = module.build_abinitio_class2d_consistency_context("P6","J4")
    assert result["success"]
    assert result["contact_sheet"]["data_url"].startswith("data:image/png;base64,")

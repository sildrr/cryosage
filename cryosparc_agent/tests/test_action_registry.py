# Fixed dry-run tests for model decision validation and planning.
import unittest
from unittest.mock import patch

from action_registry import (
    apply_hard_workflow_gates,
    build_template_repick_gate,
    build_refinement_rollback_candidates,
    execute_model_decision_payload,
    generate_candidate_actions,
    micrograph_source_from_node,
    validate_model_decision_payload,
)


class ExposureSetSubsetCandidateTests(unittest.TestCase):
    def test_split_zero_is_preferred_over_full_or_remainder_outputs(self):
        node = {
            "status": "completed",
            "workflow_node_id": "J20",
            "logical_node_id": "node_exposure_sets_tool_001",
            "cryosparc_job_uid": "J20",
            "outputs": {
                "exposures": {"available": True, "result_names": ["micrograph_blob"]},
                "split_0": {"available": True, "result_names": ["micrograph_blob"]},
                "remainder": {"available": True, "result_names": ["micrograph_blob"]},
            },
        }
        source = micrograph_source_from_node(node)
        self.assertEqual(source["source_output"], "split_0")


def candidate_actions():
    return [
        {
            "action_id": "forward_J8",
            "action_type": "forward",
            "workflow_node_id": "J8",
            "reference_job_uid": "J8",
            "job_type": "extract_micrographs_multi",
            "description": "Dry-run candidate for fixed tests.",
            "execution_mode": "dry_run_only",
            "available": True,
            "blocked_by": [],
            "required_inputs": {},
            "parameter_template": {
                "compute_num_gpus": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 4,
                },
                "box_size_pix": {
                    "type": "integer",
                    "minimum": 32,
                    "default": 400,
                },
            },
            "default_parameters": {
                "compute_num_gpus": 4,
                "box_size_pix": 400,
            },
        }
    ]


def select_2d_workflow_state():
    """Return a minimal state where select_2D can feed homo_refine_new."""
    return {
        "schema_version": "1.0",
        "generated_at": "2026-07-02T00:00:00+00:00",
        "project_uid": "P2",
        "workspace_uid": "W4",
        "workflow_status": "completed",
        "nodes": [
            {
                "workflow_node_id": "J2",
                "logical_node_id": "node_import_volumes_001",
                "cryosparc_job_uid": "J2",
                "job_type": "import_volumes",
                "title": "Import volume",
                "status": "completed",
                "updated_at": "2026-07-02T00:00:00+00:00",
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {},
                "outputs": {
                    "imported_volume_1": {
                        "type": "volume",
                        "num_items": 1,
                        "available": True,
                        "result_names": ["map"],
                        "summary_keys": [],
                        "latest_summary_stat_keys": [],
                    }
                },
                "key_parameters": {},
                "runtime": {},
                "has_error": False,
                "has_warning": False,
            },
            {
                "workflow_node_id": "J37",
                "logical_node_id": "node_select_2D_001",
                "cryosparc_job_uid": "J37",
                "job_type": "select_2D",
                "title": "Select 2D",
                "status": "completed",
                "updated_at": "2026-07-02T00:00:00+00:00",
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {},
                "outputs": {
                    "particles_selected": {
                        "type": "particle",
                        "num_items": 133812,
                        "available": True,
                        "result_names": [
                            "alignments2D",
                            "blob",
                            "ctf",
                            "location",
                            "pick_stats",
                        ],
                        "summary_keys": [],
                        "latest_summary_stat_keys": [],
                    }
                },
                "key_parameters": {},
                "runtime": {},
                "has_error": False,
                "has_warning": False,
            },
        ],
        "edges": [],
        "root_nodes": ["J2", "J37"],
        "terminal_nodes": ["J2", "J37"],
        "running_nodes": [],
        "failed_nodes": [],
        "node_mapping": {"J2": "J2", "J37": "J37"},
    }


def homo_refine_workflow_state():
    """Return a minimal state where homo_refine_new can be run in parallel."""
    state = select_2d_workflow_state()
    state["nodes"].append(
        {
            "workflow_node_id": "J39",
            "logical_node_id": "node_homo_refine_new_001",
            "cryosparc_job_uid": "J39",
            "job_type": "homo_refine_new",
            "title": "Homogeneous Refinement",
            "status": "completed",
            "updated_at": "2026-07-02T01:00:00+00:00",
            "parent_job_uids": ["J37", "J2"],
            "parent_workflow_node_ids": ["J37", "J2"],
            "parent_logical_node_ids": [
                "node_select_2D_001",
                "node_import_volumes_001",
            ],
            "child_job_uids": [],
            "child_workflow_node_ids": [],
            "child_logical_node_ids": [],
            "inputs": {
                "particles": [
                    {
                        "source_workflow_node_id": "J37",
                        "source_logical_node_id": "node_select_2D_001",
                        "source_job_uid": "J37",
                        "source_output": "particles_selected",
                        "result_names": ["blob", "ctf", "location"],
                    }
                ],
                "volume": [
                    {
                        "source_workflow_node_id": "J2",
                        "source_logical_node_id": "node_import_volumes_001",
                        "source_job_uid": "J2",
                        "source_output": "imported_volume_1",
                        "result_names": ["map"],
                    }
                ],
            },
            "outputs": {
                "volume": {
                    "type": "volume",
                    "num_items": 1,
                    "available": True,
                    "result_names": ["map"],
                    "summary_keys": [],
                    "latest_summary_stat_keys": [],
                }
            },
            "key_parameters": {},
            "runtime": {},
            "has_error": False,
            "has_warning": False,
        }
    )
    state["terminal_nodes"] = ["J39"]
    state["node_mapping"]["J39"] = "J39"
    return state


def micrograph_template_workflow_state(
    current_has_templates: bool = False,
    separate_template_node: bool = False,
):
    """Return a minimal state for synthetic picker candidate tests."""
    outputs = {
        "exposures": {
            "type": "exposure",
            "num_items": 196,
            "available": True,
            "result_names": ["ctf", "ctf_stats", "micrograph_blob", "mscope_params"],
            "summary_keys": [],
            "latest_summary_stat_keys": [],
        }
    }
    if current_has_templates:
        outputs["templates"] = {
            "type": "template",
            "num_items": 6,
            "available": True,
            "result_names": ["blob"],
            "summary_keys": [],
            "latest_summary_stat_keys": [],
        }
    nodes = [
        {
            "workflow_node_id": "J10",
            "logical_node_id": "node_patch_ctf_estimation_multi_001",
            "cryosparc_job_uid": "J10",
            "job_type": "patch_ctf_estimation_multi",
            "title": "Patch CTF",
            "status": "completed",
            "updated_at": "2026-07-02T00:00:00+00:00",
            "parent_job_uids": [],
            "parent_workflow_node_ids": [],
            "parent_logical_node_ids": [],
            "child_job_uids": [],
            "child_workflow_node_ids": [],
            "child_logical_node_ids": [],
            "inputs": {},
            "outputs": outputs,
            "key_parameters": {},
            "runtime": {},
            "has_error": False,
            "has_warning": False,
        }
    ]
    if separate_template_node:
        nodes.append(
            {
                "workflow_node_id": "J9",
                "logical_node_id": "node_blob_picker_gpu_001",
                "cryosparc_job_uid": "J9",
                "job_type": "blob_picker_gpu",
                "title": "Blob Picker",
                "status": "completed",
                "updated_at": "2026-07-02T00:00:00+00:00",
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {},
                "outputs": {
                    "templates": {
                        "type": "template",
                        "num_items": 6,
                        "available": True,
                        "result_names": ["blob"],
                        "summary_keys": [],
                        "latest_summary_stat_keys": [],
                    }
                },
                "key_parameters": {},
                "runtime": {},
                "has_error": False,
                "has_warning": False,
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at": "2026-07-02T00:00:00+00:00",
        "project_uid": "P2",
        "workspace_uid": "W9",
        "workflow_status": "completed",
        "nodes": nodes,
        "edges": [],
        "root_nodes": [node["cryosparc_job_uid"] for node in nodes],
        "terminal_nodes": [node["cryosparc_job_uid"] for node in nodes],
        "running_nodes": [],
        "failed_nodes": [],
        "node_mapping": {
            node["cryosparc_job_uid"]: node["workflow_node_id"] for node in nodes
        },
    }


def homo_refine_decision():
    """Return a model decision for homogeneous refinement."""
    return {
        "schema_version": "1.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "action_id": "forward_J37_homo_refine_new",
                "action_type": "forward",
                "workflow_node_id": "J37:homo_refine_new",
                "job_type": "homo_refine_new",
                "parameters": {
                    "compute_use_ssd": False,
                    "refine_ctf_global_refine": True,
                    "refine_defocus_refine": True,
                },
            }
        ],
        "rollback_target": None,
        "branch_plan": None,
        "reason": "Selected particles are ready for refinement.",
        "confidence": 0.94,
        "risk_flags": [],
        "evidence": ["J37 completed with selected particles."],
    }


def parallel_homo_refine_decision():
    """Return a model decision for another homogeneous refinement."""
    decision = homo_refine_decision()
    decision["selected_actions"][0].update(
        {
            "action_id": "forward_J39_homo_refine_new",
            "workflow_node_id": "J39:homo_refine_new",
        }
    )
    decision["reason"] = "Run another refinement with alternate parameters."
    return decision


def forward_decision(**overrides):
    decision = {
        "schema_version": "1.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "action_id": "forward_J8",
                "action_type": "forward",
                "workflow_node_id": "J8",
                "job_type": "extract_micrographs_multi",
                "parameters": {
                    "compute_num_gpus": 4,
                    "box_size_pix": 400,
                },
            }
        ],
        "rollback_target": None,
        "branch_plan": None,
        "reason": "Particles are ready for extraction.",
        "confidence": 0.94,
        "risk_flags": [],
        "evidence": ["Selected particles are available."],
    }
    decision.update(overrides)
    return decision

    def test_inspect_picks_is_automated(self):
        from job_specs import get_job_spec, get_parameter_template

        spec = get_job_spec("inspect_picks_v2")
        self.assertFalse(spec["interactive"])
        self.assertFalse(spec["requires_approval"])
        self.assertIn("ncc_score_thresh", get_parameter_template("inspect_picks_v2"))
        self.assertIn("keep_threshold", get_parameter_template("inspect_picks_v2"))


class ActionRegistryFixedTests(unittest.TestCase):
    def test_alternate_refinement_uses_current_outputs_and_preserves_symmetry(self):
        refine = {
            "workflow_node_id": "J9", "logical_node_id": "node_refine_001",
            "cryosparc_job_uid": "J9", "job_type": "homo_refine_new",
            "status": "completed", "parent_job_uids": [], "inputs": {},
            "outputs": {
                "particles": {"available": True, "result_names": ["blob", "ctf"]},
                "volume": {"available": True, "result_names": ["map"]},
            },
            "key_parameters": {"refine_symmetry": "D7"},
        }
        state = {"nodes": [refine]}
        quality = {"resolution_evidence": {"values": {"fsc": 3.0}}}
        with patch("action_registry.build_template_repick_gate", return_value={"allowed": False, "blocked_by": ["test"]}):
            actions = build_refinement_rollback_candidates(
                state, refine, quality, {"target_resolution_A": 2.8}, "P1",
            )
        switch = next(item for item in actions if item.get("rollback_kind") == "alternate_refinement")
        self.assertTrue(switch["available"])
        self.assertEqual(switch["job_type"], "nonuniform_refine_new")
        self.assertEqual(switch["required_inputs"]["particles"][0]["source_job_uid"], "J9")
        self.assertEqual(switch["required_inputs"]["volume"][0]["source_job_uid"], "J9")
        self.assertEqual(switch["default_parameters"]["refine_symmetry"], "D7")

        state["nodes"].append({
            "cryosparc_job_uid": "J10", "job_type": "nonuniform_refine_new",
            "status": "completed", "parent_job_uids": ["J9"],
        })
        with patch("action_registry.build_template_repick_gate", return_value={"allowed": False, "blocked_by": ["test"]}):
            repeated = build_refinement_rollback_candidates(
                state, refine, quality, {"target_resolution_A": 2.8}, "P1",
            )
        blocked = next(item for item in repeated if item.get("rollback_kind") == "alternate_refinement")
        self.assertFalse(blocked["available"])

        select = {
            "workflow_node_id": "J7", "logical_node_id": "node_select_001",
            "cryosparc_job_uid": "J7", "job_type": "select_2D",
            "status": "completed", "parent_job_uids": [], "inputs": {},
            "outputs": {}, "key_parameters": {},
        }
        refine["parent_job_uids"] = ["J7"]
        state["nodes"] = [select, refine, {
            "cryosparc_job_uid": "J10", "job_type": "nonuniform_refine_new",
            "status": "completed", "parent_job_uids": ["J7"],
        }]
        with patch("action_registry.build_template_repick_gate", return_value={"allowed": False, "blocked_by": ["test"]}):
            sibling = build_refinement_rollback_candidates(
                state, refine, quality, {"target_resolution_A": 2.8}, "P1",
            )
        blocked_sibling = next(item for item in sibling if item.get("rollback_kind") == "alternate_refinement")
        self.assertFalse(blocked_sibling["available"])

    def test_template_repick_gate_requires_near_target_resolution_and_ncc(self):
        abinitio = {
            "workflow_node_id": "J10", "cryosparc_job_uid": "J10",
            "job_type": "homo_abinit", "status": "completed",
            "parent_job_uids": [], "outputs": {},
        }
        refine = {
            "workflow_node_id": "J11", "cryosparc_job_uid": "J11",
            "job_type": "homo_refine_new", "status": "completed",
            "parent_job_uids": ["J10"], "outputs": {},
        }
        state = {"nodes": [abinitio, refine]}
        quality = {"resolution_evidence": {"values": {"fsc_final_A": 3.2}}}

        def matcher(_project, _job):
            return {"source": {"abinitio_job_uid": "J10"}, "summary": {"min_best_ncc": 0.8}}

        allowed = build_template_repick_gate(
            state, refine, quality, {"target_resolution_A": 2.8}, "P1", matcher,
        )
        self.assertTrue(allowed["allowed"])

        def weak_matcher(_project, _job):
            return {"source": {"abinitio_job_uid": "J10"}, "summary": {"min_best_ncc": 0.74}}

        weak = build_template_repick_gate(
            state, refine, quality, {"target_resolution_A": 2.8}, "P1", weak_matcher,
        )
        self.assertFalse(weak["allowed"])
        self.assertIn("min_best_ncc >= 0.75", weak["blocked_by"][0])

        far_quality = {"resolution_evidence": {"values": {"fsc_final_A": 3.31}}}
        far = build_template_repick_gate(
            state, refine, far_quality, {"target_resolution_A": 2.8}, "P1", matcher,
        )
        self.assertFalse(far["allowed"])
        self.assertIn("within 0.5 A", far["blocked_by"][0])
    def test_blob_picker_keeps_junk_detector_and_inspect_as_options(self):
        candidates, blocked = apply_hard_workflow_gates(
            [
                {"job_type": "junk_detector_v1", "action_id": "junk", "blocked_by": []},
                {"job_type": "inspect_picks_v2", "action_id": "inspect", "blocked_by": []},
                {"job_type": "extract_micrographs_multi", "action_id": "extract", "blocked_by": []},
            ],
            [],
            {"nodes": []},
            {"job_type": "blob_picker_gpu"},
        )
        self.assertEqual(
            [item["job_type"] for item in candidates],
            ["junk_detector_v1", "inspect_picks_v2", "extract_micrographs_multi"],
        )
        self.assertEqual(blocked, [])

    def test_junk_detector_requires_inspect_before_extract(self):
        candidates, blocked = apply_hard_workflow_gates(
            [
                {"job_type": "inspect_picks_v2", "action_id": "inspect", "blocked_by": []},
                {"job_type": "extract_micrographs_multi", "action_id": "extract", "blocked_by": []},
            ],
            [],
            {"nodes": []},
            {"job_type": "junk_detector_v1"},
        )
        self.assertEqual([item["job_type"] for item in candidates], ["inspect_picks_v2"])
        self.assertEqual([item["job_type"] for item in blocked], ["extract_micrographs_multi"])
        self.assertIn("accepted particles", blocked[0]["blocked_by"][-1])

    def test_valid_forward_returns_planned_action(self):
        result = execute_model_decision_payload(
            forward_decision(),
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_mode"], "dry_run")
        self.assertEqual(result["execution_plan"]["plan_version"], "1.0")
        self.assertEqual(result["execution_plan"]["status"], "planned")
        self.assertEqual(result["execution_plan"]["action_count"], 1)
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_id"],
            "forward_J8",
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["status"],
            "planned",
        )

    def test_parameter_above_maximum_is_invalid(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 16

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "parameter_above_maximum")

        execute_result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )
        self.assertFalse(execute_result["success"])
        self.assertIsNone(execute_result["execution_plan"])

    def test_null_required_parameter_is_reported_as_missing(self):
        decision = forward_decision()
        decision["selected_actions"][0] = {
            "job_type": "import_movies",
            "parameters": {
                "blob_paths": None,
                "psize_A": 0.982,
            },
        }

        result = validate_model_decision_payload(
            decision,
            candidate_actions=[],
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "missing_required_parameter")

    def test_select_2d_requires_an_explicit_model_class_list(self):
        decision = forward_decision()
        decision["selected_actions"] = [{
            "action_id": "select",
            "action_type": "forward",
            "workflow_node_id": "J20:select_2D",
            "job_type": "select_2D",
            "parameters": {},
            "connections": {},
        }]
        candidate = {
            "action_id": "select",
            "action_type": "forward",
            "workflow_node_id": "J20:select_2D",
            "job_type": "select_2D",
            "execution_mode": "create_job",
            "parameter_template": {"selected_templates": {"type": "string", "required": True}},
            "default_parameters": {},
            "required_inputs": {"templates": [{"source_job_uid": "J20", "source_output": "class_averages"}]},
        }

        result = validate_model_decision_payload(decision, candidate_actions=[candidate])

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "missing_required_parameter")

    def test_high_gpu_count_requires_approval(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 8

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        action = result["execution_plan"]["actions"][0]
        self.assertTrue(result["success"])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertTrue(action["approval_required"])
        self.assertIn("high_gpu_count", action["approval_reasons"])
        self.assertIn("high_gpu_count", result["execution_plan"]["approval_reasons"])

    def test_live_execution_requires_project_and_workspace(self):
        result = execute_model_decision_payload(
            forward_decision(),
            candidate_actions=candidate_actions(),
            dry_run=False,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution_mode"], "missing_execution_context")
        self.assertEqual(result["issues"][0]["code"], "missing_execution_context")

    def test_high_gpu_live_execution_is_blocked_by_approval(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 8

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
            dry_run=False,
            project_uid="P1",
            workspace_uid="W1",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution_mode"], "live_execution")
        self.assertEqual(
            result["execution_results"][0]["status"],
            "approval_required",
        )

    def test_unknown_action_id_becomes_generic_action(self):
        decision = forward_decision()
        decision["selected_actions"][0]["action_id"] = "forward_J999"

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["warnings"][0]["code"],
            "generic_action_without_candidate",
        )

        execute_result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )
        self.assertTrue(execute_result["success"])
        self.assertEqual(
            execute_result["execution_plan"]["actions"][0]["execution_mode"],
            "create_job",
        )

    def test_create_templates_cannot_bypass_the_evidence_gated_candidate(self):
        decision = forward_decision()
        action = decision["selected_actions"][0]
        action.update({
            "action_id": "generic_create_templates",
            "workflow_node_id": "J99:create_templates",
            "job_type": "create_templates",
            "parameters": {"n_templates": 50},
        })
        result = validate_model_decision_payload(decision, candidate_actions=[])
        self.assertFalse(result["success"])
        self.assertIn(
            "create_templates_requires_gated_candidate",
            {issue["code"] for issue in result["issues"]},
        )

    def test_stop_decision_is_valid_and_planned(self):
        decision = forward_decision(
            decision_type="stop",
            selected_actions=[],
            reason="No safe next action is available.",
            confidence=0.8,
        )

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_plan"]["decision_type"], "stop")
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_type"],
            "stop",
        )

    def test_rollback_decision_is_valid_and_planned(self):
        rollback_target = {
            "workflow_node_id": "J6",
            "job_type": "template_picker_gpu",
            "reason_code": "poor_particle_picking",
            "rollback_mode": "rerun_from_target",
        }
        decision = forward_decision(
            decision_type="rollback",
            selected_actions=[],
            rollback_target=rollback_target,
            reason="Picking should be adjusted before extraction.",
            confidence=0.83,
            risk_flags=["particle_quality_uncertain"],
        )

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["approval_reasons"],
            ["rollback_decision"],
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_type"],
            "rollback",
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["rollback_target"],
            rollback_target,
        )

    def test_class2d_completed_generates_extract_fallback_candidate(self):
        state = select_2d_workflow_state()
        extract = {
            "workflow_node_id": "J36",
            "logical_node_id": "node_extract_micrographs_multi_001",
            "cryosparc_job_uid": "J36",
            "job_type": "extract_micrographs_multi",
            "title": "Extract From Micrographs (GPU)",
            "status": "completed",
            "updated_at": "2026-07-02T00:00:00+00:00",
            "parent_job_uids": [],
            "parent_workflow_node_ids": [],
            "parent_logical_node_ids": [],
            "child_job_uids": ["J37"],
            "child_workflow_node_ids": ["J37"],
            "child_logical_node_ids": ["node_class_2D_new_001"],
            "inputs": {
                "micrographs": [{"source_job_uid": "J14", "source_output": "micrographs"}],
                "particles": [{"source_job_uid": "J14", "source_output": "particles"}],
            },
            "outputs": {},
            "key_parameters": {"box_size_pix": 320, "compute_num_gpus": 1},
            "runtime": {},
            "has_error": False,
            "has_warning": False,
        }
        state["nodes"].insert(1, extract)
        class2d = state["nodes"][-1]
        class2d.update({
            "logical_node_id": "node_class_2D_new_001",
            "job_type": "class_2D_new",
            "parent_job_uids": ["J36"],
            "parent_workflow_node_ids": ["J36"],
            "parent_logical_node_ids": ["node_extract_micrographs_multi_001"],
        })
        candidates, blocked = generate_candidate_actions(state, "J37")
        self.assertFalse(blocked)
        fallback = next(
            action for action in candidates
            if action["job_type"] == "extract_micrographs_multi"
        )
        self.assertEqual(fallback["action_type"], "forward")
        self.assertEqual(fallback["reference_job_uid"], "J36")
        self.assertEqual(fallback["default_parameters"]["box_size_pix"], 320)
        self.assertIn("box_size_pix", fallback["parameter_template"])
        self.assertEqual(
            fallback["required_inputs"]["particles"][0]["source_job_uid"],
            "J14",
        )

    def test_select_2d_completed_generates_abinitio_candidate(self):
        candidates, blocked = generate_candidate_actions(
            select_2d_workflow_state(),
            "J37",
            dataset_info={"symmetry": "D7"},
        )

        self.assertFalse(blocked)
        self.assertEqual([candidate["job_type"] for candidate in candidates], ["homo_abinit"])
        self.assertEqual(
            candidates[0]["required_inputs"]["particles"][0]["source_output"],
            "particles_selected",
        )
        self.assertEqual(candidates[0]["default_parameters"], {"abinit_symmetry": "D7"})
        self.assertEqual(
            candidates[0]["parameter_template"]["abinit_symmetry"]["enum"],
            ["D7"],
        )

    def test_template_repick_select_2d_refines_against_source_volume(self):
        state = select_2d_workflow_state()
        state["nodes"].extend([
            {
                "workflow_node_id": "J30", "logical_node_id": "node_refine_001",
                "cryosparc_job_uid": "J30", "job_type": "homo_refine_new",
                "status": "completed", "parent_job_uids": [], "inputs": {},
                "outputs": {"volume": {"available": True, "result_names": ["map"]}},
                "key_parameters": {}, "runtime": {}, "child_job_uids": ["J31"],
            },
            {
                "workflow_node_id": "J31", "logical_node_id": "node_templates_001",
                "cryosparc_job_uid": "J31", "job_type": "create_templates",
                "status": "completed", "parent_job_uids": ["J30"], "inputs": {},
                "outputs": {"templates": {"available": True, "result_names": ["blob"]}},
                "key_parameters": {}, "runtime": {}, "child_job_uids": ["J37"],
            },
        ])
        select_node = next(node for node in state["nodes"] if node["cryosparc_job_uid"] == "J37")
        select_node["parent_job_uids"] = ["J31"]
        candidates, blocked = generate_candidate_actions(
            state, "J37", dataset_info={"symmetry": "D7"},
        )
        self.assertFalse(blocked)
        self.assertEqual([candidate["job_type"] for candidate in candidates], ["homo_refine_new"])
        self.assertTrue(candidates[0]["template_repick_refinement"])
        self.assertEqual(candidates[0]["template_repick_source_refine_job_uid"], "J30")
        self.assertEqual(candidates[0]["required_inputs"]["volume"][0]["source_job_uid"], "J30")

    def test_micrographs_without_templates_blocks_template_picker(self):
        candidates, blocked = generate_candidate_actions(
            micrograph_template_workflow_state(),
            "J10",
        )

        by_type = {action["job_type"]: action for action in candidates}
        self.assertIn("blob_picker_gpu", by_type)
        self.assertEqual(
            by_type["blob_picker_gpu"]["required_inputs"]["micrographs"][0]["source_job_uid"],
            "J10",
        )
        blocked_by_type = {action["job_type"]: action for action in blocked}
        self.assertIn("template_picker_gpu", blocked_by_type)
        self.assertEqual(
            blocked_by_type["template_picker_gpu"]["missing_required_inputs"],
            ["templates"],
        )
        self.assertEqual(
            blocked_by_type["template_picker_gpu"]["required_inputs"]["templates"],
            [],
        )

    def test_micrographs_and_current_templates_enable_template_picker(self):
        candidates, blocked = generate_candidate_actions(
            micrograph_template_workflow_state(current_has_templates=True),
            "J10",
        )

        self.assertFalse(blocked)
        by_type = {action["job_type"]: action for action in candidates}
        self.assertIn("blob_picker_gpu", by_type)
        self.assertIn("template_picker_gpu", by_type)
        template_action = by_type["template_picker_gpu"]
        self.assertEqual(template_action["action_id"], "forward_J10_template_picker_gpu")
        self.assertEqual(
            template_action["required_inputs"]["micrographs"][0]["source_output"],
            "exposures",
        )
        self.assertEqual(
            template_action["required_inputs"]["templates"][0]["source_job_uid"],
            "J10",
        )
        self.assertEqual(
            template_action["required_inputs"]["templates"][0]["source_output"],
            "templates",
        )

    def test_templates_from_other_completed_node_enable_template_picker(self):
        candidates, blocked = generate_candidate_actions(
            micrograph_template_workflow_state(separate_template_node=True),
            "J10",
        )

        self.assertFalse(blocked)
        by_type = {action["job_type"]: action for action in candidates}
        self.assertIn("template_picker_gpu", by_type)
        template_action = by_type["template_picker_gpu"]
        self.assertEqual(
            template_action["required_inputs"]["micrographs"][0]["source_job_uid"],
            "J10",
        )
        self.assertEqual(
            template_action["required_inputs"]["templates"][0]["source_job_uid"],
            "J9",
        )
        self.assertEqual(
            template_action["required_inputs"]["templates"][0]["source_output"],
            "templates",
        )

    def test_homo_refine_compute_use_ssd_is_allowed(self):
        candidates, _ = generate_candidate_actions(
            homo_refine_workflow_state(),
            "J39",
        )

        result = execute_model_decision_payload(
            parallel_homo_refine_decision(),
            candidate_actions=candidates,
        )

        self.assertTrue(result["success"])
        action = result["execution_plan"]["actions"][0]
        self.assertEqual(action["job_type"], "homo_refine_new")
        self.assertFalse(action["resolved_parameters"]["compute_use_ssd"])
        self.assertEqual(
            action["connections"]["particles"],
            ("J37", "particles_selected"),
        )
        self.assertEqual(
            action["connections"]["volume"],
            ("J2", "imported_volume_1"),
        )

    def test_homo_refine_completed_allows_parallel_homo_refine_candidate(self):
        candidates, blocked = generate_candidate_actions(
            homo_refine_workflow_state(),
            "J39",
        )

        self.assertFalse(blocked)
        self.assertEqual(candidates[0]["action_id"], "forward_J39_homo_refine_new")
        self.assertEqual(candidates[0]["job_type"], "homo_refine_new")
        self.assertEqual(
            candidates[0]["required_inputs"]["particles"][0]["source_job_uid"],
            "J37",
        )
        self.assertEqual(
            candidates[0]["required_inputs"]["volume"][0]["source_job_uid"],
            "J2",
        )

    def test_parallel_homo_refine_decision_is_planned(self):
        candidates, _ = generate_candidate_actions(
            homo_refine_workflow_state(),
            "J39",
        )

        result = execute_model_decision_payload(
            parallel_homo_refine_decision(),
            candidate_actions=candidates,
        )

        self.assertTrue(result["success"])
        action = result["execution_plan"]["actions"][0]
        self.assertEqual(action["job_type"], "homo_refine_new")
        self.assertEqual(
            action["connections"]["particles"],
            ("J37", "particles_selected"),
        )
        self.assertEqual(
            action["connections"]["volume"],
            ("J2", "imported_volume_1"),
        )

    def test_generic_action_can_use_model_supplied_connections(self):
        decision = forward_decision()
        decision["selected_actions"][0] = {
            "job_type": "unknown_future_job",
            "parameters": {"custom_param": 3},
            "connections": {
                "particles": {
                    "source_job_uid": "J37",
                    "source_output": "particles_selected",
                }
            },
        }

        result = execute_model_decision_payload(
            decision,
            candidate_actions=[],
        )

        self.assertTrue(result["success"])
        action = result["execution_plan"]["actions"][0]
        self.assertEqual(action["job_type"], "unknown_future_job")
        self.assertEqual(action["connections"]["particles"], ("J37", "particles_selected"))
        self.assertEqual(action["resolved_parameters"]["custom_param"], 3)


if __name__ == "__main__":
    unittest.main()

import unittest

from reconstruction_observer import build_observation, execute_observation_tasks
from reconstruction_planner_pilot import build_planning_prompt, build_shadow_plan, validate_planning_decision


def node(uid, job_type, *, parents=None, children=None, outputs=None, parameters=None, metrics=None):
    return {
        "workflow_node_id": uid,
        "cryosparc_job_uid": uid,
        "job_type": job_type,
        "status": "completed",
        "parent_job_uids": parents or [],
        "child_job_uids": children or [],
        "inputs": {"particles": [{"source_job_uid": parents[0], "source_output": "particles"}]} if parents else {},
        "outputs": outputs or {},
        "key_parameters": parameters or {},
        "run_errors": {},
        "has_warning": False,
        "has_error": False,
        "updated_at": "2026-09-18T00:00:00Z",
    }


class ReconstructionPlannerPilotTests(unittest.TestCase):
    def test_observer_exposes_box_size_comparison_and_select_retention(self):
        extract_a = node("J10", "extract_micrographs_multi", parents=["J9"], children=["J11"], parameters={"box_size_pix": 320}, outputs={"particles": {"num_items": 1_000_000, "available": True}})
        extract_b = node("J12", "extract_micrographs_multi", parents=["J9"], children=["J13"], parameters={"box_size_pix": 384}, outputs={"particles": {"num_items": 1_000_000, "available": True}})
        class_a = node("J11", "class_2D_new", parents=["J10"], children=["J14"], outputs={"particles": {"num_items": 1_000_000, "available": True}})
        class_b = node("J13", "class_2D_new", parents=["J12"], outputs={"particles": {"num_items": 1_000_000, "available": True}})
        select = node("J14", "select_2D", parents=["J11"], outputs={"particles_selected": {"num_items": 980_000, "available": True}})
        state = {"project_uid": "P", "workspace_uid": "W", "generated_at": "now", "nodes": [extract_a, class_a, extract_b, class_b, select]}

        observation = build_observation(state, {"target_resolution_A": 3.0})

        self.assertEqual(len(observation["class2d_comparisons"]), 1)
        self.assertEqual(observation["selection_outcomes"][0]["retention_fraction"], 0.98)
        self.assertIn("permissive_select2d", {item["issue_type"] for item in observation["observed_issues"]})

    def test_shadow_plan_is_read_only_and_prioritizes_curation_review(self):
        observation = {
            "observed_issues": [{"issue_type": "permissive_select2d", "evidence": {"class2d_job_uid": "J11"}}],
            "class2d_comparisons": [{"branches": [{"class2d_job_uid": "J11"}, {"class2d_job_uid": "J13"}]}],
        }
        plan = build_shadow_plan(observation)
        prompt = build_planning_prompt(observation)

        self.assertEqual(plan["mode"], "shadow_only")
        self.assertTrue(plan["experiments"])
        self.assertIn("No experiment may be executed", plan["stop_conditions"][0])
        self.assertIn("request_observations", prompt["output_contract"]["decision_type"])

    def test_planner_can_request_bounded_read_only_lineage(self):
        blob = node("J1", "blob_picker_gpu", children=["J2"], parameters={"diameter": 160, "diameter_max": 220})
        inspect = node("J2", "inspect_picks_v2", parents=["J1"], children=["J3"])
        extract = node("J3", "extract_micrographs_multi", parents=["J2"])
        state = {"project_uid": "P", "workspace_uid": "W", "generated_at": "now", "nodes": [blob, inspect, extract]}
        decision = validate_planning_decision({
            "decision_type": "request_observations",
            "observation_tasks": [{
                "task_id": "trace",
                "tool": "trace_upstream_lineage",
                "arguments": {"start_job_uid": "J3", "stop_at_job_types": ["blob_picker_gpu"]},
            }],
        })
        results = execute_observation_tasks(state, {}, decision["observation_tasks"])

        self.assertEqual([item["job_uid"] for item in results[0]["result"]], ["J3", "J2", "J1"])
        with self.assertRaises(ValueError):
            execute_observation_tasks(state, {}, [{"tool": "submit_arbitrary_job", "arguments": {}}])

    def test_cross_branch_scale_requires_consistency_before_repick(self):
        from reconstruction_observer import _cross_branch_scale_consistency

        result = _cross_branch_scale_consistency([
            {"aggregate": {"particle_diameter_A_p80": 110.0}},
            {"aggregate": {"particle_diameter_A_p80": 180.0}},
        ])

        self.assertEqual(result["status"], "inconsistent")
        self.assertFalse(result["usable_for_blob_picker_update"])


if __name__ == "__main__":
    unittest.main()

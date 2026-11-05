"""Regression tests for per-branch extraction box-size trials."""
import unittest

from action_registry import (
    build_abinitio_baseline_refine_action,
    build_box_size_trial_candidates,
    build_pending_class_2d_trial_candidates,
    build_synthetic_homo_abinit_candidate,
)
from cryosparc_mcp_server import apply_inspect_parameter_guard, apply_mandatory_trial_guard


def extract(uid, particle_source, size):
    return {
        "workflow_node_id": uid,
        "cryosparc_job_uid": uid,
        "job_type": "extract_micrographs_multi",
        "status": "completed",
        "inputs": {
            "micrographs": [{"source_job_uid": particle_source, "source_output": "micrographs"}],
            "particles": [{"source_job_uid": particle_source, "source_output": "particles"}],
        },
        "key_parameters": {"box_size_pix": size},
    }


class BoxSizeTrialBranchingTests(unittest.TestCase):
    def test_other_picker_branch_does_not_satisfy_trials(self):
        current = extract("J30", "J20", 320)
        state = {"nodes": [current, extract("J31", "J10", 384), extract("J32", "J10", 448)]}
        actions = build_box_size_trial_candidates(state, current)
        self.assertEqual(
            {item["default_parameters"]["box_size_pix"] for item in actions},
            {384, 416, 448, 480, 504, 544, 576, 608, 640, 672, 704, 720},
        )

    def test_completed_trial_on_same_branch_is_excluded(self):
        current = extract("J30", "J20", 320)
        state = {"nodes": [current, extract("J31", "J20", 384)]}
        actions = build_box_size_trial_candidates(state, current)
        self.assertEqual(
            [item["default_parameters"]["box_size_pix"] for item in actions],
            [416, 448, 480, 504, 544, 576, 608, 640, 672, 704, 720],
        )

    def test_single_trial_is_valid_under_one_action_contract(self):
        candidates = [
            {"action_id": "trial_384", "available": True, "box_size_trial": True},
            {"action_id": "trial_440", "available": True, "box_size_trial": True},
        ]
        validation = {"success": True, "valid_actions": True, "issues": []}
        result = apply_mandatory_trial_guard(
            {"decision_type": "forward", "action_id": "trial_384"}, validation, candidates,
        )
        self.assertTrue(result["success"])

    def test_abinitio_exposes_class_zero_baseline_refine(self):
        node = {
            "workflow_node_id": "J40",
            "logical_node_id": "node_homo_abinit_001",
            "cryosparc_job_uid": "J40",
            "job_type": "homo_abinit",
            "status": "completed",
            "outputs": {
                "particles_class_0": {"available": True, "result_names": ["location", "ctf"]},
                "volume_class_0": {"available": True, "result_names": ["map"]},
            },
        }
        actions, blocked = build_abinitio_baseline_refine_action(node)
        self.assertFalse(blocked)
        self.assertEqual(actions[0]["job_type"], "homo_refine_new")
        self.assertEqual(actions[0]["required_inputs"]["particles"][0]["source_output"], "particles_class_0")
        self.assertEqual(actions[0]["required_inputs"]["volume"][0]["source_output"], "volume_class_0")

    def test_abinitio_offers_complete_select_2d_stack_as_separate_refine(self):
        selected = {
            "workflow_node_id": "J39", "logical_node_id": "node_select_2d_001",
            "cryosparc_job_uid": "J39", "job_type": "select_2D", "status": "completed",
            "parent_job_uids": [],
            "outputs": {"particles_selected": {
                "available": True, "num_items": 377234,
                "result_names": ["blob", "ctf", "location"],
            }},
        }
        abinitio = {
            "workflow_node_id": "J40", "logical_node_id": "node_homo_abinit_001",
            "cryosparc_job_uid": "J40", "job_type": "homo_abinit", "status": "completed",
            "parent_job_uids": ["J39"], "key_parameters": {"abinit_symmetry": "C1"},
            "outputs": {
                "particles_class_0": {"available": True, "num_items": 210000, "result_names": ["blob", "ctf"]},
                "volume_class_0": {"available": True, "result_names": ["map"]},
            },
        }
        actions, blocked = build_abinitio_baseline_refine_action(
            abinitio, {"nodes": [selected, abinitio]},
        )
        self.assertFalse(blocked)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]["required_inputs"]["particles"][0]["source_job_uid"], "J40")
        self.assertEqual(actions[1]["required_inputs"]["particles"][0]["source_job_uid"], "J39")
        self.assertEqual(actions[1]["required_inputs"]["volume"][0]["source_job_uid"], "J40")
        self.assertEqual(actions[1]["particle_source_comparison"]["abinitio_unused_count"], 167234)
        self.assertNotEqual(actions[0]["action_id"], actions[1]["action_id"])

    def test_inspect_auto_cluster_requires_explicit_threshold(self):
        validation = {"success": True, "valid_actions": True, "issues": []}
        result = apply_inspect_parameter_guard(
            {"job_type": "inspect_picks_v2", "parameters": {"do_auto_cluster": True}}, validation,
        )
        self.assertFalse(result["success"])

        result = apply_inspect_parameter_guard(
            {"job_type": "inspect_picks_v2", "parameters": {"ncc_score_thresh": 0.0,"lpower_thresh_max":100}}, validation,
        )
        self.assertTrue(result["success"])

    def test_completed_class2d_keeps_other_box_trials_pending(self):
        extracts = []
        for uid, size in (("J30", 320), ("J31", 384), ("J32", 440)):
            node = extract(uid, "J20", size)
            node["logical_node_id"] = f"node_extract_{uid}"
            node["outputs"] = {"particles": {"available": True, "result_names": ["location"]}}
            extracts.append(node)
        class2d = {
            "workflow_node_id": "J33", "logical_node_id": "node_class2d_J33", "cryosparc_job_uid": "J33", "job_type": "class_2D_new",
            "status": "completed", "parent_job_uids": ["J32"],
            "inputs": {"particles": [{"source_job_uid": "J32", "source_output": "particles"}]},
        }
        state = {"nodes": [*extracts, class2d]}
        pending = build_pending_class_2d_trial_candidates(state, class2d)
        self.assertEqual({item["reference_job_uid"] for item in pending}, {"J30", "J31"})

    def test_select2d_synthetic_action_starts_abinitio(self):
        action = build_synthetic_homo_abinit_candidate(
            {"workflow_node_id": "J50", "cryosparc_job_uid": "J50"},
            {"particles": [{"source_job_uid": "J50", "source_output": "particles_selected"}]}, [],
        )
        self.assertEqual(action["job_type"], "homo_abinit")
        self.assertTrue(action["available"])


if __name__ == "__main__":
    unittest.main()

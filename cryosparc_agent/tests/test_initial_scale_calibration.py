"""Tests for the bounded two-pass initial particle-scale calibration."""
import unittest
from unittest.mock import patch

from action_registry import apply_hard_workflow_gates, validate_parameter_value


def candidate(job_type, action_id, parameter_template=None, required_inputs=None):
    return {
        "action_id": action_id,
        "action_type": "forward",
        "workflow_node_id": f"J10:{job_type}",
        "job_type": job_type,
        "available": True,
        "blocked_by": [],
        "required_inputs": required_inputs or {},
        "parameter_template": parameter_template or {},
        "default_parameters": {},
    }


class InitialScaleCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            "os.environ",
            {
                "CRYOAGENT_FIXED_INITIAL_FUNNEL": "1",
                "CRYOAGENT_INITIAL_SCALE_CALIBRATION": "1",
                "CRYOAGENT_SCALE_CALIBRATION_MICROGRAPHS": "10",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_patch_ctf_starts_sampled_picker(self):
        ctf = {
            "workflow_node_id": "J9", "cryosparc_job_uid": "J9",
            "job_type": "patch_ctf_estimation_multi", "status": "completed",
        }
        picker = candidate("blob_picker_gpu", "pick", {
            "num_process": {"type": "integer", "minimum": 1},
        })
        actions, _ = apply_hard_workflow_gates(
            [picker], [], {"nodes": [ctf]}, ctf,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["parameter_template"]["num_process"]["default"], 10)
        self.assertEqual(actions[0]["calibration_phase"], "pilot_1")

    def test_pilot_sample_size_cannot_be_overridden(self):
        issues = validate_parameter_value(
            "num_process", 20, {"type": "integer", "const": 10},
            "selected_actions.0.parameters.num_process",
        )
        self.assertEqual([issue.code for issue in issues], ["parameter_const_mismatch"])

    def test_pilot_picker_forces_limited_extract_at_balanced_box(self):
        ctf = {
            "workflow_node_id": "J9", "logical_node_id": "ctf", "cryosparc_job_uid": "J9",
            "job_type": "patch_ctf_estimation_multi", "status": "completed",
            "parent_job_uids": [],
        }
        picker = {
            "workflow_node_id": "J10", "logical_node_id": "picker", "cryosparc_job_uid": "J10",
            "job_type": "blob_picker_gpu", "status": "completed", "parent_job_uids": ["J9"],
            "key_parameters": {"num_process": 10},
        }
        extraction = candidate("extract_micrographs_multi", "extract", {
            "box_size_pix": {"type": "integer", "minimum": 32, "maximum": 720},
        })
        actions, _ = apply_hard_workflow_gates(
            [extraction], [], {"nodes": [ctf, picker]}, picker,
        )
        self.assertEqual(len(actions), 1)
        defaults = actions[0]["parameter_template"]
        self.assertEqual(defaults["num_extract"]["default"], 10)
        self.assertEqual(defaults["box_size_pix"]["default"], 384)

    @patch("action_registry.build_class2d_scale_context")
    def test_first_pilot_class2d_starts_second_pilot_from_measured_scale(self, scale_context):
        scale_context.return_value = {
            "aggregate": {
                "usable_for_picker": True,
                "particle_diameter_range_A": [158.0, 164.0],
            },
            "box_size_recommendation": {
                "usable_for_extraction": True,
                "recommended_box_size_pix": 672,
            },
        }
        ctf = {
            "workflow_node_id": "J9", "logical_node_id": "ctf", "cryosparc_job_uid": "J9",
            "job_type": "patch_ctf_estimation_multi", "status": "completed",
            "parent_job_uids": [],
            "outputs": {"exposures": {
                "available": True, "num_items": 196, "result_names": ["micrograph_blob"],
            }},
        }
        picker = {
            "workflow_node_id": "J10", "logical_node_id": "picker", "cryosparc_job_uid": "J10",
            "job_type": "blob_picker_gpu", "status": "completed", "parent_job_uids": ["J9"],
            "key_parameters": {"num_process": 10},
        }
        extract = {
            "workflow_node_id": "J11", "logical_node_id": "extract", "cryosparc_job_uid": "J11",
            "job_type": "extract_micrographs_multi", "status": "completed", "parent_job_uids": ["J10"],
            "key_parameters": {"num_extract": 10, "box_size_pix": 720},
        }
        class2d = {
            "workflow_node_id": "J12", "logical_node_id": "class", "cryosparc_job_uid": "J12",
            "job_type": "class_2D_new", "status": "completed", "parent_job_uids": ["J11"],
        }
        actions, _ = apply_hard_workflow_gates(
            [], [], {"nodes": [ctf, picker, extract, class2d]}, class2d,
            project_uid="P1", dataset_info={"target_resolution_A": 2.32},
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["job_type"], "blob_picker_gpu")
        self.assertEqual(actions[0]["parameter_template"]["num_process"]["default"], 10)
        self.assertEqual(actions[0]["parameter_template"]["diameter"]["default"], 158.0)
        self.assertEqual(actions[0]["calibration_box_size_pix"], 672)

    @patch("action_registry.build_class2d_scale_context")
    def test_second_pilot_transitions_to_full_count_picker_and_final_box(self, scale_context):
        scale_context.return_value = {
            "aggregate": {
                "usable_for_picker": True,
                "particle_diameter_range_A": [160.0, 166.0],
            },
            "box_size_recommendation": {
                "usable_for_extraction": True,
                "recommended_box_size_pix": 672,
            },
        }
        ctf = {
            "workflow_node_id": "J9", "logical_node_id": "ctf", "cryosparc_job_uid": "J9",
            "job_type": "patch_ctf_estimation_multi", "status": "completed", "parent_job_uids": [],
            "outputs": {"exposures": {
                "available": True, "num_items": 196, "result_names": ["micrograph_blob"],
            }},
        }

        def pilot(uid, job_type, parent, key_parameters=None):
            return {
                "workflow_node_id": uid, "logical_node_id": uid, "cryosparc_job_uid": uid,
                "job_type": job_type, "status": "completed", "parent_job_uids": [parent],
                "key_parameters": key_parameters or {},
            }

        nodes = [
            ctf,
            pilot("J10", "blob_picker_gpu", "J9", {"num_process": 10}),
            pilot("J11", "extract_micrographs_multi", "J10", {"num_extract": 10, "box_size_pix": 384}),
            pilot("J12", "class_2D_new", "J11"),
            pilot("J13", "blob_picker_gpu", "J9", {"num_process": 10}),
            pilot("J14", "extract_micrographs_multi", "J13", {"num_extract": 10, "box_size_pix": 672}),
            pilot("J15", "class_2D_new", "J14"),
        ]
        full_picker = pilot("J16", "blob_picker_gpu", "J9", {"diameter": 160, "diameter_max": 166})
        full_extract_candidate = candidate("extract_micrographs_multi", "extract_full", {
            "box_size_pix": {"type": "integer", "minimum": 32, "maximum": 720},
        })
        full_actions, _ = apply_hard_workflow_gates(
            [full_extract_candidate], [], {"nodes": nodes + [full_picker]}, full_picker,
            project_uid="P1", dataset_info={"target_resolution_A": 2.32},
        )
        self.assertEqual(full_actions[0]["parameter_template"]["box_size_pix"]["default"], 672)
        self.assertEqual(full_actions[0]["calibration_phase"], "full_initial_extract")
        self.assertNotIn("num_extract", full_actions[0]["parameter_template"])

        second_class = nodes[-1]
        full_picker_candidate = candidate("blob_picker_gpu", "full_picker", {
            "num_process": {"type": "integer", "minimum": 1},
            "diameter": {"type": "number", "minimum": 0},
            "diameter_max": {"type": "number", "minimum": 0},
        }, required_inputs={"micrographs": [{"source_job_uid": "J9", "source_output": "exposures"}]})
        transition, _ = apply_hard_workflow_gates(
            [full_picker_candidate], [], {"nodes": nodes}, second_class,
            project_uid="P1", dataset_info={"target_resolution_A": 2.32},
        )
        self.assertEqual(transition[0]["calibration_phase"], "full_initial_picker")
        self.assertEqual(transition[0]["default_parameters"]["num_process"], 196)
        self.assertEqual(transition[0]["parameter_template"]["num_process"]["const"], 196)


if __name__ == "__main__":
    unittest.main()

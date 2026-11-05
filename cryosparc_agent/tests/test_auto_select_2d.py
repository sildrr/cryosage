import unittest

from action_registry import apply_auto_select_2d_policy


class AutoSelect2DTests(unittest.TestCase):
    def test_requires_model_selected_class_ids_without_a_fraction_default(self):
        state = {"nodes": [{
            "workflow_node_id": "J20", "cryosparc_job_uid": "J20",
            "outputs": {"class_averages": {"num_items": 50}},
        }]}
        action = {
            "job_type": "select_2D", "execution_mode": "dry_run_only",
            "parameter_template": {"selected_templates": {"type": "string"}},
            "required_inputs": {"templates": [{"source_job_uid": "J20", "source_output": "class_averages"}]},
        }
        result = apply_auto_select_2d_policy(action, state)
        self.assertNotIn("selected_templates", result["default_parameters"])
        self.assertTrue(result["parameter_template"]["selected_templates"]["required"])
        self.assertFalse(result["job_spec_metadata"]["interactive"])
        self.assertEqual(result["auto_policy"]["name"], "model_selected_templates")


if __name__ == "__main__":
    unittest.main()

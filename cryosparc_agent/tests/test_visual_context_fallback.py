"""Regression tests for unavailable visual-overlay evidence."""
import unittest

from autonomous_mcp_closed_loop import (
    build_autonomous_prompt,
    rollback_class2d_job_uids,
)
from model_direct_runner import build_workflow_decision_prompt


class VisualContextFallbackTests(unittest.TestCase):
    def test_execution_prompt_explains_cryosift_score_direction(self):
        messages = build_autonomous_prompt(
            model_input={}, candidate_context={"candidate_actions": []}, round_index=1,
            kb_tool_policy="disabled",
        )
        static = str(messages[1]["content"])
        self.assertIn("lower scores indicate better class quality", static)
        self.assertIn("all scored classes missed", static)
        self.assertIn("ASCENDING CryoSift score", static)
        self.assertIn("3.2 is the better CryoSift score", static)

    def test_pilot_selection_visual_prompt_repeats_score_direction(self):
        messages = build_autonomous_prompt({}, {"candidate_actions":[{
            "diameter_trial":{"role":"select_templates","picker_uid":"J1"}}]}, 1,
            visual_context={"kind":"class_average","contact_sheet":{}}, kb_tool_policy="disabled")
        text = str(messages[-1]['content'])
        self.assertIn("ASCENDING CryoSift score", text)
        self.assertIn("score direction does NOT reverse", text)

    def test_direct_template_selection_explains_empty_recommendation(self):
        messages = build_workflow_decision_prompt({})
        self.assertIn("ASCENDING CryoSift score", str(messages))

    def test_rollback_visual_sources_support_bounded_refinement_menu(self):
        actions = [
            {
                "quality_rollback": True,
                "rollback_source": {"class2d_job_uid": "J398"},
            },
            {
                "quality_rollback": True,
                "quality_rollback_branch": {"class2d_job_uid": "J393"},
            },
            {
                "quality_rollback": True,
                "rollback_source": {"class2d_job_uid": "J398"},
            },
        ]
        self.assertEqual(rollback_class2d_job_uids(actions), ["J398", "J393"])

    def test_autonomous_prompt_skips_missing_contact_sheet(self):
        messages = build_autonomous_prompt(
            model_input={"current_state": {"last_node_id": "J12"}},
            candidate_context={"candidate_actions": []},
            round_index=1,
            visual_context={
                "success": False,
                "kind": "pick_overlay",
                "error": "No readable micrographs were found.",
            },
            kb_tool_policy="disabled",
        )

        content = messages[-1]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("unavailable", content[0]["text"])
        self.assertNotIn("image_url", str(content))

    def test_direct_prompt_skips_missing_contact_sheet(self):
        messages = build_workflow_decision_prompt(
            {"current_state": {"last_node_id": "J12"}},
            visual_context={"success": False, "error": "overlay generation failed"},
        )

        content = messages[-1]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("unavailable", content[0]["text"])

    def test_autonomous_prompt_attaches_valid_contact_sheet(self):
        messages = build_autonomous_prompt(
            model_input={"current_state": {"last_node_id": "J12"}},
            candidate_context={"candidate_actions": []},
            round_index=1,
            visual_context={
                "kind": "pick_overlay",
                "contact_sheet": {"data_url": "data:image/png;base64,AA=="},
            },
            kb_tool_policy="disabled",
        )

        content = messages[-1]["content"]
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertEqual(content[-1]["image_url"]["url"], "data:image/png;base64,AA==")

    def test_inspect_prompt_attaches_overlay_and_score_scatter(self):
        messages = build_autonomous_prompt(
            model_input={"current_state": {"last_node_id": "J12"}},
            candidate_context={"candidate_actions": []},
            round_index=1,
            visual_context={
                "kind": "pick_inspection",
                "contact_sheet": {"data_url": "data:image/png;base64,OVERLAY=="},
                "score_scatter": {"data_url": "data:image/png;base64,SCATTER=="},
            },
            kb_tool_policy="disabled",
        )

        images = [item["image_url"]["url"] for item in messages[-1]["content"] if item["type"] == "image_url"]
        self.assertEqual(images, ["data:image/png;base64,OVERLAY==", "data:image/png;base64,SCATTER=="])


if __name__ == "__main__":
    unittest.main()

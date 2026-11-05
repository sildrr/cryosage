import unittest

from experiment_planner import (
    apply_anti_refinement_loop,
    build_planning_prompt,
    build_policy_plan,
    decision_respects_plan,
    restrict_candidate_context,
    validate_plan,
)


def action(action_id, job_type, **extra):
    return {
        "action_id": action_id,
        "job_type": job_type,
        "available": True,
        "required_inputs": extra.pop("required_inputs", {}),
        **extra,
    }


class ExperimentPlannerTests(unittest.TestCase):
    def test_prompt_prefers_near_target_nu_then_eligible_template_repicking(self):
        prompt = build_planning_prompt({}, [], [], 1)[0]["content"]
        self.assertIn("above target by less than 0.1 A", prompt)
        self.assertIn("prefer one modern NU-refine", prompt)
        self.assertIn("prefer Create Templates followed by Template Picker", prompt)
        self.assertIn("not additional hard gates", prompt)
        self.assertIn("min_best_ncc >= 0.75", prompt)
        self.assertNotIn("above-target result, prefer reclassify_selected_particles", prompt)

    def test_prompt_uses_native_diameter_box_preference_not_automatic_ctf_override(self):
        prompt = build_planning_prompt({}, [], [], 1)[0]["content"]
        self.assertIn("(1.5-2) * D_max_A / native_pixel_size_A", prompt)
        self.assertIn("nearby supported sizes", prompt)
        self.assertIn("not an automatic override", prompt)
        self.assertIn("Never approve box_size_pix above 720", prompt)
        self.assertNotIn("CTF-aware box recommendation as the next extraction candidate", prompt)

    def test_planning_prompt_explains_cryosift_score_direction_and_empty_set(self):
        prompt = build_planning_prompt({}, [], [], 1)[0]["content"]
        self.assertIn("lower scores indicate better class quality", prompt)
        self.assertIn("all scored classes missed", prompt)

    def test_target_unmet_prefers_upstream_selection_over_refinement(self):
        candidates = [
            action("refine", "nonuniform_refine_new", required_inputs={"particles": [{"source_job_uid": "J9"}]}),
            action("reselect", "select_2D", quality_rollback=True),
        ]
        plan = build_policy_plan(
            {"observed_issues": [{"issue_type": "target_resolution_unmet"}]},
            candidates, [], 4,
        )
        self.assertEqual(plan["approved_action_ids"], ["reselect"])

    def test_refinement_rollback_prefers_reselect_over_reclassify(self):
        candidates = [
            action(
                "reclassify", "class_2D_new",
                refinement_rollback=True,
                rollback_kind="reclassify_selected_particles",
            ),
            action(
                "reselect", "select_2D",
                refinement_rollback=True,
                rollback_kind="reselect_2d",
            ),
        ]
        plan = build_policy_plan(
            {"observed_issues": [{"issue_type": "target_resolution_unmet"}]},
            candidates, [], 1,
        )
        self.assertEqual(plan["approved_action_ids"], ["reselect"])

    def test_empty_cryosift_recommendation_prefers_picker_review_in_fallback(self):
        candidates = [
            action("reselect", "select_2D", refinement_rollback=True, rollback_kind="reselect_2d"),
            action("reinspect", "inspect_picks_v2", refinement_rollback=True, rollback_kind="reinspect_picks"),
        ]
        plan = build_policy_plan(
            {"cryosift_evidence": [{"success": True, "selected_class_ids": [], "class_scores": [{"score": 4.0}]}]},
            candidates, [], 1,
        )
        self.assertEqual(plan["approved_action_ids"], ["reinspect"])

    def test_failed_cryosift_does_not_trigger_empty_recommendation_policy(self):
        candidates = [
            action("reselect", "select_2D", refinement_rollback=True, rollback_kind="reselect_2d"),
            action("reinspect", "inspect_picks_v2", refinement_rollback=True, rollback_kind="reinspect_picks"),
        ]
        plan = build_policy_plan(
            {"cryosift_evidence": [{"success": False, "selected_class_ids": [], "class_scores": []}]},
            candidates, [], 1,
        )
        self.assertEqual(plan["approved_action_ids"], ["reselect"])

    def test_one_time_strategy_switch_can_survive_upstream_guard(self):
        candidates = [
            action("nu", "nonuniform_refine_new", refinement_strategy_switch=True,
                   required_inputs={"particles": [{"source_job_uid": "J9"}], "volume": [{"source_job_uid": "J9"}]}),
            action("reselect", "select_2D", quality_rollback=True),
        ]
        observation = {"observed_issues": [{"issue_type": "target_resolution_unmet"}]}
        plan = apply_anti_refinement_loop(
            {"approved_action_ids": ["nu"]}, observation, candidates, [], 1,
        )
        self.assertEqual(plan["approved_action_ids"], ["nu"])
        repeated = apply_anti_refinement_loop(
            {"approved_action_ids": ["nu"]}, observation, candidates,
            [{"job_type": "nonuniform_refine_new", "input_signature":
              '{"particles": [{"source_job_uid": "J9"}], "volume": [{"source_job_uid": "J9"}]}'}], 1,
        )
        self.assertEqual(repeated["approved_action_ids"], [])

    def test_repeated_refinement_is_blocked_for_same_inputs(self):
        candidates = [action("refine", "homo_refine_new", required_inputs={"particles": [{"source_job_uid": "J9"}]})]
        plan = apply_anti_refinement_loop(
            {"approved_action_ids": ["refine"]},
            {"observed_issues": []}, candidates,
            [{"job_type": "homo_refine_new", "input_signature": '{"particles": [{"source_job_uid": "J9"}]}' }],
            4,
        )
        self.assertEqual(plan["approved_action_ids"], [])

    def test_abinitio_requires_one_baseline_refinement_before_rollback(self):
        candidates = [
            action("refine", "homo_refine_new", required_inputs={"particles": [{"source_job_uid": "J9"}]}),
            action("repick", "blob_picker_gpu"),
        ]
        plan = apply_anti_refinement_loop(
            {"approved_action_ids": ["repick"]},
            {"dag": {"nodes": [{"job_type": "homo_abinit", "status": "completed"}]}},
            candidates, [], 4,
        )
        self.assertEqual(plan["approved_action_ids"], ["refine"])
        self.assertTrue(plan["anti_refinement_guard"]["baseline_refinement_required"])

    def test_new_abinitio_branch_requires_its_own_baseline_refinement(self):
        candidates = [
            action(
                "refine_d7",
                "homo_refine_new",
                required_inputs={
                    "particles": [{"source_job_uid": "J256", "source_output": "particles_class_0"}],
                    "volume": [{"source_job_uid": "J256", "source_output": "volume_class_0"}],
                },
            ),
            action("repick", "blob_picker_gpu"),
        ]
        plan = apply_anti_refinement_loop(
            {"approved_action_ids": ["repick"]},
            {
                "observed_issues": [{"issue_type": "target_resolution_unmet"}],
                "dag": {
                    "nodes": [
                        {"job_type": "homo_abinit", "status": "completed"},
                        {"job_type": "homo_refine_new", "status": "completed"},
                    ]
                },
            },
            candidates,
            [
                {
                    "job_type": "homo_refine_new",
                    "input_signature": '{"particles": [{"source_job_uid": "J251", "source_output": "particles_class_0"}], "volume": [{"source_job_uid": "J251", "source_output": "volume_class_0"}]}',
                }
            ],
            4,
        )
        self.assertEqual(plan["approved_action_ids"], ["refine_d7"])
        self.assertTrue(plan["anti_refinement_guard"]["baseline_refinement_required"])

    def test_execution_decision_must_name_planned_action_id(self):
        plan = {"approved_action_ids": ["select"]}
        ok, message = decision_respects_plan({"decision_type": "forward", "action_id": "select"}, plan)
        self.assertTrue(ok, message)
        ok, _ = decision_respects_plan({"decision_type": "forward", "action": "select_2D"}, plan)
        self.assertFalse(ok)

    def test_candidate_context_hides_unplanned_candidates(self):
        context = {"candidate_actions": [action("select", "select_2D"), action("refine", "homo_refine_new")], "blocked_actions": []}
        restricted = restrict_candidate_context(context, {"approved_action_ids": ["select"]})
        self.assertEqual([item["action_id"] for item in restricted["candidate_actions"]], ["select"])
        self.assertEqual(restricted["blocked_actions"][-1]["action_id"], "refine")

    def test_plan_rejects_unknown_action_id(self):
        with self.assertRaises(ValueError):
            validate_plan({"decision_type": "plan_experiments", "approved_action_ids": ["unknown"]}, [action("known", "select_2D")], 2)


if __name__ == "__main__":
    unittest.main()

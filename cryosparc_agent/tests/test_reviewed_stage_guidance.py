"""Advisory-only tests, independent of clients, execution and model APIs."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cryosparc_agent"))
from stage_guidance import (  # noqa: E402
    DEFAULT_CARDS_PATH, JOB_STAGES, ROLLBACK_STAGES, SupplementalGuidance,
    last_completed_from_model_input, last_completed_from_observation,
)


CARD_STAGES = {
    "D01": "import_movies", "D02": "patch_motion_correction",
    "D03": "ctf_estimation", "S01": "ctf_method_selection",
    "S02": "particle_picking", "D04": "extract_from_micrographs",
    "D05": "2d_classification", "D06": "ab_initio_reconstruction",
    "D07": "heterogeneous_refinement", "D08": "homogeneous_refinement",
    "D09": "non_uniform_refinement", "D10": "local_refinement",
    "D11": "local_ctf_refinement", "D12": "3d_variability_analysis",
    "R01": "particle_overlap_and_extra_density", "Q01": "fsc_mask_interpretation",
}


class ReviewedStageGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "cards.json"
        self.document = {
            "version": "reviewed-supplements-20261016", "source_sha256": "a" * 64,
            "cards": [{"card_id": cid, "stage": stage,
                       "advice": ["Check actual evidence before adopting this advice.",
                                  "Retain existing validation and stop conditions."]}
                      for cid, stage in CARD_STAGES.items()],
        }
        self.write()
        self.guidance = SupplementalGuidance(self.path)
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "contract",
                "cache_control": {"type": "ephemeral"}}]},
            {"role": "system", "content": "second system"},
            {"role": "user", "content": [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": '{"state":{},"output_schema":{"reason":"string"}}'}]},
        ]

    def write(self):
        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    def select(self, guidance=None, **kwargs):
        audit = {}
        result = (guidance or self.guidance).inject(self.messages, audit=audit, **kwargs)
        return result, audit

    def test_missing_configuration_returns_identical_messages_and_audit_error(self):
        with self.assertLogs("stage_guidance", "WARNING"):
            guidance = SupplementalGuidance(self.path.with_name("missing.json"))
            result, audit = self.select(guidance, candidate_actions=[{"job_type": "select_2D"}])
        self.assertIs(result, self.messages)
        self.assertTrue(audit["error"])
        self.assertFalse(audit["enabled"])
        self.assertIsNone(audit["actual_injected_text"])
        self.assertIsNone(audit["injection_position"])

    def test_malformed_resources_fail_open(self):
        invalid = [None, {}, {**self.document, "version": "wrong"},
                   {**self.document, "source_sha256": None},
                   {**self.document, "cards": []},
                   {**self.document, "cards": [None]},
                   {**self.document, "cards": [self.document["cards"][0]] * 2}]
        for field, value in [("card_id", "X01"), ("stage", "unknown"),
                             ("advice", []), ("advice", [""]), ("advice", [1]),
                             ("card_id", "R01")]:
            invalid.append({**self.document, "cards": [{**self.document["cards"][0], field: value}]})
        for document in invalid:
            with self.subTest(document=document), self.assertLogs("stage_guidance", "WARNING"):
                self.path.write_text(json.dumps(document), encoding="utf-8")
                result, audit = self.select(SupplementalGuidance(self.path))
                self.assertIs(result, self.messages)
                self.assertTrue(audit["error"])
        self.path.write_text("not json", encoding="utf-8")
        with self.assertLogs("stage_guidance", "WARNING"):
            result, audit = self.select(SupplementalGuidance(self.path))
        self.assertIs(result, self.messages)
        self.assertTrue(audit["error"])

    def test_invalid_budget_never_halts(self):
        for budget in [0, -1, True, 12.5, "8192", None]:
            with self.subTest(budget=budget), self.assertLogs("stage_guidance", "WARNING"):
                guidance = SupplementalGuidance(self.path, max_bytes=budget)
                result, audit = self.select(guidance, candidate_actions=[{"job_type": "select_2D"}])
                self.assertIs(result, self.messages)
                self.assertTrue(audit["error"])

    def test_small_budget_skips_whole_cards_and_continues(self):
        self.document["cards"][0]["advice"] = ["Long complete advice. " * 1000]
        self.write()
        guidance = SupplementalGuidance(self.path, max_bytes=1100)
        actions = [{"job_type": "import_movies"}, {"job_type": "patch_motion_correction"}]
        with self.assertLogs("stage_guidance", "WARNING"):
            result, audit = self.select(guidance, candidate_actions=actions)
            reversed_result, reversed_audit = self.select(guidance, candidate_actions=actions[::-1])
        self.assertEqual(audit["matched_card_ids"], ["D01", "D02"])
        self.assertEqual(audit["omitted_card_ids"], ["D01"])
        self.assertEqual(audit["injected_card_ids"], ["D02"])
        self.assertEqual(result, reversed_result)
        self.assertEqual(audit, reversed_audit)
        cards = json.loads(audit["actual_injected_text"])["supplemental_guidance"]["cards"]
        self.assertEqual(cards, [self.document["cards"][1]])
        self.assertLessEqual(audit["budget_used"], 1100)
        with self.assertLogs("stage_guidance", "WARNING"):
            result, audit = self.select(SupplementalGuidance(self.path, max_bytes=1),
                                        candidate_actions=actions)
        self.assertIs(result, self.messages)
        self.assertEqual(audit["omitted_card_ids"], ["D01", "D02"])
        self.assertEqual(audit["budget_used"], 0)
        self.assertIsNone(audit["actual_injected_text"])

    def test_unknown_jobs_and_ineligible_nodes_do_not_inject(self):
        for job in ["unknown", "import_micrographs", "my_blob_picker_gpu", "CLASS_2D_NEW"]:
            result, audit = self.select(candidate_actions=[{"job_type": job, "action_id": "select_2D"}])
            self.assertIs(result, self.messages)
            self.assertEqual(audit["matched_card_ids"], [])
        result, _ = self.select(last_completed_job={"job_type": "class_2D_new", "status": "failed"},
                                candidate_actions=[{"job_type": "class_2D_new", "available": False},
                                                   {"job_type": "select_2D", "blocked_by": ["gate"]}])
        self.assertIs(result, self.messages)

    def test_exact_aliases_and_rollback_fallback(self):
        self.assertEqual(len(JOB_STAGES), 27)
        for alias, canonical in [("patch_ctf", "patch_ctf_estimation_multi"),
                                 ("patch_ctf_estimation", "patch_ctf_estimation_multi"),
                                 ("homo_refine", "homo_refine_new")]:
            self.assertEqual(JOB_STAGES[alias], JOB_STAGES[canonical])
        for job, stages in JOB_STAGES.items():
            with self.subTest(job=job):
                _, audit = self.select(candidate_actions=[{"job_type": job}])
                expected = sorted(cid for cid, stage in CARD_STAGES.items() if stage in stages)
                self.assertEqual(audit["matched_card_ids"], expected)
        for rollback, stages in ROLLBACK_STAGES.items():
            with self.subTest(rollback=rollback):
                _, audit = self.select(candidate_actions=[{"job_type": "create_templates",
                                                           "rollback_kind": rollback}])
                self.assertEqual(audit["matched_card_ids"], sorted(
                    cid for cid, stage in CARD_STAGES.items() if stage in stages))
        _, audit = self.select(candidate_actions=[{"job_type": "class_2D_new", "rollback_kind": "repick_diameter"}])
        self.assertEqual(audit["matched_card_ids"], ["D05"])

    def test_completed_last_job_available_candidates_and_dedup(self):
        _, audit = self.select(last_completed_job={"job_type": "patch_ctf_estimation_multi", "status": "completed"},
                                candidate_actions=[{"job_type": "select_2D"}, {"job_type": "class_2D_new"}],
                                audience="planner")
        self.assertEqual(audit["matched_card_ids"], ["D03", "D05", "S01"])
        cards = json.loads(audit["actual_injected_text"])["supplemental_guidance"]["cards"]
        self.assertEqual(len(cards), 3)

    def test_q01_requires_actual_structured_fsc_evidence(self):
        positive = [
            {"resolution_evidence": {"values": {"fsc_info.radwn_final_A": 3.2}}},
            {"fsc_info.radmask_A": {"value": 3.2}},
            {"metric": "fsc_info.radwn_final_A", "resolution_A": 3.2},
            {"fsc_info": {"radwn_final_A": 3.2}},
            {"observed_metrics": {"fsc_info": {"value": {"radwn_final_A": 3.2}}}},
            {"fsc_curve": [1.0, 0.8, 0.1]},
            {"kind": "gsfsc_plot", "available": True, "local_path": "fsc.png"},
        ]
        negative = [
            {"resolution_A": 3.2}, {"ctf_fit_resolution_A": 3.2},
            {"fsc_resolution_A": None}, {"fsc_resolution_A": float("nan")},
            {"fsc_resolution_A": True}, {"fsc_resolution_A": -1},
            {"available": False, "fsc_info": {"radwn_A": 3.2}},
            {"success": False, "fsc_resolution_A": 3.2},
            {"hypotheses": [{"fsc_resolution_A": 3.2}]},
            {"parameters": {"fsc_resolution_A": 3.2}},
            {"fsc_curve": []}, {"fsc_curve": [True, 0.1]},
            {"kind": "fsc_plot", "available": True},
        ]
        for evidence in positive + negative:
            with self.subTest(evidence=evidence):
                _, audit = self.select(quality_evidence=evidence)
                self.assertEqual("Q01" in audit["matched_card_ids"], evidence in positive)
        _, audit = self.select(candidate_actions=[{"job_type": "homo_refine_new",
                                                   "parameters": {"fsc_resolution_A": 3.2}}])
        self.assertEqual(audit["matched_card_ids"], ["D08"])

    def test_r01_needs_confirmed_evidenced_overlap_not_neighbor_count(self):
        finding = {"issue_type": "particle_overlap", "observed": True,
                   "evidence": {"overlay": "J10_image"}}
        positive = [{"observed_issues": [finding]},
                    {**finding, "observed": None, "status": "confirmed"},
                    {"overlap_fraction": 0.1, "evidence_refs": ["J10"]}]
        negative = [{**finding, "observed": False},
                    {**finding, "observed": False, "status": "confirmed"},
                    {**finding, "present": False}, {**finding, "evidence": {}},
                    {"hypotheses": [finding]}, {"tasks": [finding]},
                    {"neighbor_count": 100, "evidence_refs": ["J10"]},
                    {"overlap_fraction": 0, "evidence_refs": ["J10"]},
                    {"overlap_fraction": 0.1}, {"issue_type": "particle_overlap"}]
        for evidence in positive + negative:
            with self.subTest(evidence=evidence):
                _, audit = self.select(workflow_state={"quality_evidence": evidence})
                self.assertEqual("R01" in audit["matched_card_ids"], evidence in positive)

    def test_no_inputs_mutated_preserves_system_cache_images_and_schema(self):
        job = {"job_type": "select_2D", "status": "completed", "nested": {"a": []}}
        actions = [{"job_type": "class_2D_new", "parameters": {"a": [1]}}]
        state = {"candidate_actions": actions}
        evidence = {"fsc_info": {"radwn_A": 3.2}}
        before = deepcopy([self.messages, job, actions, state, evidence])
        result, audit = self.select(last_completed_job=job, candidate_actions=actions,
                                    workflow_state=state, quality_evidence=evidence)
        self.assertEqual([self.messages, job, actions, state, evidence], before)
        self.assertEqual(audit["injection_position"], 2)
        self.assertEqual(result[2], {"role": "user", "content": audit["actual_injected_text"]})
        for original, retained in zip(self.messages, result[:2] + result[3:]):
            self.assertIs(original, retained)
        usage = json.loads(result[2]["content"])["supplemental_guidance"]["usage"]
        self.assertIn("Subordinate", usage)
        self.assertIn("optionally", usage)
        self.assertIn("no extra output", usage)

    def test_load_once_metadata_is_detached_and_content_is_immutable(self):
        metadata = self.guidance.metadata()
        self.assertEqual(metadata["source_sha256"], "a" * 64)
        self.assertEqual(metadata["normalized_sha256"], sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(metadata["version"], self.document["version"])
        self.path.unlink()
        metadata["enabled"] = False
        _, first = self.select(candidate_actions=[{"job_type": "select_2D"}])
        first["matched_card_ids"].clear()
        _, second = self.select(candidate_actions=[{"job_type": "select_2D"}])
        self.assertEqual(second["matched_card_ids"], ["D05"])
        self.assertTrue(self.guidance.metadata()["enabled"])

    def test_helpers_only_completed_detached_snapshots(self):
        model = {"current_state": {"last_node_status": "completed", "last_action": "select_2D",
                                   "last_node_info": {"nested": {"a": []}}}}
        result = last_completed_from_model_input(model)
        self.assertEqual(result["job_type"], "select_2D")
        result["nested"]["a"].append(1)
        self.assertEqual(model["current_state"]["last_node_info"]["nested"]["a"], [])
        model["current_state"]["last_node_status"] = "running"
        self.assertIsNone(last_completed_from_model_input(model))
        nodes = [{"job_uid": "J2", "status": "completed"},
                 {"cryosparc_job_uid": "J10", "status": "completed", "nested": []},
                 {"job_uid": "J11", "status": "running"}]
        result = last_completed_from_observation({"dag": {"nodes": nodes}})
        self.assertEqual(result["cryosparc_job_uid"], "J10")
        result["nested"].append(1)
        self.assertEqual(nodes[1]["nested"], [])
        for bad in [None, {}, {"dag": None}, {"dag": {"nodes": None}}]:
            self.assertIsNone(last_completed_from_observation(bad))
        for bad in [None, {}, {"current_state": []}]:
            self.assertIsNone(last_completed_from_model_input(bad))

    def test_all_fixture_cards_fit_under_8192_bytes(self):
        _, audit = self.select(candidate_actions=[{"job_type": job} for job in JOB_STAGES],
                                quality_evidence={"fsc_resolution_A": 3.2,
                                    "overlap_fraction": 0.2, "evidence_refs": ["J10"]})
        self.assertEqual(audit["injected_card_ids"], sorted(CARD_STAGES))
        self.assertEqual(audit["omitted_card_ids"], [])
        self.assertLessEqual(audit["budget_used"], 8192)
        self.assertEqual(audit["budget_used"], len(audit["actual_injected_text"].encode("utf-8")))

    @unittest.skipUnless(DEFAULT_CARDS_PATH.exists(), "reviewed resource owned by another agent")
    def test_packaged_cards_fit_under_8192_bytes(self):
        guidance = SupplementalGuidance()
        self.assertTrue(guidance.metadata()["enabled"])
        for job in JOB_STAGES:
            with self.subTest(job=job):
                _, audit = self.select(guidance, candidate_actions=[{"job_type": job}])
                self.assertTrue(audit["injected_card_ids"])
                self.assertEqual(audit["omitted_card_ids"], [])
                self.assertLessEqual(audit["budget_used"], 8192)
        for evidence, cid in [({"fsc_resolution_A": 3.2}, "Q01"),
                              ({"overlap_fraction": 0.1, "evidence_refs": ["J1"]}, "R01")]:
            _, audit = self.select(guidance, quality_evidence=evidence)
            self.assertEqual(audit["injected_card_ids"], [cid])
            self.assertEqual(audit["omitted_card_ids"], [])
            self.assertLessEqual(audit["budget_used"], 8192)


if __name__ == "__main__":
    unittest.main()

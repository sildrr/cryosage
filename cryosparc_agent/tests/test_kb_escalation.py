"""Tests for the one-time post-budget KB recovery experiment."""
import argparse
import unittest

from autonomous_mcp_closed_loop import resolve_kb_escalation


def args(**overrides):
    values = {
        "kb_escalation_after_jobs": 20,
        "kb_escalation_max_rounds": 1,
        "kb_tool_policy": "disabled",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def observation(job_count, best_resolution=None):
    issues = []
    if best_resolution is not None:
        issues.append({
            "issue_type": "target_resolution_unmet",
            "evidence": {"best_observed_resolution_A": best_resolution},
        })
    return {"dag": {"nodes": [{}] * job_count}, "observed_issues": issues}


class KbEscalationTests(unittest.TestCase):
    def test_does_not_enable_before_job_budget(self):
        policy, active, state = resolve_kb_escalation(args(), observation(19, 3.4), {})
        self.assertEqual(policy, "disabled")
        self.assertFalse(active)
        self.assertFalse(state["activated"])

    def test_requires_an_explicit_unmet_resolution(self):
        policy, active, state = resolve_kb_escalation(args(), observation(20), {})
        self.assertEqual(policy, "disabled")
        self.assertFalse(active)
        self.assertFalse(state["activated"])

    def test_enables_exactly_one_kb_guided_decision(self):
        policy, active, state = resolve_kb_escalation(args(), observation(20, 3.1), {})
        self.assertEqual(policy, "required")
        self.assertTrue(active)
        self.assertEqual(state["trigger_job_count"], 20)
        self.assertEqual(state["trigger_best_resolution_A"], 3.1)

        state["rounds_completed"] = 1
        policy, active, _ = resolve_kb_escalation(args(), observation(21, 3.0), state)
        self.assertEqual(policy, "disabled")
        self.assertFalse(active)


if __name__ == "__main__":
    unittest.main()

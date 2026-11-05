import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from job_executor import approval_reasons_for, build_queue_plan, execute_job_action
from job_specs import get_job_spec


class QueuePlanTests(unittest.TestCase):
    def test_cpu_noninteractive_job_uses_configured_lane(self):
        with patch.dict(os.environ, {"CRYOAGENT_GPU_LANE": "h20_slurm"}, clear=False):
            queue = build_queue_plan(get_job_spec("create_templates"), lane=None)
        self.assertEqual(queue["lane"], "h20_slurm")
        self.assertTrue(queue["will_queue"])

    def test_cpu_noninteractive_job_defers_lane_when_auto_select_is_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            queue = build_queue_plan(get_job_spec("create_templates"), lane=None)
        self.assertIsNone(queue["lane"])
        self.assertTrue(queue["will_queue"])

    def test_cpu_noninteractive_job_uses_default_lane_when_auto_select_is_disabled(self):
        with patch.dict(os.environ, {"CRYOAGENT_AUTO_SELECT_LANE": "0"}, clear=True):
            queue = build_queue_plan(get_job_spec("create_templates"), lane=None)
        self.assertEqual(queue["lane"], "g8m192_4090_slurm")

    def test_select_2d_queues_without_lane_when_class_ids_are_explicit(self):
        with patch.dict(os.environ, {"CRYOAGENT_GPU_LANE": "h20_slurm"}, clear=False):
            queue = build_queue_plan(get_job_spec("select_2D"), lane=None)
        self.assertIsNone(queue["lane"])
        self.assertTrue(queue["queue_without_lane"])
        self.assertTrue(queue["will_queue"])

    def test_inspect_picks_queues_without_lane(self):
        with patch.dict(os.environ, {"CRYOAGENT_GPU_LANE": "h20_slurm"}, clear=False):
            queue = build_queue_plan(get_job_spec("inspect_picks_v2"), lane=None)
        self.assertIsNone(queue["lane"])
        self.assertTrue(queue["queue_without_lane"])
        self.assertTrue(queue["will_queue"])

    def test_branch_label_does_not_block_automated_select_2d(self):
        action = {
            "action_type": "branch",
            "resolved_parameters": {"selected_templates": "1,2"},
        }
        self.assertEqual(
            approval_reasons_for(action, get_job_spec("select_2D")),
            [],
        )


class LiveQueueTests(unittest.TestCase):
    def test_obsolete_cached_plan_is_rejected_before_creation_and_budget_admission(self):
        cs = unittest.mock.Mock()
        cs.job_register.specs = [SimpleNamespace(type='fsc3D', stability='obsolete')]
        planned = dict(job_type='fsc3D', approval_required=False,
            resolved_parameters={}, connections={}, queue={'will_queue': True})
        with patch('job_executor.cryosparc_client', return_value=cs):
            result = execute_job_action('P6', 'W1', planned, dry_run=False)
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'obsolete_job_type')
        cs.find_workspace.assert_not_called()
        cs.create_job.assert_not_called()

    def test_lane_submission_does_not_also_pass_hostname(self):
        job = unittest.mock.Mock(uid="J42", status="queued")
        workspace = unittest.mock.Mock()
        workspace.create_job.return_value = job
        cs = unittest.mock.Mock()
        cs.find_workspace.return_value = workspace
        planned_action = {
            "plan_step": 1,
            "action_id": "patch_ctf",
            "action_type": "forward",
            "workflow_node_id": "patch_ctf",
            "job_type": "patch_ctf_estimation_multi",
            "execution_mode": "create_and_queue",
            "connections": {"exposures": [("J1", "imported_micrographs")]},
            "resolved_parameters": {"compute_num_gpus": 1},
            "queue": {
                "lane": "g8m192_4090_slurm",
                "hostname": "g8m192_4090_slurm",
                "gpus": [],
                "cluster_vars": {},
                "queue_without_lane": False,
                "will_queue": True,
            },
            "requires_gpu": True,
            "interactive": False,
            "approval_required": False,
            "approval_reasons": [],
        }
        with patch("job_executor.cryosparc_client", return_value=cs):
            result = execute_job_action("P19", "W1", planned_action, dry_run=False)
        self.assertTrue(result["success"])
        job.queue.assert_called_once_with(
            lane="g8m192_4090_slurm", gpus=[], cluster_vars={},
        )

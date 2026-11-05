import os
import sys
import types
import unittest
from unittest.mock import patch

from webapp import gpu_routing


class RoutingTests(unittest.TestCase):
    def test_web_overrides_stale_pin(self):
        with patch.dict(os.environ, {'CRYOAGENT_WEB_GPU_ROUTING': '1',
                'CRYOAGENT_GPU_LANE': 'g8m192_4090_slurm',
                'CRYOAGENT_GPU_LANES': 'g8m192_4090_slurm', 'CRYOAGENT_AUTO_SELECT_LANE': '0'}):
            gpu_routing.configure()
            self.assertNotIn('CRYOAGENT_GPU_LANE', os.environ)
            self.assertEqual(os.environ['CRYOAGENT_GPU_LANES'], ','.join(gpu_routing.LANES))
            self.assertEqual(os.environ['CRYOAGENT_AUTO_SELECT_LANE'], '1')

    def test_experiment_pin_unchanged(self):
        with patch.dict(os.environ, {'CRYOAGENT_GPU_LANE': 'h20_slurm',
                'CRYOAGENT_AUTO_SELECT_LANE': '0'}, clear=True):
            gpu_routing.configure()
            self.assertEqual(gpu_routing.choose_lane(), 'h20_slurm')

    def test_refresh_capacity_each_submission(self):
        fake = types.ModuleType('resource_strategy')
        from unittest.mock import Mock
        fake.select_available_lane = Mock(side_effect=[
            {'success': True, 'lane': 'h20_slurm'},
            {'success': True, 'lane': 'g8m192_4090_slurm'}, {'success': False}])
        with patch.dict(sys.modules, {'resource_strategy': fake}), patch.dict(os.environ,
                {'CRYOAGENT_WEB_GPU_ROUTING': '1'}, clear=True):
            self.assertEqual(gpu_routing.choose_lane(), 'h20_slurm')
            self.assertEqual(gpu_routing.choose_lane(), 'g8m192_4090_slurm')
            self.assertEqual(gpu_routing.choose_lane(), 'g8m192_4090_slurm')

    def test_existing_jobs_not_requeued(self):
        from unittest.mock import Mock
        fake = types.ModuleType('prepare_visual_rl_batch')
        fake.queue_and_wait = Mock(return_value='completed')
        with patch.dict(sys.modules, {'prepare_visual_rl_batch': fake}), patch.dict(os.environ,
                {'CRYOAGENT_WEB_GPU_ROUTING': '1'}), patch.object(gpu_routing, 'choose_lane') as choose:
            for status in ['queued', 'running', 'completed']:
                job = types.SimpleNamespace(status=status, uid='J1')
                gpu_routing.queue_and_wait(job, 'old', 100, 1)
            choose.assert_not_called()


if __name__ == '__main__':
    unittest.main()

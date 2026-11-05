"""Submission-time web resource routing must not change scientific inputs."""
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from webapp import gpu_routing as routing


class DynamicGpuTests(unittest.TestCase):
    def capacity(self, idle=6, per_node=3):
        return dict(success=True, lane='h20_slurm', observations=[dict(
            lane='h20_slurm', gpu_idle=idle, gpu_idle_max_node=per_node)])

    def test_count_is_same_node_bounded_and_configurable(self):
        with patch.dict(os.environ, {'CRYOAGENT_WEB_MAX_GPUS': '8'}):
            self.assertEqual(routing.allocation(self.capacity(), supports_multi=True)['gpu_count'], 3)
            self.assertEqual(routing.allocation(self.capacity(), supports_multi=False)['gpu_count'], 1)
            self.assertEqual(routing.allocation(self.capacity(0, 0), supports_multi=True)['gpu_count'], 1)
            self.assertEqual(routing.allocation(dict(success=False), supports_multi=True)['gpu_count'], 1)
        with patch.dict(os.environ, {'CRYOAGENT_WEB_MAX_GPUS': '2'}):
            self.assertEqual(routing.allocation(self.capacity(), supports_multi=True)['gpu_count'], 2)

    def test_web_removes_stale_single_gpu_cap(self):
        with patch.dict(os.environ, {'CRYOAGENT_WEB_GPU_ROUTING': '1',
                'CRYOAGENT_GPU_COUNT_CAP': '1'}, clear=True):
            routing.configure()
            self.assertEqual(os.environ['CRYOAGENT_GPU_COUNT_CAP'], '8')

    def client(self, *, multi=True, status='building', lightweight=False):
        params = {'compute_num_gpus': object()} if multi else {}
        job = SimpleNamespace(status=status, type='class_2D_new' if multi else 'nonuniform_refine_new',
            full_spec=SimpleNamespace(params=params, interactive=False, lightweight=lightweight),
            set_param=Mock())
        original = Mock(return_value='native response')
        client = SimpleNamespace(api=SimpleNamespace(jobs=SimpleNamespace(enqueue=original)),
            find_job=Mock(return_value=job))
        return client, job, original

    def test_enqueue_updates_supported_count_and_routes_once(self):
        client, job, original = self.client()
        resource = SimpleNamespace(select_available_lane=Mock(return_value=self.capacity()))
        with patch.dict(os.environ, {'CRYOAGENT_WEB_GPU_ROUTING': '1',
                'CRYOAGENT_WEB_DYNAMIC_GPUS': '1'}, clear=True), patch.dict(sys.modules, {'resource_strategy': resource}):
            routing.protect_client_enqueue(client)
            hook = client.api.jobs.enqueue
            routing.protect_client_enqueue(client)
            self.assertIs(hook, client.api.jobs.enqueue)
            response = hook('P6', 'J1', lane=routing.LANES[0], gpus=[], hostname=None)
        self.assertEqual(response, 'native response')
        job.set_param.assert_called_once_with('compute_num_gpus', 3)
        original.assert_called_once_with('P6', 'J1', lane='h20_slurm', gpus=[], hostname=None)

    def test_native_single_gpu_type_never_gets_unsupported_parameter(self):
        client, job, original = self.client(multi=False)
        resource = SimpleNamespace(select_available_lane=Mock(return_value=self.capacity()))
        with patch.dict(os.environ, {'CRYOAGENT_WEB_GPU_ROUTING': '1',
                'CRYOAGENT_WEB_DYNAMIC_GPUS': '1'}), patch.dict(sys.modules, {'resource_strategy': resource}):
            routing.protect_client_enqueue(client)
            client.api.jobs.enqueue('P6', 'J1', lane=routing.LANES[0])
        job.set_param.assert_not_called()
        original.assert_called_once()

    def test_non_gpu_pinned_and_existing_jobs_unchanged(self):
        for options, status, lightweight in [
            ({'lane': None}, 'building', False),
            ({'lane': routing.LANES[0], 'gpus': [0]}, 'building', False),
            ({'lane': routing.LANES[0], 'hostname': 'explicit'}, 'building', False),
            ({'lane': routing.LANES[0]}, 'running', False),
            ({'lane': routing.LANES[0]}, 'building', True),
        ]:
            client, job, original = self.client(status=status, lightweight=lightweight)
            with patch.dict(os.environ, {'CRYOAGENT_WEB_GPU_ROUTING': '1',
                    'CRYOAGENT_WEB_DYNAMIC_GPUS': '1'}):
                routing.protect_client_enqueue(client)
                client.api.jobs.enqueue('P6', 'J1', **options)
            job.set_param.assert_not_called()
            original.assert_called_once_with('P6', 'J1', **options)

    def test_frozen_protocol_can_disable_hook_after_client_creation(self):
        client, job, original = self.client()
        with patch.dict(os.environ, {'CRYOAGENT_WEB_GPU_ROUTING': '1',
                'CRYOAGENT_WEB_DYNAMIC_GPUS': '1'}):
            routing.protect_client_enqueue(client)
            os.environ['CRYOAGENT_WEB_DYNAMIC_GPUS'] = '0'
            client.api.jobs.enqueue('P6', 'J1', lane=routing.LANES[0])
        job.set_param.assert_not_called()
        original.assert_called_once()

    def test_inventory_excludes_drained_nodes_and_does_not_sum_node_capacity(self):
        import resource_strategy
        result = SimpleNamespace(returncode=0, stderr='', stdout='\n'.join([
            'NodeName=a Partitions=gpu State=MIXED Gres=gpu:4 GresUsed=gpu:1',
            'NodeName=b Partitions=gpu State=MIXED Gres=gpu:8 GresUsed=gpu:6',
            'NodeName=c Partitions=gpu State=IDLE+DRAIN Gres=gpu:8 GresUsed=gpu:0',
        ]))
        with patch.object(resource_strategy.subprocess, 'run', return_value=result):
            inventory = resource_strategy._query_gpu_inventory('gpu', timeout=1)
        self.assertEqual(inventory['gpu_idle'], 5)
        self.assertEqual(inventory['gpu_idle_max_node'], 3)


if __name__ == '__main__':
    unittest.main()

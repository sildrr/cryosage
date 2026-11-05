import os
import unittest
from unittest.mock import patch
from webapp.core import controller_command


class PlacementTests(unittest.TestCase):
    def test_cpu2_can_be_pinned_without_changing_script_or_gpu_lane(self):
        with patch.dict(os.environ,{'CRYOAGENT_WEB_SLURM_NODELIST':'cpu2'}):
            command=controller_command('a'*32)
            self.assertIn('--nodelist=cpu2',command)
            self.assertTrue(command[-2].endswith('controller.sbatch'))

    def test_unset_placement_preserves_scheduler_choice(self):
        with patch.dict(os.environ,{'CRYOAGENT_WEB_SLURM_NODELIST':''}):
            self.assertFalse(any(x.startswith('--nodelist=') for x in controller_command('a'*32)))

    def test_invalid_node_list_rejected(self):
        for value in ['cpu2 --mem=1G','cpu2;bad','../cpu2']:
            with patch.dict(os.environ,{'CRYOAGENT_WEB_SLURM_NODELIST':value}):
                with self.assertRaises(ValueError):
                    controller_command('a'*32)


if __name__ == '__main__':
    unittest.main()

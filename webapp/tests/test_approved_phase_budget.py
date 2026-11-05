import json
from pathlib import Path
import tempfile
import unittest
from webapp.pilot_pipeline import picker_class


class ApprovedBudgetTests(unittest.TestCase):
    def test_nested_guards_keep_reallocated_limit_without_raising_total(self):
        cls=picker_class()
        with tempfile.TemporaryDirectory() as d:
            instance=cls.__new__(cls)
            instance.root=Path(d)
            instance.manifest=dict(max_seed_jobs=80,max_new_jobs=100)
            instance.state=dict(steps={str(i):dict(job_uid='J'+str(i)) for i in range(80)})
            (instance.root/'pilot_diversity_approval.json').write_text('{}')
            (instance.root/'benchmark_budget.json').write_text(json.dumps(dict(maximum=95)))
            instance.guard_budget('new_bank')
            self.assertEqual(instance.manifest['max_seed_jobs'],95)
            instance.guard_budget('new_full_job')
            self.assertEqual(instance.manifest['max_seed_jobs'],95)
            instance.state['steps']={str(i):dict(job_uid='J'+str(i)) for i in range(95)}
            with self.assertRaises(RuntimeError):
                instance.guard_budget('too_many')


if __name__ == '__main__':
    unittest.main()

"""Production snapshot KB calls need an audit path scoped to the current task."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from webapp import pilot_pipeline


class PilotKBAuditTests(unittest.TestCase):
    def test_each_task_overrides_missing_or_inherited_audit_path(self):
        budget = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict('sys.modules', {'benchmark_budget': budget}):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop('CRYOAGENT_BENCHMARK_KB_AUDIT', None)
                    for workspace in ['W191', 'W192']:
                        root = Path(directory) / workspace
                        root.mkdir()
                        task = dict(pipeline_version=pilot_pipeline.VERSION,
                                    project='P6', workspace=workspace,
                                    config=dict(max_created_jobs=100))
                        (root / 'benchmark_budget.json').write_text(json.dumps(
                            dict(project='P6', workspace=workspace, maximum=100)))
                        pilot_pipeline.configure(task, root, Mock())
                        self.assertEqual(os.environ['CRYOAGENT_BENCHMARK_KB_AUDIT'],
                                         str(root / 'kb_queries.jsonl'))
                    self.assertEqual(budget.protect_client.call_count, 2)

    def test_legacy_configuration_is_unchanged(self):
        with patch.dict(os.environ, {'CRYOAGENT_BENCHMARK_KB_AUDIT': 'legacy.jsonl'}):
            pilot_pipeline.configure({}, Path('/unused'), Mock())
            self.assertEqual(os.environ['CRYOAGENT_BENCHMARK_KB_AUDIT'], 'legacy.jsonl')


if __name__ == '__main__':
    unittest.main()

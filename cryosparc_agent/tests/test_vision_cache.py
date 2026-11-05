"""Tests for per-deployment visual evidence cache placement."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision_inputs import vision_cache_dir


class VisionCacheTests(unittest.TestCase):
    def test_explicit_cache_directory_wins(self):
        with mock.patch.dict(os.environ, {"CRYOAGENT_VISION_CACHE_DIR": "/tmp/agent-vision"}):
            self.assertEqual(vision_cache_dir(), Path("/tmp/agent-vision"))

    def test_default_is_relative_to_running_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("vision_inputs.Path.cwd", return_value=Path(tmpdir)):
                    self.assertEqual(vision_cache_dir(), Path(tmpdir) / "logs" / "vision_inputs")


if __name__ == "__main__":
    unittest.main()

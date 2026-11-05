import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryosift_adapter import _parse_scores, load_cached_cryosift_result


class CryoSiftAdapterTests(unittest.TestCase):
    def test_converts_one_based_runner_scores_to_cryosparc_class_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            score_file = Path(directory) / "score.txt"
            score_file.write_text("1 4.0\n2 2.0\n3 1.0\n")
            previous = os.environ.get("CRYOAGENT_CRYOSIFT_SCORE_INDEX_BASE")
            os.environ["CRYOAGENT_CRYOSIFT_SCORE_INDEX_BASE"] = "1"
            try:
                selected, scores, path = _parse_scores(Path(directory), threshold=3.0)
            finally:
                if previous is None:
                    os.environ.pop("CRYOAGENT_CRYOSIFT_SCORE_INDEX_BASE", None)
                else:
                    os.environ["CRYOAGENT_CRYOSIFT_SCORE_INDEX_BASE"] = previous

        self.assertEqual(selected, [1, 2])
        self.assertEqual(scores, {0: 4.0, 1: 2.0, 2: 1.0})
        self.assertEqual(path, str(score_file))

    def test_successful_empty_recommendation_keeps_scores_and_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cryosift_eval"
            output.mkdir()
            (output / "score.txt").write_text("1 4.0\n2 5.0\n")
            with patch("cryosift_adapter._job_directory", return_value=Path(directory)):
                result = load_cached_cryosift_result("P1", "J2", 3.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["selected_class_ids"], [])
        self.assertTrue(result["no_classes_below_threshold"])
        self.assertEqual(result["score_direction"], "lower_is_better")

    def test_missing_scores_are_not_successful_empty_recommendations(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("cryosift_adapter._job_directory", return_value=Path(directory)):
                self.assertIsNone(load_cached_cryosift_result("P1", "J2"))

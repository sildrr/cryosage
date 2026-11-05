import unittest

import numpy as np

from volume_scale import measure_mask_extent


class VolumeScaleTests(unittest.TestCase):
    def test_measure_mask_extent_adds_margin_and_rounds(self):
        mask = np.zeros((80, 80, 80), dtype=np.float32)
        mask[10:70, 20:60, 20:60] = 1.0

        result = measure_mask_extent(mask, 2.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["raw_mask_diameter_A"], 120.0)
        self.assertEqual(result["recommended_particle_diameter_A"], 130)
        self.assertEqual(result["confidence"], "high")


if __name__ == "__main__":
    unittest.main()

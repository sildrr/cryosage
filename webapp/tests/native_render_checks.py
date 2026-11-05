"""Run with CryoSPARC worker Python: python -m unittest webapp.tests.native_render_checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from webapp.render_showcase_assets import draw_shape, rasterize_surface


class NativeRenderTests(unittest.TestCase):
    def test_z_buffer_retains_front_surface(self):
        points = np.array([[1, 1, 0], [7, 1, 0], [1, 7, 0],
                           [1, 1, 1], [7, 1, 1], [1, 7, 1]], dtype=np.float32)
        faces = np.array([[3, 4, 5], [0, 1, 2]], dtype=np.int32)
        shades = np.array([.2, .2, .2, .8, .8, .8], dtype=np.float32)
        pixels = rasterize_surface(points, faces, shades, 9, 9)
        self.assertEqual(int(pixels[2, 2, 0]), 204)
        self.assertEqual(pixels[8, 8].tolist(), [244, 246, 238])

    def test_vertex_shading_is_interpolated(self):
        points = np.array([[1, 1, 0], [8, 1, 0], [1, 8, 0]], dtype=np.float32)
        pixels = rasterize_surface(points, np.array([[0, 1, 2]]), np.array([.1, .9, .1]), 10, 10)
        self.assertLess(int(pixels[2, 2, 0]), int(pixels[2, 5, 0]))

    def test_shape_preserves_native_data_and_uses_gray(self):
        z, y, x = np.mgrid[-1:1:32j, -1:1:32j, -1:1:32j]
        volume = np.exp(-8*(x*x+y*y+z*z)).astype(np.float32)
        original = volume.copy()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "shape.png"
            with patch("webapp.render_showcase_assets.read_mrc_volume", return_value=(volume, (1, 1, 1))):
                result = draw_shape("unused", target)
            np.testing.assert_array_equal(volume, original)
            self.assertTrue(result["native_voxels_preserved"])
            self.assertFalse(result["mesh_decimation"])
            self.assertEqual(result["gaussian_sigma_display_voxels"], 0)
            self.assertEqual(result["display_max_voxels"], 32)
            self.assertEqual(result["disconnected_component_min_fraction"], 0)
            self.assertAlmostEqual(result["isolevel"], float(original.mean()+3*original.std()))
            pixels = np.asarray(Image.open(target))
            gray = (pixels[:, :, 0] == pixels[:, :, 1]) & (pixels[:, :, 1] == pixels[:, :, 2])
            self.assertGreater(int(gray.sum()), 1000)


if __name__ == "__main__":
    unittest.main()

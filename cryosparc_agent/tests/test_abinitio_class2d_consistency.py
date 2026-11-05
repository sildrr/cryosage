import unittest

from abinitio_class2d_consistency import (
    _masked_ncc,
    _selected_class_ids,
    _translate_with_zeros,
    build_consistency_observer_messages,
)


class _Params:
    selected_templates = "2, 10, invalid, 2, 48"


class _Job:
    params = _Params()


class AbinitioClass2DConsistencyTests(unittest.TestCase):
    def test_selected_template_ids_are_zero_based_labels_without_duplicates(self):
        self.assertEqual(_selected_class_ids(_Job()), [2, 10, 48])

    def test_visual_observer_message_carries_the_sheet_and_strict_contract(self):
        messages = build_consistency_observer_messages({
            "source": {"abinitio_job_uid": "J396"},
            "comparison_protocol": {"abinitio_views": ["XY projection"]},
            "contact_sheet": {"data_url": "data:image/png;base64,AA=="},
        })
        self.assertEqual(messages[0]["role"], "system")
        content = messages[1]["content"]
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertIn("inconsistent", content[0]["text"])
        self.assertIn("Do not call a match solely", content[0]["text"])

    def test_visual_observer_requires_an_actual_image(self):
        with self.assertRaisesRegex(ValueError, "no visual contact sheet"):
            build_consistency_observer_messages({"contact_sheet": {}})

    def test_masked_ncc_and_bounded_translation_are_deterministic(self):
        import numpy as np

        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 1.0
        shifted = _translate_with_zeros(image, 1, -2)
        self.assertEqual(shifted[5, 2], 1.0)
        self.assertEqual(float(shifted.sum()), 1.0)
        mask = np.ones_like(image, dtype=bool)
        self.assertAlmostEqual(_masked_ncc(image, image, mask), 1.0)


if __name__ == "__main__":
    unittest.main()

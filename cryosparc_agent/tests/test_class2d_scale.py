import unittest
from types import SimpleNamespace

import numpy as np

from class2d_scale import (
    _histogram_peak,
    _find_ctf_exposures,
    _measure_radial_class,
    electron_wavelength_A,
    round_box_size_up,
)
from action_registry import _apply_scale_rollback_defaults
from job_specs import get_parameter_template


class Class2DScaleTests(unittest.TestCase):
    def test_radial_profile_reports_weighted_threshold_diameters(self):
        size = 128
        yy, xx = np.indices((size, size))
        radius = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2)
        image = 0.2 + 3.0 * np.exp(-(radius ** 2) / (2 * 12.0 ** 2))

        result = _measure_radial_class(7, image, pixel_size_A=2.0, particle_count=500)

        self.assertTrue(result["valid"])
        self.assertGreater(result["diameter_5pct_A"], result["diameter_10pct_A"])
        self.assertGreater(result["particle_count"], 0)

    def test_electron_wavelength_uses_relativistic_voltage(self):
        self.assertAlmostEqual(electron_wavelength_A(300), 0.0197, delta=0.0001)
        self.assertAlmostEqual(electron_wavelength_A(200), 0.0251, delta=0.0002)

    def test_ctf_fit_peak_uses_distribution_mode_not_workflow_target(self):
        values = [3.1, 3.12, 3.14, 3.18, 3.2, 3.22, 3.7, 3.8]
        peak = _histogram_peak(values, maximum_A=4.0, bin_width_A=0.1)
        self.assertAlmostEqual(peak, 3.15)

    def test_ctf_lineage_prefers_complete_patch_ctf_set_over_pilot_subset(self):
        fields = {"ctf/df1_A": [1.0], "ctf/df2_A": [1.0]}
        full = {"ctf/df1_A": [1.0] * 196, "ctf/df2_A": [1.0] * 196}

        class Job:
            def __init__(self, uid, job_type, inputs=None, outputs=None):
                self.uid = uid
                self.type = job_type
                self.inputs = inputs or {}
                self.outputs = outputs or {}

            def load_output(self, name):
                return self.outputs[name]

        def edge(parent_uid, output):
            return SimpleNamespace(connections=[SimpleNamespace(
                job_uid=parent_uid, output=output,
            )])

        jobs = {
            "J1": Job("J1", "class_2D_new", {"particles": edge("J2", "particles")}),
            "J2": Job("J2", "extract_micrographs_multi", {"micrographs": edge("J3", "micrographs")}),
            "J3": Job("J3", "blob_picker_gpu", {"micrographs": edge("J4", "exposures")}, {"micrographs": fields}),
            "J4": Job("J4", "patch_ctf_estimation_multi", outputs={"exposures": full}),
        }

        class Project:
            def find_job(self, uid):
                return jobs[uid]

        dataset, uid = _find_ctf_exposures(Project(), jobs["J1"])
        self.assertEqual(uid, "J4")
        self.assertEqual(len(dataset["ctf/df1_A"]), 196)

    def test_box_rounding_respects_new_720_pixel_cap(self):
        self.assertEqual(round_box_size_up(670), 672)
        self.assertEqual(round_box_size_up(721), 720)

    def test_scale_evidence_populates_rollback_defaults(self):
        evidence = {
            "source_class2d_job_uid": "J20",
            "aggregate": {
                "usable_for_picker": True,
                "particle_diameter_range_A": [158.4, 164.4],
            },
            "box_size_recommendation": {
                "usable_for_extraction": True,
                "recommended_box_size_pix": 672,
            },
        }
        actions = [
            {"rollback_kind": "repick_diameter", "default_parameters": {}},
            {
                "rollback_kind": "new_box_size",
                "default_parameters": {"box_size_pix": 416},
                "parameter_template": {"box_size_pix": {"type": "integer", "minimum": 32}},
            },
        ]
        _apply_scale_rollback_defaults(
            actions,
            evidence,
            {"key_parameters": {"box_size_pix": 384}},
        )

        self.assertEqual(actions[0]["default_parameters"], {"diameter": 158.4, "diameter_max": 164.4})
        self.assertEqual(actions[1]["default_parameters"]["box_size_pix"], 672)
        self.assertEqual(actions[1]["parameter_template"]["box_size_pix"]["maximum"], 720)

    def test_extract_parameter_schema_caps_at_720(self):
        spec = get_parameter_template("extract_micrographs_multi")["box_size_pix"]
        self.assertEqual(spec["maximum"], 720)


if __name__ == "__main__":
    unittest.main()

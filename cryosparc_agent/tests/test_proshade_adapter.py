import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import proshade_adapter
from proshade_adapter import download_volume_map, parse_proshade_output, run_proshade


SAMPLE_OUTPUT = """
ProSHADE default symmetry detection algorithm claims the symmetry to be C-12 with axes:
======================================================================================
   Type     Fold       X           Y          Z           Angle        Height      Average FSC
     C       +12      +0.00000   +0.00000   +1.00000     +0.52360      +0.99912      +0.99533
"""


class ProshadeAdapterTests(unittest.TestCase):
    def test_parses_recommended_symmetry_and_axis_metrics(self):
        parsed = parse_proshade_output(SAMPLE_OUTPUT)
        self.assertEqual(parsed["recommended_symmetry"], "C12")
        self.assertEqual(parsed["primary_axis"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(parsed["average_fsc"], 0.99533)

    @patch("proshade_adapter.subprocess.run")
    def test_wraps_successful_cli_result_as_advisory_evidence(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["proshade"], 0, stdout=SAMPLE_OUTPUT, stderr=""
        )
        result = run_proshade(Path("/tmp/map.mrc"), binary=Path("/tmp/proshade"))
        self.assertTrue(result["success"])
        self.assertEqual(result["recommended_symmetry"], "C12")
        self.assertEqual(result["confidence"], "high")
        self.assertIn("--symmetry", result["command"])
        self.assertIn("--symCentre", result["command"])

    @patch("proshade_adapter.subprocess.run")
    def test_center_search_can_be_disabled(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout=SAMPLE_OUTPUT, stderr="")
        result = run_proshade(Path("/tmp/map.mrc"), binary=Path("/tmp/proshade"),
                              find_symmetry_center=False)
        self.assertNotIn("--symCentre", result["command"])

    @patch("proshade_adapter.subprocess.run")
    def test_preserves_nonzero_exit_as_structured_failure(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["proshade"], 2, stdout="", stderr="invalid map"
        )
        result = run_proshade(Path("/tmp/map.mrc"), binary=Path("/tmp/proshade"))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "proshade_failed")


def test_repository_binary_is_available_without_user_install(monkeypatch, tmp_path):
    binary = tmp_path / "proshade"
    binary.write_text("executable fixture")
    binary.chmod(0o755)
    monkeypatch.delenv("CRYOAGENT_PROSHADE_BINARY", raising=False)
    monkeypatch.setattr(proshade_adapter, "BUNDLED_PROSHADE_BINARY", binary)
    monkeypatch.setattr(proshade_adapter, "DEFAULT_PROSHADE_BINARY", tmp_path / "missing")
    assert proshade_adapter.resolve_proshade_binary() == binary


def test_explicit_binary_override_is_not_silently_ignored(monkeypatch, tmp_path):
    override = tmp_path / "unavailable"
    monkeypatch.setenv("CRYOAGENT_PROSHADE_BINARY", str(override))
    assert proshade_adapter.resolve_proshade_binary() == override


def test_download_preserves_volume_voxel_size(tmp_path):
    from cryosparc.mrc import read
    volume = np.ones((4, 4, 4), dtype=np.float32)
    job = SimpleNamespace(download_mrc=lambda _: (SimpleNamespace(nx=4, xlen=2.55), volume))
    result = download_volume_map(job, "J1/volume.mrc", tmp_path, "P1", "J1")
    header, actual = read(result)
    assert abs(header.xlen / header.nx - 0.6375) < 1e-6
    np.testing.assert_array_equal(actual, volume)


def test_volume_source_uses_primary_map_field_not_first_intermediate_file():
    job = SimpleNamespace(load_output=lambda _: {"map/path": ["J1/J1_004_volume_map.mrc"]})
    assert proshade_adapter.find_volume_source(job) == "J1/J1_004_volume_map.mrc"


def test_volume_fallback_excludes_masks_and_uses_latest_iteration():
    def unavailable(_):
        raise ValueError("No output dataset")
    job = SimpleNamespace(load_output=unavailable, list_files=lambda: [
        "J1_000_volume_map.mrc", "J1_004_volume_map_sharp.mrc",
        "J1_004_volume_mask_fsc.mrc", "J1_004_volume_map.mrc"])
    assert proshade_adapter.find_volume_source(job) == "J1_004_volume_map.mrc"


def test_volume_cache_is_scoped_to_source_iteration(tmp_path):
    volume = np.ones((4, 4, 4), dtype=np.float32)
    job = SimpleNamespace(download_mrc=lambda _: (SimpleNamespace(nx=4, xlen=2.55), volume))
    first = download_volume_map(job, "J1/J1_000_volume_map.mrc", tmp_path, "P1", "J1")
    last = download_volume_map(job, "J1/J1_004_volume_map.mrc", tmp_path, "P1", "J1")
    assert first != last

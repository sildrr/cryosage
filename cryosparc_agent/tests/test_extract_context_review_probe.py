import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "scripts/probe_extract_context_review.py"
spec = importlib.util.spec_from_file_location("extract_context_review_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def dataset():
    return {
        "location/micrograph_uid": np.array([10, 10, 20]),
        "location/center_x_frac": np.array([0.3, 0.8, 0.3]),
        "location/center_y_frac": np.array([0.5, 0.2, 0.5]),
    }


def test_match_uses_micrograph_and_coordinate_not_stack_index():
    assert probe.match_coordinate(dataset(), 20, [0.3, 0.5]) == 2
    assert probe.match_coordinate(dataset(), 10, [0.8, 0.2]) == 1
    with pytest.raises(ValueError):
        probe.match_coordinate(dataset(), 30, [0.3, 0.5])
    with pytest.raises(ValueError):
        probe.match_coordinate(dataset(), 10, [0.31, 0.5])


def test_duplicate_centers_cannot_be_silently_mismatched():
    data = dataset()
    data["location/center_x_frac"][1] = 0.3
    data["location/center_y_frac"][1] = 0.5
    with pytest.raises(ValueError, match="unique"):
        probe.match_coordinate(data, 10, [0.3, 0.5])


def test_counts_use_square_bounds_and_exclude_target():
    result = probe.neighbor_geometry([[0, 0], [100, 100], [250, 0], [320, 0]], [0, 0], [640, 440])
    assert result["extra_picked_centers_by_box_px"] == {"640": 2, "440": 1}
    assert result["nearest_other_picked_center_distance_native_px"] == pytest.approx(np.sqrt(20000))


def test_historical_result_requires_verified_ancestry(tmp_path):
    path = tmp_path / "progress.json"
    nodes = [
        {"id": "E", "parents": [], "status": "completed", "parameters": {"box_size_pix": 440}},
        {"id": "S", "parents": ["E"], "status": "completed"},
        {"id": "R", "parents": ["S"], "status": "completed", "resolution_A": 2.99},
    ]
    path.write_text(json.dumps({"nodes": nodes}))
    result = probe.verified_history(path, "E", "R")
    assert result["box_size_pix"] == 440
    assert result["corrected_FSC_resolution_A"] == 2.99
    nodes[-1]["parents"] = []
    path.write_text(json.dumps({"nodes": nodes}))
    with pytest.raises(ValueError, match="descendant"):
        probe.verified_history(path, "E", "R")


def test_no_history_condition_does_not_leak_baseline():
    bundle = {
        "native_pixel_size_A": 0.6575, "box_sizes_px": [640, 440], "sampling": "fixed",
        "display": "full field", "samples": [{
            "sample_id": "S01", "extra_picked_centers_by_box_px": {"640": 6, "440": 2},
            "nearest_other_picked_center_distance_native_px": 150,
        }],
        "historical_baseline": {"extract_job": "J592", "corrected_FSC_resolution_A": 2.9865},
    }
    without = probe.decision_facts(bundle, False)
    assert "historical_baseline" not in without
    assert "J592" not in str(without)
    assert "historical_baseline" in probe.decision_facts(bundle, True)
    early = probe.decision_facts(bundle, False, comparison_outputs=False)
    assert "candidate geometric window" in early["statistics_limitations"][-1]
    assert "actual matched outputs" not in str(early)

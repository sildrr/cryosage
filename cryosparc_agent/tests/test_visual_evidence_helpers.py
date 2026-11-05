"""Deterministic checks for the image evidence shared by visual decision backends."""

import numpy as np
import base64
import io
from PIL import Image, ImageDraw
from pathlib import Path
from unittest.mock import patch
import tempfile
import pytest

from vision_inputs import (
    _crop_square,
    _crop_views,
    render_micrograph_display,
    render_full_micrograph_preview,
    _overview_tile,
    _overview_pick_position,
    _scale_bar,
    build_class2d_feedback_context,
    build_extract_box_visual_context,
    build_micrograph_preview_visual_context,
    build_pick_inspection_visual_context,
    build_pick_score_scatter,
    percentile_ranks,
    score_rank_color,
)


class _FakeJob:
    def __init__(self, outputs, params=None):
        self.outputs = outputs
        self.params = params or {}
        self.uid = "J1"

    def load_output(self, name, **_kwargs):
        if name not in self.outputs:
            raise KeyError(name)
        return self.outputs[name]


class _FakeProject:
    def __init__(self, job):
        self.job = job

    def find_job(self, _uid):
        return self.job


class _FakeClient:
    def __init__(self, project):
        self.project = project

    def find_project(self, _uid):
        return self.project


def _synthetic_outputs(include_power=True):
    micrographs = {
        "micrograph_blob/path": ["mic1.mrc", "mic2.mrc"],
        "micrograph_blob/psize_A": np.asarray([0.65, 0.65]),
        "micrograph_blob/is_background_subtracted": [False, False],
        "uid": np.asarray([11, 12]),
    }
    particles = {
        "location/micrograph_uid": np.asarray([11, 11, 12, 12]),
        "location/center_x_frac": np.asarray([0.25, 0.75, 0.3, 0.7]),
        "location/center_y_frac": np.asarray([0.25, 0.75, 0.3, 0.7]),
        "pick_stats/ncc_score": np.asarray([0.1, 0.2, 0.3, 0.4]),
    }
    if include_power:
        particles["pick_stats/power"] = np.asarray([1.0, 2.0, 3.0, 4.0])
    return micrographs, particles


def test_percentile_ranks_preserve_missing_values_and_order():
    ranks = percentile_ranks(np.asarray([10.0, np.nan, 20.0]))
    assert ranks[0] == 0.0
    assert np.isnan(ranks[1])
    assert ranks[2] == 1.0


def test_score_overlay_palette_is_monotonic_by_relative_rank():
    low = score_rank_color(0.1, 0.1)
    high = score_rank_color(0.9, 0.9)
    assert low[0] > high[0]
    assert low[1] < high[1]


def test_micrograph_crop_and_scale_bar_are_bounded_and_calibrated():
    image = Image.new("RGB", (100, 80), "gray")
    crop = _crop_square(image, 1, 1, 64)
    assert crop.size == (64, 64)
    metadata = _scale_bar(ImageDraw.Draw(crop), crop.size, 1.0)
    assert metadata is not None
    assert metadata["length_A"] == metadata["length_pixels"]


def test_score_plot_handles_missing_power_without_crashing():
    plot = build_pick_score_scatter(
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
    )
    assert plot.size == (720, 560)


def test_score_density_plot_contains_signal_for_overlapping_points():
    values = np.full(1000, 0.5, dtype=np.float32)
    plot = build_pick_score_scatter(values, values)
    pixels = np.asarray(plot)
    assert np.any(pixels[..., 0] > pixels[..., 1])


def test_blob_preview_uses_uncropped_individual_full_fields_and_scale():
    micrographs, _ = _synthetic_outputs()
    job = _FakeJob({"micrographs": micrographs})
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.random.default_rng(2).normal(size=(1024, 1024))):
        result = build_micrograph_preview_visual_context("P1", "J1", max_micrographs=2)
    assert result["pixel_size_A_median"] == 0.65
    assert len(result["evidence_images"]) == 2
    assert all(item["local_path"].endswith(".png") for item in result["evidence_images"])
    assert all(Image.open(io.BytesIO(base64.b64decode(item["data_url"].split(",", 1)[1]))).size == (1024, 1024)
               for item in result["evidence_images"])
    assert result["visual_evidence_version"] == "full_field_v4"
    assert all(item["role"] == "full_micrograph_with_scale" for item in result["evidence_images"])
    assert all(info["full_field_retained"] for info in result["image_calibrations"])


def test_inspect_context_has_full_fields_and_does_not_fake_missing_power():
    micrographs, particles = _synthetic_outputs(include_power=False)
    job = _FakeJob({"micrographs": micrographs, "particles": particles})
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.random.default_rng(3).normal(size=(1024, 1024))):
        result = build_pick_inspection_visual_context("P1", "J1", max_micrographs=2)
    assert result["power_score_available"] is False
    assert result["score_statistics"]["power"]["count"] == 0
    assert len(result["evidence_images"]) == 1
    assert result["evidence_images"][0]["role"] == "full_micrograph_pick_overlay"
    assert result["score_examples"] == []
    assert result["score_scatter"]["data_url"].startswith("data:image/png")


def test_inspect_score_examples_are_full_images_with_calibration():
    micrographs, particles = _synthetic_outputs()
    job = _FakeJob({"micrographs": micrographs, "particles": particles})
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.zeros((1024, 900), dtype=np.float32)):
        result = build_pick_inspection_visual_context("P1", "J1", max_micrographs=1)
    assert len(result["score_examples"]) == 4
    assert all(item["role"] == "score_stratified_full_micrograph" for item in result["evidence_images"])
    assert all(v["display_processing"]["full_field_retained"] for v in result["score_examples"])
    assert all(Image.open(io.BytesIO(base64.b64decode(item["data_url"].split(",", 1)[1]))).size == (1024, 1024)
               for item in result["evidence_images"])


def test_missing_calibration_and_missing_pick_scores_are_not_fabricated():
    micrographs, particles = _synthetic_outputs()
    micrographs.pop("micrograph_blob/psize_A")
    job = _FakeJob({"micrographs": micrographs, "particles": particles})
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.zeros((1024, 1024), dtype=np.float32)):
        with pytest.raises(ValueError, match="pixel size"):
            build_micrograph_preview_visual_context("P1", "J1", max_micrographs=1)
        particles.pop("pick_stats/ncc_score")
        with pytest.raises(ValueError, match="NCC"):
            build_extract_box_visual_context("P1", "J1")


def test_full_field_messages_attach_all_images_without_base64_in_text():
    from autonomous_mcp_closed_loop import build_autonomous_prompt
    from model_direct_runner import build_workflow_decision_prompt
    context = {"kind": "micrograph_preview", "visual_evidence_version": "full_field_v4",
               "image_calibrations": [{"full_field_retained": True}],
               "evidence_images": [{"data_url": "data:image/png;base64,A"}, {"data_url": "data:image/png;base64,B"}]}
    for messages in (build_autonomous_prompt({}, {}, 1, visual_context=context), build_workflow_decision_prompt({}, context)):
        contents = [part for message in messages if isinstance(message["content"], list) for part in message["content"]]
        images = [part for part in contents if part["type"] == "image_url"]
        assert len(images) == 2
        assert all(part["image_url"]["detail"] == "high" for part in images)
        assert all("data:image" not in part["text"] for part in contents if part["type"] == "text")


def test_extract_ranks_only_inspect_accepted_particles_when_bound():
    micrographs, particles = _synthetic_outputs()
    blob = _FakeJob({"micrographs": micrographs, "particles": particles})
    accepted = {key: np.asarray(value)[:2] for key, value in particles.items()}
    inspect = _FakeJob({"particles": accepted})
    project = _FakeProject(blob)
    project.find_job = lambda uid: inspect if uid == "J2" else blob
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(project)), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.zeros((2048, 2048), dtype=np.float32)):
        result = build_extract_box_visual_context("P1", "J1", particle_job_uid="J2")
    assert result["source"]["particle_job_uid"] == "J2"
    assert len(result["examples"]) == 2
    assert {item["particle_index"] for item in result["examples"]} == {0, 1}


def test_extract_context_renders_box_size_comparisons_and_picker_facts():
    micrographs, particles = _synthetic_outputs()
    params = type("Params", (), {"model_dump": lambda self: {"diameter": 150, "diameter_max": 220}})()
    job = _FakeJob({"micrographs": micrographs, "particles": particles}, params=params)
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.random.default_rng(4).normal(size=(2048, 2048))):
        result = build_extract_box_visual_context("P1", "J1", max_examples=2)
    assert result["upstream_blob_picker_parameters"] == {"diameter": 150, "diameter_max": 220}
    assert result["pixel_size_A_median"] == 0.65
    assert result["drawn_box_sizes_pix"]
    assert len(result["evidence_images"]) == 2
    assert {v["particle_index"] for v in result["examples"]} == {0, 1, 2, 3}
    assert result["visual_evidence_version"] == "full_field_v6_extract_stratified"
    assert len(result["examples"]) > len(result["evidence_images"])
    assert all(not box["crosses_image_edge"] for item in result["examples"] for box in item["candidate_boxes"])
    assert result["examples"][0]["scale_bar"]["length_A"] == (
        result["examples"][0]["scale_bar"]["length_pixels"] * result["examples"][0]["display_processing"]["display_pixel_size_x_A"])
    assert all(v["display_processing"]["full_field_retained"] for v in result["examples"])


def test_extract_samples_multiple_spaced_high_scores_and_excludes_edges(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_BOX_SIZE_TRIALS", "128,256,384")
    micrographs, _ = _synthetic_outputs()
    particles = {
        "location/micrograph_uid": np.full(6, 11),
        "location/center_x_frac": np.array([.01, .25, .255, .75, .25, .75]),
        "location/center_y_frac": np.array([.5, .25, .255, .25, .75, .75]),
        "pick_stats/ncc_score": np.array([10, 9, 8.5, 8, 7, 6]),
    }
    job = _FakeJob({"micrographs": micrographs, "particles": particles})
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.zeros((2048, 2048), dtype=np.float32)):
        result = build_extract_box_visual_context("P1", "J1", max_examples=1)
    assert len(result["evidence_images"]) == 1
    selected_indices = {item["particle_index"] for item in result["examples"]}
    assert {3, 4, 5} <= selected_indices
    assert len(selected_indices & {1, 2}) == 1
    assert result["sampling"]["excluded_edge_count"] == 1
    assert result["sampling"]["excluded_nearby_count"] == 1
    assert result["sampling"]["affects_job_particle_selection"] is False
    assert result["power_score_available"] is False
    assert all(item["power_score"] is None for item in result["examples"])
    assert all(item["score_stratum"] == "unstratified_fallback" for item in result["examples"])
    for item in result["examples"]:
        display = item["display_processing"]
        left, top = display["offset_pixels"]
        height, width = display["image_shape_pixels"]
        for box in item["candidate_boxes"]:
            x0, y0, x1, y1 = box["bounds_display_pixels"]
            assert left <= x0 < x1 <= left + width
            assert top <= y0 < y1 <= top + height
            assert box["crosses_image_edge"] is False


def test_extract_joint_score_quota_and_labeled_contrast(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_BOX_SIZE_TRIALS", "128")
    micrographs, _ = _synthetic_outputs()
    power = np.arange(20, dtype=float)
    power[[10, 17]] = power[[17, 10]]
    particles = {
        "location/micrograph_uid": np.full(20, 11),
        "location/center_x_frac": np.tile([.2, .4, .6, .8], 5),
        "location/center_y_frac": np.repeat([.2, .35, .5, .65, .8], 4),
        "pick_stats/ncc_score": np.arange(20, dtype=float), "pick_stats/power": power,
    }
    job = _FakeJob({"micrographs": micrographs, "particles": particles})
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.zeros((2048, 2048), dtype=np.float32)):
        result = build_extract_box_visual_context("P1", "J1", max_examples=1)
    assert [item["score_stratum"] for item in result["examples"]] == ["typical", "typical", "high_ncc_moderate_power", "contrast"]
    assert all(not item["sampling_fallback"] for item in result["examples"])
    for item in result["examples"][:2]:
        assert 25 <= item["ncc_percentile"] <= 80
        assert 25 <= item["power_percentile"] <= 75
    high = result["examples"][2]
    assert 75 <= high["ncc_percentile"] <= 95
    assert 20 <= high["power_percentile"] <= 80
    control = result["examples"][3]
    assert control["power_percentile"] <= 10 or control["power_percentile"] >= 90 or control["ncc_percentile"] <= 10 or control["ncc_percentile"] >= 98


def test_extract_strata_do_not_invent_populations_for_tied_scores():
    from vision_inputs import extract_score_strata, percentile_ranks
    ranks = percentile_ranks(np.ones(10))
    slots = list(extract_score_strata(list(range(10)), ranks, ranks))
    assert slots[0][1] and slots[1][1]
    assert not slots[2][1] and not slots[3][1]


def test_extract_does_not_fall_back_to_edge_examples(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_BOX_SIZE_TRIALS", "720")
    micrographs, particles = _synthetic_outputs()
    particles["location/center_x_frac"] = np.full(4, .01)
    job = _FakeJob({"micrographs": micrographs, "particles": particles})
    with tempfile.TemporaryDirectory() as cache_dir, patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))), patch(
        "vision_inputs.vision_cache_dir", return_value=Path(cache_dir)
    ), patch("vision_inputs.load_micrograph_image", return_value=np.zeros((1024, 1024), dtype=np.float32)):
        with pytest.raises(ValueError, match="No interior scored picks"):
            build_extract_box_visual_context("P1", "J1")


def test_binned_display_matches_block_sum_contrast_and_flip_without_mutation():
    raw = np.random.default_rng(5).normal(size=(32, 24)).astype(np.float32)
    copy = raw.copy()
    rendered, metadata = render_micrograph_display(raw, 8)
    binned = raw.reshape(32, 6, 4).sum(axis=-1).reshape(8, 4, 6).sum(axis=1)
    low, high = binned.mean() - 3 * binned.std(), binned.mean() + 3 * binned.std()
    expected = np.clip((binned - low) / (high - low) * 255, 0, 255).astype(np.uint8)[::-1]
    np.testing.assert_array_equal(np.asarray(rendered)[..., 0], expected)
    assert metadata["binfactor"] == 4
    assert rendered.size == (6, 8)
    np.testing.assert_array_equal(raw, copy)


def test_edge_crop_marker_uses_real_position_not_canvas_center():
    raw = np.zeros((1024, 1024), dtype=np.float32)
    views, cx, cy, scale, _ = _crop_views(raw, 10, 20, 640, 0.65)
    assert views[0].size == (512, 512)
    assert cx == 10 * scale and cy == (640 - 1 - 20) * scale


def test_non_square_overview_keeps_aspect_and_overlay_coordinate_transform():
    raw = np.zeros((100, 200), dtype=np.float32)
    tile, info = _overview_tile(raw, 100)
    assert tile.size == (100, 100)
    assert info["display_shape_pixels"] == [50, 100]
    assert info["offset_pixels"] == [0, 25]
    x, y = _overview_pick_position(0, 0, raw.shape, info)
    assert (x, y) == (0, 74)


def test_constant_preview_is_finite_neutral_gray():
    rendered, _ = render_micrograph_display(np.ones((12, 12), dtype=np.float32), 6)
    assert np.all(np.asarray(rendered) == 127)


def test_full_field_preview_retains_edges_and_calibrates_scale():
    raw = np.zeros((101, 203), dtype=np.float32)
    raw[0, -1] = 100
    preview, info = render_full_micrograph_preview(raw, 0.65, 128)
    assert preview.size == (128, 128)
    assert info["source_shape_pixels"] == [101, 203]
    assert info["full_field_retained"]
    assert info["image_shape_pixels"] == [64, 128]
    assert info["display_pixel_size_x_A"] == 0.65 * 203 / 128
    assert info["scale_bar"]["length_A"] == info["scale_bar"]["length_pixels"] * info["display_pixel_size_x_A"]
    assert np.asarray(preview)[32 + 63, -1, 0] == 255
    assert raw[0, -1] == 100


def test_class2d_feedback_reads_class_sizes_and_resolution_fields():
    class_averages = {"blob/res_A": np.asarray([3.1, 4.2, 5.0])}
    particles = {"alignments2D/class": np.asarray([0, 0, 1, 1, 1, 2, -1])}
    job = _FakeJob({"class_averages": class_averages, "particles": particles})
    with patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeProject(job))):
        context = build_class2d_feedback_context("P1", "J1")
    assert context["input_particle_count"] == 7
    assert [item["particle_count"] for item in context["classes"]] == [2, 3, 1]
    assert [item["resolution_A"] for item in context["classes"]] == [3.1, 4.2, 5.0]

import pytest

from job_result import summarize_outputs
from model_input_builder import build_output_groups
from movie_metadata import movie_metadata, movie_summary_values
from quality_policy import extract_quality_metrics


def output(summary):
    return {"summary_values": summary, "summary_keys": list(summary),
            "latest_summary_stat_keys": [], "type": "movie", "num_items": 3640,
            "available": True, "result_names": ["movie_blob", "mscope_params"]}


def test_movie_shape_and_format_survive_both_model_facing_paths():
    observed = output({"movie_blob/shape": [40, 3710, 3838], "movie_blob/format": "TIFF",
                       "movie_blob/is_gain_corrected": 1, "movie_blob/psize_A": 0.82,
                       "arbitrary_huge_array": list(range(10000))})
    node = {"job_type": "import_movies", "outputs": {"imported_movies": observed}}
    model_group = build_output_groups(node)["groups"][0]
    result_group = summarize_outputs(node)["imported_movies"]
    for group in (model_group, result_group):
        assert group["movie_metadata"]["frame_count"] == 40
        assert group["movie_metadata"]["shape_axes"] == ["frames", "height", "width"]
        assert group["movie_metadata"]["format"] == "TIFF"
        assert group["movie_metadata"]["is_gain_corrected"] == 1
        assert "arbitrary_huge_array" not in group["summary_values"]
    assert "frame_end is optional" in build_output_groups(node)["motion_correction_guidance"]
    metrics = extract_quality_metrics(node)
    assert metrics["movie_blob/shape"]["value"] == [40, 3710, 3838]
    assert metrics["movie_blob/format"]["value"] == "TIFF"


@pytest.mark.parametrize("shape", [None, [], [3710, 3838], [0, 3710, 3838],
                                   [True, 3710, 3838], [40.5, 3710, 3838],
                                   "40,3710,3838", [[40, 3710, 3838]]])
def test_missing_or_ambiguous_shape_does_not_invent_frame_count(shape):
    metadata = movie_metadata(output({"movie_blob/shape": shape}))
    assert "frame_count" not in metadata


def test_non_movie_output_is_unchanged():
    node = {"job_type": "class_2D_new", "outputs": {"particles": output({})}}
    assert "motion_correction_guidance" not in build_output_groups(node)
    assert "movie_metadata" not in build_output_groups(node)["groups"][0]
    assert "summary_values" not in summarize_outputs(node)["particles"]


def test_missing_movie_metadata_still_explains_optional_frame_end():
    node = {"job_type": "import_movies", "outputs": {"imported_movies": output({})}}
    assert "omit it" in build_output_groups(node)["motion_correction_guidance"]
    assert movie_summary_values(output({})) == {}

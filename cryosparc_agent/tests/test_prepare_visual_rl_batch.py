import struct
from pathlib import Path

from prepare_visual_rl_batch import build_dataset_record, prepare_movie_subset


def test_movie_dataset_record_requires_movie_stack_and_uses_approved_defaults(tmp_path):
    movie_dir = tmp_path / "raw"
    movie_dir.mkdir()
    for index in range(20):
        path = movie_dir / f"movie_{index:03d}.mrc"
        path.write_bytes(struct.pack("<4i", 64, 64, 12, 2))

    source = {
        "title": "test specimen",
        "emdb_ids": [],
        "input_image_sets": [{
            "name": "raw movies",
            "category": "micrographs - multiframe",
            "directory": "raw",
            "data_format": "MRC",
            "frames_per_image": 12,
            "num_images_or_tilt_series": 20,
            "pixel_width": 0.8,
        }],
        "experimental_parameters": {
            "pixel_size_candidates": [],
            "accelerating_voltage_candidates": [],
            "spherical_aberration_candidates": [],
            "total_exposure_dose_candidates": [{"value_e_per_A2": 40}],
        },
    }

    record = build_dataset_record(
        "99999",
        source,
        directory_override=str(movie_dir),
        media_override={"media_type": "movies", "image_set_name": "raw movies"},
    )

    assert record["media_type"] == "movies"
    assert record["source_file_count"] == 20
    assert record["sample_mrc_shape"] == [64, 64, 12]
    assert record["voltage_kV"] == 300
    assert record["cs_mm"] == 2.7
    assert record["parameter_defaults_used"] == {
        "accelerating_voltage_kV": True,
        "spherical_aberration_mm": True,
    }


def test_movie_subset_is_deterministic_and_uses_symlinks(tmp_path):
    sources = []
    for index in range(25):
        path = tmp_path / f"source_{index:02d}.mrc"
        nz = 12 if index < 23 else 13
        path.write_bytes(struct.pack("<4i", 64, 64, nz, 2))
        sources.append(str(path))
    record = {"movie_files": sources, "movie_suffix": ".mrc", "directory": str(tmp_path)}

    _, first = prepare_movie_subset(record, tmp_path / "sample_a", seed=17)
    _, repeated = prepare_movie_subset(record, tmp_path / "sample_b", seed=17)

    assert len(first) == 20
    assert [path.rsplit("/", 1)[-1] for path in first] == [path.rsplit("/", 1)[-1] for path in repeated]
    assert all((tmp_path / "sample_a" / path.rsplit("/", 1)[-1]).is_symlink() for path in first)
    assert all(int(Path(path).resolve().stem.split("_")[-1]) < 23 for path in first)

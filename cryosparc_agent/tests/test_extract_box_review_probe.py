import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "scripts/probe_extract_box_review.py"
spec = importlib.util.spec_from_file_location("extract_box_review_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("mode,dtype", [(2, "<f4"), (12, "<f2")])
def test_reads_only_requested_mrc_slice_with_extended_header(tmp_path, mode, dtype):
    data = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8).astype(dtype)
    header = np.zeros(256, dtype="<i4")
    header[:4] = [8, 8, 3, mode]
    header[23] = 128
    stack = tmp_path / "particles.mrc"
    stack.write_bytes(header.tobytes() + b"\0" * 128 + data.tobytes())
    np.testing.assert_array_equal(probe.read_slice(stack, 1), data[1].astype(np.float32))
    with pytest.raises(ValueError, match="Invalid particle stack index"):
        probe.read_slice(stack, 3)


def test_sheet_supports_raw_and_filtered_views_without_mutating_particles():
    rng = np.random.default_rng(42)
    images = [rng.normal(size=(64, 64)).astype(np.float32) for _ in range(12)]
    original = images[0].copy()
    for lowpass in (None, 8):
        sheet = probe.render_sheet(images, [f"S{i:02}" for i in range(12)], 0.6575, lowpass)
        assert sheet.size == (1280, 1140)
        assert sheet.mode == "RGB"
    np.testing.assert_array_equal(images[0], original)

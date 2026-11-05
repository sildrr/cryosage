import numpy as np
import pytest

from scripts.p33_particle_count_control import sample_indices


def test_sampling_deterministic_unique_and_half_stratified():
    split = np.array([0] * 600 + [1] * 400)
    first = sample_indices(split, 101, 42)
    assert np.array_equal(first, sample_indices(split, 101, 42))
    assert len(first) == len(np.unique(first)) == 101
    assert np.sum(split[first] == 0) == 61
    assert np.sum(split[first] == 1) == 40


def test_full_sampling_preserves_all_and_invalid_counts_fail():
    split = np.array([0, 1, 0, 1])
    assert np.array_equal(sample_indices(split, 4, 1), np.arange(4))
    for count in (0, 5):
        with pytest.raises(ValueError):
            sample_indices(split, count, 1)

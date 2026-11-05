import numpy as np
import pytest

from scripts.pilot_top_view_rescue import added_indices, CLASS_PARAMETERS


def test_only_new_uids_enter_delta_and_source_order_is_preserved():
    wide=np.array([2**63+7,2**63+2,2**63+9,2**63+3],dtype=np.uint64)
    old=wide[[3,1]]
    indices=added_indices(wide,old)
    assert indices.tolist()==[0,2]
    assert not np.isin(wide[indices],old).any()
    assert np.array_equal(wide[indices],wide[[0,2]])


def test_non_nested_or_duplicate_sources_cannot_silently_create_delta():
    for wide,old in [([1,2],[1,3]),([1,1,2],[1]),([1,2],[1,1])]:
        with pytest.raises(ValueError):
            added_indices(wide,old)
    assert len(added_indices([1,2],[2,1]))==0


def test_arms_use_matched_native_classification_settings():
    assert CLASS_PARAMETERS==dict(class2D_K=20,class2D_num_full_iter=4,random_seed=26026,compute_num_gpus=1)

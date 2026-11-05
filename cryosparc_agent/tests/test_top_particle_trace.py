import numpy as np
import pytest

from scripts.trace_pilot_top_particle_locations import canonical_micrograph, nearest_locations


def particles(paths,points,scale=1.,shape=(100,200)):
    return {'uid':np.arange(len(paths))+10,'location/micrograph_path':np.array(paths),
        'location/micrograph_shape':np.tile(shape,(len(paths),1)),
        'location/micrograph_psize_A':np.full(len(paths),scale),
        'location/center_x_frac':np.array([p[0]/shape[1] for p in points]),
        'location/center_y_frac':np.array([p[1]/shape[0] for p in points]),
        'alignments2D/class':np.full(len(paths),20)}


def test_cross_project_padded_uids_are_not_micrograph_identity():
    a='J2/motioncorrected/007361360744928835187_FoilHole_1_Data_2.mrc'
    b='J1250/motioncorrected/002783369966996347241_FoilHole_1_Data_2.mrc'
    assert canonical_micrograph(a)==canonical_micrograph(b)=='FoilHole_1_Data_2.mrc'
    assert canonical_micrograph(b.encode())==canonical_micrograph(a)
    assert canonical_micrograph('FoilHole_2_Data_2.mrc')!='FoilHole_1_Data_2.mrc'


def test_location_distance_axes_angstroms_overlap_and_ambiguity():
    h=particles(['mic.mrc','outside.mrc'],[(20,30),(20,30)],scale=2.)
    c=particles(['mic.mrc','mic.mrc'],[(23,34),(26,38)],scale=2.)
    rows=nearest_locations(h,c,{'micrograph_blob/path':['mic.mrc']},radii=(5.,15.,25.))
    assert rows[0]['distance_A']==pytest.approx(10.)
    assert not rows[0]['match_5A'] and rows[0]['match_15A']
    assert not rows[0]['ambiguous_15A'] and rows[0]['ambiguous_25A']
    assert rows[0]['nearest_raw_uid']==10
    assert not rows[1]['in_current_subset'] and 'nearest_raw_uid' not in rows[1]


def test_wrong_frame_or_ambiguous_filename_is_not_silently_matched():
    h=particles(['mic.mrc'],[(20,30)])
    c=particles(['mic.mrc'],[(20,30)],scale=2.)
    with pytest.raises(ValueError,match='frames differ'):
        nearest_locations(h,c,{'micrograph_blob/path':['mic.mrc']})
    with pytest.raises(ValueError,match='identity'):
        nearest_locations(h,h,{'micrograph_blob/path':['mic.mrc','mic.mrc']})

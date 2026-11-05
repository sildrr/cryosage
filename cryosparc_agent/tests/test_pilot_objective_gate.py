from dataclasses import replace

import numpy as np
import pytest

from pilot_objective_gate import GatePolicy, pairwise_shape_ncc, shape_groups, gate_from_matrix, evaluate_gate


def image():
    y,x=np.indices((64,64))
    return (np.exp(-((x-23)**2+(y-23)**2)/25)
            +.7*np.exp(-((x-41)**2+(y-25)**2)/12)
            +.4*np.exp(-((x-29)**2+(y-43)**2)/18)).astype(np.float32)


def test_inplane_rotation_translation_mirror_and_contrast_do_not_create_views():
    a=image()
    variants=[a,np.roll(np.rot90(a),3,axis=0),-np.fliplr(a)]
    matrix=pairwise_shape_ncc(variants)
    assert np.min(matrix)>.93
    assert len(shape_groups(matrix,.85))==1


def test_flat_and_nonfinite_images_are_not_shape_evidence():
    with pytest.raises(ValueError,match='Flat'):
        pairwise_shape_ncc([np.zeros((64,64))])
    with pytest.raises(ValueError):
        pairwise_shape_ncc([np.full((64,64),np.nan)])


def test_complete_link_does_not_merge_via_similarity_bridge():
    m=np.array([[1,.92,.5],[.92,1,.91],[.5,.91,1]])
    assert shape_groups(m,.85)==[[0,1],[2]]


def test_gate_requires_quality_count_and_supported_groups_together():
    ids=[0,1,2]
    counts={0:1000,1:600,2:400}
    scores={0:2.2,1:2.5,2:2.9}
    gate=gate_from_matrix(ids,counts,scores,np.eye(3))
    assert gate['passed'] and gate['qualified_particles']==2000 and gate['supported_group_count']==3
    duplicate=gate_from_matrix(ids,counts,scores,np.ones((3,3)))
    assert not duplicate['passed'] and duplicate['supported_group_count']==1
    tiny=gate_from_matrix(ids,{0:1000,1:950,2:50},scores,np.eye(3))
    assert not tiny['passed'] and tiny['supported_group_count']==2
    few=gate_from_matrix(ids,{0:900,1:500,2:400},scores,np.eye(3))
    assert not few['passed'] and 'insufficient_cryosift_qualified_particles' in few['failed_conditions']


def context(scores=(2.2,2.5,2.8)):
    return dict(source=dict(project_uid='P1',job_uid='J1'),
        class_statistics=[dict(class_id=i,particle_count=n) for i,n in enumerate((1000,600,400))],
        cryosift=dict(success=True,project_uid='P1',job_uid='J1',class_scores=[dict(class_id=i,score=s) for i,s in enumerate(scores)]))


def test_no_cryosift_qualified_class_fails_without_a_model_vote():
    c=context((3.,4.,5.))
    c['quality']='usable'
    g=evaluate_gate(c,{})
    assert g['available'] and not g['passed']
    assert g['qualified_particles']==0
    assert 'no_classes_below_cryosift_threshold' in g['failed_conditions']


def test_wrong_source_missing_scores_and_images_never_pass():
    c=context()
    c['cryosift']['job_uid']='J2'
    assert not evaluate_gate(c,{})['available']
    c=context()
    assert not evaluate_gate(c,{})['available']


def test_nonfinite_score_cannot_enter_qualified_population():
    c=context((float('nan'),4.,5.))
    g=evaluate_gate(c,{})
    assert g['qualified_particles']==0 and not g['passed']


def test_invalid_policy_and_malformed_matrix_rejected():
    with pytest.raises(ValueError):
        GatePolicy(duplicate_ncc=float('nan'))
    with pytest.raises(ValueError):
        GatePolicy(min_particles=True)
    with pytest.raises(ValueError):
        GatePolicy(rotation_step_degrees=17)
    with pytest.raises(ValueError):
        shape_groups(np.array([[1,.3],[.8,1]]),.85)

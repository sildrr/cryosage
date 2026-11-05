import json
from types import SimpleNamespace

import numpy as np
import pytest
from cryosparc.dataset import Dataset

from coverage_rollout import heldout_indices,class_pool,validate_selection,union_from_source,CoverageRollout,external_slots
from coverage_context import planner_context


def context():
    return dict(source=dict(project_uid='P6',job_uid='J1'),
        class_statistics=[dict(class_id=i,particle_count=n) for i,n in enumerate([1500,1500,400,500,200])],
        cryosift=dict(success=True,project_uid='P6',job_uid='J1',class_scores=[
            dict(class_id=i,score=s) for i,s in enumerate([2.5,2.8,3.,4.035,4.6])]))


def test_independent_confirmation_cannot_include_pilot_micrographs():
    ids=np.array([2**63+i for i in range(10)],dtype=np.uint64)
    indices=heldout_indices(ids,ids[[0,2]],5)
    assert indices.tolist()==[1,3,4,5,6]
    with pytest.raises(ValueError): heldout_indices(ids,ids[:9],5)
    with pytest.raises(ValueError): heldout_indices(ids,np.array([0],dtype=np.uint64),5)


def test_weak_view_pool_is_not_discarded_or_declared_core():
    c=context()
    assert class_pool(c,'core')==[0,1]
    assert class_pool(c,'recovery')==[2,3]
    assert class_pool(c,'recovered')==[0,1,2,3]
    assert class_pool(c,'recovery',excluded=[2])==[3]
    c['cryosift']['job_uid']='OTHER'
    with pytest.raises(ValueError,match='source|Source'): class_pool(c,'core')


def test_no_strict_classes_can_still_have_executable_diagnostic_core():
    c=context()
    for r in c['cryosift']['class_scores']: r['score']+=2.
    assert class_pool(c,'core')
    assert class_pool(c,'recovery')==[0]
    assert validate_selection(dict(selected_class_ids=[3],reason='novel supported shape'),{2,3})
    with pytest.raises(ValueError): validate_selection(dict(selected_class_ids=[99],reason='bad source'),{2,3})


def test_union_preserves_core_and_unchanged_original_metadata():
    source=Dataset(5).add_fields([('marker',np.int64)])
    source['uid']=np.arange(5,dtype=np.uint64)+100
    source['marker']=np.arange(5)+42
    a=source.take([0,2]); b=source.take([2,4])
    output=union_from_source(source,[a,b])
    assert output['uid'].tolist()==[100,102,104]
    assert output['marker'].tolist()==[42,44,46]
    bad=source.take([1]);bad['uid'][0]=999
    with pytest.raises(ValueError): union_from_source(source,[a,bad])


def test_reference_uses_actual_average_sampling_not_native_extraction_box(monkeypatch):
    import refinement_optimization
    monkeypatch.setattr(refinement_optimization,'reference_sampling',lambda *a:128)
    instance=CoverageRollout.__new__(CoverageRollout)
    instance.project=SimpleNamespace(uid='P6')
    instance.service=lambda name,f:f()
    calls=[]
    def submit(slug,kind,sources,params):
        calls.append((kind,params));return SimpleNamespace(uid='J9')
    instance.submit=submit
    instance.reference('test','J1','J2')
    assert calls[0][1]['vol_resample_N']==128
    assert calls[0][1]['vol_lpf_A']==20.


def test_context_is_opt_in_and_workspace_scoped(tmp_path,monkeypatch):
    monkeypatch.delenv('CRYOAGENT_COVERAGE_CONTEXT',raising=False)
    assert planner_context({}) is None
    path=tmp_path/'context.json'
    value=dict(project_uid='P6',workspace_uid='Wnew',guidance='test')
    path.write_text(json.dumps(value))
    monkeypatch.setenv('CRYOAGENT_COVERAGE_CONTEXT',str(path))
    assert planner_context(dict(project_uid='P6',workspace_uid='Wnew'))==value
    assert planner_context(dict(project_uid='P6',workspace_uid='W141')) is None


def test_external_slot_names_are_not_misregistered_as_dtypes():
    source=dict(slots=[dict(name='blob',dtype='blob'),dict(name='blob_selected',dtype='blob')],
                results=[dict(name='micrograph_blob',dtype='blob')])
    assert external_slots(source,{'blob'})==[dict(name='blob',dtype='blob',required=True)]
    assert external_slots(source,{'blob_selected'})==[dict(name='blob_selected',dtype='blob',required=True)]
    assert external_slots(source,{'micrograph_blob'})==[dict(name='micrograph_blob',dtype='blob',required=True)]
    with pytest.raises(ValueError,match='authoritative'):
        external_slots(source,{'unknown'})

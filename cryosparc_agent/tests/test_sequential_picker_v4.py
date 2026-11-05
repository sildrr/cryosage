import json
from types import SimpleNamespace

import pytest

from sequential_picker_v4 import (SequentialPickerV4, signal_action,
                                   validate_picker_scale, supported_symmetry)
from test_sequential_picker_rollout import gate, thresholds, pilot_instance, serial_instance


def context(scores):
    return dict(source=dict(project_uid='P6', job_uid='J1'),
        cryosift=dict(success=True, project_uid='P6', job_uid='J1',
            class_scores=[dict(class_id=i, score=s) for i,(s,n) in enumerate(scores)]),
        class_statistics=[dict(class_id=i, particle_count=n) for i,(s,n) in enumerate(scores)])


@pytest.mark.parametrize('scores,used,expected', [
    ([(2.8,300)],False,'expand'), ([(2.8,50)],False,'adjust_parameters'),
    ([(3.3,600)],False,'expand_near'), ([(3.3,600)],True,'adjust_parameters'),
    ([(4.2,5000)],False,'adjust_parameters'), ([(3.4,50)]*12,False,'adjust_parameters')])
def test_supported_signal_controls_expansion(scores, used, expected):
    assert signal_action(context(scores), gate(1000,1), 50,200,used)[0] == expected


def test_support_heuristics_never_override_strict_promotion():
    assert signal_action(context([(3.3,5000)]),gate(0,0),200,200)[0]=='adjust_parameters'
    assert signal_action({},gate(2100,1),50,200)[0]=='core_and_recovery'
    with pytest.raises(ValueError,match='Source-scoped'):
        signal_action({},gate(0,0),50,200)


def test_larger_extent_requires_explanation_not_a_diameter_hard_cap():
    result=dict(scale_estimate=dict(diameter_max_A=250))
    with pytest.raises(ValueError,match='upstream Blob'):
        validate_picker_scale(result,dict(diameter_max=220))
    result['picker_bound_revision_reason']='Calibrated full-field extent; uncertain, review picker bound'
    assert validate_picker_scale(result,dict(diameter_max=220)) is result
    assert validate_picker_scale(dict(scale_estimate=dict(diameter_max_A=None)),dict(diameter_max=220))


@pytest.mark.parametrize('target,score,success,expected',[
    ('C2',.8,True,True),('D2',.9,True,True),('C1',.9,True,False),
    ('C2',.7,True,False),('C2',.9,False,False),('D0',.9,True,False)])
def test_symmetry_is_an_evidence_backed_hypothesis(target,score,success,expected):
    assert supported_symmetry(dict(success=success,recommended_symmetry=target,average_fsc=score)) is expected


def new_instance(tmp_path):
    e=serial_instance(tmp_path)
    e.__class__=SequentialPickerV4
    return e


def recipe():
    return dict(executable=True,count=200,box_size_pix=440,thresholds=thresholds(),
        classes='blob_classes',evidence_slug='blob',promotion_gate=gate(),bank_job='bank',best_refine='homo')


def test_same_pilot_template_comparison_reuses_blob_and_persists_resume(tmp_path):
    e=new_instance(tmp_path)
    e.state['steps']['sample_n200']=dict(job_uid='original_subset')
    e.project=SimpleNamespace(find_job=lambda uid:SimpleNamespace(uid=uid))
    calls=[]
    def route(tag,source,count,kind,bank,**kwargs):
        calls.append((source,count,kind,kwargs))
        return dict(kind=kind,executable=False,reason='measured no good classes')
    e.route=route
    assert e.confirm(recipe())['kind']=='blob'
    assert e.confirm(recipe())['kind']=='blob'
    assert len(calls)==1
    assert calls[0][0]==('original_subset','exposures')
    assert calls[0][1:3]==(200,'template')
    assert calls[0][3]==dict(thresholds=None,review=True,box=440)
    assert 'confirmations' not in e.state


def test_optional_template_review_failure_falls_back_but_transport_does_not(tmp_path):
    from sequential_picker_rollout import ProtocolExhausted
    e=new_instance(tmp_path)
    e.state['steps']['sample_n200']=dict(job_uid='subset')
    e.project=SimpleNamespace(find_job=lambda uid:SimpleNamespace(uid=uid))
    def failed(*args,**kwargs):raise ProtocolExhausted('Inspect review exhausted')
    e.route=failed
    assert e.confirm(recipe())['kind']=='blob'
    e.state['pilot_route_comparison'].pop('template')
    def transport(*args,**kwargs):raise RuntimeError('API unavailable')
    e.route=transport
    with pytest.raises(RuntimeError,match='API unavailable'):e.confirm(recipe())
    def budget(*args,**kwargs):raise ProtocolExhausted('Structured phase Job budget exhausted')
    e.route=budget
    with pytest.raises(ProtocolExhausted,match='Job budget'):e.confirm(recipe())


def test_early_no_signal_diagnoses_same_sample_not_blind_expansion(tmp_path,monkeypatch):
    e=pilot_instance(tmp_path,monkeypatch,lambda tag,count:(1000,1))
    e.__class__=SequentialPickerV4
    e.manifest['max_adjustment_rounds']=2
    p=tmp_path/'evidence/new/context.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(context([(4.2,5000)])))
    from sequential_picker_rollout import ProtocolExhausted
    with pytest.raises(ProtocolExhausted):e.pilot()
    assert [c[1] for c in e.calls if c[0]=='sample']==[50,50]
    assert [c[1] for c in e.calls if c[0]=='diagnose']==[50]


def test_symmetry_compares_same_initial_inputs_and_preserves_c1(tmp_path,monkeypatch):
    import proshade_adapter
    e=new_instance(tmp_path)
    e.manifest['pilot_core_refinement']='homo_only'
    e.project=SimpleNamespace(uid='P6')
    calls=[]
    def submit(slug,kind,sources,params):
        calls.append((slug,sources,params))
        return SimpleNamespace(uid=slug)
    e.submit=submit
    e.resolution=lambda j:4.0 if j.uid.endswith('_homo') else 3.8
    monkeypatch.setattr(proshade_adapter,'analyze_volume_symmetry',lambda *a,**k:
        dict(success=True,recommended_symmetry='D2',average_fsc=.9))
    e.model=lambda slug,msg,model,validate:validate(dict(compare=True,reason='supported hypothesis'))
    result=e.core_refinements('pilot',SimpleNamespace(uid='core'),SimpleNamespace(uid='initial'))
    assert len(result)==2
    assert calls[0][1]==calls[1][1]
    assert calls[0][2]['refine_symmetry']=='C1'
    assert calls[1][2]['refine_symmetry']=='D2'
    assert e.state['symmetry_by_refine']['pilot_symmetry_compare']=='D2'


def test_early_adjustment_can_expand_with_accepted_new_parameters(tmp_path,monkeypatch):
    e=pilot_instance(tmp_path,monkeypatch,lambda tag,count:(1000,1) if tag.startswith('pilot_r0') else
                     ((1000,1) if count==50 else (2100,3)))
    e.__class__=SequentialPickerV4
    e.manifest['max_adjustment_rounds']=2
    p=tmp_path/'evidence/new/context.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(context([(4.2,3000)])))
    diagnose=e.request_adjustment
    def adjusted_signal(*args):
        p.write_text(json.dumps(context([(2.8,300)])))
        return diagnose(*args)
    e.request_adjustment=adjusted_signal
    # No signal at 50: adjust there, then expand with the accepted new recipe.
    result=e.pilot()
    assert result['count']==100
    assert e.pilot() is result
    assert [c[1] for c in e.calls if c[0]=='sample']==[50,50,100]


def test_full_refinement_receives_promoted_symmetry_only(tmp_path,monkeypatch):
    from sequential_picker_rollout import SequentialPickerRollout
    e=new_instance(tmp_path)
    e.state['promoted_symmetry']='D2'
    calls=[]
    monkeypatch.setattr(SequentialPickerRollout,'submit',lambda self,*a:calls.append(a))
    e.submit('full_core_homo','homo_refine_new',{},dict(refine_symmetry='C1'))
    e.submit('pilot_homo','homo_refine_new',{},dict(refine_symmetry='C1'))
    assert calls[0][3]['refine_symmetry']=='D2'
    assert calls[1][3]['refine_symmetry']=='C1'

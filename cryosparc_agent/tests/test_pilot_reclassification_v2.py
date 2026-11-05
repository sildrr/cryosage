from types import SimpleNamespace

import pytest

from pilot_reclassification_v2 import next_step, selected_ids, validate_classification, diagnostic_fallback, ObjectivePilotExperiment
from pilot_objective_gate import GatePolicy


def test_model_vote_cannot_authorize_or_veto_promotion():
    assert next_step(dict(passed=True,available=True,quality='poor'),20)=='pilot_usable_for_promotion'
    assert next_step(dict(passed=False,available=True,quality='usable'),20)=='expand'
    assert next_step(dict(passed=False,available=True),200)=='pilot_exhausted_no_validated_route'
    assert 'quality' not in selected_ids(dict(selected_class_ids=[1],reason='signal',quality='usable'),{1})


def test_feedback_keeps_failure_metrics_without_resending_large_matrices():
    instance=ObjectivePilotExperiment.__new__(ObjectivePilotExperiment)
    instance.state=dict(trials=[dict(micrographs=20,finished=True,objective_gate=dict(
        passed=False,failed_conditions=['insufficient_supported_2d_shape_groups'],
        pairwise_shape_ncc=[[1]],similarity_class_ids=[0],policy={}))])
    feedback=instance.feedback()[0]['objective_gate']
    assert feedback['failed_conditions']
    assert 'pairwise_shape_ncc' not in feedback
    assert 'pairwise_shape_ncc' in instance.state['trials'][0]['objective_gate']


def test_classification_params_can_change_but_resources_and_invalid_windows_cannot():
    assert validate_classification(dict(parameters=dict(class2D_K=30,class2D_num_full_iter=3)),5000,320)
    for params in (dict(class2D_K=0),dict(class2D_K=30,compute_num_gpus=8),
                   dict(class2D_K=30,class2D_window_outer_A=400),
                   dict(class2D_K=30,class2D_window_inner_A=250,class2D_window_outer_A=200)):
        with pytest.raises(ValueError):
            validate_classification(dict(parameters=params),5000,320)


def test_empty_model_subset_can_use_score_ordered_diagnostic_not_promotion():
    c=dict(source=dict(project_uid='P1',job_uid='J1'),class_ids=[0,1,2],
        class_statistics=[dict(class_id=i,particle_count=1000) for i in range(3)],
        cryosift=dict(success=True,project_uid='P1',job_uid='J1',class_scores=[
            dict(class_id=0,score=5.),dict(class_id=1,score=4.),dict(class_id=2,score=3.2)]))
    assert diagnostic_fallback(c)==[2,1,0]
    c['cryosift']['project_uid']='Pother'
    assert diagnostic_fallback(c)==[]


def test_v2_rejects_legacy_protocol_before_connecting(tmp_path):
    import json
    (tmp_path/'manifest.json').write_text(json.dumps(dict(project='P6',workspace='W139')))
    with pytest.raises(ValueError,match='V2 requires'):
        ObjectivePilotExperiment(tmp_path)


def test_each_trial_redecides_all_parameters_and_uses_objective_result(monkeypatch,tmp_path):
    import pilot_reclassification_v2 as v2
    import vision_inputs
    import webapp.usage
    monkeypatch.setattr(vision_inputs,'build_micrograph_preview_visual_context',lambda *a,**k:{})
    monkeypatch.setattr(vision_inputs,'build_pick_inspection_visual_context',lambda *a,**k:{})
    monkeypatch.setattr(vision_inputs,'build_extract_box_visual_context',lambda *a,**k:{})
    monkeypatch.setattr(webapp.usage,'task_usage',lambda root:{'total_tokens':0})
    monkeypatch.setattr(v2,'load_class_images',lambda *a:{})
    gates=[]
    def evaluate(*args):
        gates.append(True)
        return dict(available=True,passed=len(gates)==2,qualified_particles=2500,promotion_class_ids=[0,1,2])
    monkeypatch.setattr(v2,'evaluate_gate',evaluate)
    instance=ObjectivePilotExperiment.__new__(ObjectivePilotExperiment)
    instance.root=tmp_path
    instance.manifest=dict(gate_policy={})
    instance.policy=GatePolicy()
    instance.project=SimpleNamespace(uid='P6')
    instance.state=dict(status='running',steps={},trials=[])
    instance.service=lambda name,f:f()
    instance.persist=lambda:None
    instance.report=lambda:None
    instance.write=lambda *args:None
    calls=[]
    def stage(slug,kind,visual):
        calls.append((slug,kind))
        n=int(slug.split('_')[0][1:])
        return {'blob_picker_diameter':dict(diameter=n,diameter_max=n+100),
                'inspect_picks_threshold':dict(ncc_score_thresh=n/1000,lpower_thresh_min=500,lpower_thresh_max=1500),
                'extract_box_size':dict(box_size_pix=384 if n==20 else 440)}[kind]
    instance.stage=stage
    def submit(slug,kind,sources,params):
        return SimpleNamespace(uid=slug,outputs={key:SimpleNamespace(num_items=5000)
            for key in ('particles','particles_selected')})
    instance.submit=submit
    instance.classes=lambda slug,job:dict(score_summary={},class_ids=[0,1,2])
    instance.choose=lambda *args:dict(selected_class_ids=[0,1,2],reason='diagnosis',quality='poor')
    instance.classify_parameters=lambda *args,**kwargs:dict(class2D_K=10)
    instance.run()
    assert len(calls)==6 and len(gates)==2
    a,b=instance.state['trials']
    assert a['parameters']['picker']!=b['parameters']['picker']
    assert a['parameters']['inspect']!=b['parameters']['inspect']
    assert a['parameters']['extract']!=b['parameters']['extract']
    assert instance.state['status']=='pilot_usable_for_promotion'


def test_shape_choices_and_resource_guards_are_enforced(monkeypatch,tmp_path):
    from pilot_reclassification_experiment import PilotExperiment
    response={}
    captured=[]
    def model(self,slug,messages,name,validate):
        captured.append(messages[-1]['content'])
        return validate(response)
    monkeypatch.setattr(PilotExperiment,'model',model)
    instance=ObjectivePilotExperiment.__new__(ObjectivePilotExperiment)
    response.update(parameters=dict(diameter=100,diameter_max=200,use_circle=False,use_ellipse=True,use_ring=False,min_distance=.7))
    assert instance.model('n20_picker',[],'cheap',lambda d:d)['parameters']['use_ellipse']
    assert 'NOT frozen' in captured[-1]
    response['parameters']['compute_num_gpus']=8
    with pytest.raises(ValueError,match='advertised'):
        instance.model('n50_picker',[],'cheap',lambda d:d)
    response['parameters']=dict(box_size_pix=440,bin_size_pix=220)
    with pytest.raises(ValueError,match='advertised'):
        instance.model('n50_extract',[],'cheap',lambda d:d)

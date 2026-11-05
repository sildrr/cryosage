from types import SimpleNamespace

import numpy as np
import pytest

from pilot_objective_gate import GatePolicy
from coverage_rollout import CoverageRollout
from sequential_picker_rollout import (SequentialPickerRollout, ProtocolExhausted,
    inspect_parameters, validate_visual_review, qualified_core, promotion_pass,
    next_pilot_step, combine_gate, validate_adjustment)


def thresholds(ncc=.2):
    return inspect_parameters(dict(ncc_score_thresh=ncc,lpower_thresh_min=500,lpower_thresh_max=1500))


def gate(n=2001, groups=3):
    return dict(available=True,qualified_particles=n,supported_group_count=groups,promotion_class_ids=[0,1,2])


@pytest.mark.parametrize('n,groups,core,promotion',[(2000,3,False,False),(2001,2,True,False),
    (2001,3,True,True),(1999,5,False,False)])
def test_strict_floor_is_not_total_yield_or_model_vote(n,groups,core,promotion):
    g=gate(n,groups)
    g.update(total_particles=1000000,particle_weighted_mean=1.,model_usable=True)
    assert qualified_core(g) is core
    assert promotion_pass(g) is promotion


def test_missing_evidence_does_not_become_low_quality_measurement():
    with pytest.raises(ValueError): qualified_core(dict(available=False))


def test_expand_before_changing_thresholds():
    assert next_pilot_step(gate(1999),50,200)=='expand'
    assert next_pilot_step(gate(1999),200,200)=='adjust_parameters'
    assert next_pilot_step(gate(2500,1),50,200)=='core_and_recovery'


def test_review_revisions_must_be_new_testable_parameters():
    original=thresholds()
    assert validate_visual_review(dict(action='keep',reason='observed true picks'), 'inspect',original,[])
    with pytest.raises(ValueError):
        validate_visual_review(dict(action='revise',reason='unchanged',parameters=original),'inspect',original,[])
    with pytest.raises(ValueError):
        validate_visual_review(dict(action='revise',reason='cycle',parameters=thresholds(.3)),
            'inspect',original,[thresholds(.3)])
    with pytest.raises(ValueError):
        validate_visual_review(dict(action='revise',reason='crop',parameters=dict(box_size_pix=384,bin_size_pix=128)),
            'extract',dict(box_size_pix=320),[])


def test_inspect_requires_power_upper_and_no_calibration():
    with pytest.raises(ValueError): inspect_parameters(dict(ncc_score_thresh=.2,lpower_thresh_min=500))
    with pytest.raises(ValueError): inspect_parameters(dict(ncc_score_thresh=.2,lpower_thresh_min=500,
        lpower_thresh_max=1500,calibrate_pow=True))
    assert thresholds()['do_auto_cluster'] is False


def test_diversity_recovery_preserves_source_class_namespaces():
    entries=[]
    rng=np.random.default_rng(17)
    for job,score in [('J1',2.5),('J2',2.6),('J3',4.0)]:
        c=dict(source=dict(project_uid='P6',job_uid=job),
            class_statistics=[dict(class_id=0,particle_count=1200)],
            cryosift=dict(success=True,project_uid='P6',job_uid=job,class_scores=[dict(class_id=0,score=score)]))
        entries.append((c,{0:rng.normal(size=(64,64))},[0]))
    g=combine_gate(entries,GatePolicy(min_particles=2001))
    assert g['qualified_particles']==2400  # A weak accepted view does NOT count as <3 quality support.
    assert g['supported_group_count']==2
    assert len(g['class_provenance'])==3
    assert [r['source']['job_uid'] for r in g['class_provenance']]==['J1','J2','J3']
    assert not promotion_pass(g)


def serial_instance(tmp_path):
    e=SequentialPickerRollout.__new__(SequentialPickerRollout)
    e.root=tmp_path
    e.project=SimpleNamespace(uid='P6')
    e.state=dict(steps={})
    e.manifest=dict(max_stage_reviews=3,max_inspect_rounds=2,max_seed_jobs=80)
    e.service=lambda name,f:f()
    e.write=lambda *a:None
    e.persist=lambda:None
    return e


@pytest.mark.parametrize('mode,expected', [(None, ['homo_refine_new', 'nonuniform_refine_new']),
                                         ('homo_only', ['homo_refine_new'])])
def test_pilot_core_refinement_mode_is_opt_in(tmp_path, mode, expected):
    e = serial_instance(tmp_path)
    if mode is not None:
        e.manifest['pilot_core_refinement'] = mode
    calls = []
    def submit(slug, kind, sources, params):
        calls.append((slug, kind, sources, params))
        return SimpleNamespace(uid='J' + str(len(calls)))
    e.submit = submit
    jobs = e.core_refinements('pilot', SimpleNamespace(uid='core'), SimpleNamespace(uid='abinit'))
    assert [c[1] for c in calls] == expected
    assert calls[0][2] == dict(particles=('core', 'particles_selected'), volume=('abinit', 'volume_class_0'))
    assert jobs[-1].uid == ('J1' if mode == 'homo_only' else 'J2')
    if len(calls) == 2:
        assert calls[1][2] == dict(particles=('J1', 'particles'), volume=('J1', 'volume'))


def test_invalid_core_refinement_mode_cannot_submit(tmp_path):
    e = serial_instance(tmp_path)
    e.manifest['pilot_core_refinement'] = 'unknown'
    e.submit = lambda *args: pytest.fail('Invalid mode must not create a native job')
    with pytest.raises(ValueError, match='Unsupported pilot_core_refinement'):
        e.core_refinements('pilot', SimpleNamespace(uid='core'), SimpleNamespace(uid='abinit'))


def test_inspect_is_execute_then_review_then_execute_not_a_grid(tmp_path,monkeypatch):
    import inspect_review
    e=serial_instance(tmp_path)
    events=[]
    monkeypatch.setattr(inspect_review,'build_visual_context',lambda p,j,**kw:dict(source=dict(job_uid=j)))
    def submit(slug,kind,sources,params):
        events.append(('execute',params['ncc_score_thresh']))
        return SimpleNamespace(uid=slug,outputs=dict(particles=SimpleNamespace(num_items=10)))
    e.submit=submit
    def review(slug,stage,visual,current,history):
        events.append(('review',current['ncc_score_thresh']))
        return dict(action='revise',parameters=thresholds(.3),reason='junk retained') if not history else dict(action='keep',reason='better')
    e.review_parameters=review
    _,p=e.inspect_serial('sample',SimpleNamespace(uid='picker'),thresholds())
    assert events==[('execute',.2),('review',.2),('execute',.3),('review',.3)]
    assert p['ncc_score_thresh']==.3


def test_unaccepted_last_review_never_forces_keep(tmp_path,monkeypatch):
    import inspect_review
    e=serial_instance(tmp_path)
    e.manifest['max_stage_reviews']=1
    monkeypatch.setattr(inspect_review,'build_visual_context',lambda *a,**kw:{})
    e.submit=lambda *a:SimpleNamespace(uid='J1',outputs=dict(particles=SimpleNamespace(num_items=10)))
    e.review_parameters=lambda *a:dict(action='revise',parameters=thresholds(.4),reason='still poor')
    with pytest.raises(ProtocolExhausted,match='without an accepted'):
        e.inspect_serial('sample',SimpleNamespace(uid='picker'),thresholds())


def pilot_instance(tmp_path,monkeypatch,counts):
    import vision_inputs
    e=serial_instance(tmp_path)
    e.scales=(50,100,200)
    e.calls=[]
    e.sample_picker=lambda n:e.calls.append(('sample',n)) or SimpleNamespace(uid='picker'+str(n))
    e.request_inspect=lambda slug,picker,feedback=None:e.calls.append(('parameters',picker.uid)) or thresholds(.2 if feedback is None else .3)
    e.inspect_serial=lambda tag,p,pms:(SimpleNamespace(uid='inspect'),pms)
    monkeypatch.setattr(vision_inputs,'build_extract_box_visual_context',lambda *a,**kw:{})
    e.stage=lambda *a:dict(box_size_pix=384)
    e.extract_serial=lambda tag,p,i,n,b:(SimpleNamespace(uid='extract',outputs=dict(particles=SimpleNamespace(num_items=10000))),b)
    e.submit=lambda slug,kind,src,params:SimpleNamespace(uid=slug,outputs=dict(particles=SimpleNamespace(num_items=10000)))
    def classify(tag,ex,box,picker,inspected,count,params):
        e.calls.append(('class',count,params['ncc_score_thresh'],box))
        n,groups=counts(tag,count)
        return dict(executable=True,gate=gate(n,groups),count=count,thresholds=params,box_size_pix=box,
            classes='Jnew',evidence_slug='new')
    e.classify=classify
    e.request_adjustment=lambda tag,row,history:e.calls.append(('diagnose',row['count'])) or dict(
        target='inspect',parameters=thresholds(.3),reason='missing weak picks')
    e.apply_adjustment=lambda tag,row,d,history:classify(tag,SimpleNamespace(),row['box_size_pix'],
        SimpleNamespace(),SimpleNamespace(),row['count'],d['parameters'])
    e.core_and_recovery=lambda tag,row:e.calls.append(('core',row['count'])) or dict(
        gate=row['gate'],bank_job='bank',best_refine='refine',selections=[])
    return e


def test_quality_failure_exhausts_scales_before_adjustment(tmp_path,monkeypatch):
    e=pilot_instance(tmp_path,monkeypatch,lambda tag,count:(1000,3))
    with pytest.raises(ProtocolExhausted,match='Maximum samples'):
        e.pilot()
    assert [c[1] for c in e.calls if c[0]=='sample']==[50,100,200,200]
    assert [c[1] for c in e.calls if c[0]=='parameters']==['picker50']
    assert [c[1] for c in e.calls if c[0]=='diagnose']==[200]
    assert not any(c[0]=='core' for c in e.calls)
    assert [c[2] for c in e.calls if c[0]=='class']==[.2,.2,.2,.3]


def test_core_reconstruction_waits_for_quality_and_promotion_for_diversity(tmp_path,monkeypatch):
    e=pilot_instance(tmp_path,monkeypatch,lambda tag,count:(2100,1 if count==50 else 3))
    result=e.pilot()
    assert [c[1] for c in e.calls if c[0]=='core']==[50,100]
    assert len([c for c in e.calls if c[0]=='parameters'])==1
    assert result['count']==100
    assert promotion_pass(result['promotion_gate'])
    assert e.state['bank_job']=='bank'


def test_failed_protocol_does_not_resume_into_full_data():
    e=SequentialPickerRollout.__new__(SequentialPickerRollout)
    e.state=dict(status='protocol_exhausted')
    e.pilot=lambda:pytest.fail('replayed exhausted protocol')
    assert e.run() is None


def test_phase_budget_counts_externals_and_blocks_new_jobs_not_resume(tmp_path):
    e=serial_instance(tmp_path)
    e.manifest['max_seed_jobs']=1
    e.state['steps']=dict(existing=dict(job_uid='J1'))
    e.guard_budget('existing')
    with pytest.raises(ProtocolExhausted):e.guard_budget('new_external')


def test_full_route_cannot_precede_qualified_pilot_and_confirmation(tmp_path):
    e=serial_instance(tmp_path)
    e.manifest.update(shared_jobs=['J1','J2','J3'],full_micrographs=3656,target_resolution_A=3.5)
    e.project=SimpleNamespace(find_job=lambda uid:SimpleNamespace(uid=uid))
    calls=[]
    e.pilot=lambda:calls.append('qualified_pilot') or dict(promotion_gate=gate(),bank_job='bank')
    e.confirm=lambda recipe:calls.append('qualified_confirmation') or dict(kind='template',thresholds=thresholds(),gate=gate(),box_size_pix=440)
    def full_route(*a,**kw):
        assert kw['box']==440  # The confirmed box, not the pilot's old default.
        calls.append('full_route')
        return dict(executable=True,gate=gate())
    e.route=full_route
    e.reconstruct=lambda row:calls.append('full_reconstruction') or SimpleNamespace(uid='best')
    e.resolution=lambda job:3.4
    assert e.run()=='best'
    assert calls==['qualified_pilot','qualified_confirmation','full_route','full_reconstruction']


def test_bad_pilot_cannot_bypass_gate_into_confirmation(tmp_path):
    e=serial_instance(tmp_path)
    e.pilot=lambda:dict(promotion_gate=gate(1000,1))
    e.confirm=lambda *a:pytest.fail('unqualified confirmation submitted')
    with pytest.raises(ValueError,match='No qualified pilot'):e.run()


def test_protocol_exhaustion_is_recorded_not_human_wait(tmp_path):
    e=serial_instance(tmp_path)
    def fail():raise ProtocolExhausted('Maximum samples exhausted')
    e.pilot=fail
    writes=[]
    e.write=lambda path,payload:writes.append(payload)
    assert e.run() is None
    assert e.state['status']=='protocol_exhausted'
    assert writes[-1]['awaiting_human_input'] is False
    assert writes[-1]['full_data_promoted'] is False


def test_extract_revises_only_after_actual_completed_stack_evidence(tmp_path,monkeypatch):
    import extract_review_evidence
    e=serial_instance(tmp_path)
    events=[]
    def submit(slug,kind,sources,params):
        events.append(('extract',params['box_size_pix']))
        return SimpleNamespace(uid=slug,outputs=dict(particles=SimpleNamespace(num_items=100)))
    e.submit=submit
    monkeypatch.setattr(extract_review_evidence,'build_evidence',lambda *a,**kw:
        (events.append(('actual_stack_evidence',a[1])) or [],dict(samples=[dict(sample_id='S1')])))
    def review(slug,stage,content,current,history):
        events.append(('review',current['box_size_pix']))
        return dict(action='revise',parameters=dict(box_size_pix=440),reason='clipping') if not history else dict(action='keep',reason='complete')
    e.review_parameters=review
    _,box=e.extract_serial('trial',SimpleNamespace(uid='picker'),SimpleNamespace(uid='inspect'),50,320)
    assert box==440
    assert [v[0] for v in events]==['extract','actual_stack_evidence','review','extract','actual_stack_evidence','review']


def failed_row(box=384,params=None):
    return dict(count=200,box_size_pix=box,thresholds=params or thresholds(),picker='picker',
                inspect='inspection',extract='extraction',classes='classes',evidence_slug='class_evidence')


@pytest.mark.parametrize('decision',[
    dict(target='inspect',parameters=thresholds(.3),reason='junk retained'),
    dict(target='extract',parameters=dict(box_size_pix=440),reason='central particle clipped')])
def test_maximum_sample_can_adjust_either_factor(decision):
    assert validate_adjustment(decision,failed_row())['target']==decision['target']


@pytest.mark.parametrize('decision',[
    dict(target='extract',parameters=dict(box_size_pix=384),reason='unchanged'),
    dict(target='extract',parameters=dict(box_size_pix=440,ncc_score_thresh=.3),reason='two factors'),
    dict(target='inspect',parameters=dict(thresholds(.3),box_size_pix=440),reason='two factors'),
    dict(target='inspect',parameters=thresholds(),reason='unchanged'),
    dict(target='extract',parameters=dict(box_size_pix=728),reason='invalid box')])
def test_single_factor_contract_forbids_noops_and_joint_changes(decision):
    with pytest.raises(ValueError):validate_adjustment(decision,failed_row())


def test_previously_tested_recipe_is_not_repeated():
    previous=failed_row()
    decision=dict(target='extract',parameters=dict(box_size_pix=440),reason='larger')
    with pytest.raises(ValueError):
        validate_adjustment(decision,previous,[failed_row(440)])
    assert validate_adjustment(decision,previous,[failed_row(440,thresholds(.3))])


def test_extract_adjustment_reuses_exact_inspect_and_picker(tmp_path):
    e=serial_instance(tmp_path)
    e.project=SimpleNamespace(find_job=lambda uid:SimpleNamespace(uid=uid))
    seen=[]
    e.inspect_serial=lambda *a,**kw:pytest.fail('Inspect must not change during a box trial')
    e.submit=lambda *a,**kw:pytest.fail('No separate Inspect/Blob job should be created')
    def extract(tag,picker,inspected,count,box,**kw):
        seen.append((picker.uid,inspected.uid,count,box,kw['fixed_history']))
        return SimpleNamespace(uid='new_extract'),box
    e.extract_serial=extract
    e.classify=lambda tag,job,box,picker,inspection,count,params:dict(
        extract=job.uid,box_size_pix=box,thresholds=params,inspect=inspection.uid)
    row=e.apply_adjustment('repair',failed_row(),dict(target='extract',parameters=dict(box_size_pix=440),reason='clipped'))
    assert seen[0][:4]==('picker','inspection',200,440)
    assert seen[0][4][0]['parameters']==dict(box_size_pix=384)
    assert row['thresholds']==thresholds()
    assert row['inspect']=='inspection'


def test_inspect_adjustment_keeps_box_and_does_not_redecide_extract_size(tmp_path):
    e=serial_instance(tmp_path)
    e.project=SimpleNamespace(find_job=lambda uid:SimpleNamespace(uid=uid))
    e.inspect_serial=lambda *a,**kw:(SimpleNamespace(uid='new_inspection'),thresholds(.3))
    e.extract_serial=lambda *a,**kw:pytest.fail('Cannot change box in an Inspect-only trial')
    calls=[]
    e.submit=lambda slug,kind,sources,params:calls.append((kind,sources,params)) or SimpleNamespace(uid='new_extract')
    e.classify=lambda tag,job,box,picker,inspection,count,params:dict(box_size_pix=box,thresholds=params)
    row=e.apply_adjustment('repair',failed_row(),dict(target='inspect',parameters=thresholds(.3),reason='junk'))
    assert calls[0][0]=='extract_micrographs_multi'
    assert calls[0][2]['box_size_pix']==384
    assert calls[0][1]['particles']==('new_inspection','particles')
    assert row['box_size_pix']==384
    assert row['thresholds']==thresholds(.3)


def test_partial_expansion_resume_uses_existing_stage_slugs_without_restarting_reviews(tmp_path,monkeypatch):
    e=pilot_instance(tmp_path,monkeypatch,lambda tag,count:(2100,3))
    e.state['sequential_trials']=dict(pilot_r0_n50=dict(row=dict(executable=True,
        gate=gate(1000,1),count=50,thresholds=thresholds(),box_size_pix=256,
        classes='old_classes',evidence_slug='old_classes')))
    e.request_inspect=lambda *a,**kw:pytest.fail('Do not restart initial threshold decision')
    e.extract_serial=lambda *a,**kw:pytest.fail('Do not restart initial box decision')
    submitted=[]
    e.submit=lambda slug,kind,sources,params:submitted.append((slug,params)) or SimpleNamespace(
        uid=slug,outputs=dict(particles=SimpleNamespace(num_items=10000)))
    result=e.pilot()
    assert result['count']==100
    assert [s for s,_ in submitted]==['pilot_r0_n100_frozen_inspect','pilot_r0_n100_frozen_extract']
    assert submitted[1][1]['box_size_pix']==256


def test_recovery_accepts_aged_out_controller_only_with_terminal_accounting(monkeypatch):
    from scripts import launch_fresh_coverage_rollout as launch
    results=iter([SimpleNamespace(returncode=1,stdout='',stderr='Invalid job id'),
                  SimpleNamespace(returncode=0,stdout='1398344|CANCELLED by 9031\n',stderr='')])
    monkeypatch.setattr(launch.subprocess,'run',lambda *a,**kw:next(results))
    launch.ensure_inactive_controller('1398344')


def test_recovery_refuses_unverified_or_still_active_controller(monkeypatch):
    from scripts import launch_fresh_coverage_rollout as launch
    monkeypatch.setattr(launch.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='1398344\n'))
    with pytest.raises(RuntimeError,match='still active'):launch.ensure_inactive_controller('1398344')
    results=iter([SimpleNamespace(returncode=1,stdout=''),SimpleNamespace(returncode=0,stdout='1398344|RUNNING\n')])
    monkeypatch.setattr(launch.subprocess,'run',lambda *a,**kw:next(results))
    with pytest.raises(RuntimeError,match='Cannot verify'):launch.ensure_inactive_controller('1398344')


def test_empty_completed_reference_skips_recovery_without_inventing_matches(tmp_path,monkeypatch):
    e=serial_instance(tmp_path)
    job=SimpleNamespace(uid='match',outputs=dict(templates_selected=SimpleNamespace(num_items=0)))
    monkeypatch.setattr(CoverageRollout,'reference',lambda *a:job)
    assert e.reference('ref','classes','volume') is None
    assert e.state['reference_outcomes']['ref']['status']=='empty'
    e.model=lambda *a:pytest.fail('No model should fabricate reference-supported selections')
    assert e.review('weak',dict(source=dict(job_uid='classes')),'recovered',reference=None)==[]


def test_optional_native_reference_failure_is_not_a_successful_empty_result(tmp_path,monkeypatch):
    e=serial_instance(tmp_path)
    e.state['steps']['ref_match']=dict(job_uid='Jfailed')
    e.project=SimpleNamespace(find_job=lambda uid:SimpleNamespace(type='reference_select_2D',status='failed'))
    def fail(*a):raise RuntimeError('Native Job Jfailed failed; trajectory retained')
    monkeypatch.setattr(CoverageRollout,'reference',fail)
    assert e.reference('ref','classes','volume') is None
    assert e.state['reference_outcomes']['ref']['status']=='tool_failed'
    monkeypatch.setattr(CoverageRollout,'reference',lambda *a:pytest.fail('Do not repeat failed native matching'))
    assert e.reference('ref','classes','volume') is None
    with pytest.raises(ValueError,match='identity changed'):
        e.reference('ref','another_source','volume')


@pytest.mark.parametrize('message',['network unavailable','Native Job Jlowpass failed; trajectory retained'])
def test_reference_handler_does_not_hide_transport_or_lowpass_failures(tmp_path,monkeypatch,message):
    e=serial_instance(tmp_path)
    e.state['steps']['ref_match']=dict(job_uid='Jmatch')
    def fail(*a):raise RuntimeError(message)
    monkeypatch.setattr(CoverageRollout,'reference',fail)
    with pytest.raises(RuntimeError,match=message):e.reference('ref','classes','volume')


def test_nonempty_reference_evidence_is_preserved(tmp_path,monkeypatch):
    e=serial_instance(tmp_path)
    job=SimpleNamespace(uid='match',outputs=dict(templates_selected=SimpleNamespace(num_items=2)))
    monkeypatch.setattr(CoverageRollout,'reference',lambda *a:job)
    assert e.reference('ref','classes','volume') is job
    assert e.state['reference_outcomes']['ref']['status']=='available'

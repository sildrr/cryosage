import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import pilot_promotion as promotion


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def passing_state():
    return dict(status='pilot_usable_for_promotion',trials=[dict(outcome='pilot_usable_for_promotion',
        objective_gate=dict(passed=True,available=True))])


def test_subjective_or_failed_result_cannot_trigger_full_runs():
    assert promotion.promotion_trial(dict(status='pilot_exhausted_no_validated_route',quality='usable')) is None
    with pytest.raises(ValueError):
        promotion.promotion_trial(dict(status='pilot_usable_for_promotion',trials=[]))
    s=passing_state()
    s['trials'][0]['objective_gate']['available']=False
    with pytest.raises(ValueError):
        promotion.promotion_trial(s)


def test_failed_pilot_does_not_even_connect_or_submit(tmp_path,monkeypatch):
    import cryosparc_client
    monkeypatch.setattr(cryosparc_client,'cryosparc_client',lambda:pytest.fail('No API access for failed pilot'))
    write(tmp_path/'manifest.json',dict(conditional_full_runs=dict(repeats=2)))
    write(tmp_path/'state.json',dict(status='pilot_exhausted_no_validated_route'))
    promotion.launch_full_repeats(tmp_path)
    ledger=json.loads((tmp_path/'promotion/full_runs.json').read_text())
    assert ledger['status']=='not_triggered' and ledger['runs']=={}


def recipe(tmp_path):
    kb=tmp_path/'kb.sqlite'
    kb.write_bytes(b'frozen test snapshot')
    image=Image.new('RGB',(8,8),'white')
    stream=io.BytesIO()
    image.save(stream,format='PNG')
    return dict(project='P6',pilot_workspace='Wpilot',source_class_job='J20',selected_class_ids=[0,1,2],
        objective_gate=dict(passed=True,available=True,diversity_kind='2d proxy'),
        parameters=dict(picker=dict(diameter=100,diameter_max=200),inspect={},extract=dict(box_size_pix=384)),
        classification_hints=dict(class2D_K=30),reclassification_hints=dict(class2D_K=20),
        contact_sheet=dict(data_url='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()),
        shared_jobs=['J1','J2','J3'],kb_snapshot=str(kb),kb_sha256=hashlib.sha256(kb.read_bytes()).hexdigest())


def test_passing_recipe_submits_exactly_two_independent_runs_idempotently(tmp_path,monkeypatch):
    import cryosparc_client
    import webapp.core
    r=recipe(tmp_path)
    workspaces={}
    def create_workspace(**kwargs):
        uid='W'+str(len(workspaces)+1)
        ws=SimpleNamespace(uid=uid,ids=set())
        ws.find_jobs=lambda:[SimpleNamespace(uid=x) for x in ws.ids]
        workspaces[uid]=ws
        return ws
    def link(project,uid,workspace):
        workspaces[workspace].ids.add(uid)
    project=SimpleNamespace(uid='P6',create_workspace=create_workspace,find_workspace=lambda uid:workspaces[uid],
        cs=SimpleNamespace(api=SimpleNamespace(jobs=SimpleNamespace(link_to_workspace=link))))
    monkeypatch.setattr(cryosparc_client,'cryosparc_client',lambda:SimpleNamespace(find_project=lambda uid:project))
    monkeypatch.setattr(webapp.core,'job_path',lambda uid:tmp_path/'tasks'/uid)
    monkeypatch.setattr(promotion,'build_recipe',lambda *args:r)
    calls=[]
    def submit(args,**kwargs):
        calls.append(args)
        return str(9000+len(calls))+'\n'
    monkeypatch.setattr(promotion.subprocess,'check_output',submit)
    monkeypatch.setenv('CRYOAGENT_BENCHMARK_BUDGET_FILE','original-pilot-budget')
    write(tmp_path/'manifest.json',dict(project='P6',workspace='Wpilot',conditional_full_runs=dict(
        repeats=2,target_resolution_A=3.5,max_new_jobs=40)))
    write(tmp_path/'state.json',passing_state())
    promotion.launch_full_repeats(tmp_path)
    promotion.launch_full_repeats(tmp_path)
    ledger=json.loads((tmp_path/'promotion/full_runs.json').read_text())
    assert ledger['status']=='submitted' and len(ledger['runs'])==2
    assert len(calls)==2 and len(workspaces)==2
    import os
    assert os.environ['CRYOAGENT_BENCHMARK_BUDGET_FILE']=='original-pilot-budget'
    for entry in ledger['runs'].values():
        root=Path(entry['output'])
        task=json.loads((root/'task.json').read_text())
        assert task['status']=='queued' and task['config']['target']==3.5
        assert json.loads((root/'benchmark_budget.json').read_text())['maximum']==43
        assert workspaces[entry['workspace']].ids=={'J1','J2','J3'}
    assert ledger['runs']['1']['task_id']!=ledger['runs']['2']['task_id']


def test_reference_is_opt_in_scoped_and_never_relabels_pilot_ids(tmp_path):
    r=recipe(tmp_path)
    path=tmp_path/'reference.json'
    write(path,dict(project_uid='P6',workspace_uid='Wfull',recipe=r,
        recipe_sha256=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()))
    original=[dict(role='user',content='current context')]
    assert promotion.append_reference(original,[],str(path),'P6','Wfull') is original
    messages=promotion.append_reference(original,[dict(job_type='select_2D')],str(path),'P6','Wfull')
    assert len(messages)==2 and 'MUST NOT be copied' in messages[-1]['content'][0]['text']
    with pytest.raises(ValueError,match='another full run'):
        promotion.append_reference(original,[dict(job_type='select_2D')],str(path),'P6','Wother')
    r['objective_gate']['passed']=False
    write(path,dict(project_uid='P6',workspace_uid='Wfull',recipe=r,
        recipe_sha256=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()))
    with pytest.raises(ValueError,match='Unvalidated'):
        promotion.append_reference(original,[dict(job_type='select_2D')],str(path),'P6','Wfull')

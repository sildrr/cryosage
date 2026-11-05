import copy

import pytest

from diameter_trial import trial_candidates,guard_pilot_continuation
from test_diameter_trial import fixture,node


def two_pilots():
    state,picker,_,_,classes=fixture()
    newer=copy.deepcopy(picker)
    newer.update(cryosparc_job_uid='J8',workflow_node_id='J8',key_parameters=dict(diameter=120,diameter_max=240))
    reviewed=node('J9','class_2D_new',['J8'])
    state['nodes'] += [newer,reviewed]
    return state,reviewed


def evidence(uid,values):
    return dict(source={'job_uid':uid},cryosift=dict(success=True,project_uid='P6',job_uid=uid,
        class_scores=[dict(class_id=i,score=v) for i,v in enumerate(values)]))


def test_promotion_menu_includes_previous_routes_with_bound_parameters():
    state,current=two_pilots()
    actions=trial_candidates(state,current,[])
    routes=[a for a in actions if a['diameter_trial']['role']=='accept_full']
    assert [r['diameter_trial']['class_job_uid'] for r in routes]==['J9','J5']
    assert routes[0]['default_parameters']['diameter']==120
    assert routes[1]['default_parameters']['diameter']==80
    assert all(r['required_inputs']['micrographs'][0]['source_job_uid']=='J1' for r in routes)


@pytest.mark.parametrize('terminal',['stop','complete','request_input'])
def test_all_poor_scores_still_promote_relatively_better_route(terminal):
    state,current=two_pilots()
    context=dict(project_uid='P6',candidate_actions=trial_candidates(state,current,[]))
    visual=dict(contexts=[evidence('J5',[4.0,4.2]),evidence('J9',[5.0,5.1])])
    decision=dict(decision_type=terminal,reason='All pilot classes are poor')
    chosen,audit=guard_pilot_continuation(decision,context,visual)
    assert chosen['decision_type']=='forward'
    assert chosen['action_id']=='diameter_full_J5'
    assert chosen['parameters']==dict(diameter=80,diameter_max=140,num_process=200)
    assert audit['original_decision']==decision
    assert 'unvalidated' in ' '.join(chosen['risk_flags'])


def test_missing_scores_uses_explicitly_unvalidated_fallback_and_keeps_valid_model_choices():
    state,current=two_pilots()
    context=dict(candidate_actions=trial_candidates(state,current,[]))
    chosen,audit=guard_pilot_continuation(dict(decision_type='stop'),context,{})
    assert chosen['action_id']=='diameter_full_J9'
    assert audit['score'] is None and 'unavailable' in audit['ranking_method']
    model=dict(decision_type='forward',action_id='diameter_full_J5')
    assert guard_pilot_continuation(model,context,{})==(model,None)


def test_unusable_and_unrelated_pilots_do_not_become_full_routes():
    state,current=two_pilots()
    state['nodes'][4]['outputs']['particles']={'num_items':0}
    # A different original CTF must not be silently mixed into this route.
    state['nodes'][5]['inputs']['micrographs'][0]['source_job_uid']='J_OTHER'
    assert not [a for a in trial_candidates(state,current,[]) if a['diameter_trial']['role']=='accept_full']
    decision=dict(decision_type='stop')
    assert guard_pilot_continuation(decision,dict(candidate_actions=[]),{})==(decision,None)


def test_foreign_cryosift_evidence_is_not_ranked():
    state,current=two_pilots()
    context=dict(project_uid='P6',candidate_actions=trial_candidates(state,current,[]))
    foreign=evidence('J5',[0.1]);foreign['cryosift']['project_uid']='P_OTHER'
    chosen,audit=guard_pilot_continuation(dict(decision_type='stop'),context,dict(contexts=[foreign]))
    assert chosen['action_id']=='diameter_full_J9' and audit['score'] is None

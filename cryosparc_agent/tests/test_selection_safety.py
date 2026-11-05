import copy

import pytest

from action_registry import reselection_progress
from selection_safety import guard_select_decision


def selection(ids="2,3"):
    return {"decision_type": "forward", "action": "select_2D", "action_id": "select",
            "parameters": {"selected_templates": ids}}


def context(previous=None):
    candidate = {"action_id": "select", "available": True, "job_type": "select_2D",
                 "required_inputs": {"templates": [{"source_job_uid": "J10"}]}}
    if previous:
        candidate["reselection_constraint"] = {"mode": "strict_subset_of_previous_selection",
                                              "previous_selected_templates": previous}
    return {"project_uid": "P6", "candidate_actions": [candidate]}


def evidence(job="J10", project="P6"):
    return {"cryosift": {"success": True, "job_uid": job, "project_uid": project,
                          "score_direction": "lower_is_better", "selected_class_ids": [],
                          "class_scores": [{"class_id": i, "score": v} for i, v in
                                           enumerate([3.5, 4.1, 5.4, 5.8])]}}


def test_inverted_population_is_repaired_without_changing_class_count_or_raw_decision():
    original = selection()
    saved = copy.deepcopy(original)
    corrected, audit = guard_select_decision(original, context(), evidence())
    assert original == saved
    assert corrected["parameters"]["selected_templates"] == "0,1"
    assert audit[0]["original_selected_class_ids"] == [2, 3]
    assert audit[0]["class_count_preserved"]
    assert audit[0]["all_classes_fail_threshold"]


def test_rank_guard_does_not_force_threshold_or_fixed_retention_ratio():
    corrected, audit = guard_select_decision(selection("0,1,2"), context(), evidence())
    assert not audit
    assert corrected["parameters"]["selected_templates"] == "0,1,2"


def test_one_higher_scoring_view_is_not_forbidden_if_group_ranking_is_reasonable():
    corrected, audit = guard_select_decision(selection("0,3"), context(), evidence())
    assert not audit
    assert corrected["parameters"]["selected_templates"] == "0,3"


@pytest.mark.parametrize("job,project", [("J11", "P6"), ("J10", "P7")])
def test_scores_from_other_job_or_project_are_never_used(job, project):
    corrected, audit = guard_select_decision(selection(), context(), evidence(job, project))
    assert not audit
    assert corrected == selection()


def test_strict_subset_pool_is_preserved():
    corrected, audit = guard_select_decision(selection("3"), context("1,2,3"), evidence())
    assert corrected["parameters"]["selected_templates"] == "1"
    assert audit


@pytest.mark.parametrize("visual", [None, {}, {"cryosift": {"success": False}}])
def test_missing_or_failed_scores_do_not_invent_selection(visual):
    assert guard_select_decision(selection(), context(), visual) == (selection(), [])


def test_non_select_action_is_untouched():
    raw = {"action": "inspect_picks_v2", "parameters": {}}
    assert guard_select_decision(raw, context(), evidence()) == (raw, [])


def test_branch_action_is_audited_too():
    raw = {"decision_type": "branch", "selected_actions": [selection()]}
    corrected, audit = guard_select_decision(raw, context(), evidence())
    assert corrected["selected_actions"][0]["parameters"]["selected_templates"] == "0,1"
    assert len(audit) == 1


def node(uid, kind, parents=(), resolution=None):
    return {"cryosparc_job_uid": uid, "job_type": kind, "status": "completed",
            "parent_job_uids": list(parents), "outputs": {"volume": {
                "latest_summary_stat_values": {"fsc_info": {
                    "radwn_final_A": resolution, "radwn_loose_A": 1.0}}}}}


def state(second_resolution=16.0, reclassify=False):
    nodes = [node("J1", "extract_micrographs_multi"), node("J2", "class_2D_new", ["J1"]),
             node("J3", "select_2D", ["J2"]), node("J4", "homo_refine_new", ["J3"], 15.0)]
    if reclassify:
        nodes.append(node("J5", "class_2D_new", ["J3"]))
    nodes.extend([node("J6", "select_2D", ["J5" if reclassify else "J2"]),
                  node("J7", "homo_refine_new", ["J6"], second_resolution)])
    return {"nodes": nodes}, nodes[-3] if reclassify else nodes[1]


@pytest.mark.parametrize("resolution", [16.0, 15.0, 14.98, None])
def test_no_verified_improvement_blocks_reselection_loop(resolution):
    workflow, source = state(resolution)
    progress = reselection_progress(workflow, source)
    assert progress["stagnant"]
    assert len(progress["selection_results"]) == 2


def test_genuine_corrected_fsc_improvement_allows_further_selection():
    workflow, source = state(14.0)
    assert not reselection_progress(workflow, source)["stagnant"]


def test_reclassifying_same_extracted_population_does_not_reset_loop_guard():
    workflow, source = state(16.0, reclassify=True)
    assert source["cryosparc_job_uid"] == "J5"
    assert reselection_progress(workflow, source)["stagnant"]


def test_new_extraction_population_resets_experiment_scope():
    workflow, source = state(16.0)
    new_extract = node("J8", "extract_micrographs_multi")
    new_class = node("J9", "class_2D_new", ["J8"])
    workflow["nodes"].extend([new_extract, new_class])
    assert not reselection_progress(workflow, new_class)["stagnant"]

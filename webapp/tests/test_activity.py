import json

import pytest

from webapp import core, activity


def test_activity_shows_raw_reason_and_times_without_fabrication(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DATA", tmp_path)
    identifier = "a" * 32
    root = core.job_path(identifier)
    round_dir = root / "agent/run_20261014T000000Z/round_01"
    core.write(round_dir / "model_decision.json", {"action": "select_2D", "reason": "Exclude off-center classes", "parameters": {}})
    core.write(round_dir / "execution.json", {"execution_result": {"execution_results": [{"job_uid": "J2"}]}})
    core.write(round_dir / "job_results.json", [{"result": {"job_uid": "J2", "timestamps": {"completed_at": "2026-10-14T00:00:00Z"}}}])
    result = activity.task_activity(identifier)
    assert result["decisions"][0]["reason"] == "Exclude off-center classes"
    assert result["job_details"]["J2"]["timestamps"]["completed_at"] == "2026-10-14T00:00:00Z"
    core.write(round_dir / "model_decision.json", {"action": "select_2D", "reason": "Decision produced by the visual SFT adapter."})
    result = activity.task_activity(identifier)
    assert result["decisions"][0]["reason"] is None


def test_live_instruction_is_durable_and_invalid_target_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DATA", tmp_path)
    monkeypatch.setattr(core, "public_task", lambda uid: {
        "status": "running", "controls_supported": True,
        "progress": {"nodes": [{"id": "J1", "status": "completed"}]}})
    identifier = "a" * 32
    result = activity.submit_intervention(identifier, {"action": "guide", "message": "Try a larger box", "target_node": "J1"})
    assert result["status"] == "pending"
    assert len(list((core.job_path(identifier) / "control/inbox").glob("*.json"))) == 1
    assert activity.control_status(identifier)["commands"][0]["message"] == "Try a larger box"
    with pytest.raises(ValueError, match="completed node"):
        activity.submit_intervention(identifier, {"action": "guide", "message": "x", "target_node": "J999"})


def test_old_active_controller_cannot_falsely_acknowledge_instruction(monkeypatch):
    monkeypatch.setattr(core, "public_task", lambda uid: {"status": "running"})
    with pytest.raises(ValueError, match="old or stopped"):
        activity.submit_intervention("a" * 32, {"action": "pause"})

import asyncio
import json

from operator_control import decision_boundary, save


def command(root, name, action, **kwargs):
    save(root / "inbox" / f"{name}.json", {"id": name, "action": action, **kwargs})


def test_guidance_applies_at_boundary_and_validates_anchor(tmp_path):
    calls = []

    async def validate(uid):
        calls.append(uid)

    command(tmp_path, "001", "guide", message="Use a larger extraction box", target_node="J484")
    result = asyncio.run(decision_boundary(tmp_path, "J489", validate))
    assert result["current_node"] == "J484" and result["guidance"]["text"] == "Use a larger extraction box"
    assert calls == ["J484"]
    assert json.loads((tmp_path / "receipts/001.json").read_text())["status"] == "applied"
    assert not list((tmp_path / "inbox").glob("*.json"))


def test_invalid_anchor_is_rejected_without_changing_node(tmp_path):
    async def validate(uid):
        raise ValueError("Not in current workspace")

    command(tmp_path, "001", "guide", message="Try again", target_node="J999")
    result = asyncio.run(decision_boundary(tmp_path, "J489", validate))
    assert result["current_node"] == "J489" and result["guidance"] is None
    assert json.loads((tmp_path / "receipts/001.json").read_text())["status"] == "rejected"


def test_pause_waits_until_explicit_resume(tmp_path, monkeypatch):
    import operator_control
    command(tmp_path, "001", "pause")

    async def sleep(seconds):
        assert json.loads((tmp_path / "state.json").read_text())["paused"]
        command(tmp_path, "002", "resume")

    async def validate(uid):
        pass

    monkeypatch.setattr(operator_control.asyncio, "sleep", sleep)
    result = asyncio.run(decision_boundary(tmp_path, "J489", validate))
    assert not result["stop"] and not json.loads((tmp_path / "state.json").read_text())["paused"]


def test_stop_is_durable_and_does_not_submit_or_cancel_jobs(tmp_path):
    async def validate(uid):
        raise AssertionError("No target validation needed for stop")

    command(tmp_path, "001", "stop")
    assert asyncio.run(decision_boundary(tmp_path, "J489", validate))["stop"]
    command(tmp_path, "002", "resume")
    assert not asyncio.run(decision_boundary(tmp_path, "J489", validate))["stop"]

from types import SimpleNamespace

import pytest

from webapp import core, dashboard
from cryosparc_agent.visual_decision_contract import build_stage_decision_messages


def test_reason_opt_in_preserves_training_contract():
    compact = build_stage_decision_messages("blob_picker_diameter", {})
    explained = build_stage_decision_messages("blob_picker_diameter", {}, include_reason=True)
    assert '"reason"' not in compact[1]["content"][0]["text"]
    assert '"reason"' in explained[1]["content"][0]["text"]
    assert "auditable" in explained[1]["content"][0]["text"]


def test_dashboard_assets_scoped_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DATA", tmp_path)
    monkeypatch.setattr(core, "public_task", lambda _: {"project": "P6", "progress": {"nodes": [{"id": "J1"}]}})
    asset_id = "a" * 24
    monkeypatch.setattr(dashboard, "event_images", lambda *args: [{"asset_id": asset_id, "extension": "png"}])
    calls = []

    def download(uid):
        calls.append(uid)
        return SimpleNamespace(stream=lambda: iter([b"\x89PNG\r\n\x1a\n", b"test"]))

    monkeypatch.setattr(dashboard, "client", lambda: SimpleNamespace(api=SimpleNamespace(assets=SimpleNamespace(download=download))))
    identifier = "b" * 32
    path = dashboard.image_file(identifier, "J1", asset_id)
    assert path.read_bytes().startswith(b"\x89PNG")
    assert dashboard.image_file(identifier, "J1", asset_id) == path
    assert len(calls) == 1
    with pytest.raises(ValueError, match="workspace"):
        dashboard.image_file(identifier, "J2", asset_id)
    with pytest.raises(ValueError, match="Image does not belong"):
        dashboard.image_file(identifier, "J1", "c" * 24)


def test_event_metadata_keeps_all_iterations_and_ignores_pdf(monkeypatch):
    events = [SimpleNamespace(model_dump=lambda **_: {"type": "image", "text": "Class averages", "imgfiles": [
        {"fileid": "a" * 24, "filetype": "pdf"}, {"fileid": "b" * 24, "filetype": "png"}]})]
    monkeypatch.setattr(dashboard, "client", lambda: SimpleNamespace(api=SimpleNamespace(jobs=SimpleNamespace(get_event_logs=lambda *args: events))))
    dashboard.event_images.cache_clear()
    assert dashboard.event_images("P1", "J1", 0)[0]["asset_id"] == "b" * 24
    dashboard.event_images.cache_clear()

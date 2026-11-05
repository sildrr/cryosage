import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from webapp import core, lifecycle


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DATA", tmp_path)
    identifier = "b" * 32
    core.write(core.job_path(identifier) / "task.json", {"id": identifier, "status": "finished"})
    return identifier


def test_archive_restore_and_delete_preserve_scientific_files(task):
    root = core.job_path(task)
    (root / "density.mrc").write_bytes(b"map")
    assert len(lifecycle.list_tasks()) == 1
    lifecycle.archive_task(task, True)
    assert not lifecycle.list_tasks()
    assert lifecycle.list_tasks(True)[0]["archived"] is True
    lifecycle.archive_task(task, False)
    assert len(lifecycle.list_tasks()) == 1
    lifecycle.delete_task(task)
    assert not lifecycle.list_tasks() and not lifecycle.list_tasks(True)
    assert (root / "density.mrc").read_bytes() == b"map"
    with pytest.raises(FileNotFoundError):
        lifecycle.visible_root(task)


@pytest.mark.parametrize("status", ["queued", "running", "paused", "submitting"])
def test_active_workflow_cannot_be_deleted(task, status):
    core.write(core.job_path(task) / "task.json", {"id": task, "status": status})
    with pytest.raises(ValueError, match="Stop"):
        lifecycle.delete_task(task)
    assert lifecycle.visible_root(task)


def test_stale_terminal_record_with_live_controller_cannot_be_deleted(task, monkeypatch):
    core.write(core.job_path(task) / "task.json", {"id": task, "status": "finished", "slurm_job": "123"})
    monkeypatch.setattr(lifecycle.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="RUNNING"))
    with pytest.raises(ValueError, match="still active"):
        lifecycle.delete_task(task)


def test_scheduler_error_fails_closed(task, monkeypatch):
    core.write(core.job_path(task) / "task.json", {"id": task, "status": "finished", "slurm_job": "123"})
    monkeypatch.setattr(lifecycle.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(ValueError, match="Cannot confirm"):
        lifecycle.delete_task(task)


def test_live_cryosparc_job_cannot_be_deleted(task):
    core.write(core.job_path(task) / "progress.json", {"nodes": [{"status": "running"}]})
    with pytest.raises(ValueError, match="CryoSPARC"):
        lifecycle.delete_task(task)


def test_archive_requires_boolean(task):
    with pytest.raises(ValueError):
        lifecycle.archive_task(task, "false")


def test_traversal_rejected():
    with pytest.raises(ValueError):
        lifecycle.visible_root("../../outside")


def test_authenticated_lifecycle_routes_and_deleted_detail(task, monkeypatch):
    from webapp import server
    monkeypatch.setattr(server, "TOKEN", "test-token")
    app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{app.server_port}"

    def call(path, body=None, authorized=True):
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer test-token"
        request = urllib.request.Request(base + path, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        with opener.open(request, timeout=5) as response:
            return json.load(response)
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            call(f"/api/tasks/{task}/archive", {"archived": True}, authorized=False)
        assert error.value.code == 401
        call(f"/api/tasks/{task}/archive", {"archived": True})
        assert call("/api/tasks") == []
        assert call("/api/tasks?archived=true")[0]["id"] == task
        call(f"/api/tasks/{task}/archive", {"archived": False})
        assert call("/api/tasks")[0]["id"] == task
        call(f"/api/tasks/{task}/delete", {})
        with pytest.raises(urllib.error.HTTPError) as error:
            call(f"/api/tasks/{task}")
        assert error.value.code == 404
    finally:
        app.shutdown()
        thread.join(timeout=5)
        app.server_close()

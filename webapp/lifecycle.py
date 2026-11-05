"""Web-only workflow visibility; never delete scientific inputs or CryoSPARC jobs."""
import json
import subprocess

from webapp import core


def lifecycle_metadata(root):
    path = root / "lifecycle.json"
    return json.loads(path.read_text()) if path.exists() else {"archived": False, "deleted": False}


def visible_root(identifier):
    root = core.job_path(identifier)
    if not (root / "task.json").is_file() or lifecycle_metadata(root).get("deleted"):
        raise FileNotFoundError("Workflow not found")
    return root


def list_tasks(archived=False):
    paths = sorted((core.DATA / "tasks").glob("*/task.json"),
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    result = []
    for path in paths:
        metadata = lifecycle_metadata(path.parent)
        if metadata.get("deleted") or bool(metadata.get("archived")) != archived:
            continue
        task = core.public_task(path.parent.name)
        task.update(metadata)
        result.append(task)
        if len(result) >= 100:
            break
    return result


def archive_task(identifier, archived):
    if not isinstance(archived, bool):
        raise ValueError("archived must be true or false")
    root = visible_root(identifier)
    metadata = lifecycle_metadata(root)
    metadata.update(archived=archived, visibility_updated_at=core.now())
    core.write(root / "lifecycle.json", metadata)
    return {"id": identifier, **metadata}


def delete_task(identifier):
    root = visible_root(identifier)
    task = core.read_task(identifier)
    if task.get("status") in {"submitting", "queued", "running", "paused", "recovering"}:
        raise ValueError("Stop the workflow and wait for active jobs to finish before deleting it")
    if task.get("slurm_job"):
        try:
            queue = subprocess.run(
                ["squeue", "-h", "-j", str(task["slurm_job"]), "-o", "%T"],
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("Cannot confirm that the controller has stopped; deletion refused") from exc
        # Slurm can return 'Invalid job id' after a terminal job leaves its cache.
        # Confirm that case from accounting rather than interpreting any failure as idle.
        if queue.returncode:
            account = subprocess.run(
                ["sacct", "-j", str(task["slurm_job"]), "-n", "-P", "--format=JobID,State"],
                capture_output=True, text=True, timeout=10)
            terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                        "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
            rows = [line.split("|") for line in account.stdout.splitlines()]
            state = next((row[1].split()[0] for row in rows
                          if len(row) > 1 and row[0] == str(task["slurm_job"])), None)
            if account.returncode or state not in terminal:
                raise ValueError("Cannot confirm that the controller has stopped; deletion refused")
        elif queue.stdout.strip():
            raise ValueError("The Slurm controller is still active; deletion refused")
    progress = core.public_task(identifier).get("progress", {})
    active_states = {"building", "queued", "launched", "started", "running", "waiting", "waiting_for_interactive"}
    if any(node.get("status") in active_states for node in progress.get("nodes", [])):
        raise ValueError("CryoSPARC jobs are still active; deletion refused")
    cryonet = root / "cryonet.json"
    if cryonet.exists() and json.loads(cryonet.read_text()).get("status") in {"submitting", "running", "queued"}:
        raise ValueError("CryoNet model building is still active; deletion refused")
    metadata = lifecycle_metadata(root)
    metadata.update(deleted=True, deleted_at=core.now())
    core.write(root / "lifecycle.json", metadata)
    return {"id": identifier, "deleted": True,
            "message": "Removed from the web console. Source data, audit files and CryoSPARC results are preserved."}

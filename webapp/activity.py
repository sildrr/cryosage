"""Compact UI projections of existing audit logs; never expose prompts or API keys."""
import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path

from webapp.core import DATA, REPO, job_path, read_task


@lru_cache(maxsize=512)
def load_compact(path, mtime, size):
    value = json.loads(Path(path).read_text())
    if Path(path).name == "visual_context.json":
        images = []

        def walk(item):
            if isinstance(item, dict):
                if item.get("local_path"):
                    images.append({"path": item["local_path"], "caption": item.get("role") or item.get("kind") or "Visual evidence"})
                for key, child in item.items():
                    if key not in {"data_url", "raw_text"}:
                        walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)
        walk(value)
        return images
    if Path(path).name == "job_results.json":
        return [{"job_uid": v.get("result", {}).get("job_uid"),
                 "timestamps": v.get("result", {}).get("timestamps", {})} for v in value if isinstance(v, dict)]
    if Path(path).name == "execution.json":
        return [{"job_uid": r.get("job_uid"), "parameters": (r.get("planned_action") or {}).get("resolved_parameters", {})}
                for r in (value.get("execution_result") or {}).get("execution_results", []) if r.get("job_uid")]
    return value


def read(path, default):
    if not path.exists():
        return default
    try:
        stat = path.stat()
        return load_compact(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, TypeError):
        return default


def publish_image(root, image):
    source = Path(image["path"]).resolve()
    allowed = (DATA.resolve(), (REPO / "cryosparc_agent/logs/vision_inputs").resolve(),
               (REPO / "logs/vision_inputs").resolve())
    if not any(source.is_relative_to(p) for p in allowed) or not source.is_file():
        return None
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg"} or source.stat().st_size > 12 * 1024**2:
        return None
    identity = hashlib.sha256(f"{source}:{source.stat().st_mtime_ns}".encode()).hexdigest()
    target = root / "ui_images" / (identity + source.suffix.lower())
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    return {"file": target.name, "caption": image["caption"]}


def task_activity(identifier):
    root = job_path(identifier)
    entries = []
    jobs = {}
    for directory in sorted((root / "agent").glob("run_*/round_*"))[-80:]:
        decision = read(directory / "model_decision.json", {})
        execution = read(directory / "execution.json", [])
        results = read(directory / "job_results.json", [])
        validation = read(directory / "validation.json", {})
        planner = read(directory / "planning_directive.json", {})
        images = []
        for image in read(directory / "visual_context.json", []):
            item = publish_image(root, image)
            if item and item not in images:
                images.append(item)
        entry = {"id": f"{directory.parent.name}/{directory.name}",
                 "action": decision.get("action") or decision.get("job_type") or decision.get("decision"),
                 "reason": decision.get("reason"), "parameters": decision.get("parameters", {}),
                 "planner_reason": planner.get("reason") or planner.get("rationale"),
                 "created_jobs": [e["job_uid"] for e in execution], "images": images,
                 "validation_errors": (validation.get("validation") or {}).get("issues", []),
                 "updated_at": directory.stat().st_mtime}
        if entry["reason"] == "Decision produced by the visual SFT adapter.":
            entry["reason"] = None
            entry["reason_note"] = "This round returned compact parameters without a natural-language explanation. No rationale has been inferred."
        for result in results:
            if result.get("job_uid"):
                jobs.setdefault(result["job_uid"], {}).update(timestamps=result["timestamps"])
        for result in execution:
            jobs.setdefault(result["job_uid"], {}).update(parameters=result["parameters"], decision_id=entry["id"])
        if entry["action"] or images or entry["validation_errors"]:
            entries.append(entry)
    return {"decisions": entries, "job_details": jobs}


def submit_intervention(identifier, body):
    import time
    import uuid
    from webapp.core import public_task, write, now
    task = public_task(identifier)
    action = body.get("action")
    if action == 'approve_pilot_diversity':
        if task['status'] not in {'needs_attention', 'controller_failed'}:
            raise ValueError('Diversity approval requires a stopped pilot controller')
        import subprocess
        queue = subprocess.run(['squeue', '-h', '-o', '%A'], capture_output=True, text=True, check=True, timeout=10)
        if str(task.get('slurm_job')) in queue.stdout.split():
            raise ValueError('Controller is still active; refusing duplicate approval/recovery')
        from webapp.pilot_approval import request_approval
        approval = request_approval(job_path(identifier), body.get('message') or
            'Human approved full-data reconstruction despite insufficient pilot shape diversity.')
        from webapp.resume import main
        main(identifier)
        return dict(status='queued', approval=approval)
    if action not in {"guide", "pause", "resume", "stop"}:
        raise ValueError("Supported actions: guide, pause, resume, stop")
    message = str(body.get("message", "")).strip()
    if len(message) > 4000 or action == "guide" and not message:
        raise ValueError("Guidance must contain 1-4000 characters")
    target = body.get("target_node") or None
    if target and not any(n["id"] == target and n["status"] == "completed" for n in task.get("progress", {}).get("nodes", [])):
        raise ValueError("Target must be a completed node of this task")
    stopped = task["status"] in {"needs_attention", "user_stopped", "finished", "controller_failed"}
    if stopped and action == "resume" and not target:
        completed = [n for n in task.get("progress", {}).get("nodes", []) if n["status"] == "completed"]
        target = completed[-1]["id"] if completed else None
    if not (stopped and action == "resume") and (
        not task.get("controls_supported") or task["status"] not in {"running", "paused", "queued"}
    ):
        raise ValueError("This old or stopped controller cannot consume live instructions; use Continue from a completed node")
    command = {"id": uuid.uuid4().hex, "action": action, "message": message,
               "target_node": target, "submitted_at": now()}
    filename = f"{time.time_ns():020d}-{command['id']}.json"
    write(job_path(identifier) / "control/inbox" / filename, command)
    write(job_path(identifier) / "control/history" / filename, command)
    if stopped and action == "resume":
        from webapp.resume import main
        try:
            main(identifier, target)
        except Exception:
            (job_path(identifier) / "control/inbox" / filename).unlink(missing_ok=True)
            write(job_path(identifier) / "control/receipts" / filename,
                  {"id": command["id"], "status": "rejected", "message": "Controller resubmission failed"})
            raise
    return {**command, "status": "pending", "policy": "Applied at next safe decision boundary; running jobs are not killed"}


def control_status(identifier):
    root = job_path(identifier) / "control"
    receipts = {v["id"]: v for f in (root / "receipts").glob("*.json") for v in [read(f, {})] if v.get("id")}
    return {"state": read(root / "state.json", {}), "commands": [
        {**read(f, {}), **receipts.get(read(f, {}).get("id"), {"status": "pending"})}
        for f in sorted((root / "history").glob("*.json"))[-20:]
    ]}


@lru_cache(maxsize=64)
def cached_job_detail(project, job_uid, time_bucket):
    from sys import path
    path.insert(0, str(REPO / "cryosparc_agent"))
    from cryosparc_client import cryosparc_client
    job = cryosparc_client().find_project(project).find_job(job_uid)
    model = job.model.model_dump()
    timestamps = {k: str(v) if v else None for k, v in model.items() if k.endswith("_at")}
    detail = {"id": job.uid, "status": job.status, "timestamps": timestamps,
              "parameters": job.params.model_dump(), "errors": model.get("errors_run", []),
              "cluster_job_id": model.get("cluster_job_id"), "cluster_status": model.get("cluster_job_status")}
    from webapp.dashboard import event_images
    try:
        detail["dashboard_images"] = event_images(project, job_uid, time_bucket // 2)
    except Exception:
        detail["dashboard_warning"] = "CryoSPARC dashboard images are temporarily unavailable. Please refresh later."
    volume = job.outputs.get("volume")
    if volume:
        info = volume.latest_summary_stats.get("fsc_info") or {}
        for name in ("fsc_autotight_noisesub", "fsc_autotight_noisesub_raw", "fsc_resmask", "fsc_nomask"):
            curve = info.get(name)
            if isinstance(curve, list) and curve and info.get("N") and info.get("psize"):
                detail["fsc_curve"] = {"name": name, "values": curve[:2048], "N": info["N"], "psize": info["psize"],
                                       "radwns": info.get("radwns", [])[:2048]}
                break
    return detail


def job_detail(identifier, job_uid):
    import time
    from webapp.core import public_task
    task = public_task(identifier)
    if not any(n["id"] == job_uid for n in task.get("progress", {}).get("nodes", [])):
        raise ValueError("Job does not belong to this task's workspace")
    return cached_job_detail(task["project"], job_uid, int(time.time() // 30))

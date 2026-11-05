"""Explicitly reschedule a stopped web task in its original workspace."""
import subprocess
import sys

from webapp.core import controller_command, job_path, read_task, write


def main(identifier, current_node=None):
    from webapp.lifecycle import visible_root
    visible_root(identifier)
    task = read_task(identifier)
    if current_node:
        import json
        nodes = json.loads((job_path(identifier) / "progress.json").read_text()).get("nodes", [])
        if not any(n["id"] == current_node and n["status"] == "completed" for n in nodes):
            raise ValueError("Resume node must be a completed Job in this task's recorded workspace")
    if task.get("slurm_job"):
        queue = subprocess.run(["squeue", "-h", "-j", task["slurm_job"], "-o", "%T"],
                               capture_output=True, text=True, timeout=10)
        if queue.stdout.strip():
            raise RuntimeError("Controller is still active; refusing duplicate recovery")
    result = subprocess.run(controller_command(identifier),
                            capture_output=True, text=True, check=True, timeout=30)
    task.update(status="queued", slurm_job=result.stdout.strip().split(";")[0])
    (job_path(identifier) / "progress_compat.json").unlink(missing_ok=True)
    if current_node:
        task["resume_node"] = current_node
        task.pop('service_resume_checkpoint', None)
        task.pop('service_finalize_checkpoint', None)
    task.pop("finished_at", None)
    task.pop("error", None)
    write(job_path(identifier) / "task.json", task)
    location = f"{task['project']}/{task['workspace']}" if task.get('project') and task.get('workspace') else identifier
    print(f"Rescheduled {location} from {task.get('resume_node') or task.get('import_job') or 'initialization'}: {task['slurm_job']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

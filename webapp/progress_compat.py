"""Temporary observer for a controller started before the FSC field fix."""
import json
import subprocess
import sys
import time

from webapp.core import job_path, now, read_task, write
from webapp.worker import export_result, snapshot


def main(identifier):
    root = job_path(identifier)
    task = read_task(identifier)
    controller = task["slurm_job"]
    while True:
        try:
            progress = snapshot(task["project"], task["workspace"], task["config"]["target"])
            write(root / "progress_compat.json", progress)
            queue = subprocess.run(["squeue", "-h", "-j", controller, "-o", "%T"],
                                   capture_output=True, text=True, timeout=10)
            # A failed scheduler query is not evidence that the controller ended.
            if queue.returncode != 0:
                accounting = subprocess.run(["sacct", "-n", "-P", "-j", controller,
                                             "--format=JobID,State"], capture_output=True, text=True, timeout=10)
                terminal = any(line.startswith(controller + "|") and
                               line.split("|")[1].split()[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}
                               for line in accounting.stdout.splitlines())
            else:
                terminal = not queue.stdout.strip()
            if terminal:
                task = read_task(identifier)
                if task.get("slurm_job") != controller:
                    (root / "progress_compat.json").unlink(missing_ok=True)
                    return
                export_result(task, progress)
                summaries = sorted((root / "agent").glob("run_*/summary.json"))
                summary = json.loads(summaries[-1].read_text()) if summaries else {}
                if task.get("runner_exit_code") == 0 and summary.get("success") and progress.get("best_refine"):
                    task["status"] = "finished"
                    task.pop("error", None)
                    write(root / "task.json", task)
                return
        except Exception as error:
            print(f"Progress observation will retry: {type(error).__name__}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main(sys.argv[1])

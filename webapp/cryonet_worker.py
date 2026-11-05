"""Audited handoff using the already tested CryoNet submit script, never browser keys."""
import subprocess
import sys
import json
from webapp.core import PYTHON, REPO, job_path, public_task, write


def main(identifier):
    root = job_path(identifier)
    task = public_task(identifier)
    result = task["result"]
    try:
        command = [PYTHON, str(REPO / "cryosparc_result_agent/scripts/submit_cryonet_mode22.py"),
                   "--map", str(root / "density.mrc"), "--fasta", str(root / "sequence.fasta"),
                   "--resolution-A", str(result["resolution_A"]), "--name", task["config"]["name"],
                   "--output-dir", str(root / "cryonet_submission")]
        subprocess.run(command, check=True, timeout=1800)
        response = json.loads((root / "cryonet_submission/submission.json").read_text())["response"]
        write(root / "cryonet.json", {"status": "submitted", "jobid": response.get("jobid"),
                                      "url": "https://cryonet.ai"})
    except Exception:
        write(root / "cryonet.json", {"status": "submission_failed", "url": "https://cryonet.ai",
                                      "message": "Check server-side CryoNet log before retrying; remote submission may already exist"})
        raise


if __name__ == "__main__":
    main(sys.argv[1])

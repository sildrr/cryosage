"""Optional CryoSift integration for deterministic 2D class scoring."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from cryosparc_client import cryosparc_client


AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATOR = AGENT_ROOT / "external/cryosift/cryosift_runner.py"
DEFAULT_WEIGHTS = AGENT_ROOT / "external/cryosift/class_labeling/final_model/final_model_cont.pth"
DEFAULT_CONDA_PYTHON = Path.home() / '.conda/envs/cryosift/bin/python'

CRYOSIFT_SCORE_GUIDANCE = (
    "For a successful CryoSift result, lower scores indicate better class quality. "
    "Its selected_class_ids are recommended class IDs whose scores are strictly below "
    "the threshold shown in that result (3.0 by default); they are not mandatory selections. "
    "An empty selected_class_ids list in a successful result means no class met the threshold. "
    "Compare IDs and scores only within their source Class 2D job. "
    "Do not treat higher scores as better. If image evidence supports retaining a "
    "high-scoring class, explain the disagreement. When no class passes the threshold, "
    "the score direction does NOT reverse: compare classes in ASCENDING CryoSift score "
    "order (lowest/best first), never descending. For example, with threshold 3.0, "
    "scores 3.2 and 4.8 both miss the threshold, but 3.2 is the better CryoSift score. "
    "For Select 2D and Template Picker template selection, prefer the lowest-scoring "
    "visually credible, complete, centered classes; do not force a selection if none "
    "is credible. CryoSift quality scores are NOT picker NCC/Power correlation scores; "
    "do not transfer a higher-is-better picker-score interpretation to CryoSift."
)

CRYOSIFT_EMPTY_GUIDANCE = (
    "If CryoSift succeeds and selected_class_ids is empty, all scored classes missed "
    "its quality threshold. Prefer investigating upstream particle picking and Inspect "
    "Picks thresholds before repeatedly reselecting or refining the same poor classes. "
    "This is a diagnostic clue, not proof: use the class images and other evidence to "
    "decide whether Blob Picker diameter, pick contamination, or another upstream issue "
    "actually explains the result. A failed CryoSift run is not an empty recommendation."
)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _job_directory(project_uid: str, job_uid: str) -> Path:
    cs = cryosparc_client()
    job = cs.find_project(project_uid).find_job(job_uid)
    for attr in ("dir", "job_dir", "directory", "path"):
        value = getattr(job, attr, None)
        if value:
            path = Path(str(value)).expanduser().resolve()
            if path.is_dir():
                return path
    raise FileNotFoundError(f"Could not resolve server-side directory for CryoSPARC job {job_uid}.")


def _cryosift_python() -> str:
    """Resolve a dedicated CryoSift interpreter without changing the MCP runtime."""
    configured = os.getenv("CRYOAGENT_CRYOSIFT_PYTHON")
    if configured:
        return configured
    if DEFAULT_CONDA_PYTHON.is_file():
        return str(DEFAULT_CONDA_PYTHON)
    return sys.executable


def _parse_scores(output_dir: Path, threshold: float) -> tuple[list[int], dict[int, float], str | None]:
    candidates = [output_dir / "score.txt", *sorted(output_dir.glob("score*.txt")), *sorted(output_dir.glob("score*.star"))]
    score_file = next((path for path in candidates if path.is_file()), None)
    if score_file is None:
        return [], {}, None
    # cryosift_runner.py serializes its first class as ``1`` while CryoSPARC
    # Select 2D expects zero-based class indices. Allow an override only for
    # third-party runners with a different file convention.
    try:
        index_base = int(os.getenv("CRYOAGENT_CRYOSIFT_SCORE_INDEX_BASE", "1"))
    except ValueError:
        index_base = 1
    scores: dict[int, float] = {}
    for line in score_file.read_text(errors="replace").splitlines():
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if len(numbers) < 2:
            continue
        try:
            class_id, score = int(float(numbers[0])) - index_base, float(numbers[1])
        except ValueError:
            continue
        if class_id < 0:
            continue
        scores[class_id] = score
    selected = [class_id for class_id, score in sorted(scores.items()) if score < threshold]
    return selected, scores, str(score_file)


def load_cached_cryosift_result(
    project_uid: str, job_uid: str, threshold: float = 3.0,
) -> dict[str, Any] | None:
    """Read existing scores for planning without running CryoSift again."""
    try:
        output_dir = _job_directory(project_uid, job_uid) / "cryosift_eval"
        selected, scores, score_file = _parse_scores(output_dir, threshold)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not scores:
        return None
    return {
        "success": True,
        "project_uid": project_uid,
        "job_uid": job_uid,
        "threshold": threshold,
        "score_direction": "lower_is_better",
        "score_guidance": CRYOSIFT_SCORE_GUIDANCE,
        "selection_rule": "score < threshold",
        "selected_class_ids": selected,
        "no_classes_below_threshold": not selected,
        "class_scores": [
            {"class_id": class_id, "score": score}
            for class_id, score in sorted(scores.items())
        ],
        "score_file": score_file,
        "source": "cached_score_file",
    }


def evaluate_2d_classes_with_cryosift(
    project_uid: str,
    job_uid: str,
    threshold: float = 3.0,
    output_dir: str | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run CryoSift on a completed Class 2D job and return normalized scores."""
    evaluator = _env_path("CRYOAGENT_CRYOSIFT_RUNNER", DEFAULT_EVALUATOR)
    weights = _env_path("CRYOAGENT_CRYOSIFT_WEIGHTS", DEFAULT_WEIGHTS)
    python_executable = _cryosift_python()
    conda_env = os.getenv("CRYOAGENT_CRYOSIFT_CONDA_ENV")
    missing = [str(path) for path in (evaluator, weights) if not path.is_file()]
    if missing:
        return {"success": False, "status": "not_configured", "tool": "cryosift", "project_uid": project_uid, "job_uid": job_uid, "missing_paths": missing, "message": "CryoSift evaluator or model weights are not configured."}
    try:
        classification_dir = _job_directory(project_uid, job_uid)
        result_dir = Path(output_dir).expanduser().resolve() if output_dir else classification_dir / "cryosift_eval"
        result_dir.mkdir(parents=True, exist_ok=True)
        command = []
        if conda_env:
            command.extend([os.getenv("CONDA_EXE", "conda"), "run", "-n", conda_env, python_executable])
        else:
            command.append(python_executable)
        command.extend([str(evaluator), "--input", str(classification_dir), "--output", str(result_dir), "--weights", str(weights)])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=max(timeout_seconds, 1), check=False)
        selected, scores, score_file = _parse_scores(result_dir, threshold)
        if completed.returncode != 0 or not scores:
            return {"success": False, "status": "failed" if completed.returncode != 0 else "missing_scores", "tool": "cryosift", "job_uid": job_uid, "returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-4000:]}
        return {"success": True, "status": "completed", "tool": "cryosift", "project_uid": project_uid, "job_uid": job_uid, "threshold": threshold, "score_direction": "lower_is_better", "score_guidance": CRYOSIFT_SCORE_GUIDANCE, "selection_rule": "score < threshold", "selected_class_ids": selected, "no_classes_below_threshold": not selected, "class_scores": [{"class_id": class_id, "score": score, "selected": class_id in selected} for class_id, score in sorted(scores.items())], "output_dir": str(result_dir), "score_file": score_file, "returncode": completed.returncode}
    except Exception as exc:
        return {"success": False, "status": "error", "tool": "cryosift", "project_uid": project_uid, "job_uid": job_uid, "error_type": type(exc).__name__, "message": str(exc)}

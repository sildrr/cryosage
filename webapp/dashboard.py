"""Read-only CryoSPARC event images, scoped to a task's workspace."""
import hashlib
import re
import time
import uuid
from functools import lru_cache

from webapp.core import REPO, job_path


def client():
    import sys
    sys.path.insert(0, str(REPO / "cryosparc_agent"))
    from cryosparc_client import cryosparc_client
    return cryosparc_client()


@lru_cache(maxsize=128)
def event_images(project, job_uid, bucket):
    images = []
    for event in client().api.jobs.get_event_logs(project, job_uid):
        value = event.model_dump(mode="json")
        if value.get("type") != "image":
            continue
        for asset in value.get("imgfiles", []):
            extension = str(asset.get("filetype", "")).lower()
            asset_id = asset.get("fileid", "")
            if extension not in {"png", "jpg", "jpeg"} or not re.fullmatch(r"[a-f0-9]{24}", asset_id):
                continue
            images.append({"asset_id": asset_id, "extension": extension,
                           "caption": value.get("text") or asset.get("filename") or "CryoSPARC plot",
                           "created_at": value.get("created_at"), "source": "CryoSPARC event log"})
            break
    return list(reversed(images))


def task_project(identifier, job_uid):
    from webapp.core import public_task
    task = public_task(identifier)
    if not any(n["id"] == job_uid for n in task.get("progress", {}).get("nodes", [])):
        raise ValueError("Job does not belong to this task's workspace")
    return task["project"]


def image_file(identifier, job_uid, asset_id):
    project = task_project(identifier, job_uid)
    images = event_images(project, job_uid, int(time.time() // 60))
    image = next((v for v in images if v["asset_id"] == asset_id), None)
    if image is None:
        raise ValueError("Image does not belong to this job")
    digest = hashlib.sha256(f"{project}:{job_uid}:{asset_id}".encode()).hexdigest()
    target = job_path(identifier) / "ui_images" / f"{digest}.{image['extension']}"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{uuid.uuid4().hex}.part")
    try:
        size = 0
        with temporary.open("xb") as output:
            for chunk in client().api.assets.download(asset_id).stream():
                size += len(chunk)
                if size > 24 * 1024**2:
                    raise ValueError("Dashboard image exceeds 24 MiB limit")
                output.write(chunk)
        with temporary.open("rb") as stream:
            signature = stream.read(8)
        if not (signature.startswith(b"\x89PNG\r\n\x1a\n") or signature.startswith(b"\xff\xd8\xff")):
            raise ValueError("Dashboard asset is not a PNG or JPEG image")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target

"""Submit an existing density map and FASTA to CryoNet Mode 22 safely.

CryoNet authenticates uploads by a browser-like cookie/CSRF upload session, not
by an API key. This script keeps that session in memory and writes only
non-sensitive request/response metadata to the selected audit directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://cryonet.ai"
DEFAULT_CA_BUNDLE = os.getenv('CRYOAGENT_CRYONET_CA_BUNDLE', requests.certs.where())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_success(response: requests.Response, context: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("code", "000000")) != "000000" or payload.get("error_code", 0) not in (0, "0", None):
        raise RuntimeError(f"CryoNet {context} failed: {payload}")
    return payload


def csrf_headers(token: str, origin: str) -> dict[str, str]:
    return {"Origin": origin, "X-CSRFToken": token}


def upload_file(
    session: requests.Session,
    base_url: str,
    origin: str,
    csrf_token: str,
    path: Path,
    upload_session_id: str,
    verify: str | bool,
    timeout: int,
    direct_upload: bool = False,
) -> tuple[dict[str, Any], str]:
    authorization = require_success(
        session.post(
            f"{base_url}/api/oss_upload/authorize/",
            headers={**csrf_headers(csrf_token, origin), "Content-Type": "application/json"},
            json={"filename": path.name, "size": path.stat().st_size, "upload_session_id": upload_session_id},
            verify=verify,
            timeout=timeout,
        ),
        f"authorize {path.name}",
    )
    if authorization.get("strategy") != "single":
        raise RuntimeError(
            f"{path} requires multipart upload; this smoke script only supports single uploads."
        )
    upload_urls = [authorization.get("upload_url"), authorization.get("fallback_upload_url")]
    upload_session = requests.Session()
    upload_session.trust_env = False if direct_upload else session.trust_env
    upload_error: Exception | None = None
    for upload_url in filter(None, upload_urls):
        try:
            with path.open("rb") as handle:
                response = upload_session.put(
                    upload_url,
                    data=handle,
                    headers=authorization.get("headers") or {},
                    timeout=timeout,
                )
            response.raise_for_status()
            upload_error = None
            break
        except requests.RequestException as exc:
            upload_error = exc
    if upload_error is not None:
        # Presigned object URLs contain temporary credentials; keep them out of logs.
        raise RuntimeError(f"CryoNet upload failed for {path.name}: {type(upload_error).__name__}") from None

    completed = require_success(
        session.post(
            f"{base_url}/api/oss_upload/complete_file/",
            headers={**csrf_headers(csrf_token, origin), "Content-Type": "application/json"},
            json={"key": authorization["key"], "upload_session_id": authorization["upload_session_id"]},
            verify=verify,
            timeout=timeout,
        ),
        f"complete {path.name}",
    )
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "key": completed["key"],
        "authorization": {key: authorization.get(key) for key in ("strategy", "multipart_threshold", "expires_in")},
        "complete": {key: completed.get(key) for key in ("code", "key", "size", "etag")},
    }, str(completed["upload_session_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--resolution-A", type=float, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--email")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ca-bundle", type=Path, default=Path(DEFAULT_CA_BUNDLE))
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--direct", action="store_true", help="Bypass inherited proxies for this CryoNet task only")
    parser.add_argument("--direct-upload", action="store_true",
                        help="Upload to object storage directly; preserve API proxy settings")
    args = parser.parse_args()

    for path in (args.map, args.fasta, args.ca_bundle):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.resolution_A <= 0:
        raise ValueError("--resolution-A must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    base_url = args.base_url.rstrip("/")
    origin = base_url
    session = requests.Session()
    session.trust_env = not args.direct
    verify: str | bool = str(args.ca_bundle)

    csrf = require_success(
        session.get(f"{base_url}/api/oss_upload/csrf/", verify=verify, timeout=args.timeout_seconds),
        "CSRF bootstrap",
    )
    csrf_token = str(csrf["csrf_token"])
    upload_session_id = ""
    uploaded: dict[str, Any] = {}
    for label, path in (("map", args.map), ("fasta", args.fasta)):
        record, upload_session_id = upload_file(
            session, base_url, origin, csrf_token, path, upload_session_id, verify, args.timeout_seconds,
            direct_upload=args.direct_upload,
        )
        uploaded[label] = record
        save_json(output_dir / "upload_manifest.json", {
            "created_at": utc_now(), "base_url": base_url, "uploaded": uploaded,
            "upload_session_id_present": bool(upload_session_id),
        })

    form = {
        "mode": "22",
        "mapfile": uploaded["map"]["key"],
        "seqfile": uploaded["fasta"]["key"],
        "upload_session_id": upload_session_id,
        "resolution": str(args.resolution_A),
        "name": args.name,
    }
    if args.email:
        form["email"] = args.email
    submission = require_success(
        session.post(
            f"{base_url}/api/create_job/",
            headers=csrf_headers(csrf_token, origin), data=form,
            verify=verify, timeout=args.timeout_seconds,
        ),
        "create Mode 22 job",
    )
    save_json(output_dir / "submission.json", {
        "created_at": utc_now(), "mode": 22,
        "request_fields": {key: value for key, value in form.items() if key != "upload_session_id"},
        "response": submission,
    })
    job_id = submission.get("jobid")
    if job_id:
        status = require_success(
            session.get(f"{base_url}/api/query_job/", params={"jobid": job_id}, verify=verify, timeout=args.timeout_seconds),
            "initial job query",
        )
        save_json(output_dir / "initial_status.json", {"queried_at": utc_now(), "payload": status})
    print(json.dumps({"output_dir": str(output_dir), "jobid": job_id, "status": submission.get("msg")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

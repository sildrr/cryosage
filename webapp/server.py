"""Loopback-only, authenticated standard-library web service. No model keys in the UI."""
import argparse
import hmac
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import threading
import uuid
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from webapp.core import DATA, LOGS, PYTHON, REPO, SHOWCASE, job_path, public_task, submit_task, write

STATIC = Path(__file__).parent / "static"
TOKEN_PATH = Path(os.getenv('CRYOAGENT_WEB_TOKEN_FILE', str(Path.home() / '.config/cryosage/web.token')))
TOKEN = ""
SUBMISSION_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        value = self.headers.get("Authorization", "")
        return hmac.compare_digest(value, f"Bearer {TOKEN}")

    def json_body(self):
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 65536:
            raise ValueError("Invalid request body size")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/api/showcase":
                return self.send_json(SHOWCASE)
            if path == "/api/health":
                from webapp.pilot_pipeline import VERSION
                return self.send_json({"status": "ok", "backend": "existing-agent", "live_submission": True,
                                       "pipeline_version": VERSION})
            if path.startswith("/api/"):
                if not self.authorized():
                    return self.send_json({"error": "Unlock the console with your web access token first"}, 401)
                if path == "/api/tasks":
                    from webapp.lifecycle import list_tasks
                    archived = parse_qs(urlparse(self.path).query).get("archived", ["false"])[0]
                    if archived not in {"true", "false"}:
                        raise ValueError("archived must be true or false")
                    return self.send_json(list_tasks(archived == "true"))
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "tasks"]:
                    from webapp.lifecycle import visible_root
                    visible_root(parts[2])
                if len(parts) == 7 and parts[:2] == ["api", "tasks"] and parts[3] == "jobs" and parts[5] == "assets":
                    from webapp.dashboard import image_file
                    return self.send_file(image_file(parts[2], parts[4], parts[6]))
                if len(parts) == 5 and parts[:2] == ["api", "tasks"] and parts[3] == "jobs":
                    from webapp.activity import job_detail
                    return self.send_json(job_detail(parts[2], parts[4]))
                if len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] == "activity":
                    from webapp.activity import task_activity, control_status
                    return self.send_json({**task_activity(parts[2]), "control": control_status(parts[2])})
                if len(parts) == 5 and parts[:2] == ["api", "tasks"] and parts[3] == "images":
                    import re
                    if not re.fullmatch(r"[a-f0-9]{64}\.(png|jpg|jpeg)", parts[4]):
                        raise ValueError("Invalid image identifier")
                    return self.send_file(job_path(parts[2]) / "ui_images" / parts[4])
                if len(parts) == 3 and parts[:2] == ["api", "tasks"]:
                    value = public_task(parts[2])
                    from webapp.lifecycle import lifecycle_metadata
                    value.update(lifecycle_metadata(job_path(parts[2])))
                    monitor = job_path(parts[2]) / "monitor_error.json"
                    if monitor.exists():
                        value["monitor_warning"] = json.loads(monitor.read_text())
                    return self.send_json(value)
                if len(parts) == 5 and parts[:2] == ["api", "tasks"] and parts[3] == "artifacts":
                    if parts[4] not in {"density.mrc", "report.md"}:
                        raise ValueError("Unknown artifact")
                    return self.send_file(job_path(parts[2]) / parts[4], attachment=True)
                return self.send_json({"error": "Not found"}, 404)
            filename = "index.html" if path in {"/", "/workflow"} else path.lstrip("/")
            file = (STATIC / filename).resolve()
            if not file.is_relative_to(STATIC.resolve()):
                raise ValueError("Invalid asset path")
            self.send_file(file)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except FileNotFoundError:
            self.send_json({"error": "Not found"}, 404)
        except Exception:
            traceback.print_exc()
            self.send_json({"error": "Server operation failed; inspect server logs"}, 503)

    def send_file(self, file, attachment=False):
        with file.open("rb") as stream:
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(file.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{file.name}"')
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

    def do_POST(self):
        try:
            if not self.authorized():
                return self.send_json({"error": "Unauthorized"}, 401)
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send_json({"error": "Cross-origin requests are not permitted"}, 403)
            parsed = urlparse(self.path)
            if parsed.path == "/api/uploads":
                return self.upload(parsed)
            if parsed.path == "/api/tasks":
                with SUBMISSION_LOCK:
                    return self.send_json(submit_task(self.json_body()), 201)
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["api", "tasks"]:
                from webapp.lifecycle import visible_root
                visible_root(parts[2])
            if len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] in {"archive", "delete"}:
                from webapp.lifecycle import archive_task, delete_task
                body = self.json_body()
                with SUBMISSION_LOCK:
                    value = (archive_task(parts[2], body.get("archived")) if parts[3] == "archive"
                             else delete_task(parts[2]))
                    return self.send_json(value)
            if len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] == "control":
                from webapp.activity import submit_intervention
                with SUBMISSION_LOCK:
                    return self.send_json(submit_intervention(parts[2], self.json_body()), 202)
            if len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] == "cryonet":
                return self.cryonet(parts[2], self.json_body())
            self.send_json({"error": "Not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except FileNotFoundError:
            self.send_json({"error": "Requested task or artifact is missing"}, 404)
        except Exception:
            traceback.print_exc()
            self.send_json({"error": "Operation failed; inspect server logs"}, 503)

    def upload(self, parsed):
        query = parse_qs(parsed.query)
        filename = query.get("filename", [""])[0]
        folder = query.get("group", [uuid.uuid4().hex])[0]
        if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid filename")
        if Path(filename).suffix.lower() not in {".mrc", ".mrcs", ".tif", ".tiff", ".eer", ".star"}:
            raise ValueError("Unsupported image format")
        if len(folder) != 32 or any(c not in "0123456789abcdef" for c in folder):
            raise ValueError("Invalid upload group")
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 8 * 1024**3:
            raise ValueError("Each uploaded file must be 1 byte to 8 GiB; use mounted paths for larger data")
        directory = DATA / "uploads" / folder
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / filename
        temp = directory / f".{uuid.uuid4().hex}.part"
        try:
            with temp.open("xb") as stream:
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Upload interrupted")
                    stream.write(chunk)
                    remaining -= len(chunk)
            if file.exists():
                raise ValueError("This upload group already contains that filename")
            temp.replace(file)
            file.chmod(0o644)
        finally:
            temp.unlink(missing_ok=True)
        self.send_json({"group": folder, "filename": filename, "input_path": str(directory / "*"),
                        "file_path": str(file)}, 201)

    def cryonet(self, identifier, body):
        root = job_path(identifier)
        task = public_task(identifier)
        result = task.get("result", {})
        if body.get("external_consent") is not True or result.get("status") != "available":
            raise ValueError("A completed map and explicit CryoNet upload consent are required")
        if result.get("map_kind") != "sharpened":
            raise ValueError("Sharpened map unavailable; inspect the result before model building")
        fasta = str(body.get("fasta", ""))
        if not fasta.startswith(">") or len(fasta) > 100000 or not all(c.isalpha() or c in ">_-* .\r\n0123456789|" for c in fasta):
            raise ValueError("Provide a valid FASTA sequence")
        with SUBMISSION_LOCK:
            checkpoint = root / "cryonet.json"
            if checkpoint.exists():
                raise ValueError("CryoNet submission already exists for this task")
            (root / "sequence.fasta").write_text(fasta)
            output = root / "cryonet_submission"
            command = [PYTHON, "-m", "webapp.cryonet_worker", identifier]
            LOGS.mkdir(parents=True, exist_ok=True)
            with (LOGS / f"cryonet-{identifier}.log").open("a") as log:
                # Persist first so a fast child cannot have its terminal state overwritten.
                write(checkpoint, {"status": "submitting", "url": "https://cryonet.ai"})
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=log, start_new_session=True)
            self.send_json({"status": "submitting", "url": "https://cryonet.ai"}, 202)

    def log_message(self, fmt, *args):
        # Request paths only; tokens and request bodies are never logged.
        print(fmt % args, flush=True)


def main():
    global TOKEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TOKEN_PATH.exists():
        fd = os.open(TOKEN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(secrets.token_urlsafe(32))
    TOKEN_PATH.chmod(0o600)
    TOKEN = TOKEN_PATH.read_text().strip()
    if len(TOKEN) < 24:
        raise ValueError("Web token must be at least 24 characters")
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"CryoSage: http://127.0.0.1:{args.port}; access token file: {TOKEN_PATH}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

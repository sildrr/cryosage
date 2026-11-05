"""Start/stop the private UI without keeping an interactive SSH terminal open."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from webapp.core import LOGS, PYTHON, REPO, write

PIDFILE = LOGS / "server.pid.json"
PORT = int(os.getenv("CRYOAGENT_WEB_PORT", "8777"))


def apply_runtime_profile():
    profile = Path(os.getenv('CRYOAGENT_WEB_RUNTIME_PROFILE', str(Path.home() / '.config/cryosage/web-runtime.json')))
    if not profile.exists():
        return
    values = json.loads(profile.read_text())
    allowed = {
        "CRYOAGENT_WEB_SLURM_PARTITION", "CRYOAGENT_GPU_LANE", "CRYOAGENT_AUTO_SELECT_LANE",
        "OPENAI_BASE_URL", "OPENAI_MODEL", "CRYOAGENT_FIXED_API_MODEL", "CRYOAGENT_API_ENV_FILE",
    }
    if not isinstance(values, dict) or any(k not in allowed or not isinstance(v, str) for k, v in values.items()):
        raise ValueError("Invalid per-user web runtime profile")
    os.environ.update(values)
    if values.get("CRYOAGENT_API_ENV_FILE"):
        # A selected per-user credential file must not be shadowed by an old
        # inherited provider key. Never persist secret values in this profile.
        os.environ.pop("OPENAI_API_KEY", None)


def alive(pid):
    try:
        os.kill(pid, 0)
        return "webapp.server" in open(f"/proc/{pid}/cmdline").read()
    except (OSError, ProcessLookupError):
        return False


def ensure_proxy_relay():
    from webapp.proxy import token
    token(create=True)
    proxy_pid = LOGS / "proxy.pid.json"
    if proxy_pid.exists():
        old = json.loads(proxy_pid.read_text())
        try:
            if "webapp.proxy" in open(f"/proc/{old['pid']}/cmdline").read():
                return
        except OSError:
            pass
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "proxy.log").open("a") as log:
        process = subprocess.Popen([PYTHON, "-u", "-m", "webapp.proxy"], cwd=REPO,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError("Proxy relay failed to start; inspect logs/web/proxy.log")
    write(proxy_pid, {"pid": process.pid})


def restart_proxy_relay():
    from webapp.proxy import upstream_settings
    upstream_settings()
    proxy_pid = LOGS / "proxy.pid.json"
    if proxy_pid.exists():
        old = json.loads(proxy_pid.read_text())
        try:
            if "webapp.proxy" in open(f"/proc/{old['pid']}/cmdline").read():
                os.kill(old["pid"], signal.SIGTERM)
                for _ in range(50):
                    try:
                        command = open(f"/proc/{old['pid']}/cmdline").read()
                    except OSError:
                        break
                    if "webapp.proxy" not in command:
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError("Old proxy relay did not stop; refusing replacement")
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
        proxy_pid.unlink(missing_ok=True)
    ensure_proxy_relay()
    print("Proxy relay restarted. Existing API connections may need retry; Slurm jobs were not stopped.")


def main():
    apply_runtime_profile()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status", "restart-proxy"))
    args = parser.parse_args()
    if args.action == "restart-proxy":
        restart_proxy_relay()
        return
    if args.action == "start":
        from urllib.parse import urlparse
        if any(urlparse(os.getenv(k, '')).hostname in {'localhost', '127.0.0.1', '::1'}
               for k in ('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'https_proxy', 'http_proxy', 'all_proxy')):
            ensure_proxy_relay()
    old = json.loads(PIDFILE.read_text()) if PIDFILE.exists() else {}
    running = bool(old.get("pid") and alive(old["pid"]))
    if args.action == "status":
        print(json.dumps({"running": running, "pid": old.get("pid"), "url": f"http://127.0.0.1:{PORT}"}))
    elif args.action == "stop":
        if running:
            os.kill(old["pid"], signal.SIGTERM)
        PIDFILE.unlink(missing_ok=True)
        print("Web server stopped. Existing Slurm tasks are unaffected.")
    elif running:
        print(f"Web server is already running at http://127.0.0.1:{PORT}")
    else:
        LOGS.mkdir(parents=True, exist_ok=True)
        with (LOGS / "server.log").open("a") as log:
            child = subprocess.Popen([PYTHON, "-u", "-m", "webapp.server", "--port", str(PORT)], cwd=REPO,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError(f"Web server failed to start; inspect {LOGS / 'server.log'}")
        write(PIDFILE, {"pid": child.pid})
        print(f"Web server started, PID {child.pid}, http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    main()

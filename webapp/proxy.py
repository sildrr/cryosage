"""Authenticated LAN relay to the login node's loopback HTTP proxy."""
import base64
import hmac
import os
import secrets
import select
import socket
import socketserver
from pathlib import Path
from urllib.parse import unquote, urlparse

TOKEN_FILE = Path(os.getenv('CRYOAGENT_WEB_PROXY_TOKEN_FILE', str(Path.home() / '.config/cryosage/web-proxy.token')))
HOST = os.getenv("CRYOAGENT_WEB_PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("CRYOAGENT_WEB_PROXY_PORT", "18889"))
ALLOWED_TARGET_HOSTS = frozenset({
    "api.ofox.ai", "cryonet.ai", "www.cryonet.ai",
    "packyapi.ai", "www.packyapi.ai",
})
ALLOWED_TARGET_HOSTS |= frozenset(x.strip() for x in os.getenv(
    'CRYOAGENT_WEB_PROXY_ALLOWED_HOSTS', '').split(',') if x.strip())
API4_DIRECT_TARGET = ('15.204.107.114', 3009)
PACKY_DIRECT_HOSTS = frozenset({'packyapi.ai', 'www.packyapi.ai'})


def packy_direct_target(target, method):
    return (os.getenv('CRYOAGENT_PROXY_DIRECT_PACKY') == '1'
            and method == 'CONNECT' and target.hostname in PACKY_DIRECT_HOSTS
            and (target.port or 443) == 443)


def api4_direct_target(target, method):
    return (method != 'CONNECT' and target.scheme == 'http'
            and (target.hostname, target.port or 80) == API4_DIRECT_TARGET)


def upstream_settings():
    value = os.getenv("CRYOAGENT_WEB_PROXY_UPSTREAM", "http://127.0.0.1:18888")
    parsed = urlparse(value)
    try:
        port = parsed.port or 80
    except ValueError:
        raise ValueError("Invalid upstream HTTP proxy port") from None
    if parsed.scheme != "http" or not parsed.hostname or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("CRYOAGENT_WEB_PROXY_UPSTREAM must be an http://host:port proxy URL (not SOCKS)")
    authorization = None
    if parsed.username is not None:
        credentials = f"{unquote(parsed.username)}:{unquote(parsed.password or '')}"
        authorization = b"Proxy-Authorization: Basic " + base64.b64encode(credentials.encode())
    return (parsed.hostname, port), authorization


def token(create=False):
    if create and not TOKEN_FILE.exists():
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(TOKEN_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(32))
    return TOKEN_FILE.read_text().strip()


def configure_compute_proxy():
    # Only translate loopback proxy URLs in this worker process. Preserve direct
    # LAN bypass configured by cryosparc_client, and preserve non-loopback proxies.
    translated = False
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.getenv(key)
        if value and urlparse(value).hostname in {"127.0.0.1", "localhost", "::1"}:
            os.environ[key] = f"http://cryosage:{token()}@{HOST}:{PORT}"
            translated = True
    if translated and urlparse(os.getenv('OPENAI_BASE_URL', '')).hostname == API4_DIRECT_TARGET[0]:
        # Login-node direct access must not leak into a compute-node bypass.
        # The authenticated relay reaches this exact API4 endpoint directly.
        for key in ('NO_PROXY', 'no_proxy'):
            if key in os.environ:
                os.environ[key] = ','.join(entry for entry in os.environ[key].split(',')
                    if entry.strip() not in {API4_DIRECT_TARGET[0],
                        f'{API4_DIRECT_TARGET[0]}:{API4_DIRECT_TARGET[1]}'})
    return translated


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        incoming = self.request
        incoming.settimeout(15)
        upstream = None
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = incoming.recv(8192)
                if not chunk or len(data) + len(chunk) > 65536:
                    return
                data += chunk
            head, body = data.split(b"\r\n\r\n", 1)
            lines = head.split(b"\r\n")
            authorization = next((line.split(b":", 1)[1].strip() for line in lines[1:]
                                  if line.lower().startswith(b"proxy-authorization:")), b"")
            expected = b"Basic " + base64.b64encode(f"cryosage:{self.server.secret}".encode())
            if not hmac.compare_digest(authorization, expected):
                incoming.sendall(b'HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm="CryoSage"\r\nContent-Length: 0\r\n\r\n')
                return
            # Never permit tunneling to arbitrary LAN services through this relay.
            request_parts = lines[0].decode("ascii").split()
            target = urlparse("//" + request_parts[1]) if request_parts[0] == "CONNECT" else urlparse(request_parts[1])
            direct_api4 = api4_direct_target(target, request_parts[0])
            direct_packy = packy_direct_target(target, request_parts[0])
            if target.hostname not in ALLOWED_TARGET_HOSTS and not direct_api4:
                print(f"Proxy relay denied target host: {target.hostname}", flush=True)
                incoming.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            cleaned = [line for line in lines if not line.lower().startswith(b"proxy-authorization:")]
            if direct_packy:
                address = (target.hostname, 443)
            elif direct_api4:
                # Compute nodes cannot reach this endpoint directly; the login
                # node can. Keep the PC proxy out of this exact API4 route.
                address = API4_DIRECT_TARGET
                path = target.path or '/'
                if target.query:
                    path += '?' + target.query
                cleaned[0] = f'{request_parts[0]} {path} {request_parts[2]}'.encode('ascii')
            else:
                address, upstream_authorization = upstream_settings()
                if upstream_authorization:
                    cleaned.append(upstream_authorization)
            upstream = socket.create_connection(address, timeout=15)
            if direct_packy:
                # TLS remains end-to-end; never send the CONNECT headers or relay secret to the API.
                incoming.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                if body:
                    upstream.sendall(body)
            else:
                upstream.sendall(b"\r\n".join(cleaned) + b"\r\n\r\n" + body)
            incoming.settimeout(None)
            upstream.settimeout(None)
            while True:
                readable, _, _ = select.select([incoming, upstream], [], [], 300)
                if not readable:
                    return
                for connection in readable:
                    chunk = connection.recv(65536)
                    if not chunk:
                        return
                    (upstream if connection is incoming else incoming).sendall(chunk)
        except (OSError, ValueError, IndexError):
            if upstream is None:
                try:
                    incoming.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                except OSError:
                    pass
            return
        finally:
            if upstream:
                upstream.close()


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    upstream_settings()  # Reject unsupported upstream schemes before accepting clients.
    with ProxyServer((HOST, PORT), Relay) as server:
        server.secret = token(create=True)
        print(f"Authenticated model proxy relay listening on {HOST}:{PORT}", flush=True)
        server.serve_forever()

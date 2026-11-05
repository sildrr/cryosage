import base64
import socket
import threading

import pytest

from webapp import proxy
from urllib.parse import urlparse


def test_default_upstream(monkeypatch):
    monkeypatch.delenv("CRYOAGENT_WEB_PROXY_UPSTREAM", raising=False)
    assert proxy.upstream_settings() == (("127.0.0.1", 18888), None)


@pytest.mark.parametrize('host,method,allowed', [
    ('www.packyapi.ai:443','CONNECT',True),
    ('packyapi.ai:443','CONNECT',True),
    ('www.packyapi.ai:80','CONNECT',False),
    ('www.packyapi.ai.evil.test:443','CONNECT',False),
    ('172.16.1.2:443','CONNECT',False),
    ('www.packyapi.ai:443','GET',False),
])
def test_packy_direct_is_explicit_and_exact(monkeypatch,host,method,allowed):
    monkeypatch.delenv('CRYOAGENT_PROXY_DIRECT_PACKY',raising=False)
    assert not proxy.packy_direct_target(urlparse('//'+host),method)
    monkeypatch.setenv('CRYOAGENT_PROXY_DIRECT_PACKY','1')
    assert proxy.packy_direct_target(urlparse('//'+host),method) is allowed


def test_packy_connect_tunnels_bytes_without_pc_proxy_or_credentials(monkeypatch):
    monkeypatch.setenv('CRYOAGENT_PROXY_DIRECT_PACKY','1')
    received=[]
    create_connection=socket.create_connection
    with socket.socket() as target:
        target.bind(('127.0.0.1',0));target.listen();target.settimeout(5)
        def connect(address,timeout):
            assert address==('www.packyapi.ai',443)
            return create_connection(target.getsockname(),timeout=timeout)
        monkeypatch.setattr(proxy.socket,'create_connection',connect)
        monkeypatch.setattr(proxy,'upstream_settings',lambda:pytest.fail('Used PC proxy'))
        def serve():
            connection,_=target.accept()
            with connection:
                connection.settimeout(5)
                received.append(connection.recv(4096))
                connection.sendall(b'test-reply')
        threading.Thread(target=serve,daemon=True).start()
        relay=proxy.ProxyServer(('127.0.0.1',0),proxy.Relay);relay.secret='test-secret'
        threading.Thread(target=relay.serve_forever,daemon=True).start()
        try:
            with create_connection(relay.server_address,timeout=5) as client:
                auth=base64.b64encode(b'cryosage:test-secret')
                client.sendall(b'CONNECT www.packyapi.ai:443 HTTP/1.1\r\nProxy-Authorization: Basic '+auth+b'\r\n\r\n')
                assert b'200 Connection Established' in client.recv(4096)
                client.sendall(b'test-tls-bytes')
                assert client.recv(4096)==b'test-reply'
            assert received==[b'test-tls-bytes']
        finally:
            relay.shutdown();relay.server_close()


@pytest.mark.parametrize('url,method,allowed', [
    ('http://15.204.107.114:3009/v1/chat/completions','POST',True),
    ('http://15.204.107.114:80/v1','POST',False),
    ('https://15.204.107.114:3009/v1','POST',False),
    ('http://15.204.107.114.evil.test:3009/v1','POST',False),
    ('http://15.204.107.114:3009/v1','CONNECT',False),
])
def test_api4_exact_host_port_and_plain_http_only(url,method,allowed):
    assert proxy.api4_direct_target(urlparse(url),method) is allowed


def test_api4_direct_route_rewrites_absolute_url_and_keeps_secret_private(monkeypatch):
    received=[]
    with socket.socket() as target:
        target.bind(('127.0.0.1',0))
        target.listen()
        target.settimeout(5)
        monkeypatch.setattr(proxy,'API4_DIRECT_TARGET',target.getsockname())
        def no_pc_proxy():
            raise AssertionError('API4 must not use the PC upstream proxy')
        monkeypatch.setattr(proxy,'upstream_settings',no_pc_proxy)
        def serve():
            connection,_=target.accept()
            with connection:
                connection.settimeout(5)
                received.append(connection.recv(4096))
                connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK')
        worker=threading.Thread(target=serve,daemon=True)
        worker.start()
        relay=proxy.ProxyServer(('127.0.0.1',0),proxy.Relay)
        relay.secret='private-relay-token'
        threading.Thread(target=relay.serve_forever,daemon=True).start()
        try:
            with socket.create_connection(relay.server_address,timeout=5) as client:
                auth=base64.b64encode(b'cryosage:private-relay-token')
                client.sendall(f'POST http://127.0.0.1:{target.getsockname()[1]}/v1/chat/completions?probe=1 HTTP/1.1\r\n'.encode()+
                    b'Proxy-Authorization: Basic '+auth+b'\r\nAuthorization: Bearer synthetic-key\r\nContent-Length: 2\r\n\r\n{}')
                assert b'200 OK' in client.recv(4096)
            worker.join(5)
            assert received[0].startswith(b'POST /v1/chat/completions?probe=1 HTTP/1.1')
            assert auth not in received[0]
            assert b'Authorization: Bearer synthetic-key' in received[0]
        finally:
            relay.shutdown()
            relay.server_close()


def test_configurable_upstream_credentials(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_WEB_PROXY_UPSTREAM", "http://user:p%40ss@localhost:18891")
    address, auth = proxy.upstream_settings()
    assert address == ("localhost", 18891)
    assert auth == b"Proxy-Authorization: Basic " + base64.b64encode(b"user:p@ss")


@pytest.mark.parametrize("url", ["socks5://localhost:7890", "https://localhost:7890", "http://localhost:no", "http://localhost:9/path", "http://"])
def test_reject_unsupported_upstream(monkeypatch, url):
    monkeypatch.setenv("CRYOAGENT_WEB_PROXY_UPSTREAM", url)
    with pytest.raises(ValueError):
        proxy.upstream_settings()


@pytest.mark.parametrize("request_line", [
    b"GET http://api.ofox.ai/v1/models HTTP/1.1",
    b"CONNECT www.packyapi.ai:443 HTTP/1.1",
    b"CONNECT packyapi.ai:443 HTTP/1.1",
    b"CONNECT cryonet.ai:443 HTTP/1.1",
])
def test_forward_to_configured_upstream_without_relay_secret(monkeypatch, request_line):
    received = []
    with socket.socket() as upstream:
        upstream.bind(("127.0.0.1", 0))
        upstream.listen()
        upstream.settimeout(5)
        monkeypatch.setenv("CRYOAGENT_WEB_PROXY_UPSTREAM", f"http://user:pass@127.0.0.1:{upstream.getsockname()[1]}")

        def serve():
            connection, _ = upstream.accept()
            with connection:
                connection.settimeout(5)
                data = b""
                while b"\r\n\r\n" not in data:
                    data += connection.recv(4096)
                received.append(data)
                connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        relay = proxy.ProxyServer(("127.0.0.1", 0), proxy.Relay)
        relay.secret = "private-relay-token"
        thread = threading.Thread(target=relay.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.create_connection(relay.server_address, timeout=5) as client:
                auth = base64.b64encode(b"cryosage:private-relay-token")
                client.sendall(request_line + b"\r\nProxy-Authorization: Basic " + auth + b"\r\n\r\n")
                assert b"200 OK" in client.recv(4096)
            worker.join(5)
            assert len(received) == 1
            assert base64.b64encode(b"user:pass") in received[0]
            assert auth not in received[0]
        finally:
            relay.shutdown()
            relay.server_close()


@pytest.mark.parametrize("host,authenticated,status", [
    ("172.16.1.2:61000", True, b"403 Forbidden"),
    ("www.packyapi.ai.evil.test:443", True, b"403 Forbidden"),
    ("unrelated.test:443", True, b"403 Forbidden"),
    ("www.packyapi.ai:443", False, b"407 Proxy Authentication Required"),
])
def test_new_provider_does_not_bypass_target_or_auth_checks(host, authenticated, status):
    relay = proxy.ProxyServer(("127.0.0.1", 0), proxy.Relay)
    relay.secret = "test-secret"
    thread = threading.Thread(target=relay.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(relay.server_address, timeout=5) as client:
            auth = base64.b64encode(b"cryosage:test-secret") if authenticated else b"invalid"
            client.sendall(f"CONNECT {host} HTTP/1.1\r\n".encode() +
                           b"Proxy-Authorization: Basic " + auth + b"\r\n\r\n")
            assert status in client.recv(4096)
    finally:
        relay.shutdown()
        relay.server_close()

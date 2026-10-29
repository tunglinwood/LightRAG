"""Offline tests for the MCP server bootstrap: argv sanitizing, dotenv, lifespan."""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from lightrag.mcp import server as mcp_server

pytestmark = pytest.mark.offline


class _FakeMCPServer:
    """Records construction; run() is a no-op so main() returns in tests."""

    instances = []

    def __init__(self, name, lifespan=None):
        self.name = name
        self.lifespan = lifespan
        self.ran = False
        _FakeMCPServer.instances.append(self)

    def run(self, transport="stdio", **kwargs):
        self.ran = True
        self.transport = transport
        self.run_kwargs = kwargs


def _mock_bootstrap(monkeypatch, tmp_path):
    """Patch everything env/network-touching so main() runs offline."""
    monkeypatch.setattr(sys, "argv", ["lightrag-mcp"])
    monkeypatch.setattr(
        mcp_server,
        "initialize_config",
        lambda: SimpleNamespace(working_dir=str(tmp_path)),
    )
    monkeypatch.setattr(mcp_server, "MCPServer", _FakeMCPServer)
    monkeypatch.setattr(mcp_server, "register_tools", lambda mcp: None)
    _FakeMCPServer.instances.clear()


class FakeRag:
    def __init__(self):
        self.initialized = False
        self.finalized = False

    async def initialize_storages(self):
        self.initialized = True

    async def finalize_storages(self):
        self.finalized = True


def test_main_sanitizes_argv_before_initialize_config(monkeypatch, tmp_path):
    captured = {}
    _mock_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["lightrag-mcp", "--port", "1", "--json"])

    real_initialize = mcp_server.initialize_config

    def spy_initialize_config():
        captured["argv"] = list(sys.argv)
        return real_initialize()

    monkeypatch.setattr(mcp_server, "initialize_config", spy_initialize_config)

    mcp_server.main()

    assert captured["argv"] == ["lightrag-mcp"]
    assert _FakeMCPServer.instances[0].name == "lightrag"
    assert _FakeMCPServer.instances[0].ran


def test_bootstrap_restores_stdout_fd(monkeypatch, tmp_path):
    _mock_bootstrap(monkeypatch, tmp_path)
    before = os.fstat(1)
    try:
        mcp_server._run_bootstrap()
    finally:
        after = os.fstat(1)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_bootstrap_diverts_stdout_during_config_parse(monkeypatch, tmp_path):
    """While bootstrap runs, fd 1 must point at fd 2 (stderr), not the wire."""
    _mock_bootstrap(monkeypatch, tmp_path)
    before = os.fstat(1)[:2]
    seen = {}

    def spy_initialize_config():
        seen["stdout_is_stderr"] = os.fstat(1)[:2] == os.fstat(2)[:2]
        return SimpleNamespace(working_dir=str(tmp_path))

    monkeypatch.setattr(mcp_server, "initialize_config", spy_initialize_config)
    try:
        mcp_server._run_bootstrap()
    finally:
        after = os.fstat(1)[:2]
    assert seen["stdout_is_stderr"]
    assert after == before


def test_main_honors_dotenv_path(monkeypatch, tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text("X=1\n")
    _mock_bootstrap(monkeypatch, tmp_path)

    monkeypatch.setenv("DOTENV_PATH", str(env_file))
    calls = []
    real_load_dotenv = mcp_server.load_dotenv

    def spy_load_dotenv(*args, **kwargs):
        calls.append(kwargs.get("dotenv_path"))
        return real_load_dotenv(*args, **kwargs)

    monkeypatch.setattr(mcp_server, "load_dotenv", spy_load_dotenv)

    mcp_server.main()

    assert calls == [str(env_file)]


def test_lifespan_initializes_then_finalizes(monkeypatch, tmp_path):
    args = SimpleNamespace(working_dir=str(tmp_path))
    rag = FakeRag()
    built = {}

    def fake_build(a):
        built["args"] = a
        return rag

    monkeypatch.setattr(mcp_server, "build_rag_from_args", fake_build)

    lifespan = mcp_server._lifespan(args)

    async def drive():
        async with lifespan(None) as ctx:
            assert rag.initialized
            assert not rag.finalized
            assert ctx["rag"] is rag
            assert ctx["args"] is args
        assert rag.finalized

    asyncio.run(drive())
    assert built["args"] is args


def test_lifespan_finalizes_even_on_exception(monkeypatch, tmp_path):
    args = SimpleNamespace(working_dir=str(tmp_path))
    rag = FakeRag()
    monkeypatch.setattr(mcp_server, "build_rag_from_args", lambda a: rag)

    lifespan = mcp_server._lifespan(args)

    async def drive():
        with pytest.raises(RuntimeError, match="tool crashed"):
            async with lifespan(None):
                raise RuntimeError("tool crashed")
        assert rag.finalized

    asyncio.run(drive())


class TestTransportSettings:
    def test_defaults_to_stdio(self, monkeypatch):
        monkeypatch.delenv("LIGHTRAG_MCP_TRANSPORT", raising=False)
        transport, kwargs = mcp_server._transport_settings()
        assert transport == "stdio"
        assert kwargs == {}

    def test_streamable_http_uses_default_host_and_port(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_MCP_TRANSPORT", "streamable-http")
        monkeypatch.delenv("LIGHTRAG_MCP_HOST", raising=False)
        monkeypatch.delenv("LIGHTRAG_MCP_PORT", raising=False)
        transport, kwargs = mcp_server._transport_settings()
        assert transport == "streamable-http"
        assert kwargs == {"host": "127.0.0.1", "port": 9623}

    def test_streamable_http_honors_host_and_port(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("LIGHTRAG_MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("LIGHTRAG_MCP_PORT", "9700")
        transport, kwargs = mcp_server._transport_settings()
        assert kwargs == {"host": "0.0.0.0", "port": 9700}

    def test_unknown_transport_raises(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_MCP_TRANSPORT", "tcp")
        with pytest.raises(ValueError, match="unknown LIGHTRAG_MCP_TRANSPORT"):
            mcp_server._transport_settings()


def test_main_passes_transport_kwargs_to_run(monkeypatch, tmp_path):
    _mock_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setenv("LIGHTRAG_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("LIGHTRAG_MCP_PORT", "9701")

    mcp_server.main()

    server = _FakeMCPServer.instances[0]
    assert server.ran
    assert server.transport == "streamable-http"
    assert server.run_kwargs == {"host": "127.0.0.1", "port": 9701}

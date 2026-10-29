"""Bootstrap and run the LightRAG MCP server.

Transports (selected via ``LIGHTRAG_MCP_TRANSPORT``):
- ``stdio`` (default): the client spawns the server and owns its lifecycle —
  one process per client session, nothing to run persistently.
- ``streamable-http``: a persistent daemon listening on
  ``LIGHTRAG_MCP_HOST``/``LIGHTRAG_MCP_PORT`` (default 127.0.0.1:9623) that
  any number of clients can reach over HTTP, like the REST servers.

Rules for callers:
- Run ``lightrag-mcp`` from the repo root (or point ``DOTENV_PATH`` at the
  server's ``.env``) so LLM/embedding/storage configuration resolves exactly
  like the REST server's.
- stdout is the MCP protocol channel: never print to it. The ``lightrag``
  logger already writes to stderr (``SafeStreamHandler``), so startup logs do
  not corrupt the protocol. Bootstrap additionally diverts fd 1 to stderr
  while config parsing runs (pipmaster's spinner writes to stdout), restoring
  it before the transport claims the wire; tool-time lazy installs are safe
  because the transport holds that claim for the whole session.
- Bad environment fails fast before the protocol starts (argparse validators
  may call ``sys.exit``) — the client sees the process die, not a broken
  session.
"""

import os
import sys
from contextlib import asynccontextmanager, suppress
from typing import Any

from dotenv import load_dotenv
from mcp.server import MCPServer

from lightrag.api.config import initialize_config
from lightrag.api.rag_factory import build_rag_from_args
from lightrag.utils import logger

from .tools import register_tools

DEFAULT_MCP_HTTP_PORT = 9623


def main() -> None:
    """Console entry point: configure from env, then serve MCP."""
    mcp = _run_bootstrap()
    transport, kwargs = _transport_settings()
    endpoint = (
        f"{kwargs['host']}:{kwargs['port']}" if transport == "streamable-http" else "stdio"
    )
    logger.info(f"lightrag-mcp starting ({transport}, {endpoint})")
    mcp.run(transport, **kwargs)


def _transport_settings() -> tuple[str, dict[str, Any]]:
    """Resolve the transport from the environment.

    Returns ``(transport, run_kwargs)`` for ``MCPServer.run``. stdio is the
    default (client-spawned); streamable-http turns the server into a
    persistent daemon. Unknown values raise instead of silently falling
    back, so a typo does not masquerade as a working stdio server.
    """
    transport = os.environ.get("LIGHTRAG_MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "stdio":
        return "stdio", {}
    if transport == "streamable-http":
        host = os.environ.get("LIGHTRAG_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("LIGHTRAG_MCP_PORT", str(DEFAULT_MCP_HTTP_PORT)))
        return transport, {"host": host, "port": port}
    raise ValueError(
        f"unknown LIGHTRAG_MCP_TRANSPORT {transport!r} (expected 'stdio' or 'streamable-http')"
    )


def _run_bootstrap() -> MCPServer:
    """Configure the server; keep fd-1 pollution off the protocol wire.

    Config parsing auto-installs missing provider packages (pipmaster), whose
    spinner writes to stdout. stdout is the MCP wire, so during bootstrap fd 1
    is diverted to stderr and restored before ``mcp.run()`` — the transport
    then claims fd 1 itself and redirects stray output for the session's
    remainder (handlers run under that claim, so tool-time lazy installs are
    already safe). The diversion is best-effort: if the dup fails, bootstrap
    output goes to the wire exactly as it would unpatched.
    """
    saved_stdout: int | None = None
    with suppress(OSError):
        saved_stdout = os.dup(1)
        os.dup2(2, 1)
    try:
        load_dotenv(dotenv_path=os.environ.get("DOTENV_PATH", ".env"), override=False)
        # Keep argparse off the MCP client's argv (the transport passes its
        # own arguments through sys.argv when the server is spawned).
        sys.argv = [sys.argv[0]]
        args = initialize_config()
        mcp = MCPServer("lightrag", lifespan=_lifespan(args))
        register_tools(mcp)
        return mcp
    finally:
        if saved_stdout is not None:
            with suppress(OSError):
                os.dup2(saved_stdout, 1)
                os.close(saved_stdout)


def _lifespan(args):
    """Own the RAG instance lifecycle: construct, init storages, guarantee finalize."""

    @asynccontextmanager
    async def lifespan(server):
        rag = build_rag_from_args(args)
        await rag.initialize_storages()
        try:
            yield {"rag": rag, "args": args}
        finally:
            await rag.finalize_storages()

    return lifespan


if __name__ == "__main__":
    main()

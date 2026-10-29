"""MCP tools over a factory-built LightRAG instance.

Design: every tool has a module-level pure async function taking ``rag`` as
its first argument (directly unit-testable, no SDK objects), and
``register_tools`` wraps each in a thin ``@mcp.tool()`` shell that fetches
``rag`` from the lifespan context and delegates. Anticipated failures map to
``ToolError`` (message reaches the client, no traceback); unexpected errors
propagate so they surface as server crashes rather than silent wrong answers.
"""

import json
from typing import Literal, Optional
from uuid import uuid4

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from lightrag import QueryParam
from lightrag.base import DocStatus
from lightrag.constants import MAX_QUERY_CHARS, MAX_QUERY_TOP_K
from lightrag.utils import logger

_QUERY_MODES = ("local", "global", "hybrid", "naive", "mix", "bypass")
_DOC_STATUSES = ("PENDING", "PARSING", "ANALYZING", "PROCESSING", "PROCESSED", "FAILED")


def _rag(ctx: Context):
    return ctx.request_context.lifespan_context["rag"]


def _doc_to_json(doc_id: str, status) -> dict:
    """Serialize a (doc_id, DocProcessingStatus) row defensively.

    Reads fields via getattr so a duck-typed status (or a row from a backend
    that stores a superset of fields) still serializes. DocStatus is emitted
    as its wire value ("processed"), not the enum repr ("DocStatus.PROCESSED").
    """
    doc_status = getattr(status, "status", None)
    if isinstance(doc_status, DocStatus):
        doc_status = doc_status.value
    return {
        "id": doc_id,
        "status": doc_status,
        "file_path": getattr(status, "file_path", None),
        "track_id": getattr(status, "track_id", None),
        "created_at": getattr(status, "created_at", None),
        "updated_at": getattr(status, "updated_at", None),
        "error_msg": getattr(status, "error_msg", None),
    }


async def run_query(
    rag,
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix",
    response_type: Optional[str] = None,
    only_need_context: bool = False,
    top_k: Optional[int] = None,
    chunk_top_k: Optional[int] = None,
) -> str:
    """Query the LightRAG knowledge base.

    mode selects the retrieval strategy (local/global/hybrid/naive/mix/bypass).
    top_k and chunk_top_k bound retrieval breadth (1..1000); unset keeps the
    server's configured defaults. only_need_context returns the retrieved
    context without LLM generation. response_type overrides the answer format
    (e.g. "Bullet Points").
    """
    if not 1 <= len(query) <= MAX_QUERY_CHARS:
        raise ToolError(
            f"query length must be between 1 and {MAX_QUERY_CHARS} characters, got {len(query)}"
        )
    for name, value in (("top_k", top_k), ("chunk_top_k", chunk_top_k)):
        if value is not None and not 1 <= value <= MAX_QUERY_TOP_K:
            raise ToolError(f"{name} must be between 1 and {MAX_QUERY_TOP_K}, got {value}")

    param = QueryParam(
        mode=mode,
        stream=False,
        only_need_context=only_need_context,
        **({"response_type": response_type} if response_type else {}),
        **({"top_k": top_k} if top_k else {}),
        **({"chunk_top_k": chunk_top_k} if chunk_top_k else {}),
    )
    try:
        return await rag.aquery(query, param=param)
    except Exception as e:
        logger.error(f"MCP query failed: {e}")
        raise ToolError(f"Query failed: {e}") from e


async def insert_text(rag, text: str, file_source: str) -> str:
    """Insert a text document into the knowledge base.

    file_source names the document for citation/dedup (e.g. "notes.md").
    Returns JSON with the processing status and a track_id for progress
    polling via get_insert_status.
    """
    text = text.strip()
    if not text:
        raise ToolError("text must not be empty")
    if not file_source.strip():
        raise ToolError("file_source must not be empty")

    track_id = f"mcp-{uuid4().hex[:12]}"
    try:
        await rag.apipeline_enqueue_documents(
            input=[text], file_paths=[file_source], track_id=track_id
        )
        await rag.apipeline_process_enqueue_documents()
    except Exception as e:
        logger.error(f"MCP insert failed (track_id={track_id}): {e}")
        raise ToolError(f"Insert failed: {e}") from e
    return json.dumps(
        {"status": "processed", "track_id": track_id, "file_source": file_source}
    )


async def list_documents(
    rag,
    status_filter: Optional[
        Literal["PENDING", "PARSING", "ANALYZING", "PROCESSING", "PROCESSED", "FAILED"]
    ] = None,
    page: int = 1,
    page_size: int = 50,
) -> str:
    """List documents with pagination (page_size clamped to 10..200).

    Returns JSON with the page of documents, the total count, and per-status
    counts. status_filter restricts to one processing status.
    """
    if status_filter is not None:
        try:
            status = DocStatus[status_filter]
        except KeyError:
            raise ToolError(
                f"invalid status_filter {status_filter!r}; expected one of {_DOC_STATUSES}"
            ) from None
    else:
        status = None
    page_size = max(10, min(200, page_size))
    page = max(1, page)

    try:
        rows, total = await rag.doc_status.get_docs_paginated(
            status_filter=status, page=page, page_size=page_size
        )
        counts = await rag.doc_status.get_all_status_counts()
    except Exception as e:
        logger.error(f"MCP list_documents failed: {e}")
        raise ToolError(f"List documents failed: {e}") from e
    return json.dumps(
        {
            "documents": [_doc_to_json(doc_id, s) for doc_id, s in rows],
            "total": total,
            "page": page,
            "status_counts": counts,
        }
    )


async def get_insert_status(rag, track_id: str) -> str:
    """Get processing status for every document sharing a track_id.

    Returns JSON in the same document shape as list_documents. Unknown
    track_ids are an error so callers do not mistake a typo for an empty
    (still-processing) batch.
    """
    try:
        docs = await rag.aget_docs_by_track_id(track_id)
    except Exception as e:
        logger.error(f"MCP get_insert_status failed (track_id={track_id}): {e}")
        raise ToolError(f"Get insert status failed: {e}") from e
    if not docs:
        raise ToolError(f"unknown track_id: {track_id}")
    return json.dumps(
        {"track_id": track_id, "documents": [_doc_to_json(i, s) for i, s in docs.items()]}
    )


def register_tools(mcp) -> None:
    """Register the four tools on an MCPServer instance.

    Each wrapper only resolves ``rag`` from the lifespan context; all logic
    lives in the module-level functions above.
    """

    @mcp.tool()
    async def query(
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix",
        response_type: Optional[str] = None,
        only_need_context: bool = False,
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
        ctx: Context = None,
    ) -> str:
        """Query the LightRAG knowledge base (mode: local/global/hybrid/naive/mix/bypass)."""
        return await run_query(
            _rag(ctx), query, mode, response_type, only_need_context, top_k, chunk_top_k
        )

    @mcp.tool(name="insert_text")
    async def insert_text_tool(text: str, file_source: str, ctx: Context = None) -> str:
        """Insert a text document into the knowledge base under the given source name."""
        return await insert_text(_rag(ctx), text, file_source)

    @mcp.tool(name="list_documents")
    async def list_documents_tool(
        status_filter: Optional[
            Literal[
                "PENDING", "PARSING", "ANALYZING", "PROCESSING", "PROCESSED", "FAILED"
            ]
        ] = None,
        page: int = 1,
        page_size: int = 50,
        ctx: Context = None,
    ) -> str:
        """List documents with pagination; optional status_filter and per-status counts."""
        return await list_documents(_rag(ctx), status_filter, page, page_size)

    @mcp.tool(name="get_insert_status")
    async def get_insert_status_tool(track_id: str, ctx: Context = None) -> str:
        """Get processing status for a track_id returned by insert_text."""
        return await get_insert_status(_rag(ctx), track_id)

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

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from lightrag import QueryParam
from lightrag.base import DocStatus
from lightrag.constants import MAX_QUERY_CHARS, MAX_QUERY_TOP_K
from lightrag.utils import logger

_DOC_STATUSES = ("PENDING", "PARSING", "ANALYZING", "PROCESSING", "PROCESSED", "FAILED")


def _rag(ctx: Context):
    return ctx.request_context.lifespan_context["rag"]


def _validate_query(query: str, top_k: Optional[int], chunk_top_k: Optional[int]) -> None:
    """Shared bounds for the query tools, checked before any storage touch."""
    if not 1 <= len(query) <= MAX_QUERY_CHARS:
        raise ToolError(
            f"query length must be between 1 and {MAX_QUERY_CHARS} characters, got {len(query)}"
        )
    for name, value in (("top_k", top_k), ("chunk_top_k", chunk_top_k)):
        if value is not None and not 1 <= value <= MAX_QUERY_TOP_K:
            raise ToolError(f"{name} must be between 1 and {MAX_QUERY_TOP_K}, got {value}")


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


async def query_data(
    rag,
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix",
    top_k: Optional[int] = None,
    chunk_top_k: Optional[int] = None,
) -> str:
    """Retrieve raw structured graph data without LLM generation.

    Returns the same JSON the REST POST /query/data endpoint serves:
    retrieved entities, relationships, chunks, references, plus retrieval
    metadata (query mode, extracted keywords, processing counts). Use this
    when you want the knowledge-graph evidence itself — to feed your own
    reasoning or chain into another tool — instead of a synthesized answer.
    top_k and chunk_top_k bound retrieval breadth (1..1000); unset keeps the
    server's configured defaults.
    """
    _validate_query(query, top_k, chunk_top_k)

    param = QueryParam(
        mode=mode,
        stream=False,
        **({"top_k": top_k} if top_k else {}),
        **({"chunk_top_k": chunk_top_k} if chunk_top_k else {}),
    )
    try:
        result = await rag.aquery_data(query, param=param)
    except Exception as e:
        logger.error(f"MCP query_data failed: {e}")
        raise ToolError(f"Query data failed: {e}") from e
    return json.dumps(result)


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


def register_tools(mcp) -> None:
    """Register the tools on an MCPServer instance.

    Each wrapper only resolves ``rag`` from the lifespan context; all logic
    lives in the module-level functions above.
    """

    @mcp.tool(name="query_data")
    async def query_data_tool(
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix",
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
        ctx: Context = None,
    ) -> str:
        """Retrieve structured entities/relationships/chunks without LLM generation (same data as POST /query/data)."""
        return await query_data(_rag(ctx), query, mode, top_k, chunk_top_k)

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

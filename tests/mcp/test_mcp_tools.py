"""Offline unit tests for the MCP tool functions (pure, rag-first signatures).

FakeRag is duck-typed on purpose: LightRAG is @final and these tests must not
need real storages or a live LLM.
"""

import json
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.constants import MAX_QUERY_CHARS, MAX_QUERY_TOP_K
from lightrag.mcp.tools import (
    list_documents,
    query_data,
)

pytestmark = pytest.mark.offline


def _doc(status=DocStatus.PROCESSED, track_id="t-1"):
    return DocProcessingStatus(
        content_summary="hello world",
        content_length=100,
        file_path="notes.md",
        status=status,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-02T00:00:00",
        track_id=track_id,
    )


class FakeRag:
    def __init__(self):
        self.seen_params = []
        self.fail_query = False

    async def aquery_data(self, query, param=None):
        if self.fail_query:
            raise RuntimeError("backend exploded")
        self.seen_params.append((query, param))
        return {
            "status": "success",
            "message": "Query executed successfully",
            "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
            "metadata": {"query_mode": param.mode if param else None, "keywords": {}},
        }

    @property
    def doc_status(self):
        rag = self

        class _DocStatus:
            async def get_docs_paginated(
                self, status_filter=None, page=1, page_size=50, **kwargs
            ):
                rag.paginated_call = {
                    "status_filter": status_filter,
                    "page": page,
                    "page_size": page_size,
                }
                return rag.paginated_rows, rag.paginated_total

            async def get_all_status_counts(self):
                return rag.status_counts

        return _DocStatus()


def fake_ctx(rag):
    return SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"rag": rag})
    )


class TestQueryData:
    async def test_returns_structured_json_and_passes_params(self):
        rag = FakeRag()
        result = json.loads(await query_data(rag, "what is X?", mode="hybrid", top_k=5))
        assert result["status"] == "success"
        assert result["data"]["entities"] == []
        query, param = rag.seen_params[0]
        assert query == "what is X?"
        assert param.mode == "hybrid"
        assert param.top_k == 5
        assert param.stream is False
        assert result["metadata"]["query_mode"] == "hybrid"

    async def test_query_length_and_top_k_bounds_enforced(self):
        rag = FakeRag()
        with pytest.raises(ToolError):
            await query_data(rag, "")
        with pytest.raises(ToolError):
            await query_data(rag, "x" * (MAX_QUERY_CHARS + 1))
        with pytest.raises(ToolError):
            await query_data(rag, "q", top_k=0)
        with pytest.raises(ToolError):
            await query_data(rag, "q", top_k=MAX_QUERY_TOP_K + 1)
        with pytest.raises(ToolError):
            await query_data(rag, "q", chunk_top_k=-1)
        assert rag.seen_params == []

    async def test_backend_failure_maps_to_tool_error(self):
        rag = FakeRag()
        rag.fail_query = True
        with pytest.raises(ToolError, match="Query data failed: backend exploded"):
            await query_data(rag, "q")


class TestListDocuments:
    async def test_serializes_rows_and_counts(self):
        rag = FakeRag()
        rag.paginated_rows = [("doc-1", _doc()), ("doc-2", _doc(DocStatus.FAILED))]
        rag.paginated_total = 2
        rag.status_counts = {"processed": 1, "failed": 1}
        result = json.loads(await list_documents(rag))
        assert result["total"] == 2
        assert result["status_counts"] == {"processed": 1, "failed": 1}
        assert [d["id"] for d in result["documents"]] == ["doc-1", "doc-2"]
        assert result["documents"][0]["file_path"] == "notes.md"
        assert result["documents"][0]["track_id"] == "t-1"
        # DocStatus serializes as its wire value, not the enum repr
        assert result["documents"][0]["status"] == "processed"
        assert result["documents"][1]["status"] == "failed"

    async def test_status_filter_maps_to_enum(self):
        rag = FakeRag()
        rag.paginated_rows = []
        rag.paginated_total = 0
        rag.status_counts = {}
        await list_documents(rag, status_filter="FAILED")
        assert rag.paginated_call["status_filter"] == DocStatus.FAILED

    async def test_invalid_status_filter_rejected_before_storage_touch(self):
        rag = FakeRag()
        with pytest.raises(ToolError):
            await list_documents(rag, status_filter="BOGUS")
        assert not hasattr(rag, "paginated_call")

    async def test_page_size_clamped(self):
        rag = FakeRag()
        rag.paginated_rows = []
        rag.paginated_total = 0
        rag.status_counts = {}
        await list_documents(rag, page_size=1)
        assert rag.paginated_call["page_size"] == 10
        await list_documents(rag, page_size=9999)
        assert rag.paginated_call["page_size"] == 200

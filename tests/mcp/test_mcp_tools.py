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
    get_insert_status,
    insert_text,
    list_documents,
    run_query,
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
        self.enqueue_calls = []
        self.process_calls = 0
        self.fail_query = False
        self.docs_by_track = {}

    async def aquery(self, query, param=None):
        if self.fail_query:
            raise RuntimeError("backend exploded")
        self.seen_params.append((query, param))
        return "canned answer"

    async def apipeline_enqueue_documents(self, **kwargs):
        self.enqueue_calls.append(kwargs)
        return "ok"

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1

    async def aget_docs_by_track_id(self, track_id):
        return self.docs_by_track.get(track_id, {})

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


class TestRunQuery:
    async def test_mode_and_top_k_passthrough(self):
        rag = FakeRag()
        result = await run_query(rag, "what is X?", mode="local", top_k=5)
        assert result == "canned answer"
        query, param = rag.seen_params[0]
        assert query == "what is X?"
        assert param.mode == "local"
        assert param.top_k == 5
        assert param.stream is False

    async def test_defaults_and_optional_overrides(self):
        rag = FakeRag()
        await run_query(rag, "q")
        _, param = rag.seen_params[0]
        assert param.mode == "mix"
        assert param.response_type == "Multiple Paragraphs"  # QueryParam default

        await run_query(rag, "q", response_type="Bullet Points", chunk_top_k=7)
        _, param2 = rag.seen_params[1]
        assert param2.response_type == "Bullet Points"
        assert param2.chunk_top_k == 7

    async def test_empty_and_oversized_query_rejected(self):
        rag = FakeRag()
        with pytest.raises(ToolError):
            await run_query(rag, "")
        with pytest.raises(ToolError):
            await run_query(rag, "x" * (MAX_QUERY_CHARS + 1))
        assert rag.seen_params == []

    async def test_top_k_bounds(self):
        rag = FakeRag()
        with pytest.raises(ToolError):
            await run_query(rag, "q", top_k=0)
        with pytest.raises(ToolError):
            await run_query(rag, "q", top_k=MAX_QUERY_TOP_K + 1)
        with pytest.raises(ToolError):
            await run_query(rag, "q", chunk_top_k=-1)

    async def test_backend_failure_maps_to_tool_error(self):
        rag = FakeRag()
        rag.fail_query = True
        with pytest.raises(ToolError, match="Query failed: backend exploded"):
            await run_query(rag, "q")


class TestInsertText:
    async def test_enqueue_then_process(self):
        rag = FakeRag()
        result = json.loads(await insert_text(rag, "  some text  ", "notes.md"))
        assert result["status"] == "processed"
        assert result["file_source"] == "notes.md"
        assert result["track_id"].startswith("mcp-")
        assert rag.enqueue_calls[0]["input"] == ["some text"]
        assert rag.enqueue_calls[0]["file_paths"] == ["notes.md"]
        assert rag.enqueue_calls[0]["track_id"] == result["track_id"]
        assert rag.process_calls == 1

    async def test_whitespace_only_rejected(self):
        rag = FakeRag()
        with pytest.raises(ToolError):
            await insert_text(rag, "   \n ", "notes.md")
        assert rag.enqueue_calls == []

    async def test_empty_file_source_rejected(self):
        rag = FakeRag()
        with pytest.raises(ToolError):
            await insert_text(rag, "text", "  ")

    async def test_pipeline_failure_maps_to_tool_error(self):
        rag = FakeRag()

        async def boom(**kwargs):
            raise RuntimeError("pipeline down")

        rag.apipeline_enqueue_documents = boom
        with pytest.raises(ToolError, match="Insert failed: pipeline down"):
            await insert_text(rag, "text", "notes.md")


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


class TestGetInsertStatus:
    async def test_unknown_track_id_is_error(self):
        rag = FakeRag()
        with pytest.raises(ToolError, match="unknown track_id"):
            await get_insert_status(rag, "nope")

    async def test_serializes_matching_docs(self):
        rag = FakeRag()
        rag.docs_by_track["mcp-abc"] = {"doc-1": _doc(track_id="mcp-abc")}
        result = json.loads(await get_insert_status(rag, "mcp-abc"))
        assert result["track_id"] == "mcp-abc"
        assert result["documents"][0]["id"] == "doc-1"

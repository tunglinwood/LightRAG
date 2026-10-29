"""First-party MCP server: expose a factory-built LightRAG instance over stdio.

The server reuses ``lightrag.api.rag_factory.build_rag_from_args`` so every
storage backend selectable via environment variables works here exactly as it
does for the REST server — there is no backend-specific code in this package.
"""

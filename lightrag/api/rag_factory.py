"""Build a server-equivalent LightRAG instance from parsed configuration arguments.

Rules for callers:
- Pass the argparse Namespace produced by ``lightrag.api.config.parse_args``
  (or ``initialize_config``) so env-driven LLM/embedding/storage selection is
  honored. Every storage backend LightRAG supports is reachable through this
  factory; there is deliberately no backend-specific code here.
- ``selected_chunker`` is the resolved custom chunker (or None). Callers load
  it themselves because selection failures are surfaced with caller-specific
  messaging (the API server prints a startup banner and exits; other callers
  may want something else).
- The returned instance is NOT storage-initialized: callers must still run
  ``await rag.initialize_storages()`` and ``await rag.finalize_storages()``.

This module is a pure relocation of the construction block that used to live
inside ``create_app`` so in-process tools (the MCP server, offline rebuild
tools) construct LightRAG exactly like the REST server does.
"""

import json
import os
from pathlib import Path
from typing import Any

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.api.config import (
    PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS,
    get_default_host,
    resolve_asymmetric_embedding_opt_in,
)
from lightrag.utils import EmbeddingFunc, logger, register_paper_id_lookup


class LLMConfigCache:
    """Smart LLM and Embedding configuration cache class"""
    def __init__(self, args):
        self.args = args
        # Initialize configurations based on binding conditions
        self.openai_llm_options = None
        self.gemini_llm_options = None
        self.gemini_embedding_options = None
        self.ollama_llm_options = None
        self.ollama_embedding_options = None
        self.bedrock_llm_options = None
        # Only initialize and log OpenAI options when using OpenAI-related bindings
        if args.llm_binding in ["openai", "azure_openai"]:
            from lightrag.llm.binding_options import OpenAILLMOptions
            self.openai_llm_options = OpenAILLMOptions.options_dict(args)
            logger.info(f"OpenAI LLM Options: {self.openai_llm_options}")
        if args.llm_binding == "gemini":
            from lightrag.llm.binding_options import GeminiLLMOptions
            self.gemini_llm_options = GeminiLLMOptions.options_dict(args)
            logger.info(f"Gemini LLM Options: {self.gemini_llm_options}")
        if args.llm_binding == "bedrock":
            from lightrag.llm.binding_options import BedrockLLMOptions
            self.bedrock_llm_options = BedrockLLMOptions.options_dict(args)
            logger.info(f"Bedrock LLM Options: {self.bedrock_llm_options}")
        # Only initialize and log Ollama LLM options when using Ollama LLM binding
        if args.llm_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions
                self.ollama_llm_options = OllamaLLMOptions.options_dict(args)
                logger.info(f"Ollama LLM Options: {self.ollama_llm_options}")
            except ImportError:
                logger.warning(
                    "OllamaLLMOptions not available, using default configuration"
                )
                self.ollama_llm_options = {}
        # Only initialize and log Ollama Embedding options when using Ollama Embedding binding
        if args.embedding_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaEmbeddingOptions
                self.ollama_embedding_options = OllamaEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Ollama Embedding Options: {self.ollama_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "OllamaEmbeddingOptions not available, using default configuration"
                )
                self.ollama_embedding_options = {}
        # Only initialize and log Gemini Embedding options when using Gemini Embedding binding
        if args.embedding_binding == "gemini":
            try:
                from lightrag.llm.binding_options import GeminiEmbeddingOptions
                self.gemini_embedding_options = GeminiEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Gemini Embedding Options: {self.gemini_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "GeminiEmbeddingOptions not available, using default configuration"
                )
                self.gemini_embedding_options = {}
_PROVIDER_LOG_LABELS = {
    "azure_openai": "Azure OpenAI",
    "bedrock": "Bedrock",
    "gemini": "Gemini",
    "lollms": "Lollms",
    "ollama": "Ollama",
    "openai": "OpenAI",
}
def create_optimized_embedding_function(
    config_cache: LLMConfigCache,
    binding,
    model,
    host,
    api_key,
    args,
    document_prefix=None,
    query_prefix=None,
) -> EmbeddingFunc:
    """
    Create optimized embedding function and return an EmbeddingFunc instance
    with proper max_token_size inheritance from provider defaults.
    This function:
    1. Imports the provider embedding function
    2. Extracts max_token_size and embedding_dim from provider if it's an EmbeddingFunc
    3. Creates an optimized wrapper that calls the underlying function directly (avoiding double-wrapping)
    4. Returns a properly configured EmbeddingFunc instance
    Configuration Rules:
    - When EMBEDDING_MODEL is not set: Uses provider's default model and dimension
      (e.g., jina-embeddings-v4 with 2048 dims, text-embedding-3-small with 1536 dims)
    - When EMBEDDING_MODEL is set to a custom model: User MUST also set EMBEDDING_DIM
      to match the custom model's dimension (e.g., for jina-embeddings-v3, set EMBEDDING_DIM=1024)
    Note: The embedding_dim parameter is automatically injected by EmbeddingFunc wrapper
    when send_dimensions=True (enabled for Jina and Gemini bindings). This wrapper calls
    the underlying provider function directly (.func) to avoid double-wrapping, so we must
    explicitly pass embedding_dim to the provider's underlying function.
    """
    # Step 1: Import provider function and extract default attributes
    provider_func = None
    provider_max_token_size = None
    provider_embedding_dim = None
    provider_supports_asymmetric = False
    try:
        if binding == "openai":
            from lightrag.llm.openai import openai_embed
            provider_func = openai_embed
        elif binding == "ollama":
            from lightrag.llm.ollama import ollama_embed
            provider_func = ollama_embed
        elif binding == "gemini":
            from lightrag.llm.gemini import gemini_embed
            provider_func = gemini_embed
        elif binding == "jina":
            from lightrag.llm.jina import jina_embed
            provider_func = jina_embed
        elif binding == "azure_openai":
            from lightrag.llm.azure_openai import azure_openai_embed
            provider_func = azure_openai_embed
        elif binding == "bedrock":
            from lightrag.llm.bedrock import bedrock_embed
            provider_func = bedrock_embed
        elif binding == "lollms":
            from lightrag.llm.lollms import lollms_embed
            provider_func = lollms_embed
        elif binding == "voyageai":
            from lightrag.llm.voyageai import voyageai_embed
            provider_func = voyageai_embed
        # Extract attributes if provider is an EmbeddingFunc
        if provider_func and isinstance(provider_func, EmbeddingFunc):
            provider_max_token_size = provider_func.max_token_size
            provider_embedding_dim = provider_func.embedding_dim
            provider_supports_asymmetric = provider_func.supports_asymmetric
            logger.debug(
                f"Extracted from {binding} provider: "
                f"max_token_size={provider_max_token_size}, "
                f"embedding_dim={provider_embedding_dim}, "
                f"supports_asymmetric={provider_supports_asymmetric}"
            )
    except ImportError as e:
        logger.warning(f"Could not import provider function for {binding}: {e}")
    # Fail-fast guard: require explicit EMBEDDING_DIM when a non-default
    # embedding model is configured. Without this, the provider's decorator
    # dimension silently applies regardless of the actual model selected,
    # causing vector-store write failures at runtime.
    # See: https://github.com/HKUDS/LightRAG/issues/3644
    # Note: lollms is excluded because it ignores the model parameter entirely.
    _BINDINGS_WITH_DIM_GUARD = frozenset(
        ["ollama", "openai", "jina", "gemini", "bedrock", "voyageai"]
    )
    # `not args.embedding_dim` (rather than `is None`) keeps the guard aligned
    # with the truthiness-based dimension resolution below: a 0 would otherwise
    # pass the guard and then silently resolve to the provider default.
    if (
        binding in _BINDINGS_WITH_DIM_GUARD
        and model
        and not args.embedding_dim
        and provider_func is not None
    ):
        default_model = getattr(provider_func, "model_name", None)
        if default_model:
            # The `:latest` suffix is an Ollama/OCI convention; stripping it
            # is a no-op for other bindings but keeps one unified comparison.
            configured_model = model.removesuffix(":latest")
            normalized_default = default_model.removesuffix(":latest")
            if configured_model != normalized_default:
                raise ValueError(
                    "EMBEDDING_DIM must be set when EMBEDDING_MODEL selects a "
                    f"custom {binding} model ({model!r}); the provider default "
                    f"dimension only applies to {default_model!r}"
                )
    # Azure OpenAI uses deployment names that never match a universal default,
    # so any configured model requires an explicit EMBEDDING_DIM.
    # AZURE_EMBEDDING_DEPLOYMENT wins over the configured model at runtime
    # (see azure_openai_embed: `os.getenv("AZURE_EMBEDDING_DEPLOYMENT") or model`),
    # so the error message must resolve it in the same order.
    azure_effective_model = (
        (os.environ.get("AZURE_EMBEDDING_DEPLOYMENT") or model)
        if binding == "azure_openai"
        else None
    )
    if (
        binding == "azure_openai"
        and azure_effective_model
        and not args.embedding_dim
        and provider_func is not None
    ):
        raise ValueError(
            "EMBEDDING_DIM must be set when using Azure OpenAI with a "
            f"configured deployment ({azure_effective_model!r}); Azure deployment "
            f"names require an explicit dimension. Note: AZURE_EMBEDDING_DEPLOYMENT "
            f"takes precedence over EMBEDDING_MODEL as the effective deployment name"
        )
    # Step 2: Apply priority (user config > provider default)
    # For max_token_size: explicit env var > provider default > None
    final_max_token_size = args.embedding_token_limit or provider_max_token_size
    # For embedding_dim: user config (always has value) takes priority
    # Only use provider default if user config is explicitly None (which shouldn't happen)
    final_embedding_dim = (
        args.embedding_dim if args.embedding_dim else provider_embedding_dim
    )
    # Asymmetric embedding is explicit opt-in only. Provider-specific
    # validation decides whether task parameters or prefixes are required.
    asymmetric_opt_in = resolve_asymmetric_embedding_opt_in(
        binding=binding,
        embedding_asymmetric=args.embedding_asymmetric,
        embedding_asymmetric_configured=args.embedding_asymmetric_configured,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        query_prefix_configured=args.embedding_query_prefix_configured,
        document_prefix_configured=args.embedding_document_prefix_configured,
    )
    # Step 3: Create optimized embedding function (calls underlying function directly)
    # Note: When model is None, each binding will use its own default model
    async def optimized_embedding_function(
        texts, embedding_dim=None, context="document"
    ):
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_embed
                # Get real function, skip EmbeddingFunc wrapper if present
                actual_func = (
                    lollms_embed.func
                    if isinstance(lollms_embed, EmbeddingFunc)
                    else lollms_embed
                )
                # lollms embed_model is not used (server uses configured vectorizer)
                # Only pass base_url and api_key
                return await actual_func(texts, base_url=host, api_key=api_key)
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_embed
                # Get real function, skip EmbeddingFunc wrapper if present
                actual_func = (
                    ollama_embed.func
                    if isinstance(ollama_embed, EmbeddingFunc)
                    else ollama_embed
                )
                # Use pre-processed configuration if available
                if config_cache.ollama_embedding_options is not None:
                    ollama_options = config_cache.ollama_embedding_options
                else:
                    from lightrag.llm.binding_options import OllamaEmbeddingOptions
                    ollama_options = OllamaEmbeddingOptions.options_dict(args)
                # Pass embed_model only if provided, let function use its default (bge-m3:latest)
                kwargs = {
                    "texts": texts,
                    "host": host,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                    "options": ollama_options,
                }
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    if query_prefix:
                        kwargs["query_prefix"] = query_prefix
                    if document_prefix:
                        kwargs["document_prefix"] = document_prefix
                if model:
                    kwargs["embed_model"] = model
                return await actual_func(**kwargs)
            elif binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_embed
                actual_func = (
                    azure_openai_embed.func
                    if isinstance(azure_openai_embed, EmbeddingFunc)
                    else azure_openai_embed
                )
                # Pass model only if provided, let function use its default otherwise
                kwargs = {
                    "texts": texts,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    if query_prefix:
                        kwargs["query_prefix"] = query_prefix
                    if document_prefix:
                        kwargs["document_prefix"] = document_prefix
                return await actual_func(**kwargs)
            elif binding == "bedrock":
                from lightrag.llm.bedrock import bedrock_embed
                actual_func = (
                    bedrock_embed.func
                    if isinstance(bedrock_embed, EmbeddingFunc)
                    else bedrock_embed
                )
                # Pass model only if provided, let function use its default otherwise
                kwargs = {
                    "texts": texts,
                    "aws_region": getattr(args, "aws_region", None),
                    "aws_access_key_id": getattr(args, "aws_access_key_id", None),
                    "aws_secret_access_key": getattr(
                        args, "aws_secret_access_key", None
                    ),
                    "aws_session_token": getattr(args, "aws_session_token", None),
                }
                if host is not None:
                    kwargs["endpoint_url"] = host
                if model:
                    kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "jina":
                from lightrag.llm.jina import jina_embed
                actual_func = (
                    jina_embed.func
                    if isinstance(jina_embed, EmbeddingFunc)
                    else jina_embed
                )
                # Pass model only if provided, let function use its default (jina-embeddings-v4)
                kwargs = {
                    "texts": texts,
                    "embedding_dim": embedding_dim,
                    "base_url": host,
                    "api_key": api_key,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    kwargs["task"] = None
                return await actual_func(**kwargs)
            elif binding == "gemini":
                from lightrag.llm.gemini import gemini_embed
                actual_func = (
                    gemini_embed.func
                    if isinstance(gemini_embed, EmbeddingFunc)
                    else gemini_embed
                )
                # Use pre-processed configuration if available
                if config_cache.gemini_embedding_options is not None:
                    gemini_options = config_cache.gemini_embedding_options
                else:
                    from lightrag.llm.binding_options import GeminiEmbeddingOptions
                    gemini_options = GeminiEmbeddingOptions.options_dict(args)
                # Pass model only if provided, let function use its default (gemini-embedding-001)
                kwargs = {
                    "texts": texts,
                    "base_url": host,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                task_type = gemini_options.get("task_type")
                if task_type is not None:
                    kwargs["task_type"] = task_type
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                return await actual_func(**kwargs)
            elif binding == "voyageai":
                from lightrag.llm.voyageai import voyageai_embed
                actual_func = (
                    voyageai_embed.func
                    if isinstance(voyageai_embed, EmbeddingFunc)
                    else voyageai_embed
                )
                kwargs = {
                    "texts": texts,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                return await actual_func(**kwargs)
            else:  # openai and compatible
                from lightrag.llm.openai import openai_embed
                actual_func = (
                    openai_embed.func
                    if isinstance(openai_embed, EmbeddingFunc)
                    else openai_embed
                )
                # Pass model only if provided, let function use its default (text-embedding-3-small)
                kwargs = {
                    "texts": texts,
                    "base_url": host,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    if query_prefix:
                        kwargs["query_prefix"] = query_prefix
                    if document_prefix:
                        kwargs["document_prefix"] = document_prefix
                return await actual_func(**kwargs)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} embedding: {e}")
    # Step 4: Wrap in EmbeddingFunc and return
    embedding_func_instance = EmbeddingFunc(
        embedding_dim=final_embedding_dim,
        func=optimized_embedding_function,
        max_token_size=final_max_token_size,
        send_dimensions=False,  # Will be set later based on binding requirements
        model_name=model,
        supports_asymmetric=provider_supports_asymmetric and asymmetric_opt_in,
    )
    # Log final embedding configuration. Only include prefix info when
    # prefixes will actually be applied (prefix-based asymmetric mode).
    prefix_info = ""
    if (
        asymmetric_opt_in
        and binding in PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS
        and (document_prefix or query_prefix)
    ):
        prefix_info = f" document_prefix={repr(document_prefix)} query_prefix={repr(query_prefix)}"
    logger.info(
        f"Embedding config: binding={binding} model={model} "
        f"embedding_dim={final_embedding_dim} max_token_size={final_max_token_size}{prefix_info}"
    )
    return embedding_func_instance
def create_embedding_function_from_args(
    args, config_cache: LLMConfigCache | None = None
) -> EmbeddingFunc:
    """Build the fully configured EmbeddingFunc used by the LightRAG server.
    Combines the provider embedding factory with the send_dimensions policy
    so that offline tools (e.g. lightrag-rebuild-vdb) can embed in exactly
    the same vector space as the running server.
    """
    import inspect
    if config_cache is None:
        config_cache = LLMConfigCache(args)
    # Create the EmbeddingFunc instance (now returns complete EmbeddingFunc with max_token_size)
    embedding_func = create_optimized_embedding_function(
        config_cache=config_cache,
        binding=args.embedding_binding,
        model=args.embedding_model,
        host=args.embedding_binding_host,
        api_key=None
        if args.embedding_binding == "bedrock"
        else args.embedding_binding_api_key,
        args=args,
        document_prefix=args.embedding_document_prefix,
        query_prefix=args.embedding_query_prefix,
    )
    # Get embedding_send_dim from centralized configuration
    embedding_send_dim = args.embedding_send_dim
    # Check if the underlying function signature has embedding_dim parameter
    sig = inspect.signature(embedding_func.func)
    has_embedding_dim_param = "embedding_dim" in sig.parameters
    # Determine send_dimensions value based on binding type
    # Jina and Gemini REQUIRE dimension parameter (forced to True)
    # OpenAI and others: controlled by EMBEDDING_SEND_DIM environment variable
    if args.embedding_binding in ["jina", "gemini"]:
        # Jina and Gemini APIs require dimension parameter - always send it
        send_dimensions = has_embedding_dim_param
        dimension_control = f"forced by {args.embedding_binding.title()} API"
    else:
        # For OpenAI and other bindings, respect EMBEDDING_SEND_DIM setting
        send_dimensions = embedding_send_dim and has_embedding_dim_param
        if send_dimensions or not embedding_send_dim:
            dimension_control = "by env var"
        else:
            dimension_control = "by not hasparam"
    # Set send_dimensions on the EmbeddingFunc instance
    embedding_func.send_dimensions = send_dimensions
    logger.info(
        f"Send embedding dimension: {send_dimensions} {dimension_control} "
        f"(dimensions={embedding_func.embedding_dim}, has_param={has_embedding_dim_param}, "
        f"binding={args.embedding_binding})"
    )
    # Log max_token_size source
    if embedding_func.max_token_size:
        source = (
            "env variable"
            if args.embedding_token_limit
            else f"{args.embedding_binding} provider default"
        )
        logger.info(
            f"Embedding max_token_size: {embedding_func.max_token_size} (from {source})"
        )
    else:
        logger.info(
            "Embedding max_token_size: None (Embedding token limit is disabled)."
        )
    return embedding_func

def _provider_log_label(binding: Any) -> str:
    binding_name = str(binding)
    return _PROVIDER_LOG_LABELS.get(
        binding_name, binding_name.replace("_", " ").title()
    )
def _log_role_provider_options(rag: Any) -> None:
    """Log sanitized provider options for every role LLM."""
    try:
        role_configs = rag.get_llm_role_config()
    except Exception as e:
        logger.warning(f"Failed to read role LLM configuration for logging: {e}")
        return
    logger.info("Role LLM Option:")
    for spec in ROLES:
        role_config = role_configs.get(spec.name)
        if not isinstance(role_config, dict):
            continue
        metadata = role_config.get("metadata") or {}
        binding = role_config.get("binding") or metadata.get("binding")
        if not binding:
            continue
        provider_options = metadata.get("provider_options") or {}
        logger.info(
            " - %s: %s %s",
            spec.name,
            _provider_log_label(binding),
            provider_options,
        )

def _create_llm_model_kwargs(binding: str, args, llm_timeout: int) -> dict:
    """
    Create LLM model kwargs based on binding type.
    Uses lazy import for binding-specific options.
    """
    if binding == "lollms":
        return {
            "timeout": llm_timeout,
            # "options" is an Ollama-only payload; lollms_model_if_cache()
            # never reads it. Pin it to an empty dict instead of deriving it
            # from OllamaLLMOptions.options_dict(args), which only happens to
            # return {} because its arguments are registered for the ollama
            # binding alone.
            "options": {},
            "api_key": args.llm_binding_api_key,
            # lollms_model_if_cache()'s parameter is named base_url, not
            # host -- unlike ollama's AsyncClient(host=...). Passing "host"
            # here would silently land in its **kwargs and never be read.
            "base_url": args.llm_binding_host,
        }
    if binding == "ollama":
        try:
            from lightrag.llm.binding_options import OllamaLLMOptions
            options = OllamaLLMOptions.options_dict(args)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} options: {e}")
        # Imported lazily (the module installs the ollama package on import)
        # and only for the binding that actually forwards think= -- lollms
        # never reaches the ollama client.
        from lightrag.llm.ollama import ensure_think_supported
        ensure_think_supported(options, context="the base LLM binding")
        return {
            "timeout": llm_timeout,
            "options": options,
            "api_key": args.llm_binding_api_key,
            "host": args.llm_binding_host,
        }
    return {}

def build_rag_from_args(args, selected_chunker=None):
    """Construct a LightRAG instance from parsed server-style arguments.

    Applies the same binding validation, default-host backfill, rerank setup,
    role-LLM wiring, and constructor arguments as the REST server's startup.
    Raises on invalid configuration (fail fast, before any storage is touched).
    """

    # Create configuration cache (this will output configuration logs)
    config_cache = LLMConfigCache(args)

    # Verify that bindings are correctly setup
    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "bedrock",
        "gemini",
    ]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "bedrock",
        "jina",
        "gemini",
        "voyageai",
    ]:
        raise Exception(f"embedding binding '{args.embedding_binding}' not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)

    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    def create_optimized_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized OpenAI LLM function with pre-processed configuration"""

        async def optimized_openai_alike_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            from lightrag.llm.openai import openai_complete_if_cache

            if history_messages is None:
                history_messages = []

            # Use pre-processed configuration to avoid repeated parsing.
            # response_format and legacy keyword_extraction/entity_extraction
            # flags flow through **kwargs; openai_complete_if_cache handles
            # the deprecation shim for the legacy booleans.
            kwargs["timeout"] = llm_timeout
            if config_cache.openai_llm_options:
                kwargs.update(config_cache.openai_llm_options)

            return await openai_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=args.llm_binding_host,
                api_key=args.llm_binding_api_key,
                **kwargs,
            )

        return optimized_openai_alike_model_complete

    def create_optimized_azure_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Azure OpenAI LLM function with pre-processed configuration"""

        async def optimized_azure_openai_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            from lightrag.llm.azure_openai import azure_openai_complete_if_cache

            if history_messages is None:
                history_messages = []

            # response_format and legacy extraction booleans flow through kwargs
            # to azure_openai_complete_if_cache, which handles deprecation shims.
            kwargs["timeout"] = llm_timeout
            if config_cache.openai_llm_options:
                kwargs.update(config_cache.openai_llm_options)

            return await azure_openai_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=args.llm_binding_host,
                api_key=os.getenv("AZURE_OPENAI_API_KEY", args.llm_binding_api_key),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                **kwargs,
            )

        return optimized_azure_openai_model_complete

    def create_optimized_gemini_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Gemini LLM function with cached configuration"""

        async def optimized_gemini_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            from lightrag.llm.gemini import gemini_complete_if_cache

            if history_messages is None:
                history_messages = []

            # response_format and legacy extraction booleans flow through kwargs
            # to gemini_complete_if_cache, which handles deprecation shims.
            kwargs["timeout"] = llm_timeout
            if (
                config_cache.gemini_llm_options is not None
                and "generation_config" not in kwargs
            ):
                kwargs["generation_config"] = dict(config_cache.gemini_llm_options)

            return await gemini_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=args.llm_binding_api_key,
                base_url=args.llm_binding_host,
                **kwargs,
            )

        return optimized_gemini_model_complete

    def create_llm_model_func(binding: str):
        """
        Create LLM model function based on binding type.
        Uses optimized functions for OpenAI bindings and lazy import for others.
        """
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_model_complete

                return lollms_model_complete
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_model_complete

                return ollama_model_complete
            elif binding == "bedrock":
                return bedrock_model_complete  # Already defined locally
            elif binding == "azure_openai":
                # Use optimized function with pre-processed configuration
                return create_optimized_azure_openai_llm_func(
                    config_cache, args, llm_timeout
                )
            elif binding == "gemini":
                return create_optimized_gemini_llm_func(config_cache, args, llm_timeout)
            else:  # openai and compatible
                # Use optimized function with pre-processed configuration
                return create_optimized_openai_llm_func(config_cache, args, llm_timeout)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} LLM binding: {e}")

    def resolve_role_llm_settings(
        role: str, override_meta: dict | None = None
    ) -> dict[str, Any]:
        attr = role.lower()
        override_meta = override_meta or {}

        role_binding = (
            override_meta.get("binding")
            or getattr(args, f"{attr}_llm_binding", None)
            or args.llm_binding
        )
        role_model = (
            override_meta.get("model")
            or getattr(args, f"{attr}_llm_model", None)
            or args.llm_model
        )
        role_host = (
            override_meta.get("host")
            or getattr(args, f"{attr}_llm_binding_host", None)
            or args.llm_binding_host
        )
        explicit_role_apikey = override_meta.get("api_key") or getattr(
            args, f"{attr}_llm_binding_api_key", None
        )
        if role_binding == "bedrock":
            if explicit_role_apikey:
                raise ValueError(
                    f"Bedrock role '{role}' does not support role-specific "
                    "LLM_BINDING_API_KEY; use role-specific SigV4 AWS_* "
                    "variables or process-level AWS_BEARER_TOKEN_BEDROCK."
                )
            role_apikey = None
        else:
            role_apikey = explicit_role_apikey or args.llm_binding_api_key
        role_timeout = (
            override_meta.get("timeout")
            or getattr(args, f"{attr}_llm_timeout", None)
            or llm_timeout
        )
        role_max_async = override_meta.get("max_async")
        if role_max_async is None:
            role_max_async = getattr(args, f"{attr}_llm_max_async", None)
        is_cross_provider = role_binding != args.llm_binding

        role_provider_options = override_meta.get("provider_options")
        if role_provider_options is None:
            if role_binding in ["openai", "azure_openai"]:
                from lightrag.llm.binding_options import OpenAILLMOptions

                role_provider_options = OpenAILLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            elif role_binding == "gemini":
                from lightrag.llm.binding_options import GeminiLLMOptions

                role_provider_options = GeminiLLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            elif role_binding in ["lollms", "ollama"]:
                from lightrag.llm.binding_options import OllamaLLMOptions

                role_provider_options = OllamaLLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            elif role_binding == "bedrock":
                from lightrag.llm.binding_options import BedrockLLMOptions

                role_provider_options = BedrockLLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            else:
                role_provider_options = {}

        if role_binding == "ollama":
            # Validated after the whole resolution above (including the
            # override_meta short-circuit), so what is checked is exactly what
            # the role will call with -- inherited global OLLAMA_LLM_THINK
            # included. Every role is resolved once while create_app builds
            # role_llm_configs, so an unsupported think= stops the server at
            # startup rather than at the role's first call.
            from lightrag.llm.ollama import ensure_think_supported

            ensure_think_supported(role_provider_options, context=f"LLM role '{role}'")

        bedrock_aws_options = {}
        if role_binding == "bedrock":
            override_bedrock_aws_options = override_meta.get("bedrock_aws_options", {})
            bedrock_aws_options = {
                "aws_region": override_meta.get("aws_region")
                or override_bedrock_aws_options.get("aws_region")
                or getattr(args, f"{attr}_aws_region", None)
                or getattr(args, "aws_region", None),
                "aws_access_key_id": override_meta.get("aws_access_key_id")
                or override_bedrock_aws_options.get("aws_access_key_id")
                or getattr(args, f"{attr}_aws_access_key_id", None)
                or getattr(args, "aws_access_key_id", None),
                "aws_secret_access_key": override_meta.get("aws_secret_access_key")
                or override_bedrock_aws_options.get("aws_secret_access_key")
                or getattr(args, f"{attr}_aws_secret_access_key", None)
                or getattr(args, "aws_secret_access_key", None),
                "aws_session_token": override_meta.get("aws_session_token")
                or override_bedrock_aws_options.get("aws_session_token")
                or getattr(args, f"{attr}_aws_session_token", None)
                or getattr(args, "aws_session_token", None),
            }

        return {
            "binding": role_binding,
            "model": role_model,
            "host": role_host,
            "api_key": role_apikey,
            "timeout": role_timeout,
            "max_async": role_max_async,
            "provider_options": role_provider_options,
            "is_cross_provider": is_cross_provider,
            "bedrock_aws_options": bedrock_aws_options,
        }

    def create_role_llm_func(role: str, override_meta: dict | None = None):
        """Create an independent raw LLM function for a role."""
        settings = resolve_role_llm_settings(role, override_meta)
        role_binding = settings["binding"]
        role_model = settings["model"]
        role_host = settings["host"]
        role_apikey = settings["api_key"]
        role_timeout = settings["timeout"]
        role_provider_options = settings["provider_options"]
        bedrock_aws_options = settings["bedrock_aws_options"]

        try:
            if role_binding == "ollama":
                from lightrag.llm.ollama import _ollama_model_if_cache

                async def role_ollama_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    enable_cot: bool = False,
                    **kwargs,
                ):
                    # response_format and legacy extraction booleans flow
                    # through kwargs to _ollama_model_if_cache, which handles
                    # the deprecation shim and emits a single warning.
                    if history_messages is None:
                        history_messages = []
                    if role_provider_options:
                        kwargs.setdefault("options", dict(role_provider_options))
                    return await _ollama_model_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        enable_cot=enable_cot,
                        host=role_host,
                        timeout=role_timeout,
                        api_key=role_apikey,
                        **kwargs,
                    )

                return role_ollama_complete
            if role_binding == "lollms":
                from lightrag.llm.lollms import lollms_model_if_cache

                async def role_lollms_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    enable_cot: bool = False,
                    **kwargs,
                ):
                    # response_format and legacy extraction booleans flow
                    # through kwargs to lollms_model_if_cache, which drops
                    # them and emits deprecation warnings when booleans are set.
                    if history_messages is None:
                        history_messages = []
                    if role_provider_options:
                        kwargs = {**role_provider_options, **kwargs}
                    return await lollms_model_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        enable_cot=enable_cot,
                        base_url=role_host,
                        api_key=role_apikey,
                        timeout=role_timeout,
                        **kwargs,
                    )

                return role_lollms_complete
            if role_binding == "bedrock":
                from lightrag.llm.bedrock import bedrock_complete_if_cache

                async def role_bedrock_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    **kwargs,
                ) -> str:
                    if history_messages is None:
                        history_messages = []
                    # Server-configured timeout overrides any caller-passed value,
                    # matching the OpenAI/Azure/Gemini role wrappers.
                    kwargs["timeout"] = role_timeout
                    if role_provider_options:
                        kwargs = {**role_provider_options, **kwargs}
                    return await bedrock_complete_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        endpoint_url=role_host,
                        **bedrock_aws_options,
                        **kwargs,
                    )

                return role_bedrock_complete
            if role_binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_complete_if_cache

                async def role_azure_openai_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    **kwargs,
                ) -> str:
                    if history_messages is None:
                        history_messages = []
                    kwargs["timeout"] = role_timeout
                    if role_provider_options:
                        kwargs.update(role_provider_options)
                    return await azure_openai_complete_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        base_url=role_host,
                        api_key=role_apikey or os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=os.getenv(
                            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
                        ),
                        **kwargs,
                    )

                return role_azure_openai_complete
            if role_binding == "gemini":
                from lightrag.llm.gemini import gemini_complete_if_cache

                async def role_gemini_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    **kwargs,
                ) -> str:
                    if history_messages is None:
                        history_messages = []
                    kwargs["timeout"] = role_timeout
                    if role_provider_options and "generation_config" not in kwargs:
                        kwargs["generation_config"] = dict(role_provider_options)
                    return await gemini_complete_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=role_apikey,
                        base_url=role_host,
                        **kwargs,
                    )

                return role_gemini_complete

            from lightrag.llm.openai import openai_complete_if_cache

            async def role_openai_complete(
                prompt,
                system_prompt=None,
                history_messages=None,
                **kwargs,
            ) -> str:
                if history_messages is None:
                    history_messages = []
                kwargs["timeout"] = role_timeout
                if role_provider_options:
                    kwargs.update(role_provider_options)
                return await openai_complete_if_cache(
                    role_model,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    base_url=role_host,
                    api_key=role_apikey,
                    **kwargs,
                )

            return role_openai_complete
        except ImportError as e:
            raise Exception(f"Failed to create LLM for role '{role}': {e}")

    def create_role_llm_model_kwargs(
        role: str, override_meta: dict | None = None
    ) -> dict[str, Any] | None:
        """Create role-specific kwargs for runtime wrapper injection.

        Role functions built above already encapsulate provider host/model/api_key/options,
        so we intentionally return an empty dict here to prevent base kwargs inheritance
        from polluting cross-provider role calls.
        """
        _ = role
        _ = override_meta
        return {}

    llm_timeout = args.llm_timeout
    embedding_timeout = args.embedding_timeout

    async def bedrock_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs,
    ) -> str:
        # Lazy import
        from lightrag.llm.bedrock import bedrock_complete_if_cache

        if history_messages is None:
            history_messages = []

        # Bedrock Converse API has no JSON mode; response_format and the legacy
        # extraction booleans flow through kwargs to bedrock_complete_if_cache,
        # which drops them and emits deprecation warnings when booleans are set.
        if config_cache.bedrock_llm_options:
            kwargs = {**config_cache.bedrock_llm_options, **kwargs}
        # Server-configured timeout overrides any caller-passed value, matching
        # the OpenAI wrapper (create_optimized_openai_llm_func).
        kwargs["timeout"] = llm_timeout

        return await bedrock_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            endpoint_url=args.llm_binding_host,
            aws_region=getattr(args, "aws_region", None),
            aws_access_key_id=getattr(args, "aws_access_key_id", None),
            aws_secret_access_key=getattr(args, "aws_secret_access_key", None),
            aws_session_token=getattr(args, "aws_session_token", None),
            **kwargs,
        )

    # Create embedding function with optimized configuration and max_token_size inheritance
    import inspect

    embedding_func = create_embedding_function_from_args(args, config_cache)

    # Configure rerank function based on args.rerank_bindingparameter
    rerank_model_func = None
    if args.rerank_binding != "null":
        from lightrag.rerank import (
            cohere_rerank,
            jina_rerank,
            ali_rerank,
            DEFAULT_RERANK_MAX_TOKENS_PER_DOC,
            MIN_PRACTICAL_RERANK_MAX_TOKENS,
        )

        # Map rerank binding to corresponding function
        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
        }

        # Select the appropriate rerank function based on binding
        selected_rerank_func = rerank_functions.get(args.rerank_binding)
        if not selected_rerank_func:
            logger.error(f"Unsupported rerank binding: {args.rerank_binding}")
            raise ValueError(f"Unsupported rerank binding: {args.rerank_binding}")

        # Get default values from selected_rerank_func if args values are None
        if args.rerank_model is None or args.rerank_binding_host is None:
            sig = inspect.signature(selected_rerank_func)

            # Set default model if args.rerank_model is None
            if args.rerank_model is None and "model" in sig.parameters:
                default_model = sig.parameters["model"].default
                if default_model != inspect.Parameter.empty:
                    args.rerank_model = default_model

            # Set default base_url if args.rerank_binding_host is None
            if args.rerank_binding_host is None and "base_url" in sig.parameters:
                default_base_url = sig.parameters["base_url"].default
                if default_base_url != inspect.Parameter.empty:
                    args.rerank_binding_host = default_base_url

        # Cohere binding supports optional document chunking (useful for models with
        # token limits like ColBERT). Parse and validate its config ONCE at startup so a
        # misconfigured RERANK_MAX_TOKENS_PER_DOC fails fast here instead of surfacing on
        # the first user query, and so the value is not re-parsed on every rerank call.
        rerank_enable_chunking = False
        rerank_max_tokens_per_doc = DEFAULT_RERANK_MAX_TOKENS_PER_DOC
        if args.rerank_binding == "cohere":
            rerank_enable_chunking = (
                os.getenv("RERANK_ENABLE_CHUNKING", "false").lower() == "true"
            )
            raw_max_tokens = os.getenv(
                "RERANK_MAX_TOKENS_PER_DOC", str(DEFAULT_RERANK_MAX_TOKENS_PER_DOC)
            )
            try:
                rerank_max_tokens_per_doc = int(raw_max_tokens)
            except ValueError as e:
                raise ValueError(
                    f"RERANK_MAX_TOKENS_PER_DOC must be an integer, got {raw_max_tokens!r}"
                ) from e
            if rerank_max_tokens_per_doc < 1:
                raise ValueError(
                    f"RERANK_MAX_TOKENS_PER_DOC must be >= 1, got {rerank_max_tokens_per_doc}"
                )
            if (
                rerank_enable_chunking
                and rerank_max_tokens_per_doc < MIN_PRACTICAL_RERANK_MAX_TOKENS
            ):
                logger.warning(
                    f"RERANK_MAX_TOKENS_PER_DOC={rerank_max_tokens_per_doc} is below the "
                    f"practical minimum ({MIN_PRACTICAL_RERANK_MAX_TOKENS}); chunking will "
                    f"split each document into many tiny, low-signal chunks."
                )

        async def server_rerank_func(
            query: str, documents: list, top_n: int = None, extra_body: dict = None
        ):
            """Server rerank function with configuration from environment variables"""
            # Prepare kwargs for rerank function
            kwargs = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_key": args.rerank_binding_api_key,
                "model": args.rerank_model,
                "base_url": args.rerank_binding_host,
            }

            # Add Cohere-specific parameters if using cohere binding (validated at startup)
            if args.rerank_binding == "cohere":
                kwargs["enable_chunking"] = rerank_enable_chunking
                kwargs["max_tokens_per_doc"] = rerank_max_tokens_per_doc

            return await selected_rerank_func(**kwargs, extra_body=extra_body)

        rerank_model_func = server_rerank_func
        logger.info(
            f"Reranking is enabled: {args.rerank_model or 'default model'} using {args.rerank_binding} provider"
        )
    else:
        logger.info("Reranking is disabled")

    # Create ollama_server_infos from command line arguments
    from lightrag.api.config import OllamaServerInfos

    ollama_server_infos = OllamaServerInfos(
        name=args.simulated_model_name, tag=args.simulated_model_tag
    )

    # LightRAG.__post_init__ normalizes addon_params and backfills env-based defaults
    # (SUMMARY_LANGUAGE, ENTITY_TYPE_PROMPT_FILE, ...), so we only need to pass the
    # API-level overrides here.
    addon_params = {
        "language": args.summary_language,
    }

    role_llm_configs = {
        spec.name: {
            **resolve_role_llm_settings(spec.name),
            "func": create_role_llm_func(spec.name),
            "kwargs": create_role_llm_model_kwargs(spec.name),
        }
        for spec in ROLES
    }

    # Initialize RAG with unified configuration
    try:
        rag = LightRAG(
            **({"chunking_func": selected_chunker} if selected_chunker else {}),
            working_dir=args.working_dir,
            workspace=args.workspace,
            llm_model_func=create_llm_model_func(args.llm_binding),
            llm_model_name=args.llm_model,
            llm_model_max_async=args.max_async,
            summary_max_tokens=args.summary_max_tokens,
            summary_context_size=args.summary_context_size,
            chunk_token_size=int(args.chunk_size),
            chunk_overlap_token_size=int(args.chunk_overlap_size),
            embedding_chunk_overlap_token_size=int(
                args.embedding_chunk_overlap_token_size
            ),
            llm_model_kwargs=_create_llm_model_kwargs(
                args.llm_binding, args, llm_timeout
            ),
            embedding_func=embedding_func,
            default_llm_timeout=llm_timeout,
            default_embedding_timeout=embedding_timeout,
            kv_storage=args.kv_storage,
            graph_storage=args.graph_storage,
            vector_storage=args.vector_storage,
            doc_status_storage=args.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": args.cosine_threshold
            },
            enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
            enable_llm_cache=args.enable_llm_cache,
            vlm_process_enable=args.vlm_process_enable,
            rerank_model_func=rerank_model_func,
            rerank_model_max_async=args.rerank_max_async,
            default_rerank_timeout=args.rerank_timeout,
            max_parallel_insert=args.max_parallel_insert,
            pipeline_scheduling_page_size=args.pipeline_scheduling_page_size,
            pipeline_require_strict_storage_reads=args.pipeline_require_strict_storage_reads,
            max_pending_documents=args.max_pending_documents,
            max_graph_nodes=args.max_graph_nodes,
            addon_params=addon_params,
            ollama_server_infos=ollama_server_infos,
            role_llm_configs={
                spec.name: RoleLLMConfig(
                    func=role_llm_configs[spec.name]["func"],
                    kwargs=role_llm_configs[spec.name]["kwargs"],
                    max_async=role_llm_configs[spec.name]["max_async"],
                    timeout=role_llm_configs[spec.name]["timeout"],
                    metadata={
                        "base_binding": args.llm_binding,
                        "binding": role_llm_configs[spec.name]["binding"],
                        "model": role_llm_configs[spec.name]["model"],
                        "host": role_llm_configs[spec.name]["host"],
                        "api_key": role_llm_configs[spec.name]["api_key"],
                        "provider_options": role_llm_configs[spec.name][
                            "provider_options"
                        ],
                        "bedrock_aws_options": role_llm_configs[spec.name][
                            "bedrock_aws_options"
                        ],
                        "is_cross_provider": role_llm_configs[spec.name][
                            "is_cross_provider"
                        ],
                    },
                )
                for spec in ROLES
            },
        )
    except Exception as e:
        logger.error(f"Failed to initialize LightRAG: {e}")
        raise

    # Load truncated-sha256 -> full paper_id lookup so derived/stored
    # paper-lib html_urls resolve to valid full-length URLs (see
    # lightrag.utils.register_paper_id_lookup).
    paper_id_map_path = Path(args.working_dir) / "paper_id_map.json"
    if paper_id_map_path.is_file():
        try:
            with open(paper_id_map_path) as f:
                register_paper_id_lookup(json.load(f))
            logger.info(f"Loaded paper_id map from {paper_id_map_path}")
        except Exception as e:
            logger.warning(f"Failed to load {paper_id_map_path}: {e}")

    _log_role_provider_options(rag)

    rag.register_role_llm_builder(
        lambda role, meta: (
            create_role_llm_func(role, meta),
            create_role_llm_model_kwargs(role, meta),
        )
    )

    return rag

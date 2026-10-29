"""
LightRAG FastAPI Server
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import asyncio
import json
import os
import re
import logging
import logging.config
import sys
import textwrap
import time
import uuid
import uvicorn
import pipmaster as pm
from typing import Any
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    get_auth_status_dependency,
    display_splash_screen,
    check_env_file,
    internal_server_error,
)
from lightrag.api.admission_middleware import AdmissionMiddleware
from lightrag.api.body_limit_middleware import BodyLimitMiddleware, resolve_body_limits
from lightrag.api.rag_factory import (
    LLMConfigCache as LLMConfigCache,
    _create_llm_model_kwargs as _create_llm_model_kwargs,
    build_rag_from_args,
    create_embedding_function_from_args as create_embedding_function_from_args,
    create_optimized_embedding_function as create_optimized_embedding_function,
)
from .config import (
    global_args,
    normalize_api_prefix,
    update_uvicorn_mode_config,
)
from lightrag import __version__ as core_version
from lightrag.utils import get_env_value
from lightrag.api import __api_version__
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
)
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes,
)
from lightrag.parser.docx.smart_heading.nlp import SmartHeadingNLPError
from lightrag.parser.plugins import load_third_party_parsers
from lightrag.chunker.plugins import load_and_resolve_chunker
from lightrag.chunker.registry import log_chunker_selection
from lightrag.parser.routing import (
    parser_rules_from_env,
    validate_parser_routing_config,
    validate_smart_heading_dependencies,
)
from lightrag.parser.external.mineru.cache import MinerUParserOptions
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
from lightrag.api.routers.graph3d_routes import create_graph3d_routes
from lightrag.api.routers.ollama_api import OllamaAPI
from lightrag.api.routers.ui_customization_routes import create_ui_customization_routes
from lightrag.api.ui_customization import (
    WEBUI_CHROME_LOCALES,
    locales_without_chrome_translation,
    resolve_ui_customization_snapshot,
)

from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_default_workspace,
    # set_default_workspace,
    cleanup_keyed_lock,
    drain_reserved_background_tasks,
    finalize_share_data,
    get_pipeline_ingress,
)
from lightrag import pipeline_metrics
from lightrag.utils_pipeline import describe_doc_status_capabilities
from fastapi.security import OAuth2PasswordRequestForm
from lightrag.api.auth import auth_handler
from lightrag.api.login_rate_limit import LoginRateLimiter

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=os.environ.get("DOTENV_PATH", ".env"), override=False)


webui_title = os.getenv("WEBUI_TITLE")
webui_description = os.getenv("WEBUI_DESCRIPTION")

# Global authentication configuration
auth_configured = bool(auth_handler.accounts)


def _inject_swagger_theme(html: str, theme: str) -> str:
    if theme not in {"dark", "light"}:
        theme = "auto"

    # The script resolves dark / light / (auto + prefers-color-scheme) into a
    # single boolean attribute `data-lightrag-docs-dark` on <html>. CSS below
    # only matches when that attribute is present, so light/auto-light paths
    # leave Swagger UI's default palette untouched.
    theme_snippet = textwrap.dedent(
        f"""
        <script>
          (function () {{
            var ALLOWED = {{ dark: 1, light: 1, auto: 1 }};
            var currentTheme = {json.dumps(theme)};
            var mql = window.matchMedia('(prefers-color-scheme: dark)');
            function resolveDark(value) {{
              if (value === 'dark') return true;
              if (value === 'auto') return mql.matches;
              return false;
            }}
            function apply(value) {{
              currentTheme = ALLOWED[value] ? value : 'auto';
              var root = document.documentElement;
              if (resolveDark(currentTheme)) {{
                root.setAttribute('data-lightrag-docs-dark', '1');
              }} else {{
                root.removeAttribute('data-lightrag-docs-dark');
              }}
            }}
            apply(currentTheme);
            // Re-resolve when the OS theme flips while `theme=auto` is active.
            var onMqlChange = function () {{ apply(currentTheme); }};
            if (mql.addEventListener) mql.addEventListener('change', onMqlChange);
            else if (mql.addListener) mql.addListener(onMqlChange);
            window.addEventListener('message', function (event) {{
              var data = event.data;
              if (!data || data.type !== 'lightrag:set-docs-theme') return;
              apply(data.theme);
            }});
          }})();
        </script>
        <style>
          html[data-lightrag-docs-dark="1"] {{
            color-scheme: dark;
          }}

          html[data-lightrag-docs-dark="1"] body,
          html[data-lightrag-docs-dark="1"] .swagger-ui,
          html[data-lightrag-docs-dark="1"] .swagger-ui .scheme-container,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-box,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock,
          html[data-lightrag-docs-dark="1"] .swagger-ui .dialog-ux .modal-ux,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container {{
            background: #0f172a;
            color: #e5e7eb;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .info .title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock-tag,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock .opblock-summary-description,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .parameter__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .parameter__type,
          html[data-lightrag-docs-dark="1"] .swagger-ui .response-col_status,
          html[data-lightrag-docs-dark="1"] .swagger-ui .response-col_description,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container h4,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container label,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .markdown p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .markdown li,
          html[data-lightrag-docs-dark="1"] .swagger-ui .renderedMarkdown p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .renderedMarkdown li,
          html[data-lightrag-docs-dark="1"] .swagger-ui table thead tr th,
          html[data-lightrag-docs-dark="1"] .swagger-ui table tbody tr td,
          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li,
          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li button.tablinks {{
            color: #e5e7eb;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock-description-wrapper p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock-external-docs-wrapper p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .response-col_links {{
            color: #cbd5f5;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui input,
          html[data-lightrag-docs-dark="1"] .swagger-ui textarea,
          html[data-lightrag-docs-dark="1"] .swagger-ui select {{
            background: #020617;
            border-color: #334155;
            color: #f8fafc;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .markdown code,
          html[data-lightrag-docs-dark="1"] .swagger-ui .renderedMarkdown code,
          html[data-lightrag-docs-dark="1"] .swagger-ui .highlight-code,
          html[data-lightrag-docs-dark="1"] .swagger-ui .highlight-code pre,
          html[data-lightrag-docs-dark="1"] .swagger-ui .microlight,
          html[data-lightrag-docs-dark="1"] .swagger-ui .body-param__example,
          html[data-lightrag-docs-dark="1"] .swagger-ui .example,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-example pre {{
            background: #020617;
            color: #e2e8f0;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui table thead tr th,
          html[data-lightrag-docs-dark="1"] .swagger-ui table tbody tr td {{
            border-color: #334155;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li.active button.tablinks,
          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li.tabitem.active {{
            color: #f8fafc;
            border-bottom-color: #34d399;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .btn.authorize,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-wrapper .authorize {{
            background: #064e3b;
            border-color: #34d399;
            color: #d1fae5;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .btn.authorize svg {{
            fill: #d1fae5;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .dialog-ux .modal-ux,
          html[data-lightrag-docs-dark="1"] .swagger-ui .scheme-container,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock {{
            border-color: #334155;
            box-shadow: none;
          }}

          /* Schemas panel: section.models contains its own grey-on-grey
             buttons (`Schemas` header, each model row, "Expand all") that
             ignore the top-level body color. Force the whole subtree to
             use surface backgrounds and bright text. */
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models.is-open,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .model-container,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-box {{
            background: #111827;
            border-color: #334155;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 button,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 a,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 span,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control button,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .model-toggle,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .model-title__text,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .property,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .prop-name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .prop-type,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .prop-format,
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation,
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation span {{
            color: #e5e7eb;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-toggle::after,
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion__icon svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion__icon svg path {{
            fill: #e5e7eb;
          }}

          /* The "Expand all" pill and per-row toggle buttons inherit a light
             grey background from Swagger; clear it so they don't punch a
             pale rectangle into the dark panel. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 button,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control button {{
            background: transparent;
          }}

          /* Swagger's new JSON Schema 2020-12 renderer hard-codes light-mode
             greys (#505050 / #3b4151 / #afaeae / #6b6b6b) on every title,
             keyword, attribute and json-viewer node — completely independent
             from the .model / .swagger-ui ancestors we already restyled.
             Override the whole renderer subtree so model/property names,
             types, and the per-row "Expand all" button stay readable. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-property .json-schema-2020-12__title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-expand-deep-button,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--const .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--const .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--default .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--default .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--enum .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--enum .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--examples .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--examples .json-schema-2020-12-json-viewer__value {{
            color: #e5e7eb;
          }}

          /* Secondary / extension / description text — keep them visible but
             dimmer than primary titles. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--description,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer-extension-keyword .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer-extension-keyword .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute--muted {{
            color: #cbd5f5;
          }}

          /* The deep-expand button inside each schemas row has its own
             background and shouldn't paint a pale capsule on dark surface. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-expand-deep-button,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion {{
            background-color: transparent;
          }}

          /* Restore Swagger's red warning palette. The broad keyword__value /
             __attribute / json-viewer__value overrides above otherwise win
             the cascade over `.json-schema-2020-12-*--warning` (higher
             specificity), flattening deprecated/schema-warning markers into
             plain text. Re-declared *after* the generic rules so equal-
             specificity selectors lose to these explicit ones. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--warning,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--warning,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute--warning {{
            color: #fca5a5;
            border-color: #fca5a5;
          }}

          /* `.model-toggle::after` paints its caret with a `background:url(
             data:image/svg+xml,…<path d=…/>)` embedded SVG whose path has no
             fill attribute and no currentColor reference — `fill` rules can't
             touch it. Invert the rendered pixels so the black arrow flips to
             white on the dark schema surface. The glyph is single-color, so
             invert has no perceptible side effect. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-toggle::after {{
            filter: invert(1);
          }}

          /* Per-operation Authorize lock icon. Swagger renders it via
             `<symbol id="locked|unlocked">` whose <path> has no fill attr
             and no currentColor reference; Swagger's CSS also never sets
             fill on .authorization__btn svg, leaving the path at the SVG
             default (black). Set fill on the outer <svg> — fill is inherited
             through <use> into the referenced symbol because the path itself
             is unstyled, so one declaration colors both locked and unlocked
             states. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .authorization__btn svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .authorization__btn .locked,
          html[data-lightrag-docs-dark="1"] .swagger-ui .authorization__btn .unlocked {{
            fill: #e5e7eb;
          }}
        </style>
        """
    ).strip()

    needle = "</head>"
    if needle not in html:
        logger.warning(
            "Swagger UI HTML missing </head> tag; theme patch was skipped. "
            "FastAPI's swagger template may have changed."
        )
        return html
    return html.replace(needle, f"{theme_snippet}\n{needle}", 1)


# Fixed WebUI mount path. Used as `app.mount(WEBUI_PATH, ...)` and as the
# in-app component of `webuiPrefix` injected into window.__LIGHTRAG_CONFIG__
# (which the browser sees as `LIGHTRAG_API_PREFIX + WEBUI_PATH + "/"`).
# Not user-configurable: a single mount path simplifies the operator surface
# and matches how LightRAG is deployed in practice. See
# docs/MultiSiteDeployment.md.
WEBUI_PATH = "/webui"

# Fixed mount path of the query-user entry (the "workspace" UI). Serves the
# same static build directory as WEBUI_PATH but with workspace.html as its
# directory index, so the entry identity is decided entirely by the URL.
# NOTE: this is a WebUI URL, not the knowledge-base `workspace` parameter used
# for data isolation — the two are unrelated (see the workspace-entry PRD).
WORKSPACE_PATH = "/workspace"

# Entry HTML filenames inside the shared build directory. Each mount serves
# only its own file as the directory index and returns 404 for the other
# entry's HTML (otherwise both files would be reachable under both mounts,
# creating fully working aliases like /webui/workspace.html).
WEBUI_INDEX_FILENAME = "index.html"
WORKSPACE_INDEX_FILENAME = "workspace.html"


class _RootPathNormalizationMiddleware:
    """Make Mount sub-apps work when the reverse proxy strips the API prefix.

    When ``LIGHTRAG_API_PREFIX=/site01`` and nginx strips ``/site01`` before
    forwarding, the backend sees ``scope["path"]="/webui/"`` while FastAPI's
    ``__call__`` sets ``scope["root_path"]="/site01"``. Starlette's outer
    Mount.matches still hits via ``get_route_path`` 's fallback branch (path
    not starting with root_path is returned unchanged), but it mutates the
    child scope to ``root_path="/site01/webui"`` without touching
    ``scope["path"]``. The inner ``StaticFiles.get_path`` then sees a
    non-overlapping pair and falls through to a literal ``webui`` filename
    lookup → 404 on the actual file system.

    Prepending ``root_path`` to a non-prefixed ``scope["path"]`` restores the
    canonical ASGI form (path always contains root_path), matching what a
    verbatim-forwarding proxy produces natively. Plain Routes are unaffected
    because their handlers do not redo nested ``get_route_path`` resolution.

    See docs/MultiSiteDeployment.md for the deployment modes this enables.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            root_path = scope.get("root_path", "")
            path = scope.get("path", "")
            if root_path and not path.startswith(root_path):
                scope = {**scope, "path": root_path + path}
                raw_path = scope.get("raw_path")
                if isinstance(raw_path, (bytes, bytearray)):
                    scope["raw_path"] = root_path.encode("ascii") + bytes(raw_path)
        await self.app(scope, receive, send)


def _clean_workspace_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_storage_workspace(storage: Any) -> str | None:
    if storage is None:
        return None

    effective_workspace = _clean_workspace_value(
        getattr(storage, "effective_workspace", None)
    )
    if effective_workspace:
        return effective_workspace

    final_namespace = _clean_workspace_value(getattr(storage, "final_namespace", None))
    namespace = _clean_workspace_value(getattr(storage, "namespace", None))
    if final_namespace and namespace:
        suffix = f"_{namespace}"
        if final_namespace.endswith(suffix):
            workspace = final_namespace[: -len(suffix)]
            if workspace:
                return workspace

    return _clean_workspace_value(getattr(storage, "workspace", None))


def _get_storage_workspaces(rag: Any) -> dict[str, str | None]:
    return {
        "kv_storage": _get_storage_workspace(getattr(rag, "full_docs", None)),
        "doc_status_storage": _get_storage_workspace(getattr(rag, "doc_status", None)),
        "graph_storage": _get_storage_workspace(
            getattr(rag, "chunk_entity_relation_graph", None)
        ),
        "vector_storage": _get_storage_workspace(getattr(rag, "entities_vdb", None)),
    }


def _build_mineru_status() -> dict[str, Any]:
    """Snapshot MinerU-related env vars for the /health endpoint.

    Reads env directly (no MinerURawClient instantiation — that has
    side effects like token validation). Reuses MinerUParserOptions to
    share defaulting logic with the actual parser path.
    """
    api_mode_raw = os.getenv("MINERU_API_MODE", "").strip().lower()
    api_mode: str | None = api_mode_raw or None
    endpoint = ""
    if api_mode == "official":
        endpoint = os.getenv("MINERU_OFFICIAL_ENDPOINT", "").strip()
    elif api_mode == "local":
        endpoint = os.getenv("MINERU_LOCAL_ENDPOINT", "").strip()

    options: dict[str, Any] = {}
    if api_mode in ("official", "local"):
        try:
            opts = MinerUParserOptions.from_env(api_mode=api_mode)
        except Exception:
            opts = None
        if opts is not None:
            options = {
                "language": opts.language,
                "enable_table": opts.enable_table,
                "enable_formula": opts.enable_formula,
            }
            if opts.api_mode == "official":
                options["model_version"] = opts.model_version
                options["is_ocr"] = opts.is_ocr
            else:
                options["local_backend"] = opts.local_backend
                options["local_parse_method"] = opts.local_parse_method
                options["local_image_analysis"] = opts.local_image_analysis

    return {"endpoint": endpoint, "api_mode": api_mode, "options": options}


def _build_docling_status() -> dict[str, Any]:
    """Snapshot Docling-related env vars for the /health endpoint."""
    endpoint = os.getenv("DOCLING_ENDPOINT", "").strip()
    if not endpoint:
        return {"endpoint": "", "options": {}}
    return {
        "endpoint": endpoint,
        "options": {
            "do_ocr": get_env_value("DOCLING_DO_OCR", True, bool),
            "force_ocr": get_env_value("DOCLING_FORCE_OCR", True, bool),
            "ocr_engine": os.getenv("DOCLING_OCR_ENGINE", "auto").strip() or "auto",
            "ocr_lang": os.getenv("DOCLING_OCR_LANG", "").strip(),
            "do_formula_enrichment": get_env_value(
                "DOCLING_DO_FORMULA_ENRICHMENT", False, bool
            ),
        },
    }


def check_frontend_build():
    """Check if frontend is built and optionally check if source is up-to-date

    Returns:
        tuple: (assets_exist: bool, is_outdated: bool)
            - assets_exist: True if WebUI build files exist
            - is_outdated: True if source is newer than build (only in dev environment)
    """
    webui_dir = Path(__file__).parent / "webui"
    index_html = webui_dir / "index.html"

    # 1. Check if build files exist
    if not index_html.exists():
        ASCIIColors.yellow("\n" + "=" * 80)
        ASCIIColors.yellow("WARNING: Frontend Not Built")
        ASCIIColors.yellow("=" * 80)
        ASCIIColors.yellow("The WebUI frontend has not been built yet.")
        ASCIIColors.yellow("The API server will start without the WebUI interface.")
        ASCIIColors.yellow(
            "\nTo enable WebUI, build the frontend using these commands:\n"
        )
        ASCIIColors.cyan("    cd lightrag_webui")
        ASCIIColors.cyan("    bun install --frozen-lockfile")
        ASCIIColors.cyan("    bun run build")
        ASCIIColors.cyan("    cd ..")
        ASCIIColors.yellow("\nThen restart the service.\n")
        ASCIIColors.cyan(
            "Note: Make sure you have Bun installed. Visit https://bun.sh for installation."
        )
        ASCIIColors.yellow("=" * 80 + "\n")
        return (False, False)  # Assets don't exist, not outdated

    # 2. Check if this is a development environment (source directory exists)
    try:
        source_dir = Path(__file__).parent.parent.parent / "lightrag_webui"
        src_dir = source_dir / "src"

        # Determine if this is a development environment: source directory exists and contains src directory
        if not source_dir.exists() or not src_dir.exists():
            # Production environment, skip source code check
            logger.debug(
                "Production environment detected, skipping source freshness check"
            )
            return (True, False)  # Assets exist, not outdated (prod environment)

        # Development environment, perform source code timestamp check
        logger.debug("Development environment detected, checking source freshness")

        # Source code file extensions (files to check)
        source_extensions = {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",  # TypeScript/JavaScript
            ".css",
            ".scss",
            ".sass",
            ".less",  # Style files
            ".json",
            ".jsonc",  # Configuration/data files
            ".html",
            ".htm",  # Template files
            ".md",
            ".mdx",  # Markdown
        }

        # Key configuration files (in lightrag_webui root directory)
        key_files = [
            source_dir / "package.json",
            source_dir / "bun.lock",
            source_dir / "vite.config.ts",
            source_dir / "tsconfig.json",
            source_dir / "tailraid.config.js",
            source_dir / "index.html",
        ]

        # Get the latest modification time of source code
        latest_source_time = 0

        # Check source code files in src directory
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                # Only check source code files, ignore temporary files and logs
                if file_path.suffix.lower() in source_extensions:
                    mtime = file_path.stat().st_mtime
                    latest_source_time = max(latest_source_time, mtime)

        # Check key configuration files
        for key_file in key_files:
            if key_file.exists():
                mtime = key_file.stat().st_mtime
                latest_source_time = max(latest_source_time, mtime)

        # Get build time
        build_time = index_html.stat().st_mtime

        # Compare timestamps (5 second tolerance to avoid file system time precision issues)
        if latest_source_time > build_time + 5:
            ASCIIColors.yellow("\n" + "=" * 80)
            ASCIIColors.yellow("WARNING: Frontend Source Code Has Been Updated")
            ASCIIColors.yellow("=" * 80)
            ASCIIColors.yellow(
                "The frontend source code is newer than the current build."
            )
            ASCIIColors.yellow(
                "This might happen after 'git pull' or manual code changes.\n"
            )
            ASCIIColors.cyan(
                "Recommended: Rebuild the frontend to use the latest changes:"
            )
            ASCIIColors.cyan("    cd lightrag_webui")
            ASCIIColors.cyan("    bun install --frozen-lockfile")
            ASCIIColors.cyan("    bun run build")
            ASCIIColors.cyan("    cd ..")
            ASCIIColors.yellow("\nThe server will continue with the current build.")
            ASCIIColors.yellow("=" * 80 + "\n")
            return (True, True)  # Assets exist, outdated
        else:
            logger.info("Frontend build is up-to-date")
            return (True, False)  # Assets exist, up-to-date

    except Exception as e:
        # If check fails, log warning but don't affect startup
        logger.warning(f"Failed to check frontend source freshness: {e}")
        return (True, False)  # Assume assets exist and up-to-date on error


def check_workspace_frontend_build() -> bool:
    """Whether the workspace query entry's HTML (workspace.html) is bundled.

    Checked independently from ``check_frontend_build()``: a stale build
    directory (new server + old WebUI bundle) legitimately contains only
    index.html, and that must degrade ONLY the /workspace entry — treating
    the whole WebUI as "not built" would needlessly take the admin /webui
    down with it.
    """
    workspace_html = Path(__file__).parent / "webui" / WORKSPACE_INDEX_FILENAME
    if workspace_html.exists():
        return True
    ASCIIColors.yellow(
        "WARNING: workspace.html is missing from the WebUI build directory. "
        "The /workspace query entry will be unavailable (admin /webui is not "
        "affected). Rebuild the frontend to enable it."
    )
    return False


def _build_capability_status(rag) -> dict:
    """Strict-capability report, or ``{}`` when it cannot be determined.

    /health is a liveness probe first: one unavailable diagnostic must never turn
    it into a 500.
    """
    doc_status = getattr(rag, "doc_status", None)
    if doc_status is None:
        return {}
    try:
        return describe_doc_status_capabilities(doc_status)
    except Exception as capability_error:  # pragma: no cover - defensive
        logger.debug(f"Capability probe unavailable for /health: {capability_error}")
        return {}


def _build_scheduling_status(pipeline_snapshot: dict, ingress_counts: dict) -> dict:
    """Curated scheduling/observability view for /health (LR2 Phase 6 items 2/4).

    Answers the questions an operator actually has during a manual retry or a
    scan: which phase the manual channel is in, which request holds the freeze
    and since when, what the drain is still waiting for, and how full the sticky
    channel is. The manual owner's ``owner_token`` is omitted on purpose — it
    authorizes releasing a reservation, so publishing it would turn a status page
    into a control surface.
    """
    owner = pipeline_snapshot.get("manual_owner") or {}
    freeze_started = pipeline_snapshot.get("manual_freeze_started_at")
    freeze_seconds = None
    if isinstance(freeze_started, (int, float)) and freeze_started > 0:
        # Wall clock, because the freeze may be held by another process; a
        # negative value (clock stepped back) is reported as 0 rather than as a
        # nonsensical duration.
        freeze_seconds = max(0.0, round(time.time() - float(freeze_started), 3))
    pending_enqueues = int(pipeline_snapshot.get("pending_enqueues", 0) or 0)
    return {
        "manual_phase": pipeline_snapshot.get("manual_phase") or "idle",
        "manual_freeze_requested": bool(
            pipeline_snapshot.get("manual_freeze_requested", False)
        ),
        "manual_resetting": bool(pipeline_snapshot.get("manual_resetting", False)),
        "manual_freeze_seconds": freeze_seconds,
        "manual_owner_request_id": (
            owner.get("request_id") if isinstance(owner, dict) else None
        ),
        "manual_owner_pid": owner.get("pid") if isinstance(owner, dict) else None,
        # What a DRAIN_TO_IDLE is still waiting for: reservations whose rows are
        # not written yet, plus whether a processing run still holds busy.
        "drain_pending_enqueues": pending_enqueues,
        "drain_waiting_on_workers": bool(pipeline_snapshot.get("busy", False)),
        "manual_retries_queued": ingress_counts.get("manual_retries"),
        "manual_retries_capacity": ingress_counts.get("manual_retries_capacity"),
        "document_notifications_queued": ingress_counts.get("documents"),
        "document_notification_overflows": ingress_counts.get("document_overflows"),
        "auto_rescan_pending": ingress_counts.get("auto_rescan_pending"),
    }


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render a request-validation refusal in the shape each endpoint promises.

    Module-level rather than a closure inside ``create_app`` so the real
    handler can be exercised without standing up the whole application.
    """
    # Check if this is a request to /query/data endpoint
    if request.url.path.endswith("/query/data"):
        # Extract error details
        error_details = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append(f"{field_path}: {error['msg']}")

        error_message = "; ".join(error_details)

        # Return in the expected format for /query/data
        return JSONResponse(
            status_code=400,
            content={
                "status": "failure",
                "message": f"Validation error: {error_message}",
                "data": {},
                "metadata": {},
            },
        )

    # For other endpoints, return the default FastAPI validation error.
    # `jsonable_encoder` is not optional here: a rejection raised as a
    # ValueError inside a validator (the RAG query minimum, the non-empty
    # check, the keyword and history checks, the aggregate text budget)
    # carries the exception object itself in `ctx.error`, which json.dumps
    # cannot serialize — without the encoder this handler raises and the
    # client gets a 500 in place of the 422 the refusal actually is.
    return JSONResponse(
        status_code=422, content={"detail": jsonable_encoder(exc.errors())}
    )


def create_app(args):
    # A server with no configured embedding model cannot protect its vectors.
    # The marker every vector backend records is what detects a swap to a
    # DIFFERENT model of the same dimension -- the one change no dimension
    # check can see -- and with no name to record, every container this server
    # provisions is unprotected for life.
    #
    # Refusing at startup rather than degrading is the same call the rest of
    # this feature makes: LightRAG supports neither multi-process configuration
    # propagation nor rolling updates, so an embedding-model change means
    # stopping the server and rebuilding the indexes with lightrag-rebuild-vdb.
    # A deployment that cannot tell which model wrote its vectors has no safe
    # way through that, and finding out at query time means wrong answers, not
    # an error. Only the SERVER refuses: the library stays usable without a
    # model name, and lightrag-rebuild-vdb in particular must keep running,
    # since it is the way out.
    #
    # An args object that never carried the field is refused the same way. The
    # ``default_ui`` tolerance below is for a cosmetic default; a safety guard
    # with an exemption for "the attribute was never set" is a guard with a
    # bypass, and the server's own parser always sets this one.
    if not (getattr(args, "embedding_model", None) or "").strip():
        raise SystemExit(
            "EMBEDDING_MODEL is not set. The server refuses to start without it: "
            "the embedding model's name is what every vector storage records "
            "alongside its vectors, and it is the only thing that detects a later "
            "switch to a different model of the same dimension -- which otherwise "
            "returns confidently wrong results with no error anywhere. Set "
            "EMBEDDING_MODEL to the model this deployment embeds with (and "
            "EMBEDDING_DIM when it is not the binding's default). Changing it "
            "later requires stopping the server and running lightrag-rebuild-vdb; "
            "rolling an embedding-model change through a live deployment is not "
            "supported."
        )

    # Check frontend build first and get status. The two entries are checked
    # independently: an old build directory may carry only index.html, which
    # degrades /workspace alone (see check_workspace_frontend_build).
    webui_assets_exist, is_frontend_outdated = check_frontend_build()
    workspace_assets_exist = check_workspace_frontend_build()
    # Scope of default_ui is exactly one behavior: the '/' redirect target.
    # getattr default keeps create_app working for programmatic callers that
    # build args without this field (tests, embedding).
    default_ui = getattr(args, "default_ui", "webui")

    # UI customization bundle (workspace-entry PRD §8). Completely orthogonal
    # to the entry-HTML checks above: a configured-but-broken bundle fails
    # startup regardless of which entry products exist, and a missing
    # workspace.html never suppresses bundle validation. bundle_revision is
    # logged HERE only — it appears in no /health tier.
    ui_templates_dir = (getattr(args, "ui_templates_dir", "") or "").strip()
    ui_customization_snapshot = None
    if ui_templates_dir:
        # Raises UICustomizationError → server startup fails (fail-fast; a
        # silent fallback would show LightRAG content while the operator
        # believes the customer branding is live). Returns None for the one
        # sanctioned exception: a configured directory that holds no
        # manifest.json yet, i.e. the unpopulated mount every default Docker
        # deployment starts with (see resolve_ui_customization_snapshot).
        ui_customization_snapshot = resolve_ui_customization_snapshot(ui_templates_dir)
    if ui_customization_snapshot is not None:
        logger.info(
            "UI customization: bundle %s %s",
            ui_customization_snapshot.bundle_revision,
            sorted(ui_customization_snapshot.locales),
        )
        # A bundle may declare any valid BCP 47 locale, but the WebUI chrome
        # only exists in the languages it ships. Content in another language
        # renders fine while the buttons and settings around it stay in the
        # visitor's resolved UI language — nothing the frontend can fix, so
        # say it here instead of leaving the operator to discover it from a
        # user's screenshot.
        untranslated = locales_without_chrome_translation(
            ui_customization_snapshot.locales
        )
        if untranslated:
            logger.warning(
                "UI customization: the WebUI ships no interface translation for "
                "%s — content in %s locale(s) will render beside controls in the "
                "visitor's resolved UI language (supported: %s)",
                untranslated,
                len(untranslated),
                sorted(WEBUI_CHROME_LOCALES),
            )
    elif ui_templates_dir:
        # Not an error, but not silent either: the only way to tell "I have
        # not written my bundle yet" from "my mount did not land where I
        # thought" is the path this names.
        logger.warning(
            "UI customization: UI_TEMPLATES_DIR=%s holds no manifest.json — "
            "serving the built-in LightRAG branding. Put a bundle there and "
            "restart to activate it; if you expected one, check that the "
            "directory is the one you populated.",
            ui_templates_dir,
        )
    else:
        logger.info("UI customization: no bundle configured (UI_TEMPLATES_DIR unset)")

    # Create unified API version display with warning symbol if frontend is outdated
    api_version_display = (
        f"{__api_version__}⚠️" if is_frontend_outdated else __api_version__
    )

    # Setup logging
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)
    # Discover third-party parser engines (``lightrag.parsers`` entry points)
    # BEFORE validating routing rules, so LIGHTRAG_PARSER may reference them.
    load_third_party_parsers()
    # Same reason as the smart_heading block below: an invalid CUSTOM_CHUNKER
    # is an operator configuration mistake, and a raw ValueError traceback
    # buries the one sentence that says how to fix it. Re-raised, never
    # swallowed — an unselectable chunker must still abort startup (the
    # deployment would otherwise chunk its C documents the wrong way).
    try:
        selected_chunker = load_and_resolve_chunker(getattr(args, "custom_chunker", ""))
    except ValueError as exc:
        # markup=False: ASCIIColors interprets "[...]" as rich markup tags.
        ASCIIColors.red("\n" + "=" * 80, markup=False)
        ASCIIColors.red("ERROR: invalid CUSTOM_CHUNKER selection", markup=False)
        ASCIIColors.red("=" * 80, markup=False)
        ASCIIColors.red(str(exc), markup=False)
        ASCIIColors.red("\nAuthoring and selection guide:", markup=False)
        ASCIIColors.cyan("    docs/ThirdPartyChunker.md", markup=False)
        ASCIIColors.red("=" * 80 + "\n", markup=False)
        raise
    log_chunker_selection(selected_chunker)
    validate_parser_routing_config()
    # Fail fast when DOCX_SMART_HEADING / a LIGHTRAG_PARSER rule enables
    # smart_heading but the pinned spaCy models are missing — surfacing the
    # install step at startup instead of failing mid-pipeline. Runs in
    # create_app so both the uvicorn and gunicorn (preload) paths hit it.
    # Caught here (instead of letting it propagate as a raw traceback) so the
    # missing-dependency message reads like the other boxed startup notices.
    try:
        validate_smart_heading_dependencies()
    except SmartHeadingNLPError as exc:
        # markup=False: ASCIIColors interprets "[...]" as rich markup tags and
        # silently drops anything it doesn't recognize (e.g. "[api]").
        ASCIIColors.red("\n" + "=" * 80, markup=False)
        ASCIIColors.red("ERROR: smart_heading dependencies missing", markup=False)
        ASCIIColors.red("=" * 80, markup=False)
        ASCIIColors.red(exc.problem, markup=False)
        ASCIIColors.red("\nInstall with:", markup=False)
        ASCIIColors.cyan(
            "    pip install lightrag-hku[api] && lightrag-download-cache --spacy-install",
            markup=False,
        )
        ASCIIColors.red(
            "(offline: see requirements-offline-smart-heading.txt)", markup=False
        )
        ASCIIColors.red("=" * 80 + "\n", markup=False)
        sys.exit(1)

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize document manager with workspace support for data isolation
    doc_manager = DocumentManager(args.input_dir, workspace=args.workspace)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()

        try:
            # Initialize database connections
            # Note: initialize_storages() now auto-initializes pipeline_status for rag.workspace
            await rag.initialize_storages()

            # Data migration regardless of storage implementation
            await rag.check_and_migrate_data()

            # Admission control needs a doc_status backend that can count
            # strictly (LR2 §9.1). Probe once here so an unsupported backend
            # fails at startup instead of turning every upload into a 503.
            if getattr(rag, "max_pending_documents", 0) > 0:
                from lightrag.utils_pipeline import count_active_documents

                try:
                    active_now = await count_active_documents(rag.doc_status)
                except Exception as admission_probe_error:
                    raise RuntimeError(
                        "MAX_PENDING_DOCUMENTS is set but the configured "
                        f"doc_status backend cannot count strictly: "
                        f"{admission_probe_error}"
                    ) from admission_probe_error
                logger.info(
                    f"Admission control enabled: capacity "
                    f"{rag.max_pending_documents}, {active_now} document(s) "
                    "currently active"
                )

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Cancel and join all reserved background tasks FIRST, so each
            # child's finally releases its reservation while shared state is
            # still alive. Resists repeated cancellation; a deferred shutdown
            # cancellation is re-raised only after storage/shared-state cleanup.
            shutdown_cancel = await drain_reserved_background_tasks(
                app.state.background_tasks
            )

            # Clean up database connections
            await rag.finalize_storages()

            if "LIGHTRAG_GUNICORN_MODE" not in os.environ:
                # Only perform cleanup in Uvicorn single-process mode
                logger.debug("Uvicorn Mode: finalizing shared storage...")
                finalize_share_data()
            else:
                # In Gunicorn mode with preload_app=True, cleanup is handled by on_exit hooks
                logger.debug(
                    "Gunicorn Mode: postpone shared storage finalization to master process"
                )

            # Re-raise a shutdown cancellation only after all cleanup is done.
            if shutdown_cancel is not None:
                raise shutdown_cancel

    # Single switch for every interactive API documentation surface: /docs,
    # /docs/oauth2-redirect, /redoc, /openapi.json and the /static/swagger-ui
    # mount. All five must stay conditioned on this one flag (one flag) — a route audit that special-cases only the APIRoutes would
    # diverge from the real route table.
    api_docs_enabled = bool(getattr(args, "enable_api_docs", True))

    # Whether the two query UIs label every answer as AI-generated
    # (ENABLE_AI_CONTENT_NOTICE). Deployment-level display configuration, so
    # it rides the same responses as webui_title/webui_description: both
    # entries read it once at boot from /auth-status (or /login), and the
    # admin shell keeps it fresh from its /health poll. getattr keeps
    # create_app working for callers that build args programmatically.
    ai_content_notice_enabled = bool(getattr(args, "enable_ai_content_notice", False))

    # The WebUI mount path is fixed at "/webui" — see
    # docs/MultiSiteDeployment.md for the rationale. Computed before
    # swagger_description below, which embeds it in the ReDoc link.
    api_prefix = normalize_api_prefix(getattr(args, "api_prefix", None))
    webui_path = WEBUI_PATH

    base_description = (
        "Providing API for LightRAG core, Web UI and Ollama Model Emulation"
    )
    swagger_description = (
        base_description
        + (" (API-Key Enabled)" if api_key else "")
        + f"\n\n[View ReDoc documentation]({api_prefix}/redoc)"
    )

    app_kwargs = {
        "title": "LightRAG Server API",
        "description": swagger_description,
        "version": __api_version__,
        "openapi_url": "/openapi.json" if api_docs_enabled else None,
        "docs_url": None,  # custom endpoint for offline Swagger support
        "redoc_url": "/redoc" if api_docs_enabled else None,
        "root_path": api_prefix if api_prefix else None,
        "lifespan": lifespan,
    }

    # Configure Swagger UI parameters
    # Enable persistAuthorization and tryItOutEnabled for better user experience
    app_kwargs["swagger_ui_parameters"] = {
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    }

    app = FastAPI(**app_kwargs)

    # Custom validation error handler, shaped per endpoint (see the
    # module-level function for why it is not a closure).
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Last-resort handler for any exception that escapes a route without being
    # converted to an HTTPException. It guarantees raw exception text never
    # reaches the client (CWE-209): the full detail and traceback are logged
    # server-side under a correlation id, and the response body carries only a
    # generic message plus that id. HTTPException (including the sanitized 500s
    # raised via internal_server_error) is still handled by FastAPI's own
    # handler and is intentionally not intercepted here.
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        error_id = uuid.uuid4().hex[:12]
        logger.error(
            f"Unhandled exception [error_id={error_id}] on "
            f"{request.method} {request.url.path}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error (error_id: {error_id})"},
        )

    def get_cors_origins():
        """Get allowed origins from global_args.

        Returns a list of allowed origins. The wildcard default ["*"] applies
        only when CORS_ORIGINS is unset (config defaults the value to "*"). An
        explicitly empty or origin-less value (e.g. CORS_ORIGINS= or a stray
        comma) fails closed, returning an empty list so that no cross-origin
        browser access is granted rather than silently widening to "*". Empty
        entries (e.g. from a trailing comma) are dropped.
        """
        origins_str = global_args.cors_origins
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",") if origin.strip()]

    # Normalize scope["path"] for proxy-strip deployments so the WebUI
    # Mount (and any other Mount) routes correctly. Added before CORS so it
    # runs first in the middleware stack — see _RootPathNormalizationMiddleware
    # docstring.
    if api_prefix:
        app.add_middleware(_RootPathNormalizationMiddleware)

    # Pre-body admission control (LR2 §9.3). Installed only when there is a
    # capacity to enforce; the ingestion routes keep their own reservation, so an
    # absent middleware costs economy (the body is read before the refusal), not
    # correctness. ``rag`` is built further down in this function, hence the lazy
    # getter.
    #
    # Added BEFORE the CORS middleware on purpose: the most recently added
    # middleware runs outermost, so CORS ends up wrapping this one and its 401 /
    # 429 responses carry the CORS headers a browser needs to read the status
    # (without them the WebUI would see an opaque network error instead of "at
    # capacity"). It therefore also sees the un-normalized path, which is why it
    # strips ``api_prefix`` itself.
    if args.max_pending_documents > 0:
        app.add_middleware(
            AdmissionMiddleware,
            rag_getter=lambda: rag,
            api_key=api_key,
            api_prefix=api_prefix,
        )

    # Raw request-body ceilings (GHSA-r8jh-295g-vv42). Added AFTER the admission
    # middleware so it ends up outside it: an oversized body is then refused
    # before it can take a capacity slot, and the reservation a mid-body 413
    # would otherwise strand is released by admission's own finally block as the
    # exception travels back out. Still added before CORS, for the same reason
    # admission is — a browser has to be able to read the 413.
    body_limits = resolve_body_limits(args)
    if body_limits is not None:
        app.add_middleware(BodyLimitMiddleware, api_prefix=api_prefix, **body_limits)

    # Add CORS middleware
    cors_origins = get_cors_origins()
    # Per the Fetch spec, the wildcard origin "*" and credentialed requests are
    # mutually exclusive: a server must not pair "Access-Control-Allow-Origin: *"
    # with "Access-Control-Allow-Credentials: true". LightRAG authenticates via
    # the Authorization (Bearer) and X-API-Key request headers, never via cookies
    # or other ambient credentials, so credentials are only ever meaningful for an
    # explicit origin allowlist. When origins are wildcarded we therefore disable
    # credentials to keep the configuration spec-compliant and avoid the permissive
    # "reflect any origin with credentials" behavior that Starlette would otherwise
    # apply to cookie-bearing cross-origin requests.
    #
    # Starlette treats ANY allow_origins list that contains "*" as allow-all, so we
    # must test membership rather than exact equality: a mixed config such as
    # "*,https://app.example.com" is still allow-all and must not enable credentials.
    # An empty list is a fail-closed (no-origin) config, which also gets no
    # credentials header.
    allow_credentials = bool(cors_origins) and "*" not in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-New-Token"
        ],  # Expose token renewal header for cross-origin requests
    )

    # Create combined auth dependency for all endpoints
    combined_auth = get_combined_auth_dependency(api_key)
    # Non-enforcing dependency: reports whether the caller is authenticated so
    # /health can stay a public liveness probe while gating sensitive config.
    auth_status = get_auth_status_dependency(api_key)

    def get_workspace_from_request(request: Request) -> str | None:
        """
        Extract workspace from HTTP request header or use default.

        This enables multi-workspace API support by checking the custom
        'LIGHTRAG-WORKSPACE' header. If not present, falls back to the
        server's default workspace configuration.

        Args:
            request: FastAPI Request object

        Returns:
            Workspace identifier (may be empty string for global namespace)
        """
        # Check custom header first
        workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()

        if not workspace:
            workspace = None
        else:
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)
            if sanitized != workspace:
                logger.warning(
                    f"Workspace header '{workspace}' contains invalid characters. "
                    f"Sanitized to '{sanitized}'."
                )
                workspace = sanitized

        return workspace

    rag = build_rag_from_args(args, selected_chunker=selected_chunker)


    # Add routes
    # root_path is set on the app for reverse proxy support;
    # routes stay at their natural paths and are prefixed by the proxy or uvicorn --root-path
    app.include_router(create_document_routes(rag, doc_manager, api_key))
    app.include_router(create_query_routes(rag, api_key, args.top_k))
    app.include_router(create_graph_routes(rag, api_key))
    app.include_router(create_graph3d_routes(rag, api_key))
    # Public read-only customization surface — registered unconditionally:
    # without a bundle it answers 200 {"customized": false, ...}.
    app.include_router(
        create_ui_customization_routes(
            ui_customization_snapshot, webui_title, webui_description
        )
    )

    # Add Ollama API routes
    ollama_api = OllamaAPI(rag, top_k=args.top_k, api_key=api_key)
    app.include_router(ollama_api.router, prefix="/api")

    if api_docs_enabled:
        # Custom Swagger UI endpoint for offline support
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html(request: Request):
            """Custom Swagger UI HTML with local static files"""
            root = request.scope.get("root_path", "").rstrip("/")
            response = get_swagger_ui_html(
                openapi_url=f"{root}{app.openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=f"{root}/docs/oauth2-redirect",
                swagger_js_url=f"{root}/static/swagger-ui/swagger-ui-bundle.js",
                swagger_css_url=f"{root}/static/swagger-ui/swagger-ui.css",
                swagger_favicon_url=f"{root}/static/swagger-ui/favicon-32x32.png",
                swagger_ui_parameters=app.swagger_ui_parameters,
            )
            html = response.body.decode("utf-8")
            html = _inject_swagger_theme(
                html, request.query_params.get("theme", "auto").lower()
            )
            return HTMLResponse(content=html)

        @app.get("/docs/oauth2-redirect", include_in_schema=False)
        async def swagger_ui_redirect():
            """OAuth2 redirect for Swagger UI"""
            return get_swagger_ui_oauth2_redirect_html()

    def service_info_response(request: Request) -> JSONResponse:
        """Fixed JSON fallback when neither the WebUI nor /docs can be served.

        HTTP 200 with a root_path-aware health_url, so multi-site deployments
        behind LIGHTRAG_API_PREFIX get a correct absolute path.
        """
        root = request.scope.get("root_path", "")
        return JSONResponse(
            {
                "status": "healthy",
                "service": "LightRAG Server",
                "api_version": api_version_display,
                "message": (
                    "WebUI assets are not bundled and API docs are disabled "
                    "(ENABLE_API_DOCS=false)."
                ),
                "health_url": f"{root}/health",
            }
        )

    def workspace_unavailable_response(request: Request) -> JSONResponse:
        """Fixed JSON degradation for the /workspace query entry.

        Unlike the /webui fallback, this NEVER redirects to — nor links or
        points at — the API docs, regardless of ENABLE_API_DOCS: sending a
        query user to /docs is the same cross-entry leak as sending them to
        the admin login page. HTTP 200 with a root_path-aware health_url.
        """
        root = request.scope.get("root_path", "")
        return JSONResponse(
            {
                "status": "healthy",
                "service": "LightRAG Server",
                "api_version": api_version_display,
                "message": (
                    "The workspace query entry is not available: "
                    "workspace.html is not bundled with this server. "
                    "Rebuild the frontend to enable it."
                ),
                "health_url": f"{root}/health",
            }
        )

    @app.get("/")
    async def redirect_to_default_ui(request: Request):
        """Redirect root path to the configured default UI entry.

        Prepend the ASGI root_path so that, behind a reverse proxy, the
        absolute redirect target keeps the configured prefix instead of
        bypassing it. The '/' behavior equals the chosen entry's behavior
        INCLUDING its degradation branch: when the default entry's assets
        are missing, '/' answers with that entry's own fallback rather than
        rerouting to the other entry (or, for workspace, to the API docs).
        """
        root = request.scope.get("root_path", "")
        if default_ui == "workspace":
            if workspace_assets_exist:
                return RedirectResponse(url=f"{root}{WORKSPACE_PATH}/")
            return workspace_unavailable_response(request)
        if webui_assets_exist:
            return RedirectResponse(url=f"{root}{webui_path}/")
        elif api_docs_enabled:
            return RedirectResponse(url=f"{root}/docs")
        else:
            return service_info_response(request)

    # The verifier ignores WHITELIST_PATHS: its whole purpose is to report
    # credential validity, and a whitelist entry covering it (e.g. "/auth/*")
    # would otherwise turn it into an unconditional 200 while the protected
    # routes still reject.
    credential_verify_auth = get_combined_auth_dependency(
        api_key, respect_whitelist=False
    )

    @app.get("/auth/verify", dependencies=[Depends(credential_verify_auth)])
    async def verify_credentials():
        """Verify that the caller's credentials satisfy the combined auth.

        /health deliberately stays on the default whitelist as an
        unauthenticated liveness probe, so it can never distinguish valid,
        invalid and missing credentials. This endpoint ALWAYS runs the
        combined dependency — WHITELIST_PATHS is deliberately ignored here —
        so a missing or wrong X-API-Key fails with the standard 403 detail
        ("API Key required" / "Invalid API Key"), which the WebUI's API-key
        dialogs key on, while valid credentials get a trivial 200. No data is
        exposed.
        """
        return {"status": "ok"}

    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status and guest token if auth is not configured"""

        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
                "ai_content_notice_enabled": ai_content_notice_enabled,
            }

        return {
            "auth_configured": True,
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
            "ai_content_notice_enabled": ai_content_notice_enabled,
        }

    # Brute-force protection for /login (CWE-307): throttle failed attempts per
    # client IP + username. Checked before the bcrypt verification, so a locked
    # key is also rejected without paying the bcrypt cost.
    # getattr defaults keep create_app working for callers that build args
    # programmatically (e.g. tests, embedding) without these newer fields.
    login_rate_limiter = LoginRateLimiter(
        max_attempts=getattr(args, "login_max_failed_attempts", 5),
        window_seconds=getattr(args, "login_lockout_window_seconds", 300.0),
    )

    @app.post("/login")
    async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
                "ai_content_notice_enabled": ai_content_notice_enabled,
            }
        username = form_data.username
        # Rate-limit key is client IP + username. X-Forwarded-For is NOT trusted
        # (spoofable); behind a reverse proxy all clients may share the proxy IP,
        # so buckets are separated by username to avoid one attacker locking out
        # unrelated accounts. request.client can be None for some transports.
        client_ip = request.client.host if request.client else "unknown"
        rate_limit_key = f"{client_ip}:{username}"

        retry_after = login_rate_limiter.retry_after(rate_limit_key)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many failed login attempts. Please try again later.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        # Reserve this attempt BEFORE the bcrypt await. verify runs on a worker
        # thread, so without pre-reserving, many concurrent requests would all
        # pass the check above before any resolved, bypassing the limit (TOCTOU).
        # retry_after + reserve here are synchronous with no await between them,
        # so the decision is consistent. The reservation is always released in
        # the finally; only a confirmed wrong password becomes a real failure.
        login_rate_limiter.reserve_attempt(rate_limit_key)
        try:
            # verify_password runs a CPU-bound bcrypt for every attempt (incl.
            # unknown usernames, to equalize timing). Run it in a worker thread
            # so a login flood cannot block the event loop and starve the whole
            # API (unauthenticated DoS).
            password_ok = await asyncio.to_thread(
                auth_handler.verify_password, username, form_data.password
            )
            # Record the outcome BEFORE releasing the reservation, so the entry
            # still carries the failure (release removes a now-idle entry).
            if not password_ok:
                # Confirmed failure -> count it (and emit the lockout alert only
                # here, so a correct password on the Nth attempt never does).
                login_rate_limiter.commit_failure(rate_limit_key)
                raise HTTPException(status_code=401, detail="Incorrect credentials")
            # Success clears the key's earlier failures.
            login_rate_limiter.reset(rate_limit_key)
        finally:
            # Always drop the in-flight reservation (also frees the slot once the
            # key is fully idle, e.g. after a successful login).
            login_rate_limiter.release(rate_limit_key)

        # Regular user login
        user_token = auth_handler.create_token(
            username=username, role="user", metadata={"auth_mode": "enabled"}
        )
        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
            "ai_content_notice_enabled": ai_content_notice_enabled,
        }

    @app.get(
        "/health",
        dependencies=[Depends(combined_auth)],
        summary="Get system health and configuration status",
        description=(
            "Always reachable as a liveness probe (HTTP 200). Unauthenticated "
            "callers receive only liveness signals (status, versions, auth_mode, "
            "pipeline_busy). The full configuration and operational metrics are "
            "returned only to authenticated callers (valid JWT or X-API-Key)."
        ),
        response_description="System health status; configuration included only when authenticated",
        responses={
            200: {
                "description": "Successful response with system status",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "healthy",
                            "webui_available": True,
                            "api_docs_available": True,
                            "working_directory": "/path/to/working/dir",
                            "input_directory": "/path/to/input/dir",
                            "configuration": {
                                "llm_binding": "openai",
                                "llm_model": "gpt-4",
                                "embedding_binding": "openai",
                                "embedding_model": "text-embedding-ada-002",
                                "workspace": "default",
                                "storage_workspaces": {
                                    "kv_storage": "default",
                                    "doc_status_storage": "default",
                                    "graph_storage": "default",
                                    "vector_storage": "default",
                                },
                                "parser_routing": "pdf:mineru",
                                "mineru": {
                                    "endpoint": "http://localhost:8080",
                                    "api_mode": "local",
                                    "options": {
                                        "language": "ch",
                                        "enable_table": True,
                                        "enable_formula": True,
                                        "local_backend": "pipeline",
                                        "local_parse_method": "auto",
                                        "local_image_analysis": False,
                                    },
                                },
                                "docling": {
                                    "endpoint": "",
                                    "options": {},
                                },
                            },
                            "auth_mode": "enabled",
                            "pipeline_busy": False,
                            "core_version": "0.0.1",
                            "api_version": "0.0.1",
                        }
                    }
                },
            }
        },
    )
    async def get_status(request: Request, authenticated: bool = Depends(auth_status)):
        """Get current system status including WebUI availability.

        Stays a public liveness probe: unauthenticated callers receive only
        liveness signals; sensitive configuration is returned only when the
        caller is authenticated (see get_auth_status_dependency).
        """
        try:
            workspace = get_workspace_from_request(request)
            default_workspace = get_default_workspace()
            if workspace is None:
                workspace = default_workspace
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )
            # One DictProxy RPC in multi-worker mode; keep /health read-only and
            # avoid one cross-process ``get`` per field.
            pipeline_snapshot = pipeline_status.copy()
            pipeline_busy = bool(pipeline_snapshot.get("busy", False))
            pipeline_scanning = bool(pipeline_snapshot.get("scanning", False))
            pipeline_destructive_busy = bool(
                pipeline_snapshot.get("destructive_busy", False)
            )
            pipeline_pending_enqueues = int(
                pipeline_snapshot.get("pending_enqueues", 0) or 0
            )
            pipeline_active = (
                pipeline_busy
                or pipeline_scanning
                or pipeline_destructive_busy
                or pipeline_pending_enqueues > 0
            )

            # Ingress channel depths (bounded counters only — never the
            # messages). Best-effort: a mailbox that is not bootstrapped yet must
            # not turn a liveness probe into a 500.
            ingress_counts: dict[str, Any] = {}
            try:
                ingress_counts = dict(
                    (await get_pipeline_ingress(workspace)).counts() or {}
                )
            except Exception as ingress_error:
                logger.debug(f"Ingress counts unavailable for /health: {ingress_error}")

            if not auth_configured:
                auth_mode = "disabled"
            else:
                auth_mode = "enabled"

            # Liveness payload — always returned, even to unauthenticated
            # callers, so /health stays a usable liveness probe (HTTP 200).
            # Every field here is either a pure liveness signal or is already
            # exposed by the unauthenticated /auth-status endpoint, so it leaks
            # nothing new.
            status_data = {
                "status": "healthy",
                "auth_mode": auth_mode,
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_available": webui_assets_exist,
                # Same liveness tier as webui_available (both are trivially
                # probeable by requesting the mount path). The two fields are
                # independent checks over their own entry HTML — /health must
                # be able to express "admin UI up, query entry missing".
                "workspace_available": workspace_assets_exist,
                # Whether /docs, /redoc and /openapi.json are served — same
                # liveness tier as webui_available: the state is trivially
                # probeable by requesting /docs, so it leaks nothing.
                "api_docs_available": api_docs_enabled,
                "webui_title": webui_title,
                "webui_description": webui_description,
                "ai_content_notice_enabled": ai_content_notice_enabled,
                "pipeline_busy": pipeline_busy,
                "pipeline_active": pipeline_active,
            }

            # Sensitive runtime configuration and operational diagnostics
            # (filesystem paths, LLM/embedding provider + model + host, storage
            # backends, queue status, keyed locks, ...) are revealed only to
            # authenticated callers. The skipped queue-status
            # and keyed-lock-cleanup calls also keep unauthenticated probes cheap.
            if not authenticated:
                return status_data

            # Cleanup expired keyed locks and get status
            keyed_lock_info = cleanup_keyed_lock()

            status_data.update(
                {
                    "working_directory": str(args.working_dir),
                    "input_directory": str(args.input_dir),
                    "configuration": {
                        # LLM configuration binding/host address (if applicable)/model (if applicable)
                        "llm_binding": args.llm_binding,
                        "llm_binding_host": args.llm_binding_host,
                        "llm_model": args.llm_model,
                        # embedding model configuration binding/host address (if applicable)/model (if applicable)
                        "embedding_binding": args.embedding_binding,
                        "embedding_binding_host": args.embedding_binding_host,
                        "embedding_model": args.embedding_model,
                        "summary_max_tokens": args.summary_max_tokens,
                        "summary_context_size": args.summary_context_size,
                        "kv_storage": args.kv_storage,
                        "doc_status_storage": args.doc_status_storage,
                        "graph_storage": args.graph_storage,
                        "vector_storage": args.vector_storage,
                        "enable_llm_cache_for_extract": args.enable_llm_cache_for_extract,
                        "enable_llm_cache": args.enable_llm_cache,
                        "vlm_process_enable": args.vlm_process_enable,
                        "workspace": default_workspace,
                        "storage_workspaces": _get_storage_workspaces(rag),
                        "max_graph_nodes": args.max_graph_nodes,
                        # Rerank configuration
                        "enable_rerank": rag.rerank_model_func is not None,
                        "rerank_binding": args.rerank_binding,
                        "rerank_model": args.rerank_model
                        if rag.rerank_model_func
                        else None,
                        "rerank_binding_host": args.rerank_binding_host
                        if rag.rerank_model_func
                        else None,
                        "rerank_max_async": args.rerank_max_async,
                        "rerank_timeout": args.rerank_timeout,
                        # Environment variable status (requested configuration)
                        "summary_language": args.summary_language,
                        "force_llm_summary_on_merge": args.force_llm_summary_on_merge,
                        "max_parallel_insert": args.max_parallel_insert,
                        "cosine_threshold": args.cosine_threshold,
                        "min_rerank_score": args.min_rerank_score,
                        "related_chunk_number": args.related_chunk_number,
                        "max_async": args.max_async,
                        "llm_timeout": args.llm_timeout,
                        "embedding_func_max_async": args.embedding_func_max_async,
                        "embedding_batch_num": args.embedding_batch_num,
                        "embedding_timeout": args.embedding_timeout,
                        "role_llm_config": rag.get_llm_role_config(),
                        # Parser routing snapshot — surfaced in the WebUI status card
                        "parser_routing": parser_rules_from_env(),
                        "mineru": _build_mineru_status(),
                        "docling": _build_docling_status(),
                    },
                    "server_mode": "gunicorn"
                    if os.environ.get("LIGHTRAG_GUNICORN_MODE")
                    else "uvicorn",
                    "workers": getattr(args, "workers", 1),
                    "pipeline_scanning": pipeline_scanning,
                    "pipeline_destructive_busy": pipeline_destructive_busy,
                    "pipeline_pending_enqueues": pipeline_pending_enqueues,
                    # Curated scheduling view (LR2 Phase 6 items 2 & 4). The raw
                    # manual_* fields stay hidden from /pipeline_status — they are
                    # coordination internals — but an operator watching a freeze
                    # needs to see WHICH request holds it, for how long, and what
                    # the drain is still waiting for. The owner token is
                    # deliberately omitted: it is a capability, not a status.
                    "scheduling": _build_scheduling_status(
                        pipeline_snapshot, ingress_counts
                    ),
                    "capabilities": _build_capability_status(rag),
                    # Per-worker counters/durations for the paths the bounded
                    # rework introduced (LR2 Phase 6 item 3); see
                    # lightrag/pipeline_metrics.py for the boundary.
                    "scheduling_metrics": pipeline_metrics.snapshot(),
                    "keyed_locks": keyed_lock_info,
                    "llm_queue_status": await rag.get_llm_queue_status(
                        include_base=True
                    ),
                    "embedding_queue_status": await rag.get_embedding_queue_status(),
                    "rerank_queue_status": await rag.get_rerank_queue_status(),
                }
            )
            return status_data
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            raise internal_server_error(e)

    # Pre-render the runtime-config <script> once. The browser-visible URL
    # prefixes are NOT baked into the bundle anymore — index.html ships with
    # a placeholder comment that we replace with this snippet on every HTML
    # response, so one build serves any reverse-proxy mount point.
    #
    # `</` → `<\/` escaping prevents an embedded "</script>" sequence from
    # breaking out of the inline script (defense-in-depth — values come from
    # admin config, not user input).
    #
    # `webuiTitle` (WEBUI_TITLE) rides along and is applied to `document.title`
    # in the same snippet: the tag sits in <head> right after the static
    # <title>, so the tab carries the deployment's own name from the FIRST
    # paint. Waiting for the SPA to read it from /health would show "LightRAG"
    # until the bundle boots, and browsers cache that first title for history
    # entries and bookmarks. `or None` normalizes an unset/empty variable to a
    # single falsy value, which the client reads as "no custom title".
    _runtime_config_payload = json.dumps(
        {
            "apiPrefix": api_prefix,
            "webuiPrefix": f"{api_prefix}{webui_path}/",
            "webuiTitle": webui_title or None,
        }
    ).replace("</", "<\\/")
    runtime_config_script = (
        f"<script>window.__LIGHTRAG_CONFIG__ = {_runtime_config_payload};"
        "if(window.__LIGHTRAG_CONFIG__.webuiTitle)"
        "{document.title=window.__LIGHTRAG_CONFIG__.webuiTitle;}</script>"
    )

    # Custom StaticFiles class for smart caching + runtime config injection
    class SmartStaticFiles(StaticFiles):  # Renamed from NoCacheStaticFiles
        # Replaced in every entry HTML on each request. Keep in sync with
        # lightrag_webui/index.html AND lightrag_webui/workspace.html — the
        # placeholder must exist in both entry files.
        RUNTIME_CONFIG_PLACEHOLDER = b"<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"

        def __init__(
            self,
            *args,
            index_file: str = WEBUI_INDEX_FILENAME,
            blocked_html: tuple[str, ...] = (),
            **kwargs,
        ):
            """``index_file``: directory-index filename for this mount.

            Starlette's StaticFiles hardcodes ``index.html`` under
            ``html=True`` with no configuration hook, so the workspace mount
            overrides it here to serve workspace.html at its root.

            ``blocked_html``: the OTHER entry's HTML filename(s); requests
            for them return 404 on this mount. Both entry files live in one
            shared directory, so without this each mount would also serve a
            fully working alias of the other entry (/webui/workspace.html,
            /workspace/index.html) and the URL would stop indicating the
            entry. Same-entry explicit filenames (/webui/index.html,
            /workspace/workspace.html) stay reachable.

            Matched case-INSENSITIVELY: on a case-insensitive filesystem
            (macOS, Windows) the underlying StaticFiles would happily serve
            /webui/WORKSPACE.HTML, so an exact-case check would leave the
            alias open exactly where the OS opens it.
            """
            super().__init__(*args, **kwargs)
            self.index_file = index_file
            self.blocked_html = {name.lower() for name in blocked_html}

        async def get_response(self, path: str, scope):
            if path.lower() in self.blocked_html:
                raise HTTPException(status_code=404, detail="Not Found")

            # Rewrite the mount-root request ('' / '.') to this mount's own
            # index file. The Router's redirect_slashes has already forced a
            # trailing slash on the mount root, so this rewrite cannot bypass
            # any redirect.
            if path in ("", "."):
                path = self.index_file

            response = await super().get_response(path, scope)

            # `path` is empty when accessing the mount root (StaticFiles
            # rewrites it to index.html internally) — match on media_type
            # too so we still inject in that case.
            is_html = (
                path.endswith(".html")
                or path == ""
                or path.endswith("/")
                or getattr(response, "media_type", None) == "text/html"
            )

            if (
                is_html
                and getattr(response, "status_code", 0) == 200
                and isinstance(response, FileResponse)
            ):
                response = self._inject_runtime_config(response)

            if is_html:
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            elif path.startswith("assets/") or "/assets/" in path:
                # Assets (JS, CSS, images, fonts) generated by Vite with hash
                # in filename. Inside a Mount, `path` is relative (no leading
                # slash) — "assets/x.js" — so match the top-level assets dir
                # explicitly.
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            # Add other rules here if needed for non-HTML, non-asset files

            # Ensure correct Content-Type
            if path.endswith(".js"):
                response.headers["Content-Type"] = "application/javascript"
            elif path.endswith(".css"):
                response.headers["Content-Type"] = "text/css"

            return response

        def _inject_runtime_config(self, response: FileResponse) -> Response:
            """Replace the runtime-config placeholder in index.html.

            Returns the original FileResponse if the placeholder is absent
            (older build, or a non-index HTML file) — avoids breaking
            previously-working bundles during upgrades.
            """
            try:
                content = Path(response.path).read_bytes()
            except OSError as e:
                logger.warning(
                    "Could not read %s for runtime config injection: %s",
                    response.path,
                    e,
                )
                return response

            if self.RUNTIME_CONFIG_PLACEHOLDER not in content:
                return response

            new_content = content.replace(
                self.RUNTIME_CONFIG_PLACEHOLDER,
                runtime_config_script.encode("utf-8"),
            )
            return Response(content=new_content, media_type="text/html")

    # Mount Swagger UI static files for offline support
    swagger_static_dir = Path(__file__).parent / "static" / "swagger-ui"
    if api_docs_enabled and swagger_static_dir.exists():
        app.mount(
            "/static/swagger-ui",
            StaticFiles(directory=swagger_static_dir),
            name="swagger-ui-static",
        )

    # Conditionally mount each UI entry only if its own entry HTML exists.
    # Both mounts serve the SAME build directory; entry identity is decided
    # by which file each mount uses as its directory index.
    static_dir = Path(__file__).parent / "webui"
    if webui_assets_exist:
        static_dir.mkdir(exist_ok=True)
        app.mount(
            webui_path,
            SmartStaticFiles(
                directory=static_dir,
                html=True,
                check_dir=True,
                index_file=WEBUI_INDEX_FILENAME,
                blocked_html=(WORKSPACE_INDEX_FILENAME,),
            ),
            name="webui",
        )
        logger.info(f"Admin WebUI mounted at {api_prefix}{webui_path}/")
    else:
        logger.info("WebUI assets not available, WebUI route not mounted")

        # Add redirect for WebUI path when assets are not available
        @app.get(webui_path)
        @app.get(f"{webui_path}/")
        async def webui_redirect_to_docs(request: Request):
            """Redirect WebUI path to /docs when WebUI is not available.

            With docs disabled there is no page to redirect to, so answer
            with the JSON service info instead of a 404.
            """
            if not api_docs_enabled:
                return service_info_response(request)
            root = request.scope.get("root_path", "")
            return RedirectResponse(url=f"{root}/docs")

    if workspace_assets_exist:
        app.mount(
            WORKSPACE_PATH,
            SmartStaticFiles(
                directory=static_dir,
                html=True,
                check_dir=True,
                index_file=WORKSPACE_INDEX_FILENAME,
                blocked_html=(WEBUI_INDEX_FILENAME,),
            ),
            name="workspace",
        )
        logger.info(f"Workspace query UI mounted at {api_prefix}{WORKSPACE_PATH}/")
    else:
        logger.info(
            "Workspace entry HTML not available, /workspace serves the "
            "fixed service-info degradation"
        )

        # Fixed degradation, explicitly registered for both trailing-slash
        # forms. Never redirects to /docs (or anywhere else) — see
        # workspace_unavailable_response.
        @app.get(WORKSPACE_PATH)
        @app.get(f"{WORKSPACE_PATH}/")
        async def workspace_unavailable(request: Request):
            return workspace_unavailable_response(request)

    return app


def get_application(args=None):
    """Factory function for creating the FastAPI application"""
    if args is None:
        args = global_args
    return create_app(args)


def configure_logging():
    """Configure logging for uvicorn startup"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    # Get log directory path from environment variable
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

    print(f"\nLightRAG log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
    log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                # Configure all uvicorn related loggers
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
            },
            "filters": {
                "path_filter": {
                    "()": "lightrag.utils.LightragPathFilter",
                },
            },
        }
    )


def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "uvicorn",
        "tiktoken",
        "fastapi",
        # Add other required packages here
    ]

    for package in required_packages:
        if not pm.is_installed(package):
            print(f"Installing {package}...")
            pm.install(package)
            print(f"{package} installed successfully")


def main():
    # On Windows, ProactorEventLoop (default since Python 3.8) has known
    # race conditions with uvicorn's socket binding that can cause the server
    # to report it's running while the port is never actually bound.
    # Using SelectorEventLoop resolves this issue.
    # See: https://github.com/HKUDS/LightRAG/issues/2438
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Explicitly initialize configuration for clarity
    # (The proxy will auto-initialize anyway, but this makes intent clear)
    from .config import initialize_config

    initialize_config()

    # Check if running under Gunicorn
    if "GUNICORN_CMD_ARGS" in os.environ:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check .env file
    if not check_env_file():
        sys.exit(1)

    # Check and install dependencies
    check_and_install_dependencies()

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    configure_logging()
    update_uvicorn_mode_config()
    display_splash_screen(global_args)

    # Note: Signal handlers are NOT registered here because:
    # - Uvicorn has built-in signal handling that properly calls lifespan shutdown
    # - Custom signal handlers can interfere with uvicorn's graceful shutdown
    # - Cleanup is handled by the lifespan context manager's finally block

    # Create application instance directly instead of using factory function
    app = create_app(global_args)

    # Start Uvicorn in single process mode. Do not pass root_path here;
    # the prefix lives only on FastAPI's app.root_path. See
    # docs/MultiSiteDeployment.md.
    uvicorn_config = {
        "app": app,  # Pass application instance directly instead of string path
        "host": global_args.host,
        "port": global_args.port,
        "log_config": None,  # Disable default config
    }

    if global_args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": global_args.ssl_certfile,
                "ssl_keyfile": global_args.ssl_keyfile,
            }
        )

    print(
        f"Starting Uvicorn server in single-process mode on {global_args.host}:{global_args.port}"
    )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()

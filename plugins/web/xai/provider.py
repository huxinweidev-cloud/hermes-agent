"""xAI Web Search — plugin form.

Routes ``web_search`` tool calls through xAI's agentic Web Search tool
(server-side ``web_search`` on the Responses API). Grok runs the actual
searching and page-browsing server-side; we ask it to return the top
results as structured JSON so we can hand back the same
``{title, url, description, position}`` rows every other Hermes web
provider produces.

Reference: https://docs.x.ai/developers/tools/web-search

Config keys this provider responds to::

    web:
      search_backend: "xai"           # explicit per-capability
      backend: "xai"                  # shared fallback

Optional knobs (under ``web.xai`` in ``config.yaml``)::

    web:
      xai:
        model: "grok-4.3"             # reasoning model required by web_search
        allowed_domains: ["x.ai"]     # max 5 — mutually exclusive with excluded_domains
        excluded_domains: ["bad.com"] # max 5 — mutually exclusive with allowed_domains
        timeout: 90                   # seconds (default 90)

Auth: reuses :func:`tools.xai_http.resolve_xai_http_credentials`, which
prefers Hermes-managed xAI Grok OAuth (via ``hermes auth``) and falls back
to ``XAI_API_KEY`` (resolved through ``~/.hermes/.env``, then
``os.environ``).
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider
from tools.xai_http import (
    get_env_value,
    has_xai_credentials,
    hermes_xai_user_agent,
    resolve_xai_http_credentials,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "grok-4.3"
DEFAULT_TIMEOUT = 90
_MAX_DOMAIN_FILTERS = 5  # xAI hard cap on allowed_domains / excluded_domains

# Match the JSON object Grok is asked to emit. Tolerates leading/trailing
# prose since reasoning models occasionally narrate before the JSON block
# even when explicitly asked not to.
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _model_name_for_xai_web_search(raw_model: str) -> str:
    model = str(raw_model or "").strip()
    if model.startswith("@"):
        parts = model.split(":")
        if parts:
            model = parts[-1].strip()
    return model


@dataclass(frozen=True)
class XAIWebEndpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float
    provider: str = "xai"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_xai_web_config() -> Dict[str, Any]:
    """Read ``web.xai`` from config.yaml (returns {} on miss)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        xai_section = web_section.get("xai") if isinstance(web_section, dict) else None
        return xai_section if isinstance(xai_section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load web.xai config: %s", exc)
        return {}


def _coerce_domain_list(value: Any) -> List[str]:
    """Coerce a config value to a clean list of <=5 domain strings."""
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        if len(cleaned) >= _MAX_DOMAIN_FILTERS:
            break
    return cleaned


def _coerce_timeout(value: Any, default: float = DEFAULT_TIMEOUT) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _redact_error_text(value: Any, *, limit: int = 300) -> str:
    text = str(value or "")[:limit]
    for marker in ("Authorization", "Bearer ", "api_key", "token", "secret"):
        if marker.lower() in text.lower():
            return "[REDACTED]"
    return text


def _load_configured_endpoints(cfg: Dict[str, Any]) -> List[XAIWebEndpoint]:
    raw_endpoints = cfg.get("endpoints")
    if not isinstance(raw_endpoints, list):
        return []

    raw_default_model = cfg.get("model") if isinstance(cfg.get("model"), str) else DEFAULT_MODEL
    default_model = _model_name_for_xai_web_search(raw_default_model or DEFAULT_MODEL) or DEFAULT_MODEL
    default_timeout = _coerce_timeout(cfg.get("timeout", DEFAULT_TIMEOUT))
    endpoints: List[XAIWebEndpoint] = []

    for index, raw in enumerate(raw_endpoints):
        if not isinstance(raw, dict):
            continue
        if raw.get("enabled", True) is False:
            continue
        name = str(raw.get("name") or f"endpoint_{index + 1}").strip() or f"endpoint_{index + 1}"
        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        api_key = str(raw.get("api_key") or "").strip()
        api_key_env = str(raw.get("api_key_env") or "").strip()
        if not api_key and api_key_env:
            api_key = str(get_env_value(api_key_env) or "").strip()
        raw_model = raw.get("model") if isinstance(raw.get("model"), str) else default_model
        model = _model_name_for_xai_web_search(raw_model or default_model) or default_model
        timeout = _coerce_timeout(raw.get("timeout", default_timeout), default_timeout)
        if not base_url or not api_key:
            logger.warning("Skipping xAI web endpoint %s: missing base_url or api_key", name)
            continue
        endpoints.append(XAIWebEndpoint(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
        ))
    return endpoints


def _has_configured_xai_endpoint(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Cheaply check whether web.xai.endpoints contains a usable endpoint.

    This intentionally avoids OAuth/runtime-provider resolution. It only reads
    config + env/.env values, so backend availability checks can call it during
    tool registration without triggering network refreshes.
    """
    cfg = cfg if isinstance(cfg, dict) else _load_xai_web_config()
    raw_endpoints = cfg.get("endpoints")
    if not isinstance(raw_endpoints, list):
        return False
    for raw in raw_endpoints:
        if not isinstance(raw, dict):
            continue
        if raw.get("enabled", True) is False:
            continue
        base_url = str(raw.get("base_url") or "").strip()
        api_key = str(raw.get("api_key") or "").strip()
        api_key_env = str(raw.get("api_key_env") or "").strip()
        if not api_key and api_key_env:
            api_key = str(get_env_value(api_key_env) or "").strip()
        if base_url and api_key:
            return True
    fallback = cfg.get("fallback")
    if isinstance(fallback, dict) and fallback.get("enabled") is True:
        if str(fallback.get("type") or "").strip().lower() == "current_model":
            return True
    return False


def _ordered_endpoints(endpoints: List[XAIWebEndpoint], strategy: Any) -> List[XAIWebEndpoint]:
    if not endpoints:
        return []
    strategy_name = str(strategy or "failover").strip().lower()
    if strategy_name == "random_start_failover" and len(endpoints) > 1:
        start = random.randrange(len(endpoints))
        return endpoints[start:] + endpoints[:start]
    return list(endpoints)


def _fallback_current_model_endpoint(cfg: Dict[str, Any]) -> tuple[Optional[XAIWebEndpoint], Optional[Dict[str, Any]]]:
    fallback = cfg.get("fallback")
    if not isinstance(fallback, dict) or fallback.get("enabled") is not True:
        return None, None
    if str(fallback.get("type") or "").strip().lower() != "current_model":
        return None, None
    if fallback.get("require_native_web_search") is not True:
        return None, {
            "endpoint": "current_model",
            "error": "current_model fallback requires require_native_web_search: true",
        }

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider()
    except Exception as exc:  # noqa: BLE001
        return None, {
            "endpoint": "current_model",
            "error": f"Could not resolve current model runtime: {_redact_error_text(exc)}",
        }

    api_key = str(runtime.get("api_key") or "").strip() if isinstance(runtime, dict) else ""
    base_url = str(runtime.get("base_url") or "").strip().rstrip("/") if isinstance(runtime, dict) else ""
    api_mode = str(runtime.get("api_mode") or "").strip().lower() if isinstance(runtime, dict) else ""
    if api_mode != "codex_responses":
        return None, {
            "endpoint": "current_model",
            "error": "current model does not advertise native Responses API web_search support",
        }
    model = _model_name_for_xai_web_search(runtime.get("model") or runtime.get("default") or "") if isinstance(runtime, dict) else ""
    if not model:
        try:
            from hermes_cli.config import load_config

            full_cfg = load_config()
            model_cfg = full_cfg.get("model") if isinstance(full_cfg, dict) else None
            if isinstance(model_cfg, dict):
                model = _model_name_for_xai_web_search(model_cfg.get("default") or model_cfg.get("model") or "")
            elif isinstance(model_cfg, str):
                model = _model_name_for_xai_web_search(model_cfg)
        except Exception:
            model = ""
    timeout = _coerce_timeout(fallback.get("timeout", cfg.get("timeout", DEFAULT_TIMEOUT)))
    if not api_key or not base_url or not model:
        return None, {
            "endpoint": "current_model",
            "error": "current model does not expose native Responses API web_search credentials/model",
        }
    return XAIWebEndpoint(
        name="current_model",
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        provider=str(runtime.get("provider") or "current_model"),
    ), None


def _legacy_endpoint_from_credentials(creds: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[XAIWebEndpoint]:
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        return None
    base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")
    raw_model = cfg.get("model") if isinstance(cfg.get("model"), str) else DEFAULT_MODEL
    model = _model_name_for_xai_web_search(raw_model or DEFAULT_MODEL) or DEFAULT_MODEL
    return XAIWebEndpoint(
        name="default",
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=_coerce_timeout(cfg.get("timeout", DEFAULT_TIMEOUT)),
        provider=str(creds.get("provider") or "xai"),
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class XAIWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by xAI's agentic Web Search tool.

    Sends a structured prompt to Grok with ``tools=[{"type": "web_search"}]``
    enabled and asks it to return the top *limit* results as JSON. Falls
    back to the Responses API ``citations`` list if Grok ignores the JSON
    schema instruction (rare for grok-4.3 but cheap insurance).

    No extract capability — pair with Firecrawl / Tavily / Exa for
    ``web_extract`` if you need page content.

    Trust model
    -----------
    Unlike index-backed providers (Brave / Tavily / Exa) which return
    verbatim search-engine results, this backend is an LLM in a trench
    coat: Grok decides which URLs to surface, generates the titles and
    descriptions itself, and is influenced by the *content of the query*.
    A maliciously crafted query (e.g. injected via untrusted upstream
    input the agent picked up) can in principle steer Grok into emitting
    attacker-chosen URLs. Callers that pipe untrusted text directly into
    ``web_search`` should treat returned URLs the same way they would
    treat any model-generated link — validate before fetching.
    """

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI Web Search (Grok)"

    def is_available(self) -> bool:
        """Cheap availability probe — configured endpoint OR env/auth credentials.

        This deliberately avoids :func:`resolve_xai_http_credentials`: provider
        availability is checked during registry resolution and tool UI refreshes,
        so it must not trigger OAuth refreshes or network calls. Token freshness
        / refresh is handled inside :meth:`search`.
        """
        return _has_configured_xai_endpoint() or has_xai_credentials()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def supports_crawl(self) -> bool:
        return False

    # -- Search -----------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Grok-backed web search.

        Returns ``{"success": True, "data": {"web": [{title, url, description, position}, ...]}}``
        on success, ``{"success": False, "error": str}`` on failure.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:  # noqa: BLE001 — interrupt module is best-effort
            pass

        # Clamp limit to the same range the caller (web_search_tool) accepts,
        # so we don't silently downgrade explicit limits. Grok happily
        # produces longer lists; cost scales linearly with the requested
        # count via reasoning tokens, but that's the caller's call to make.
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 100))

        cfg = _load_xai_web_config()
        allowed = _coerce_domain_list(cfg.get("allowed_domains"))
        excluded = _coerce_domain_list(cfg.get("excluded_domains"))
        if allowed and excluded:
            # xAI explicitly rejects this combo — surface a clear error
            # rather than a 400 from the API.
            return {
                "success": False,
                "error": (
                    "web.xai.allowed_domains and web.xai.excluded_domains "
                    "cannot both be set (xAI restriction)."
                ),
            }

        web_search_tool: Dict[str, Any] = {"type": "web_search"}
        if allowed:
            web_search_tool["filters"] = {"allowed_domains": allowed}
        elif excluded:
            web_search_tool["filters"] = {"excluded_domains": excluded}

        prompt = self._build_prompt(query, limit)

        configured_endpoints = _load_configured_endpoints(cfg)
        endpoints: List[XAIWebEndpoint] = []
        if configured_endpoints:
            endpoints = _ordered_endpoints(configured_endpoints, cfg.get("strategy"))
        else:
            creds = resolve_xai_http_credentials()
            legacy_endpoint = _legacy_endpoint_from_credentials(creds, cfg)
            if legacy_endpoint:
                endpoints = [legacy_endpoint]

        details: List[Dict[str, Any]] = []
        for endpoint in endpoints:
            result = self._search_endpoint(endpoint, prompt, limit, web_search_tool)
            if result.get("success") is True:
                return result
            details.append(result.get("detail") or {
                "endpoint": endpoint.name,
                "error": result.get("error", "unknown error"),
            })

        fallback_endpoint, fallback_error = _fallback_current_model_endpoint(cfg)
        if fallback_endpoint is not None:
            result = self._search_endpoint(fallback_endpoint, prompt, limit, web_search_tool)
            if result.get("success") is True:
                return result
            details.append(result.get("detail") or {
                "endpoint": "current_model",
                "error": result.get("error", "unknown error"),
            })
        elif fallback_error is not None:
            details.append(fallback_error)

        if configured_endpoints:
            error = "All xAI web search endpoints failed"
            if any(d.get("endpoint") == "current_model" for d in details):
                error += ", and current model fallback failed or is unavailable"
            return {"success": False, "error": error, "details": details}

        first_error = details[0].get("error") if details else "xAI web search failed"
        return {"success": False, "error": str(first_error)}

    def _search_endpoint(
        self,
        endpoint: XAIWebEndpoint,
        prompt: str,
        limit: int,
        web_search_tool: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": endpoint.model,
            "input": [{"role": "user", "content": prompt}],
            "tools": [web_search_tool],
            "include": ["no_inline_citations"],
        }

        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        }

        try:
            import httpx
        except ImportError:
            return {
                "success": False,
                "error": "httpx is not installed (required for xAI web search)",
                "detail": {"endpoint": endpoint.name, "error": "httpx is not installed"},
            }

        logger.info(
            "xAI web search via %s: (limit=%d, model=%s, endpoint=%s)",
            endpoint.base_url, limit, endpoint.model, endpoint.name,
        )

        resp = None
        max_attempts = 2 if endpoint.provider == "xai-oauth" else 1
        current_endpoint = endpoint
        for attempt in range(max_attempts):
            headers["Authorization"] = f"Bearer {current_endpoint.api_key}"
            try:
                resp = httpx.post(
                    f"{current_endpoint.base_url}/responses",
                    headers=headers,
                    json=payload,
                    timeout=current_endpoint.timeout,
                )
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 401 and attempt == 0 and current_endpoint.provider == "xai-oauth":
                    logger.info(
                        "xAI web search got 401 on first attempt; forcing OAuth "
                        "refresh and retrying once.",
                    )
                    try:
                        refreshed = resolve_xai_http_credentials(force_refresh=True)
                        refreshed_key = str(refreshed.get("api_key") or "").strip()
                        if refreshed_key and refreshed_key != current_endpoint.api_key:
                            refreshed_base_url = str(
                                refreshed.get("base_url") or current_endpoint.base_url
                            ).strip().rstrip("/")
                            current_endpoint = XAIWebEndpoint(
                                name=current_endpoint.name,
                                base_url=refreshed_base_url or current_endpoint.base_url,
                                api_key=refreshed_key,
                                model=current_endpoint.model,
                                timeout=current_endpoint.timeout,
                                provider=str(refreshed.get("provider") or current_endpoint.provider),
                            )
                            continue
                    except Exception as refresh_exc:  # noqa: BLE001
                        logger.warning(
                            "xAI web search OAuth refresh after 401 failed: %s",
                            _redact_error_text(refresh_exc),
                        )
                body = ""
                try:
                    body = _redact_error_text(exc.response.text if exc.response is not None else "")
                except Exception:
                    body = ""
                logger.warning("xAI web search endpoint %s HTTP %d: %s", current_endpoint.name, status, body)
                error = f"xAI web search returned HTTP {status}: {body}".rstrip()
                return {
                    "success": False,
                    "error": error,
                    "detail": {"endpoint": current_endpoint.name, "status": status, "error": error},
                }
            except httpx.RequestError as exc:
                message = _redact_error_text(exc)
                logger.warning("xAI web search endpoint %s request error: %s", current_endpoint.name, message)
                return {
                    "success": False,
                    "error": f"Could not reach xAI: {message}",
                    "detail": {"endpoint": current_endpoint.name, "error": message},
                }

        if resp is None:
            return {
                "success": False,
                "error": "xAI web search produced no response",
                "detail": {"endpoint": endpoint.name, "error": "no response"},
            }

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            message = _redact_error_text(exc)
            logger.warning("xAI web search endpoint %s bad JSON: %s", endpoint.name, message)
            return {
                "success": False,
                "error": "Could not parse xAI Responses API reply as JSON",
                "detail": {"endpoint": endpoint.name, "error": "bad JSON"},
            }

        api_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(api_error, dict):
            err_msg = (
                api_error.get("message")
                or api_error.get("code")
                or "unknown error"
            )
            err_msg = _redact_error_text(err_msg)
            logger.warning("xAI web search endpoint %s returned error envelope: %s", endpoint.name, err_msg)
            return {
                "success": False,
                "error": f"xAI returned an error: {err_msg}",
                "detail": {"endpoint": endpoint.name, "error": err_msg},
            }

        web_results = self._extract_results(data, limit=limit)
        if not web_results:
            return {"success": True, "data": {"web": []}}

        return {"success": True, "data": {"web": web_results}}

    # -- Prompt + parsing -------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, limit: int) -> str:
        """Compose the prompt that asks Grok to act as a search engine.

        We deliberately ask for a JSON object (not bare array) so we can
        match it cheaply with ``_JSON_BLOCK_RE``; we explicitly forbid
        prose, markdown fences, and inline-citation links to keep the
        payload parseable.
        """
        return (
            "Use the web_search tool to find current information for the query below, "
            "then respond with ONLY a single JSON object — no prose, no markdown "
            "fences, no inline citation links — matching this exact schema:\n\n"
            '{"results": [{"title": "string", "url": "string", '
            '"description": "1-2 sentence summary"}]}\n\n'
            f'Return at most {limit} results, ordered by relevance, with absolute '
            "https:// URLs. If no usable results exist, return "
            '{"results": []}.\n\n'
            f"Query: {query}"
        )

    @classmethod
    def _extract_results(
        cls,
        response_data: Dict[str, Any],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Pull a ``[{title, url, description, position}, ...]`` list out of a
        Responses-API reply.

        Strategy:

        1. Walk ``output[*].content[*].text`` for ``output_text`` blocks and
           try to parse the first JSON object that has a ``results`` list.
        2. If the JSON path fails, fall back to the message annotations
           (``url_citation`` entries) — every annotation carries a URL and
           a ``title`` (citation number); we pair those URLs with surrounding
           text from the message body as a best-effort description.
        """
        text_blocks, annotations = cls._collect_output_text(response_data)

        # Primary path: parse the JSON object Grok was asked for.
        for block in text_blocks:
            parsed = cls._try_parse_json_results(block, limit=limit)
            if parsed:
                return parsed

        # Secondary path: derive results from message annotations + raw text.
        # Only short-circuit when annotations actually yielded usable rows;
        # otherwise fall through to the citations list. (xAI currently only
        # emits ``url_citation`` annotations, but future annotation types
        # would silently produce an empty result set if we returned here
        # unconditionally — masking real data in ``citations``.)
        if annotations:
            joined_text = "\n".join(text_blocks)
            annotation_results = cls._results_from_annotations(
                annotations, joined_text, limit=limit,
            )
            if annotation_results:
                return annotation_results

        # Last-ditch: raw citations list (no titles or descriptions).
        citations = response_data.get("citations") or []
        if isinstance(citations, list):
            return [
                {
                    "title": "",
                    "url": str(u),
                    "description": "",
                    "position": i + 1,
                }
                for i, u in enumerate(citations[:limit])
                if isinstance(u, str) and u.strip()
            ]

        return []

    @staticmethod
    def _collect_output_text(
        response_data: Dict[str, Any],
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Return (text_blocks, annotations) extracted from ``response.output``."""
        text_blocks: List[str] = []
        annotations: List[Dict[str, Any]] = []
        output = response_data.get("output")
        if not isinstance(output, list):
            return text_blocks, annotations

        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict) or chunk.get("type") != "output_text":
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text.strip():
                    text_blocks.append(text)
                chunk_annotations = chunk.get("annotations")
                if isinstance(chunk_annotations, list):
                    for ann in chunk_annotations:
                        if isinstance(ann, dict):
                            annotations.append(ann)
        return text_blocks, annotations

    @staticmethod
    def _try_parse_json_results(
        text: str,
        *,
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse a JSON object with a ``results`` array out of ``text``.

        Returns the normalized result list on success, ``None`` when the
        block has no valid JSON object or no ``results`` key. Tolerates
        leading/trailing prose because reasoning models sometimes prefix a
        short narration even when told not to.
        """
        # Try the whole string first — cheapest path when Grok obeys.
        candidates = [text]
        match = _JSON_BLOCK_RE.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            results = parsed.get("results")
            if not isinstance(results, list):
                continue
            normalized: List[Dict[str, Any]] = []
            for row in results[:limit]:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url", "")).strip()
                if not url:
                    continue
                normalized.append(
                    {
                        "title": str(row.get("title", "")).strip(),
                        "url": url,
                        "description": str(row.get("description", "")).strip(),
                        # Renumber from the kept results, not the raw input
                        # index, so a dropped malformed row doesn't leave a
                        # gap in the positions handed back to the agent.
                        "position": len(normalized) + 1,
                    }
                )
            if normalized:
                return normalized
        return None

    @staticmethod
    def _results_from_annotations(
        annotations: List[Dict[str, Any]],
        joined_text: str,
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Best-effort fallback when JSON parsing fails.

        Uses each ``url_citation`` annotation's ``url`` (the citation
        title is just the integer label, so we don't surface it) and
        slices ~200 characters of surrounding text as the description.
        """
        seen: set[str] = set()
        results: List[Dict[str, Any]] = []
        for ann in annotations:
            if ann.get("type") != "url_citation":
                continue
            url = str(ann.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)

            description = ""
            start = ann.get("start_index")
            end = ann.get("end_index")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(joined_text):
                window_start = max(0, start - 200)
                description = joined_text[window_start:start].strip()
                if len(description) > 200:
                    description = description[-200:].strip()

            results.append(
                {
                    "title": "",
                    "url": url,
                    "description": description,
                    "position": len(results) + 1,
                }
            )
            if len(results) >= limit:
                break
        return results

    # -- Setup picker -----------------------------------------------------

    def get_setup_schema(self) -> Dict[str, Any]:
        # Auth resolution is delegated to the shared ``xai_grok`` post_setup
        # hook (same one image_gen.xai and tts.xai use) so users see the
        # familiar OAuth-or-API-key prompt for every xAI service.
        return {
            "name": "xAI Web Search (Grok)",
            "badge": "paid",
            "tag": (
                "Agentic web search via Grok's web_search tool — uses xAI "
                "Grok OAuth or XAI_API_KEY."
            ),
            "env_vars": [],
            "post_setup": "xai_grok",
        }

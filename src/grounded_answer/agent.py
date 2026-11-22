"""Web-grounded answers via the Perplexity Agent API (``POST /v1/agent``).

The SDK (``perplexityai``) reads ``PERPLEXITY_API_KEY`` from the environment and
retries 408/409/429/5xx with backoff, honouring ``Retry-After``. This module
adds a thin, typed wrapper that returns the answer text plus de-duplicated sources.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import perplexity
from perplexity import Perplexity

API_KEY_ENV = "PERPLEXITY_API_KEY"
DEFAULT_EFFORT = "low"
WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search"}


class AgentError(RuntimeError):
    """Raised for any failure talking to the Agent API. Never contains the API key."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MissingAPIKeyError(AgentError):
    def __init__(self) -> None:
        super().__init__(
            f"{API_KEY_ENV} is not set. Create a key at https://console.perplexity.ai "
            f"and export it in your shell: export {API_KEY_ENV}=..."
        )


@dataclass(frozen=True)
class Source:
    url: str
    title: str | None = None
    date: str | None = None


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    sources: list[Source]
    response_id: str
    model: str
    status: str
    total_cost_usd: float | None = None
    search_queries: list[str] = field(default_factory=list)


def make_client(**kwargs: Any) -> Perplexity:
    """Build an SDK client, failing early with an actionable message if the key is absent."""
    if "api_key" not in kwargs and not os.environ.get(API_KEY_ENV):
        raise MissingAPIKeyError()
    return Perplexity(**kwargs)


def ask(
    question: str,
    *,
    client: Perplexity | None = None,
    effort: str | None = None,
    model: str | None = None,
    instructions: str | None = None,
    previous_response_id: str | None = None,
    response_format: Mapping[str, Any] | None = None,
    max_output_tokens: int | None = None,
) -> GroundedAnswer:
    """Answer ``question`` with web grounding.

    Pass ``model`` (e.g. ``"openai/gpt-5.6-sol"``) to pick a model directly; the
    ``web_search`` tool is then added explicitly, since web search is opt-in for
    direct-model requests. Otherwise an effort level (default ``"low"``) is used,
    which bundles model, tools and limits. ``effort`` and ``model`` are exclusive.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if effort and model:
        raise ValueError("pass either effort or model, not both")

    params: dict[str, Any] = {"input": question}
    if model:
        params["model"] = model
        params["tools"] = [WEB_SEARCH_TOOL]
    else:
        params["effort"] = effort or DEFAULT_EFFORT
    if instructions:
        params["instructions"] = instructions
    if previous_response_id:
        params["previous_response_id"] = previous_response_id
    if response_format:
        params["response_format"] = dict(response_format)
    if max_output_tokens:
        params["max_output_tokens"] = max_output_tokens

    sdk = client or make_client()
    try:
        response = sdk.responses.create(**params)
    except perplexity.AuthenticationError as exc:
        raise AgentError(
            "Authentication failed (401): check PERPLEXITY_API_KEY; rotate it at "
            "https://console.perplexity.ai if it may have been exposed.",
            status_code=exc.status_code,
        ) from None
    except perplexity.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after")
        hint = f" Retry after {retry_after}s." if retry_after else ""
        raise AgentError(
            f"Rate limited (429) after SDK retries.{hint}", status_code=exc.status_code
        ) from None
    except perplexity.APIStatusError as exc:
        raise AgentError(
            f"Agent API error {exc.status_code}: {exc.message}", status_code=exc.status_code
        ) from None
    except perplexity.APIConnectionError as exc:
        raise AgentError(f"Could not reach the Agent API: {exc.message}") from None

    return to_grounded_answer(response)


def to_grounded_answer(response: Any) -> GroundedAnswer:
    """Map an Agent API response to ``GroundedAnswer``.

    Output items are dispatched on their ``type`` discriminator rather than SDK
    classes: the SDK's runtime models (``perplexity.generated.api``) are not the
    classes exported from ``perplexity.types``.
    """
    if response.status != "completed":
        detail = response.error.message if response.error else "no error detail"
        raise AgentError(f"Response {response.id} ended with status {response.status}: {detail}")

    sources: dict[str, Source] = {}
    queries: list[str] = []
    for item in response.output:
        if item.type == "search_results":
            queries.extend(item.queries or [])
            for result in item.results:
                sources.setdefault(
                    result.url, Source(url=result.url, title=result.title, date=result.date)
                )
        elif item.type == "message":
            for part in item.content or []:
                for ann in part.annotations or []:
                    if ann.url:
                        sources.setdefault(ann.url, Source(url=ann.url, title=ann.title))

    usage = response.usage
    cost = usage.cost.total_cost if usage and usage.cost else None
    return GroundedAnswer(
        text=str(response.output_text),
        sources=list(sources.values()),
        response_id=response.id,
        model=response.model,
        status=response.status,
        total_cost_usd=cost,
        search_queries=queries,
    )

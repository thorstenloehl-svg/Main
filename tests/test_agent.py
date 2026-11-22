from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from perplexity import Perplexity

from grounded_answer import AgentError, MissingAPIKeyError, ask, make_client
from grounded_answer.cli import main

RESPONSE: dict[str, Any] = {
    "id": "resp_123",
    "created_at": 1771891737,
    "model": "openai/gpt-5.6-sol",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "search_results",
            "queries": ["capital of france"],
            "results": [
                {
                    "id": 1,
                    "title": "Paris - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "snippet": "Paris is the capital",
                    "date": "2026-01-01",
                    "source": "web",
                },
            ],
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Paris is the capital of France.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "https://en.wikipedia.org/wiki/Paris",
                            "title": "Paris",
                        },
                        {"type": "url_citation", "url": "https://www.britannica.com/place/Paris"},
                    ],
                }
            ],
        },
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 8,
        "total_tokens": 18,
        "cost": {"currency": "USD", "input_cost": 0.001, "output_cost": 0.002, "total_cost": 0.003},
    },
}


def client_with(handler: Any) -> tuple[Perplexity, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        result: httpx.Response = handler(request)
        return result

    http = httpx.Client(transport=httpx.MockTransport(wrapped))
    return Perplexity(api_key="test-key", http_client=http, max_retries=0), seen


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=RESPONSE)


def test_effort_request_and_parsing() -> None:
    client, seen = client_with(ok)
    result = ask("Capital of France?", client=client)

    # SDK uses /v1/responses, the documented alias of POST /v1/agent
    assert seen[0].url.path in {"/v1/agent", "/v1/responses"}
    assert json.loads(seen[0].content) == {"input": "Capital of France?", "effort": "low"}
    assert result.text == "Paris is the capital of France."
    assert [s.url for s in result.sources] == [
        "https://en.wikipedia.org/wiki/Paris",
        "https://www.britannica.com/place/Paris",
    ]
    assert result.sources[0].title == "Paris - Wikipedia"
    assert result.search_queries == ["capital of france"]
    assert result.total_cost_usd == 0.003


def test_model_mode_adds_web_search_and_multiturn() -> None:
    client, seen = client_with(ok)
    ask(
        "Follow-up?",
        client=client,
        model="openai/gpt-5.6-sol",
        previous_response_id="resp_0",
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
    )
    body = json.loads(seen[0].content)
    assert body["model"] == "openai/gpt-5.6-sol"
    assert body["tools"] == [{"type": "web_search"}]
    assert body["previous_response_id"] == "resp_0"
    assert body["response_format"]["type"] == "json_schema"
    assert "effort" not in body


def test_effort_and_model_are_exclusive() -> None:
    with pytest.raises(ValueError):
        ask("q", effort="low", model="openai/gpt-5.6-sol", client=client_with(ok)[0])


@pytest.mark.parametrize(
    ("status", "headers", "fragment"),
    [
        (401, {}, "Authentication failed"),
        (429, {"retry-after": "7"}, "Retry after 7s"),
        (400, {}, "Agent API error 400"),
    ],
)
def test_http_errors(status: int, headers: dict[str, str], fragment: str) -> None:
    client, _ = client_with(
        lambda _r: httpx.Response(status, json={"error": {"message": "boom"}}, headers=headers)
    )
    with pytest.raises(AgentError, match=fragment) as info:
        ask("q", client=client)
    assert info.value.status_code == status
    assert "test-key" not in str(info.value)


def test_failed_status_raises() -> None:
    client, _ = client_with(
        lambda _r: httpx.Response(
            200,
            json={
                **RESPONSE,
                "status": "failed",
                "output": [],
                "error": {"message": "model overloaded"},
            },
        )
    )
    with pytest.raises(AgentError, match="model overloaded"):
        ask("q", client=client)


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="console.perplexity.ai"):
        make_client()


def test_cli_missing_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    assert main(["hello"]) == 1
    assert "PERPLEXITY_API_KEY is not set" in capsys.readouterr().err

# grounded-answer

Web-grounded answers with source citations via the Perplexity Agent API (`POST https://api.perplexity.ai/v1/agent`, Python SDK `perplexityai`).

## Setup

```bash
uv sync
export PERPLEXITY_API_KEY=...   # create at https://console.perplexity.ai — never commit it
```

## Usage

```bash
uv run grounded-answer "What changed in the EU PPWR in 2026?"                 # effort=low (default)
uv run grounded-answer "..." --effort medium                                  # fast|low|medium|high|xhigh|ultra
uv run grounded-answer "..." --model openai/gpt-5.6-sol                       # direct model + web_search tool
uv run grounded-answer "And in Austria?" --previous-response-id resp_...      # multi-turn
uv run grounded-answer "..." --json
```

```python
from grounded_answer import ask

result = ask("Latest polyolefin recycling capacity announcements in Europe?", effort="low")
print(result.text)
for s in result.sources:
    print(s.title, s.url)
```

Structured output: pass `response_format={"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` and parse `result.text`.

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run python scripts/smoke_test.py   # live request; prints HTTP status + response shape only
```

Errors: 401 → key problem (rotate in console if exposed); 429 → rate limited. The SDK retries 429/5xx (default 2 retries, honours `Retry-After`) before `AgentError` is raised.

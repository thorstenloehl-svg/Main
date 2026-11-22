"""Minimal live request. Prints HTTP status and response shape only — never the key."""

from __future__ import annotations

import os
import sys

import perplexity
from perplexity import Perplexity


def main() -> int:
    if not os.environ.get("PERPLEXITY_API_KEY"):
        print("PERPLEXITY_API_KEY is not set (create one at https://console.perplexity.ai)")
        return 2
    client = Perplexity(max_retries=0)
    try:
        raw = client.responses.with_raw_response.create(
            effort="fast", input="What is the capital of France?"
        )
    except perplexity.APIStatusError as exc:
        print(f"HTTP {exc.status_code}")
        if exc.status_code == 429:
            print(f"retry-after: {exc.response.headers.get('retry-after')}")
        return 1
    print(f"HTTP {raw.status_code}")
    resp = raw.parse()
    print(f"status={resp.status} model={resp.model} output_types={[o.type for o in resp.output]}")
    print(f"output_text_chars={len(resp.output_text)}")
    return 0 if raw.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())

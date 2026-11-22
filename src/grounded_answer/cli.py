"""CLI: ``grounded-answer "question" [--effort low | --model provider/model] [--json]``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

from .agent import DEFAULT_EFFORT, AgentError, ask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grounded-answer", description=__doc__)
    p.add_argument("question")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--effort",
        help=f"effort level (fast|low|medium|high|xhigh|ultra; default {DEFAULT_EFFORT})",
    )
    mode.add_argument("--model", help='direct model, e.g. "openai/gpt-5.6-sol" (adds web_search)')
    p.add_argument("--previous-response-id", help="continue a prior response (multi-turn)")
    p.add_argument("--max-output-tokens", type=int)
    p.add_argument("--json", action="store_true", help="emit the full result as JSON")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = ask(
            args.question,
            effort=args.effort,
            model=args.model,
            previous_response_id=args.previous_response_id,
            max_output_tokens=args.max_output_tokens,
        )
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        return 0

    print(result.text)
    if result.sources:
        print("\nSources:")
        for i, src in enumerate(result.sources, 1):
            print(f"  [{i}] {src.title or src.url} — {src.url}")
    print(f"\n(response_id={result.response_id}, model={result.model})", file=sys.stderr)
    return 0
